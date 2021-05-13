import argparse
from pathlib import Path

import yaml
import wandb
import sys
import shutil
import numpy as np
import soundfile as sf
from collections import OrderedDict
from argparse import Namespace

import src.training_manager as manager
import torch
import torch.nn as nn
import torch.nn.functional as F

from datetime import datetime

import download_weights

# melgan
import melgan_models

# hifigan
import hifi_models


def read_yaml(fp):
    with open(fp) as file:
        # return yaml.load(file)
        return yaml.load(file, Loader=yaml.Loader)


class RCBlock(nn.Module):
    def __init__(self, feat_dim, ks, dilation, num_groups):
        super().__init__()
        # ks = 3  # kernel size
        ksm1 = ks - 1
        mfd = feat_dim
        di = dilation
        self.num_groups = num_groups

        self.relu = nn.LeakyReLU()

        self.rec = nn.GRU(mfd, mfd, num_layers=1, batch_first=True, bidirectional=True)
        self.conv = nn.Conv1d(
            mfd, mfd, ks, 1, ksm1 * di // 2, dilation=di, groups=num_groups
        )
        self.gn = nn.GroupNorm(num_groups, mfd)

    def init_hidden(self, batch_size, hidden_size):
        num_layers = 1
        num_directions = 2
        hidden = torch.zeros(num_layers * num_directions, batch_size, hidden_size)
        hidden.normal_(0, 1)
        return hidden

    def forward(self, x):
        bs, mfd, nf = x.size()

        hidden = self.init_hidden(bs, mfd).to(x.device)

        r = x.transpose(1, 2)
        r, _ = self.rec(r, hidden)
        r = r.transpose(1, 2).view(bs, 2, mfd, nf).sum(1)
        c = self.relu(self.gn(self.conv(r)))
        x = x + r + c

        return x


class BodyGBlock(nn.Module):
    def __init__(self, input_dim, output_dim, middle_dim, num_groups):
        super().__init__()

        ks = 3  # filter size
        mfd = middle_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mfd = mfd
        self.num_groups = num_groups

        # ### Main body ###
        block = [
            nn.Conv1d(input_dim, mfd, 3, 1, 1),
            nn.GroupNorm(num_groups, mfd),
            nn.LeakyReLU(),
            RCBlock(mfd, ks, dilation=1, num_groups=num_groups),
            nn.Conv1d(mfd, output_dim, 3, 1, 1),
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x):

        # ### Main ###
        x = self.block(x)

        return x


class HierarchicalGenerator(nn.Module):
    def __init__(self, feat_dim, z_dim, z_scale_factors):
        super().__init__()

        # ks = 3  # filter size
        mfd = 512
        num_groups = 4
        self.num_groups = num_groups
        self.mfd = mfd

        self.feat_dim = feat_dim
        self.z_dim = z_dim
        self.z_scale_factors = z_scale_factors

        # ### Main body ###
        self.block0 = BodyGBlock(z_dim, mfd, mfd, num_groups)
        self.head0 = nn.Conv1d(mfd, feat_dim, 3, 1, 1)

        blocks = []
        heads = []
        for scale_factor in z_scale_factors:
            block = BodyGBlock(mfd, mfd, mfd, num_groups)
            blocks.append(block)

            head = nn.Conv1d(mfd, feat_dim, 3, 1, 1)
            heads.append(head)

        self.blocks = nn.ModuleList(blocks)
        self.heads = nn.ModuleList(heads)

        # ### Head ###
        # self.head = nn.Conv1d(mfd, feat_dim, 3, 1, 1)

    def forward(self, z):

        # SBlock0
        z_scale_factors = self.z_scale_factors
        # nf = min(z.size(2), cond_.size(2))
        # zc = torch.cat([z[:, :, :nf], cond_[:, :, :nf]], dim=1)
        x_body = self.block0(z)
        x_head = self.head0(x_body)

        # print(len(self.blocks))
        for ii, (block, head, scale_factor) in enumerate(
            zip(self.blocks, self.heads, z_scale_factors)
        ):
            x_body = F.interpolate(x_body, scale_factor=scale_factor, mode="nearest")
            x_head = F.interpolate(x_head, scale_factor=scale_factor, mode="nearest")

            # print(total_scale_factor, x.shape, cond_.shape)
            # nf = min(x.size(2), cond_.size(2))
            # c = torch.cat([x[:, :, :nf], cond_[:, :, :nf]], dim=1)

            x_body = x_body + block(x_body)

            x_head = x_head + head(x_body)

        # Head
        # shape=(bs, feat_dim, nf)
        # x = torch.sigmoid(self.head(x))
        # x = torch.sigmoid(x)

        return x_head


class NonHierarchicalGenerator(nn.Module):
    def __init__(self, feat_dim, z_dim):
        super().__init__()

        ks = 3  # filter size
        mfd = 512
        num_groups = 4

        self.num_groups = num_groups
        self.mfd = mfd

        self.feat_dim = feat_dim
        self.z_dim = z_dim

        # ### Main body ###
        blocks = [
            nn.Conv1d(z_dim, mfd, 3, 1, 1),
            nn.GroupNorm(num_groups, mfd),
            nn.LeakyReLU(),
            RCBlock(mfd, ks, dilation=2, num_groups=num_groups),
            RCBlock(mfd, ks, dilation=4, num_groups=num_groups),
        ]
        self.body = nn.Sequential(*blocks)

        # ### All heads ###
        self.head = nn.Conv1d(mfd, feat_dim, 3, 1, 1)

    def forward(self, z):

        # Body
        x = self.body(z)

        # Head
        # shape=(bs, feat_dim, nf)
        x = self.head(x)

        return x


def main(
    output_folder=None,
    duration=10,
    num_samples=5,
    gid=1,
    seed=123,
    melgan_run_id=None,
    unagan_run_id=None,
    hifigan_run_id=None,
):

    if melgan_run_id and hifigan_run_id:
        raise Exception("Can only set one of [melgan_run_id, hifigan_run_id], not both")

    if not unagan_run_id:
        raise Exception("unagan_run_id should not be empty")

    download_weights.main(
        model_dir=Path("models/custom"),
        melgan_run_id=melgan_run_id,
        unagan_run_id=unagan_run_id,
        hifigan_run_id=hifigan_run_id,
    )
    # ### Data type ###
    # assert data_type in ["singing", "speech", "piano", "violin"]

    # ### Architecture type ###
    # if data_type == "singing":
    #     assert arch_type in ["nh", "h", "hc"]
    # elif data_type == "speech":
    #     assert arch_type in ["h", "hc"]
    # elif data_type == "piano":
    #     assert arch_type in ["hc"]
    # elif data_type == "violin":
    #     assert arch_type in ["hc"]

    # if arch_type == "nh":
    #     arch_type = "nonhierarchical"
    # elif arch_type == "h":
    #     arch_type = "hierarchical"
    # elif arch_type == "hc":
    #     arch_type = "hierarchical_with_cycle"

    data_type = "custom"
    arch_type = "hierarchical_with_cycle"

    # ### Model type ###
    model_type = f"{data_type}.{arch_type}"

    # ### Model info ###
    if output_folder is None:
        output_folder = Path("generated_samples") / model_type
    output_folder = Path(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)

    # also save to all_generated_audio_dir is the folder exists,
    # but do not save if in unagan training (both run id None)
    if not unagan_run_id:
        all_generated_audio_dir = None
    else:
        try:
            all_generated_audio_dir = Path(
                "/content/drive/My Drive/PUBLICATIONS/The Replicant/AUDIO DATABASE/UNAGAN OUTPUT/AUDIOS/"
            )
            all_generated_audio_dir.mkdir(parents=True, exist_ok=True)
            print(
                "generated audio files will also saved to:",
                str(all_generated_audio_dir),
            )
        except:
            all_generated_audio_dir = None
            print(
                "the path '",
                str(all_generated_audio_dir),
                "' not exists. Only save audio files to:",
                str(output_folder),
            )

    api = wandb.Api()
    previous_run = api.run(f"demiurge/unagan/{unagan_run_id}")
    unagan_config = Namespace(**previous_run.config)

    ################# unagan config parameters ##################
    z_dim = unagan_config.z_dim
    z_scale_factors = unagan_config.z_scale_factors
    z_total_scale_factor = np.prod(z_scale_factors)
    feat_dim = unagan_config.feat_dim
    ##################

    param_fp = f"models/{data_type}/params.generator.{arch_type}.pt"

    mean_fp = f"models/{data_type}/mean.mel.npy"
    std_fp = f"models/{data_type}/std.mel.npy"

    mean = torch.from_numpy(np.load(mean_fp)).float().view(1, feat_dim, 1)
    std = torch.from_numpy(np.load(std_fp)).float().view(1, feat_dim, 1)
    if gid >= 0:
        mean = mean.cuda(gid)
        std = std.cuda(gid)

    ###############################################################
    ### Vocoder info ###

    ### MELGAN ###
    if melgan_run_id:
        api = wandb.Api()
        previous_run = api.run(f"demiurge/melgan/{melgan_run_id}")
        melgan_config = Namespace(**previous_run.config)

        ################# melgan config parameters ##################
        # melgan only parameters
        n_mel_channels = 80
        ngf = 32
        n_residual_layers = 3

        # also applied to unagan generate
        sampling_rate = 44100
        hop_length = 256

        if n_mel_channels != feat_dim:
            print(
                f"Warning!!! melgan n_mel_channels {n_mel_channels} != unagan feat_dim {feat_dim}"
            )

        if hasattr(melgan_config, "hop_length"):
            hop_length = melgan_config.hop_length
        if hasattr(melgan_config, "sampling_rate"):
            sampling_rate = melgan_config.sampling_rate
        if hasattr(melgan_config, "n_mel_channels"):
            n_mel_channels = melgan_config.n_mel_channels
        if hasattr(melgan_config, "ngf"):
            ngf = melgan_config.ngf
        if hasattr(melgan_config, "n_residual_layers"):
            n_residual_layers = melgan_config.n_residual_layers

        ########################

        # ### Vocoder Model ###
        vocoder_model_dir = Path("models") / data_type / "vocoder"

        if data_type == "speech":
            vocoder_name = "OriginalGenerator"
        else:
            vocoder_name = "GRUGenerator"
        MelGAN = getattr(melgan_models, vocoder_name)
        vocoder = MelGAN(n_mel_channels, ngf, n_residual_layers)
        vocoder.eval()

        vocoder_param_fp = vocoder_model_dir / "params.pt"
        vocoder_state_dict = torch.load(vocoder_param_fp)
        try:
            vocoder.load_state_dict(vocoder_state_dict)
        except RuntimeError as e:
            print(e)
            print("Fixing model by removing .module prefix")
            vocoder_state_dict = OrderedDict(
                (k.split(".", 1)[1], v) for k, v in vocoder_state_dict.items()
            )
            vocoder.load_state_dict(vocoder_state_dict)

        if gid >= 0:
            vocoder = vocoder.cuda(gid)

    ### HIFI-GAN ###

    if hifigan_run_id:
        api = wandb.Api()
        previous_run = api.run(f"demiurge/hifi-gan/{hifigan_run_id}")
        hifigan_config = Namespace(**previous_run.config)

        # parameters applied to unagan generate
        sampling_rate = 44100
        hop_length = 256
        if hasattr(hifigan_config, "hop_size"):
            hop_length = hifigan_config.hop_size
        if hasattr(hifigan_config, "sampling_rate"):
            sampling_rate = hifigan_config.sampling_rate

        vocoder_model_dir = Path("models") / data_type / "vocoder"
        vocoder = hifi_models.Generator(hifigan_config)
        vocoder.eval()

        vocoder_state_dict = torch.load(vocoder_model_dir / "g")
        vocoder.load_state_dict(vocoder_state_dict["generator"])

        if gid >= 0:
            vocoder = vocoder.cuda(gid)

    ###################################################################

    # ### Generator ###
    if arch_type == "nonhierarchical":
        generator = NonHierarchicalGenerator(feat_dim, z_dim)
    elif arch_type.startswith("hierarchical"):
        generator = HierarchicalGenerator(feat_dim, z_dim, z_scale_factors)

    generator.eval()
    for p in generator.parameters():
        p.requires_grad = False

    manager.load_model(param_fp, generator, device_id="cpu")

    if gid >= 0:
        generator = generator.cuda(gid)

    # ### Process ###
    torch.manual_seed(seed)
    # information for filename
    filename_base = datetime.utcnow().strftime("%Y-%m-%d_%H-%M")

    if melgan_run_id:
        filename_base += "_mel-" + melgan_run_id

    if unagan_run_id:
        filename_base += "_una-" + unagan_run_id

    if hifigan_run_id:
        filename_base += "_hifi-" + hifigan_run_id

    num_frames = int(np.ceil(duration * (sampling_rate / hop_length)))

    audio_array = []
    for ii in range(num_samples):
        out_fp_wav = Path(output_folder) / f"{filename_base}_sample{ii}.wav"
        print(f"Generating {out_fp_wav}")

        if arch_type == "nonhierarchical":
            z = torch.zeros((1, z_dim, num_frames)).normal_(0, 1).float()
        elif arch_type.startswith("hierarchical"):
            z = (
                torch.zeros((1, z_dim, int(np.ceil(num_frames / z_total_scale_factor))))
                .normal_(0, 1)
                .float()
            )

        if gid >= 0:
            z = z.cuda(gid)

        with torch.set_grad_enabled(False):
            with torch.cuda.device(gid):
                # Generator
                melspec_voc = generator(z)
                melspec_voc = (melspec_voc * std) + mean

                # Vocoder
                audio = vocoder(melspec_voc)
                audio = audio.squeeze().cpu().numpy()

        # keep generated audio as array to log to wandb
        if not unagan_run_id:
            audio_array.append(audio)
        else:
            # Save to wav
            sf.write(out_fp_wav, audio, sampling_rate)
            # Save also to all_generated_audio_dir
            if all_generated_audio_dir:
                out2_fp_wav = (
                    Path(all_generated_audio_dir) / f"{filename_base}_sample{ii}.wav"
                )
                sf.write(out2_fp_wav, audio, sampling_rate)
    return audio_array, sampling_rate


def parse_argument():
    parser = argparse.ArgumentParser(
        description="Uncondtional Singing Voice Generation"
    )

    parser.add_argument(
        "--data_type",
        "-d",
        dest="data_type",
        default="custom",
        help='Data type. Options: "custom"(Default)|"speech"|"piano"|"violin"',
    )

    parser.add_argument(
        "--arch_type",
        "-a",
        dest="arch_type",
        default="hc",
        help='Architecture type. Options: \
        "nh" for non-hierarchical, available to singing|\
        "h" for hierarchical, available to singing and speech|\
        "hc" (Default) for hierarchical with cycle, available to all',
    )

    parser.add_argument(
        "--output_folder",
        "-o",
        dest="output_folder",
        default=None,
        help="Output folder",
    )

    parser.add_argument(
        "--duration",
        dest="duration",
        default=10,
        help="Sample duration (second)",
        type=float,
    )

    parser.add_argument(
        "--num_samples",
        "-ns",
        dest="num_samples",
        default=5,
        help="Number of samples to be generated",
        type=int,
    )

    parser.add_argument(
        "--gid",
        dest="gid",
        default=-1,
        type=int,
        help="GPU id. Default: -1 for using cpu",
    )

    parser.add_argument(
        "--seed", dest="seed", default=123, help="Random seed. Default: 123"
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_argument()

    main(**vars(args))
