#!/usr/bin/env ipython
import shutil
import tempfile
from argparse import ArgumentParser
from pathlib import Path

import wandb

MODEL_PATHS = {
    "unagan": [
        (
            "model/params.Generator.latest.torch",
            "params.generator.hierarchical_with_cycle.pt",
        ),
        ("training_data/exp_data/mean.mel.npy", "mean.mel.npy"),
        ("training_data/exp_data/std.mel.npy", "std.mel.npy"),
    ],
    "melgan": [
        ("args.yml", "vocoder/args.yml"),
        ("best_netG.pt", "vocoder/params.pt"),
        ("mel2wav/modules.py", "vocoder/modules.py"),
    ],
}


def download_files_from_run(run, model_dir, paths):

    for wandb_path, path_in_model_dir in paths:

        wandb_file = run.file(wandb_path)

        with tempfile.TemporaryDirectory() as temp_dir:

            local_path = model_dir / path_in_model_dir
            print(f"Downloading {wandb_path} -> {local_path}")
            wandb_file.download(root=temp_dir)

            Path(local_path.parent).mkdir(exist_ok=True, parents=True)
            shutil.copy(Path(temp_dir) / wandb_path, local_path)


def main(
    melgan_run_id,
    unagan_run_id,
    model_dir,
):
    model_dir.mkdir(parents=True, exist_ok=True)
    run_ids = {"melgan": melgan_run_id, "unagan": unagan_run_id}

    wandb.login()
    api = wandb.Api()

    for model in ["melgan", "unagan"]:
        run_id = run_ids[model]
        run = api.run(f"demiurge/{model}/{run_id}")
        download_files_from_run(run, model_dir, MODEL_PATHS[model])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--melgan-run-id", required=True)
    parser.add_argument("--unagan-run-id", required=True)
    parser.add_argument("--model-dir", required=True, type=Path)
    args = parser.parse_args()
    main(**vars(args))
