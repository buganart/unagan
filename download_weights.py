#!/usr/bin/env ipython
import collections
import shutil
import tempfile
from argparse import ArgumentParser
from pathlib import Path
from typing import List


import wandb

MODEL_PATHS = {
    "unagan": [
        (
            "params.Generator.latest.torch",
            "params.generator.hierarchical_with_cycle.pt",
        ),
        ("mean.mel.npy", "mean.mel.npy"),
        ("std.mel.npy", "std.mel.npy"),
    ],
    "melgan": [
        ("args.yml", "vocoder/args.yml"),
        ("best_netG.pt", "vocoder/params.pt"),
        ("modules.py", "vocoder/modules.py"),
    ],
}


def group_filenames(files: List):

    grouped_by_name = collections.defaultdict(list)

    for file in files:
        grouped_by_name[Path(file.name).name].append(file)

    return grouped_by_name


def download_files_from_run(run, model_dir, paths):

    files = list(run.files())

    wandb_files_grouped_by_filename = group_filenames(files)

    for filename, path_in_model_dir in paths:

        try:
            wandb_files = wandb_files_grouped_by_filename[filename]
        except KeyError:
            raise ValueError(f"File {filename} not found in {run}.")

        if len(wandb_files) > 1:
            # raise ValueError(
            #     f"Run {run} has more than one file with filename"
            #     f" {filename}: {wandb_files}"
            # )
            print(f"Run {run} has more than one file with filename")
            print(f" {filename}: {wandb_files}")

        wandb_file = wandb_files[0]
        wandb_path = wandb_file.name

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
