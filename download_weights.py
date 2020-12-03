#!/usr/bin/env ipython
import collections
import pprint
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


def find_duplicate_filenames(paths: List[Path]):

    grouped_by_name = collections.defaultdict(list)

    for path in paths:
        grouped_by_name[path.name].append(path)

    return {name: files for name, files in grouped_by_name.items() if len(files) > 1}


def download_files_from_run(run, model_dir, paths):

    files = list(run.files())

    duplicates = find_duplicate_filenames([Path(f.name) for f in files])

    if duplicates:
        pprint.pprint(duplicates)
        raise ValueError("This run has duplicate file names.")

    filename_to_file = {Path(f.name).name: f for f in files}

    for filename, path_in_model_dir in paths:

        try:
            wandb_file = filename_to_file[filename]
        except KeyError:
            raise ValueError(f"File {filename} not found in {run}.")

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
