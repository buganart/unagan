import warnings
from pathlib import Path

import numpy as np
import librosa
import click

from joblib import Parallel, delayed
import soundfile as sf


def make_clip(path, out_dir, sampling_rate, start, end, subclip_duration):
    out_path = out_dir / f"{path.name}.{start}_{end}.wav"
    if out_path.exists():
        return

    subclip, _ = librosa.load(
        path, sr=sampling_rate, offset=start, duration=subclip_duration
    )
    sf.write(out_path, subclip, sampling_rate)


def process_one(path, out_dir):

    print(f"Processing {path}")

    duration = librosa.get_duration(filename=path)
    _, existing_sampling_rate = librosa.load(path, duration=0, sr=None)

    if existing_sampling_rate != 44100:
        print(
            f"Sampling rate is {existing_sampling_rate}: "
            "To avoid slow resampling use 44.1 kHz audio."
        )

    subclip_duration = 10
    num_subclips = int(np.ceil(duration / subclip_duration))

    args = []
    for ii in range(num_subclips):
        args.append(
            dict(
                path=path,
                out_dir=out_dir,
                sampling_rate=44100,
                start=ii * subclip_duration,
                end=(ii + 1) * subclip_duration,
                subclip_duration=subclip_duration,
            )
        )

    Parallel(verbose=0, n_jobs=-1, backend="multiprocessing")(
        delayed(make_clip)(**kwargs) for kwargs in args
    )


@click.command()
@click.option(
    "--audio-dir",
    "-a",
    required=True,
    type=click.Path(exists=True),
    help="Clean audios or separated audios from mixture",
)
@click.option("--extension", "-e", default="*")
@click.option(
    "--out-dir",
    "-o",
    type=click.Path(),
    default="./training_data/clips",
)
def collect_audio_clips(audio_dir, out_dir, extension):
    audio_dir = Path(audio_dir)
    out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(audio_dir.glob(f"*.{extension}"))

    for path in paths:
        process_one(path, out_dir)

    print("Done.")


if __name__ == "__main__":
    collect_audio_clips()
