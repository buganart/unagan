from pathlib import Path

import numpy as np
import librosa
from pydub import AudioSegment
import click

from multiprocessing import Pool


def process_one(args):

    path, out_dir = args
    print(f"Processing {path}")

    duration = librosa.get_duration(filename=path)
    subclip_duration = 10
    num_subclips = int(np.ceil(duration / subclip_duration))

    song = AudioSegment.from_wav(path)

    for ii in range(num_subclips):
        start = ii * subclip_duration
        end = (ii + 1) * subclip_duration
        print(path, start, end)

        out_path = out_dir / f"{path.name}.{start}_{end}.mp3"
        if out_path.exists():
            print(f"Path {out_path} exists. Done before.")
            continue

        subclip = song[start * 1000 : end * 1000]
        subclip.export(out_path, format="mp3")


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
    args_list = [(path, out_dir) for path in paths]

    pool = Pool()
    pool.map(process_one, args_list)


if __name__ == "__main__":
    collect_audio_clips()
