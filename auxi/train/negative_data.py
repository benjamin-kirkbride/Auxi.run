import logging
import zipfile
from multiprocessing import Pool
from pathlib import Path
from typing import TYPE_CHECKING

import datasets
import numpy as np
import openwakeword.data
import pydub
import pydub.exceptions
import requests
import scipy
from huggingface_hub import login
from tqdm import tqdm

from train import TRAINING_DIR, config

NEGATIVE_DIR = TRAINING_DIR / "negative"

if TYPE_CHECKING:
    import io

logger = logging.getLogger(__name__)

login(token=config["huggingface_token"])


class _CV_13_Processor:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def process(self, i_example):
        i, example = i_example
        example_stripped_extension_path = example["path"][0:-4]
        assert isinstance(example_stripped_extension_path, str)
        example_path = self.output_dir / f"{example_stripped_extension_path}.wav"
        example_path.parent.mkdir(exist_ok=True)

        # convert to 16-bit PCM format
        wav_data = (example["audio"]["array"] * 32767).astype(np.int16)
        scipy.io.wavfile.write(example_path, 16000, wav_data)


def _prepare_cv_13() -> None:
    limit = 20_000
    # Prepare Mozilla Common Voice data from HuggingFace
    cv_13_dir = NEGATIVE_DIR / Path("cv_13")
    cv_13_dir.mkdir(exist_ok=True)

    cv_13_lock = Path(cv_13_dir) / "FINISHED"

    if cv_13_lock.is_file():
        logger.info("'common_voice_13' dataset already prepared")
        return

    logger.info("querying 'common_voice_13' dataset")
    cv_13 = datasets.load_dataset(
        "mozilla-foundation/common_voice_13_0", "en", split="train", streaming=True
    )

    logger.info("converting audio to 16-khz")
    cv_13 = cv_13.cast_column("audio", datasets.Audio(sampling_rate=16000, mono=True))

    # TODO: is converting to wav even necessary?
    # TODO: can we just feed it to the model in one shot, with no on-disk step?
    # TODO: ^ should be optional - cache vs one-shot mode
    cv_13_iter = zip(
        tqdm(range(limit), desc="downloading cv_13 samples", position=0),
        cv_13,
        strict=False,
    )

    processor = _CV_13_Processor(cv_13_dir)

    logger.info("converting audio to wav")
    with tqdm(
        desc="job completions", total=limit, position=1
    ) as progress, Pool() as pool:
        for _ in pool.imap(processor.process, cv_13_iter, chunksize=100):
            progress.update()

    # create lock file, ensuring we won't prep this data again (unless removed)
    cv_13_lock.touch()


def _download_zip(name: str, url: str, dest_dir: Path):
    # Make sure the directory exists
    dest_dir.mkdir(exist_ok=True)

    logger.info(f"downloading {url}")
    with requests.Session() as session:
        response = session.get(url, stream=True)
        response.raise_for_status()

        total_size_in_bytes = int(response.headers.get("content-length", 0))
        with tqdm(
            total=total_size_in_bytes,
            unit="iB",
            unit_scale=True,
            desc=f"{name}.zip download",
        ) as progress:
            zip_path = dest_dir / f"{name}.zip"
            with zip_path.open("wb") as out_file:
                for chunk in response.iter_content(chunk_size=1024):
                    progress.update(len(chunk))
                    out_file.write(chunk)


class _FMA_Processor:
    def __init__(self, dir_: Path, zip_: Path):
        self.dir = dir_
        self.zip_ = zip_

    def process(self, zip_info: zipfile.ZipInfo):
        if zip_info.filename[-4:] != ".mp3":
            logger.info(f"skipping {zip_info.filename}")
            return
        stripped_extension_path = zip_info.filename[0:-4]
        output_filepath = self.dir / f"{stripped_extension_path}.wav"
        output_filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(self.zip_) as zip_:
                mp3 = zip_.open(zip_info)
                sound = pydub.AudioSegment.from_mp3(mp3)

            sound.export(
                output_filepath,
                format="wav",
                # -ac 1 specifies that the output should be mono.
                # -ar 16000 specifies the sample rate of 16 kHz.
                # -sample_fmt s16 specifies the 16-bit PCM format.
                parameters=["-ac", "1", "-ar", "16000", "-sample_fmt", "s16"],
            )
        except pydub.exceptions.CouldntDecodeError:
            logger.critical(f"failed on {zip_info.filename}")


def _prepare_fma():
    fma_dataset = "fma_small"
    fma_dataset_file_qty = 8_000

    fma_dir = NEGATIVE_DIR / Path(fma_dataset)
    fma_dir.mkdir(exist_ok=True)
    # if some files are missing we are not going to sweat it
    if len(list(fma_dir.rglob("*.wav"))) >= (fma_dataset_file_qty * 0.9):
        logger.info(f"{fma_dataset} dataset already prepared")
        return

    logger.info(f"preparing {fma_dataset}")
    fma_dataset_zip = fma_dir / f"{fma_dataset}.zip"
    if not fma_dataset_zip.is_file() and not zipfile.ZipFile(fma_dataset_zip).testzip():
        _download_zip(
            name=fma_dataset,
            url="https://os.unil.cloud.switch.ch/fma/fma_small.zip",
            dest_dir=fma_dir,
        )
    else:
        logger.info(f"{fma_dataset}.zip already downloaded")

    processor = _FMA_Processor(dir_=fma_dir, zip_=fma_dataset_zip)

    with zipfile.ZipFile(fma_dataset_zip, "r") as zip_:
        infolist = zip_.infolist()

    # v this is how you would have a tqdm that updates in realtime no batching
    # https://stackoverflow.com/a/74334558/1342874
    with tqdm(
        desc="convert mp3 to wav", total=len(zip_.infolist())
    ) as progress, Pool() as pool:
        for _ in pool.imap_unordered(processor.process, infolist, chunksize=25):
            progress.update()


class _FSD50K_Processor:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def process(self, i_example):
        i, example = i_example
        raw_path = example["audio"]["path"]
        assert isinstance(raw_path, str)
        relevant_path_segments = raw_path.split("/")[-3:]
        example_path = self.output_dir / "/".join(relevant_path_segments)
        example_path.parent.mkdir(parents=True, exist_ok=True)

        # convert to 16-bit PCM format
        wav_data = (example["audio"]["array"] * 32767).astype(np.int16)
        scipy.io.wavfile.write(example_path, 16000, wav_data)


def _prepare_fsd50k():
    limit = 20_000
    # Prepare Mozilla Common Voice data from HuggingFace
    fsd50k_dir = NEGATIVE_DIR / Path("fsd50k")
    fsd50k_dir.mkdir(exist_ok=True)

    fsd50k_lock = Path(fsd50k_dir) / "FINISHED"

    if fsd50k_lock.is_file():
        logger.info("'fsd50k' dataset already prepared")
        return

    logger.info("querying 'fsd50k' dataset")
    fsd50k = datasets.load_dataset("Fhrozen/FSD50k", split="test", streaming=True)

    logger.info("converting audio to 16-khz")
    fsd50k = fsd50k.cast_column("audio", datasets.Audio(sampling_rate=16000, mono=True))

    fsd50k_iter = zip(
        tqdm(range(limit), desc="downloading fsd50k samples", position=0),
        fsd50k,
        strict=False,
    )

    processor = _FSD50K_Processor(fsd50k_dir)

    logger.info("converting audio to wav")
    with tqdm(
        desc="job completions", total=limit, position=1
    ) as progress, Pool() as pool:
        for _ in pool.imap(processor.process, fsd50k_iter, chunksize=1):
            progress.update()

    # create lock file, ensuring we won't prep this data again (unless removed)
    fsd50k_lock.touch()


def _prepare() -> None:
    """Prepare negative training data."""
    _prepare_cv_13()
    _prepare_fma()
    _prepare_fsd50k()


def get():
    _prepare()
    # very annoying filter_audio_paths doesn't support recursive globbing
    training_dirs = [
        training_dir
        for training_dir in TRAINING_DIR.rglob("*")
        if training_dir.is_dir()
    ]
    negative_clips, negative_durations = openwakeword.data.filter_audio_paths(
        training_dirs,
        min_length_secs=1.0,  # minimum clip length in seconds
        max_length_secs=60 * 30,  # maximum clip length in seconds
        duration_method="header",  # use the file header to calculate duration
        glob_filter="*.wav",
    )
    logger.info(
        f"{len(negative_clips)} negative clips after filtering, "
        f"representing ~{sum(negative_durations)//3600} hours"
    )
