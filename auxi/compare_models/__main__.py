"""Compare whisper model performance.

Generates a CSV with performance data.
"""
import csv
import logging
import time
from pathlib import Path

from faster_whisper import WhisperModel

from compare_models import config


class _CustomFormatter(logging.Formatter):
    def format(self, record):
        elapsed_time = time.perf_counter() - start_time
        record.elapsed_time = f"{elapsed_time:.2f}"

        return super().format(record)


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.propagate = False
handler = logging.StreamHandler()
handler.setFormatter(_CustomFormatter("%(levelname)s [%(elapsed_time)s]: %(message)s"))
logger.addHandler(handler)

faster_whisper_logger = logging.getLogger("faster_whisper")
faster_whisper_logger.propagate = False

start_time = time.perf_counter()


MODEL_NAMES = [
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v1",
    "large-v2",
]


def _get_transcription(model_name: str, input_file: Path) -> tuple[str, float, float]:
    start_init_time = time.perf_counter()
    logger.info(f"initializing model '{model_name}'")
    model = WhisperModel(model_name, device="cuda", compute_type="float16")
    end_init_time = time.perf_counter()
    logger.info(f"model '{model_name}' initialized")

    init_time = end_init_time - start_init_time

    start_comp_time = time.perf_counter()
    segments, info = model.transcribe(str(input_file), beam_size=5)
    segment_list = list(segments)
    logger.info("transcription completed")
    end_comp_time = time.perf_counter()

    comp_time = end_comp_time - start_comp_time

    logger.info(
        f"Detected language {info.language} with"
        f" probability {info.language_probability}",
    )

    return "".join([segment.text for segment in segment_list]), init_time, comp_time


def _main():
    file = Path(__file__).parent.parent.parent / "test.m4a"
    assert file.is_file()

    csv_path = Path(config.get("filename"))
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["model", "init_time", "comp_time"])
        for model_name in MODEL_NAMES:
            transcript, init_time, comp_time = _get_transcription(
                model_name=model_name,
                input_file=file,
            )
            writer.writerow([model_name, init_time, comp_time])


if __name__ == "__main__":
    _main()
