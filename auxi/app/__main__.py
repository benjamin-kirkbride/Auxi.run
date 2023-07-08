import tkinter.filedialog
from pathlib import Path

from faster_whisper import WhisperModel

from app import config


def _main():
    # assert config

    model_size = "large-v2"

    # Run on GPU with FP16
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    # or run on GPU with INT8
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    # or run on CPU with INT8
    # model = WhisperModel(model_size, device="cpu", compute_type="int8")

    file = Path(__file__).parent.parent.parent / "test.m4a"
    print(file)
    assert file.is_file()
    segments, info = model.transcribe(str(file), beam_size=5)

    print(
        "Detected language '%s' with probability %f"
        % (info.language, info.language_probability)
    )

    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))


if __name__ == "__main__":
    _main()
