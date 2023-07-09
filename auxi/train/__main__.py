import logging

from train import TRAINING_DIR, negative_data

# Define log format and set it as default for all loggers.
log_format = "%(levelname)s %(filename)s: %(message)s"
logging.basicConfig(format=log_format, level=logging.INFO)

logger = logging.getLogger(__name__)


def main():
    logger.info(f"{TRAINING_DIR=}")
    logger.info("Gathering Training Data")
    negative_data.get()

    logger.info("Done!")


if __name__ == "__main__":
    main()
