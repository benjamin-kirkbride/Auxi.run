"""Top level init."""

import logging

import openai
from dotenv import dotenv_values

config = dotenv_values(".env")

openai.api_key = config["openai_key"]

# Define log format and set it as default for all loggers.
log_format = "%(levelname)s %(filename)s: %(message)s"
logging.basicConfig(format=log_format, level=logging.DEBUG)
