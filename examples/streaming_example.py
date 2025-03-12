import logging
import sys
import time

from gpt_task.inference import run_task

import dotenv
dotenv.load_dotenv()

logging.basicConfig(
    format="[{asctime}] [{levelname:<8}] {name}: {message}",
    datefmt="%Y-%m-%d %H:%M:%S",
    style="{",
    level=logging.INFO,
)

_logger = logging.getLogger(__name__)

messages = [
    {"role": "user", "content": "Write a short story about a magical forest."}
]

print("Starting generation...", flush=True)

# Enable streaming output
try:
    for chunk in run_task(
        model="mistralai/Mistral-7B-Instruct-v0.1",
        messages=messages,
        stream=True,
        generation_config={
            "repetition_penalty": 1.1,
            "do_sample": True,
            "temperature": 0.7,
            "max_new_tokens": 100
        },
        seed=42,
        dtype="float16"
    ):
        content = chunk["choices"][0]["delta"].get("content", "")
        finish_reason = chunk["choices"][0].get("finish_reason")

        _logger.debug(f"Received chunk: {chunk}")

        if content:
            sys.stdout.write(content)
            sys.stdout.flush()

        if finish_reason:
            sys.stdout.write("\n")
            sys.stdout.flush()
            _logger.debug(f"Generation finished with reason: {finish_reason}")

        time.sleep(0.1)

except Exception as e:
    _logger.exception("Error during generation")
    raise

print("\nGeneration complete!", flush=True)
