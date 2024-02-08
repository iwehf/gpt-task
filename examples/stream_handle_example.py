import logging
from gpt_task.inference import run_task


logging.basicConfig(
    format="[{asctime}] [{levelname:<8}] {name}: {message}",
    datefmt="%Y-%m-%d %H:%M:%S",
    style="{",
    level=logging.INFO,
)

messages = [{"role": "user", "content": "I want to create a chat bot. Any suggestions?"}]


def stream_handle(token: str):
    print(token)

res = run_task(
    model="gpt2",
    messages=messages,
    seed=42,
    stream=True,
    stream_handle=stream_handle
)
print(res)
