import logging
from gpt_task.inference import TaskRunner


logging.basicConfig(
    format="[{asctime}] [{levelname:<8}] {name}: {message}",
    datefmt="%Y-%m-%d %H:%M:%S",
    style="{",
    level=logging.INFO,
)

messages1 = [{"role": "user", "content": "I want to create a chat bot. Any suggestions?"}]
messages2 = [{"role": "user", "content": "What is the highest mountain in the world?"}]

runner = TaskRunner(
    model="gpt2"
)

res1 = runner.run(
    messages1,
    generation_config={
        "max_new_tokens": 30,
        "temperature": 1
    },
    seed=42
)
print(res1)

res2 = runner.run(
    messages2,
    generation_config={
        "max_new_tokens": 30,
        "temperature": 1
    },
    seed=42
)
print(res2)
