from __future__ import annotations

import logging
from typing import Any, List, Literal, Mapping, Sequence

import torch
from pydantic import TypeAdapter
from transformers import AutoConfig, AutoTokenizer, pipeline, set_seed

from gpt_task import models
from gpt_task.config import Config

from .errors import wrap_error
from .utils import load_model_kwargs, use_deterministic_mode

_logger = logging.getLogger(__name__)

use_deterministic_mode()


def _find_prompt_tokens(input_tokens: List[int], output_tokens: List[int]) -> int:
    start = output_tokens.index(input_tokens[0])
    if start == -1:
        return 0
    end = output_tokens.index(input_tokens[-1], start + len(input_tokens) - 1)
    if end == -1:
        return 0
    return end + 1


class TaskRunner(object):
    def __init__(
        self,
        model: str,
        dtype: Literal["float16", "bfloat16", "float32", "auto"] = "auto",
        quantize_bits: Literal[4, 8] | None = None,
        config: Config | None = None,
    ):
        self.model = model
        self.dtype = dtype
        self.quantize_bits = quantize_bits

        _logger.info("Start loading pipeline")
        _logger.debug(f"task model: {model}")
        _logger.debug(f"task model dtype: {dtype}")
        if quantize_bits is not None:
            _logger.debug(f"task model quantize bits: {quantize_bits}")

        torch_dtype = None
        if dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "float32":
            torch_dtype = torch.float32
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16

        model_kwargs = load_model_kwargs(config=config)
        _logger.debug(f"model kwargs: {model_kwargs}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model, use_fast=False, trust_remote_code=True, **model_kwargs
        )
        model_config = AutoConfig.from_pretrained(
            model,
            _from_pipeline="text-generation",
            trust_remote_code=True,
            **model_kwargs,
        )

        if quantize_bits == 4:
            model_kwargs["load_in_4bit"] = True
        elif quantize_bits == 8:
            model_kwargs["load_in_8bit"] = True

        self.pipeline = pipeline(
            "text-generation",
            model=model,
            config=model_config,
            tokenizer=self.tokenizer,
            trust_remote_code=True,
            use_fast=False,
            device_map="auto",
            torch_dtype=torch_dtype,
            model_kwargs=dict(
                offload_folder="offload",
                offload_state_dict=True,
                **model_kwargs,
            ),
        )

        _logger.info("Loading pipeline completes")

    def run(
        self,
        messages: Sequence[models.Message],
        generation_config: models.GPTGenerationConfig | None = None,
        seed: int = 0,
    ) -> models.GPTTaskResponse:
        _logger.info("Start text generation")

        set_seed(seed)

        final_generation_config: models.GPTGenerationConfig = {"num_return_sequences": 1, "max_new_tokens": 256}
        if generation_config is not None:
            for k, v in generation_config.items():
                if v is not None:
                    final_generation_config[k] = v

        chats = [dict(**m) for m in messages]
        if self.tokenizer.chat_template is not None:
            inputs = self.tokenizer.apply_chat_template(
                chats, tokenize=False, add_generation_prompt=True
            )
        else:
            inputs = "\n".join(c["content"] for c in chats)

        output = self.pipeline(
            inputs,
            return_tensors=True,
            **final_generation_config,
        )
        assert output is not None
        assert isinstance(output, list)

        res_token_ids = []
        for single in output:
            assert isinstance(single, dict)
            res_token_ids.append(single["generated_token_ids"])

        assert len(res_token_ids) > 0

        input_tokens = self.tokenizer.encode(inputs, add_special_tokens=False)
        prompt_tokens = _find_prompt_tokens(input_tokens, res_token_ids[0])

        completion_tokens = 0
        output_texts = []
        finish_reasons = []
        for token_ids in res_token_ids:
            # when the last token is eos token, finish reason is stop, otherwise is length
            if token_ids[-1] == self.tokenizer.eos_token_id:
                finish_reason = "stop"
            else:
                finish_reason = "length"
            finish_reasons.append(finish_reason)

            completion_tokens += len(token_ids) - prompt_tokens

            text = self.tokenizer.decode(
                token_ids[prompt_tokens:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            output_texts.append(text)

        usage: models.Usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

        choices: List[models.ResponseChoice] = [
            {
                "finish_reason": reason,
                "message": {"role": "assistant", "content": text},
                "index": i,
            }
            for i, (reason, text) in enumerate(zip(finish_reasons, output_texts))
        ]

        resp: models.GPTTaskResponse = {
            "model": self.model,
            "choices": choices,
            "usage": usage,
        }

        _logger.info("Text generation completes")
        return resp


@wrap_error
def run_task(
    args: models.GPTTaskArgs | None = None,
    *,
    model: str | None = None,
    messages: Sequence[models.Message | Mapping[str, Any]] | None = None,
    generation_config: models.GPTGenerationConfig | Mapping[str, Any] | None = None,
    seed: int = 0,
    dtype: Literal["float16", "bfloat16", "float32", "auto"] = "auto",
    quantize_bits: Literal[4, 8] | None = None,
    config: Config | None = None,
) -> models.GPTTaskResponse:
    if args is None:
        args = models.GPTTaskArgs.model_validate(
            {
                "model": model,
                "messages": messages,
                "generation_config": generation_config,
                "seed": seed,
                "dtype": dtype,
                "quantize_bits": quantize_bits,
            }
        )

    _logger.info("Task starts")

    runner = TaskRunner(
        model=args.model,
        dtype=args.dtype,
        quantize_bits=args.quantize_bits,
        config=config,
    )

    return runner.run(
        messages=args.messages, generation_config=args.generation_config, seed=args.seed
    )
