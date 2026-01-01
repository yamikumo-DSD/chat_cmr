import llama_cpp
from llama_cpp import Llama, LlamaGrammar
from llama_cpp.llama import StoppingCriteriaList, LogitsProcessorList
from llama_cpp.llama_types import CreateCompletionResponse, CreateCompletionStreamResponse
import llama_cpp.llama_types as llama_types
from typing import (
    Union,
    List,
    Iterator,
    Optional,
    Dict,
    Any,
    Literal,
    Tuple,
    Union,
    Protocol,
    cast,
)
import os
import sys
import uuid
import time
import json
import ctypes
import typing
import random
import fnmatch
import warnings
import contextlib
import multiprocessing
import numpy as np

from llama_cpp.llama_chat_format import Llava15ChatHandler




class LlamaCppSpecial(Llama):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def token_count(self, text, add_bos=False, special=True) -> int:
        """
        Returns length of tokenized sequence. 
        Args:
            text: text to tokenize
            add_bos: when True, adding BOS token at the beginning of the given text
            special: whether to tokenize special tokens as special/normal sequence
        """
        token_bos = self.detokenize([self.token_bos()], special=True).decode()
        text = text.split(token_bos)[1] if text.startswith(token_bos) else text
        return len(self.tokenize(text.encode("utf-8"), add_bos=add_bos, special=special))

    def add_bos(self) -> bool:
        return self._model.get_add_bos()
        
    def _create_completion(
        self,
        prompt: Union[str, List[int]],
        suffix: Optional[str] = None,
        max_tokens: Optional[int] = 16,
        temperature: float = 0.8,
        top_p: float = 0.95,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        logprobs: Optional[int] = None,
        echo: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.0,
        top_k: int = 40,
        top_n_sigma: float = -1.00,
        stream: bool = False,
        seed: Optional[int] = None,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        xtc_threshold: float = 0.1,
        xtc_probability: float = 0.0,
        dry_multiplier: float = 0.0,
        dry_base: float = 1.75,
        dry_allowed_length: int = 2,
        dry_penalty_last_n:int = 0,
        dry_seq_breakers: list[str] = ["\n", ":", "\"", "*"],
        model: Optional[str] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logit_bias: Optional[Dict[int, float]] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        grammar: Optional[LlamaGrammar] = None,
        detokenize_special = True,
    ) -> Union[
        Iterator[CreateCompletionResponse], Iterator[CreateCompletionStreamResponse]
    ]:
        """
        This just gives argument to switch whether to detokenize special tokens.
        Other part of the original code is preserved as it is.
        """
        assert suffix is None or suffix.__class__ is str

        # User alternative detokenize function.
        def detokenize(tokens, prev_tokens=None):
            return self.detokenize(tokens, prev_tokens, special=True) if detokenize_special else self.detokenize(tokens, prev_tokens, special=False)

        completion_id: str = f"cmpl-{str(uuid.uuid4())}"
        created: int = int(time.time())
        bos_token_id: int = self._model.token_bos()
        eos_token_id: int = self._model.token_eos()
        sep_token_id: int = self._model.token_sep()
        prefix_token_id: int = self._model.token_fim_pre()
        middle_token_id: int = self._model.token_fim_mid()
        suffix_token_id: int = self._model.token_fim_suf()
        add_space_prefix: bool = (
            self.metadata.get("tokenizer.ggml.add_space_prefix", "true") == "true"
        )
        bos_tokens: List[int] = [bos_token_id]
        eos_tokens: List[int] = [
            sep_token_id if self._model.get_add_sep() else eos_token_id
        ]

        if (
            (isinstance(prompt, list) and suffix is None)
            or not self._model.get_add_bos()
            or bos_tokens[:1] == [-1]
        ):
            bos_tokens = []

        if (isinstance(prompt, list) and suffix is None) or (
            not self._model.get_add_eos() and not self._model.get_add_sep()
        ):
            eos_tokens = []

        suffix_space_prefix: int = 0
        # Tokenizer hack to remove leading space
        if add_space_prefix and suffix_token_id >= 0 and suffix:
            suffix = "☺" + suffix
            suffix_space_prefix = 2

        # If prompt is empty, initialize completion with BOS token to avoid
        # detokenization including a space at the beginning of the completion
        completion_tokens: List[int] = [] if len(prompt) > 0 else [bos_token_id]
        # Add blank space to start of prompt to match OG llama tokenizer
        prefix_tokens: List[int] = (
            [prefix_token_id] if prefix_token_id >= 0 and suffix is not None else []
        ) + (
            (
                self.tokenize(
                    prompt.encode("utf-8"),
                    add_bos=False,
                    special=(prefix_token_id < 0 or suffix is None),
                )
                if prompt != ""
                else []
            )
            if isinstance(prompt, str)
            else prompt
        )
        suffix_tokens: List[int] = (
            (
                [suffix_token_id]
                + (
                    self.tokenize(suffix.encode("utf-8"), add_bos=False, special=False)[
                        suffix_space_prefix:
                    ]
                    if suffix
                    else []
                )
            )
            if suffix_token_id >= 0 and suffix is not None
            else []
        )
        middle_tokens: List[int] = (
            [middle_token_id] if middle_token_id >= 0 and suffix is not None else []
        )
        prompt_tokens: List[int] = (
            bos_tokens
            + (
                (suffix_tokens + prefix_tokens + middle_tokens)
                if self.spm_infill
                else (prefix_tokens + suffix_tokens + middle_tokens)
            )
            + eos_tokens
        )
        text: bytes = b""
        returned_tokens: int = 0
        stop = (
            stop if isinstance(stop, list) else [stop] if isinstance(stop, str) else []
        )
        model_name: str = model if model is not None else self.model_path

        if prompt_tokens[:2] == [self.token_bos()] * 2:
            warnings.warn(
                f'Detected duplicate leading "{self._model.token_get_text(self.token_bos())}" in prompt, this will likely reduce response quality, consider removing it...',
                RuntimeWarning,
            )

        # NOTE: This likely doesn't work correctly for the first token in the prompt
        # because of the extra space added to the start of the prompt_tokens
        if logit_bias is not None:
            logit_bias_map = {int(k): float(v) for k, v in logit_bias.items()}

            def logit_bias_processor(
                input_ids: npt.NDArray[np.intc],
                scores: npt.NDArray[np.single],
            ) -> npt.NDArray[np.single]:
                new_scores = np.copy(
                    scores
                )  # Does it make sense to copy the whole array or can we just overwrite the original one?
                for input_id, score in logit_bias_map.items():
                    new_scores[input_id] = score + scores[input_id]
                return new_scores

            _logit_bias_processor = LogitsProcessorList([logit_bias_processor])
            if logits_processor is None:
                logits_processor = _logit_bias_processor
            else:
                logits_processor = logits_processor.extend(_logit_bias_processor)

        if self.verbose:
            self._ctx.reset_timings()

        if len(prompt_tokens) >= self._n_ctx:
            raise ValueError(
                f"Requested tokens ({len(prompt_tokens)}) exceed context window of {llama_cpp.llama_n_ctx(self.ctx)}"
            )

        if max_tokens is None or max_tokens <= 0:
            # Unlimited, depending on n_ctx.
            max_tokens = self._n_ctx - len(prompt_tokens)

        # Truncate max_tokens if requested tokens would exceed the context window
        max_tokens = (
            max_tokens
            if max_tokens + len(prompt_tokens) < self._n_ctx
            else (self._n_ctx - len(prompt_tokens))
        )

        if stop != []:
            stop_sequences = [s.encode("utf-8") for s in stop]
        else:
            stop_sequences = []

        if logprobs is not None and self._logits_all is False:
            raise ValueError(
                "logprobs is not supported for models created with logits_all=False"
            )

        if self.cache:
            try:
                cache_item = self.cache[prompt_tokens]
                cache_prefix_len = Llama.longest_token_prefix(
                    cache_item.input_ids.tolist(), prompt_tokens
                )
                eval_prefix_len = Llama.longest_token_prefix(
                    self._input_ids.tolist(), prompt_tokens
                )
                if cache_prefix_len > eval_prefix_len:
                    self.load_state(cache_item)
                    if self.verbose:
                        print("Llama._create_completion: cache hit", file=sys.stderr)
            except KeyError:
                if self.verbose:
                    print("Llama._create_completion: cache miss", file=sys.stderr)

        if seed is not None:
            self.set_seed(seed)
        else:
            self.set_seed(random.Random(self._seed).randint(0, 2 ** 32))

        finish_reason = "length"
        multibyte_fix = 0
        for token in self.generate(
            prompt_tokens,
            top_k=top_k,
            top_n_sigma=top_n_sigma,
            top_p=top_p,
            min_p=min_p,
            typical_p=typical_p,
            temp=temperature,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            xtc_threshold=xtc_threshold,
            xtc_probability=xtc_probability,
            dry_multiplier=dry_multiplier,
            dry_base=dry_base,
            dry_allowed_length=dry_allowed_length,
            dry_penalty_last_n=dry_penalty_last_n,
            dry_seq_breakers=dry_seq_breakers,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            stopping_criteria=stopping_criteria,
            logit_bias=logit_bias,
            logits_processor=logits_processor,
            grammar=grammar,
        ):
            if llama_cpp.llama_token_is_eog(self._model.vocab, token):
                text = detokenize(completion_tokens, prev_tokens=prompt_tokens)
                finish_reason = "stop"
                break

            completion_tokens.append(token)

            all_text = detokenize(completion_tokens, prev_tokens=prompt_tokens)

            # Contains multi-byte UTF8
            for k, char in enumerate(all_text[-3:]):
                k = 3 - k
                for num, pattern in [(2, 192), (3, 224), (4, 240)]:
                    # Bitwise AND check
                    if num > k and pattern & char == pattern:
                        multibyte_fix = num - k

            # Stop incomplete bytes from passing
            if multibyte_fix > 0:
                multibyte_fix -= 1
                continue

            any_stop = [s for s in stop_sequences if s in all_text]
            if len(any_stop) > 0:
                first_stop = any_stop[0]
                text = all_text[: all_text.index(first_stop)]
                finish_reason = "stop"
                break

            if stream:
                remaining_tokens = completion_tokens[returned_tokens:]
                remaining_text = detokenize(
                    remaining_tokens,
                    prev_tokens=prompt_tokens + completion_tokens[:returned_tokens],
                )
                remaining_length = len(remaining_text)

                # We want to avoid yielding any characters from
                # the generated text if they are part of a stop
                # sequence.
                first_stop_position = 0
                for s in stop_sequences:
                    for i in range(min(len(s), remaining_length), 0, -1):
                        if remaining_text.endswith(s[:i]):
                            if i > first_stop_position:
                                first_stop_position = i
                            break

                token_end_position = 0

                if logprobs is not None:
                    # not sure how to handle this branch when dealing
                    # with CJK output, so keep it unchanged
                    for token in remaining_tokens:
                        if token == bos_token_id:
                            continue
                        token_end_position += len(
                            detokenize(
                                [token],
                                prev_tokens=prompt_tokens
                                + completion_tokens[:returned_tokens],
                            )
                        )
                        # Check if stop sequence is in the token
                        if token_end_position > (
                            remaining_length - first_stop_position
                        ):
                            break
                        token_str = detokenize(
                            [token],
                            prev_tokens=prompt_tokens
                            + completion_tokens[:returned_tokens],
                        ).decode("utf-8", errors="ignore")
                        text_offset = len(prompt) + len(
                            detokenize(
                                completion_tokens[:returned_tokens],
                                prev_tokens=prompt_tokens
                                + completion_tokens[:returned_tokens],
                            ).decode("utf-8", errors="ignore")
                        )
                        token_offset = len(prompt_tokens) + returned_tokens
                        logits = self._scores[token_offset - 1, :]
                        current_logprobs = Llama.logits_to_logprobs(logits).tolist()
                        sorted_logprobs = list(
                            sorted(
                                zip(current_logprobs, range(len(current_logprobs))),
                                reverse=True,
                            )
                        )
                        top_logprob = {
                            detokenize([i]).decode(
                                "utf-8", errors="ignore"
                            ): logprob
                            for logprob, i in sorted_logprobs[:logprobs]
                        }
                        top_logprob.update({token_str: current_logprobs[int(token)]})
                        logprobs_or_none = {
                            "tokens": [
                                detokenize(
                                    [token],
                                    prev_tokens=prompt_tokens
                                    + completion_tokens[:returned_tokens],
                                ).decode("utf-8", errors="ignore")
                            ],
                            "text_offset": [text_offset],
                            "token_logprobs": [current_logprobs[int(token)]],
                            "top_logprobs": [top_logprob],
                        }
                        returned_tokens += 1
                        yield {
                            "id": completion_id,
                            "object": "text_completion",
                            "created": created,
                            "model": model_name,
                            "choices": [
                                {
                                    "text": detokenize(
                                        [token],
                                        prev_tokens=prompt_tokens
                                        + completion_tokens[:returned_tokens],
                                    ).decode("utf-8", errors="ignore"),
                                    "index": 0,
                                    "logprobs": logprobs_or_none,
                                    "finish_reason": None,
                                }
                            ],
                        }
                else:
                    while len(remaining_tokens) > 0:
                        decode_success = False
                        for i in range(1, len(remaining_tokens) + 1):
                            try:
                                bs = detokenize(
                                    remaining_tokens[:i],
                                    prev_tokens=prompt_tokens
                                    + completion_tokens[:returned_tokens],
                                )
                                ts = bs.decode("utf-8")
                                decode_success = True
                                break
                            except UnicodeError:
                                pass
                        else:
                            break
                        if not decode_success:
                            # all remaining tokens cannot be decoded to a UTF-8 character
                            break
                        token_end_position += len(bs)
                        if token_end_position > (
                            remaining_length - first_stop_position
                        ):
                            break
                        remaining_tokens = remaining_tokens[i:]
                        returned_tokens += i

                        yield {
                            "id": completion_id,
                            "object": "text_completion",
                            "created": created,
                            "model": model_name,
                            "choices": [
                                {
                                    "text": ts,
                                    "index": 0,
                                    "logprobs": None,
                                    "finish_reason": None,
                                }
                            ],
                        }

            if len(completion_tokens) >= max_tokens:
                text = detokenize(completion_tokens, prev_tokens=prompt_tokens)
                finish_reason = "length"
                break

        if stopping_criteria is not None and stopping_criteria(
            self._input_ids, self._scores[-1, :]
        ):
            text = detokenize(completion_tokens, prev_tokens=prompt_tokens)
            finish_reason = "stop"

        if self.verbose:
            self._ctx.print_timings()

        if stream:
            remaining_tokens = completion_tokens[returned_tokens:]
            remaining_text = detokenize(
                remaining_tokens,
                prev_tokens=prompt_tokens + completion_tokens[:returned_tokens],
            )
            any_stop = [s for s in stop_sequences if s in remaining_text]
            if len(any_stop) > 0:
                end = min(remaining_text.index(stop) for stop in any_stop)
            else:
                end = len(remaining_text)

            token_end_position = 0
            for token in remaining_tokens:
                token_end_position += len(
                    detokenize(
                        [token],
                        prev_tokens=prompt_tokens + completion_tokens[:returned_tokens],
                    )
                )

                logprobs_or_none: Optional[CompletionLogprobs] = None
                if logprobs is not None:
                    if token == bos_token_id:
                        continue
                    token_str = detokenize([token]).decode(
                        "utf-8", errors="ignore"
                    )
                    text_offset = len(prompt) + len(
                        detokenize(
                            completion_tokens[:returned_tokens],
                            prev_tokens=prompt_tokens
                            + completion_tokens[:returned_tokens],
                        )
                    )
                    token_offset = len(prompt_tokens) + returned_tokens - 1
                    logits = self._scores[token_offset, :]
                    current_logprobs = Llama.logits_to_logprobs(logits).tolist()
                    sorted_logprobs = list(
                        sorted(
                            zip(current_logprobs, range(len(current_logprobs))),
                            reverse=True,
                        )
                    )
                    top_logprob = {
                        detokenize([i]).decode("utf-8", errors="ignore"): logprob
                        for logprob, i in sorted_logprobs[:logprobs]
                    }
                    top_logprob.update({token_str: current_logprobs[int(token)]})
                    logprobs_or_none = {
                        "tokens": [
                            detokenize([token]).decode("utf-8", errors="ignore")
                        ],
                        "text_offset": [text_offset],
                        "token_logprobs": [current_logprobs[int(token)]],
                        "top_logprobs": [top_logprob],
                    }

                if token_end_position >= end:
                    last_text = detokenize([token])
                    if token_end_position == end - 1:
                        break
                    returned_tokens += 1
                    yield {
                        "id": completion_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "text": last_text[
                                    : len(last_text) - (token_end_position - end)
                                ].decode("utf-8", errors="ignore"),
                                "index": 0,
                                "logprobs": logprobs_or_none,
                                "finish_reason": None,
                            }
                        ],
                    }
                    break
                returned_tokens += 1
                yield {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "text": detokenize([token]).decode(
                                "utf-8", errors="ignore"
                            ),
                            "index": 0,
                            "logprobs": logprobs_or_none,
                            "finish_reason": None,
                        }
                    ],
                }
            yield {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "text": "",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": finish_reason,
                    }
                ],
            }
            if self.cache:
                if self.verbose:
                    print("Llama._create_completion: cache save", file=sys.stderr)
                self.cache[prompt_tokens + completion_tokens] = self.save_state()
                if self.verbose:
                    print("Llama._create_completion: cache saved", file=sys.stderr)
            return

        if self.cache:
            if self.verbose:
                print("Llama._create_completion: cache save", file=sys.stderr)
            self.cache[prompt_tokens + completion_tokens] = self.save_state()

        text_str = text.decode("utf-8", errors="ignore")

        if echo:
            text_str = prompt + text_str

        if suffix_token_id < 0 and suffix is not None:
            text_str = text_str + suffix

        logprobs_or_none: Optional[CompletionLogprobs] = None
        if logprobs is not None:
            text_offset = 0 if echo else len(prompt)
            token_offset = 0 if echo else len(prompt_tokens[1:])
            text_offsets: List[int] = []
            token_logprobs: List[Optional[float]] = []
            tokens: List[str] = []
            top_logprobs: List[Optional[Dict[str, float]]] = []

            if echo:
                # Remove leading BOS token if exists
                all_tokens = (
                    prompt_tokens[1 if prompt_tokens[0] == self.token_bos() else 0 :]
                    + completion_tokens
                )
            else:
                all_tokens = completion_tokens

            all_token_strs = [
                detokenize([token], prev_tokens=all_tokens[:i]).decode(
                    "utf-8", errors="ignore"
                )
                for i, token in enumerate(all_tokens)
            ]
            all_logprobs = Llama.logits_to_logprobs(self._scores)[token_offset:]
            # TODO: may be able to change this loop to use np.take_along_dim
            for idx, (token, token_str, logprobs_token) in enumerate(
                zip(all_tokens, all_token_strs, all_logprobs)
            ):
                if token == bos_token_id:
                    continue
                text_offsets.append(
                    text_offset
                    + len(
                        detokenize(all_tokens[:idx]).decode(
                            "utf-8", errors="ignore"
                        )
                    )
                )
                tokens.append(token_str)
                sorted_logprobs = list(
                    sorted(
                        zip(logprobs_token, range(len(logprobs_token))), reverse=True
                    )
                )
                token_logprobs.append(logprobs_token[int(token)])
                top_logprob: Optional[Dict[str, float]] = {
                    detokenize([i], prev_tokens=all_tokens[:idx]).decode(
                        "utf-8", errors="ignore"
                    ): logprob
                    for logprob, i in sorted_logprobs[:logprobs]
                }
                top_logprob.update({token_str: logprobs_token[int(token)]})
                top_logprobs.append(top_logprob)
            # Weird idosincracy of the OpenAI API where
            # token_logprobs and top_logprobs are null for
            # the first token.
            if echo and len(all_tokens) > 0:
                token_logprobs[0] = None
                top_logprobs[0] = None
            logprobs_or_none = {
                "tokens": tokens,
                "text_offset": text_offsets,
                "token_logprobs": token_logprobs,
                "top_logprobs": top_logprobs,
            }

        yield {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "text": text_str,
                    "index": 0,
                    "logprobs": logprobs_or_none,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": len(completion_tokens),
                "total_tokens": len(prompt_tokens) + len(completion_tokens),
            },
        }




    def create_completion(
        self,
        prompt: Union[str, List[int]],
        suffix: Optional[str] = None,
        max_tokens: Optional[int] = 16,
        temperature: float = 0.8,
        top_p: float = 0.95,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        logprobs: Optional[int] = None,
        echo: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.0,
        top_k: int = 40,
        top_n_sigma: float = -1.00,
        stream: bool = False,
        seed: Optional[int] = None,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        xtc_threshold: float = 0.1,
        xtc_probability: float = 0.0,
        dry_multiplier: float = 0.0,
        dry_base: float = 1.75,
        dry_allowed_length: int = 2,
        dry_penalty_last_n:int = 0,
        dry_seq_breakers: list[str] = ["\n", ":", "\"", "*"],
        model: Optional[str] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logit_bias: Optional[Dict[int, float]] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        grammar: Optional[LlamaGrammar] = None,
        detokenize_special: bool = True,
    ) -> Union[CreateCompletionResponse, Iterator[CreateCompletionStreamResponse]]:
        """Generate text from a prompt.

        Args:
            prompt: The prompt to generate text from.
            suffix: A suffix to append to the generated text. If None, no suffix is appended.
            max_tokens: The maximum number of tokens to generate. If max_tokens <= 0 or None, the maximum number of tokens to generate is unlimited and depends on n_ctx.
            temperature: The temperature to use for sampling.
            top_p: The top-p value to use for nucleus sampling. Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
            min_p: The min-p value to use for minimum p sampling. Minimum P sampling as described in https://github.com/ggml-org/llama.cpp/pull/3841
            typical_p: The typical-p value to use for sampling. Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
            logprobs: The number of logprobs to return. If None, no logprobs are returned.
            echo: Whether to echo the prompt.
            stop: A list of strings to stop generation when encountered.
            frequency_penalty: The penalty to apply to tokens based on their frequency in the prompt.
            presence_penalty: The penalty to apply to tokens based on their presence in the prompt.
            repeat_penalty: The penalty to apply to repeated tokens.
            top_k: The top-k value to use for sampling. Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
            top_n_sigma: Limit the next token selection to a subset of tokens with pre-softmax logits that are within n * σ less than the max logit (default: -1.00, -1.00 = disabled).
            stream: Whether to stream the results.
            seed: The seed to use for sampling.
            mirostat_mode: The mirostat sampling mode.
            mirostat_tau: The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
            mirostat_eta: The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
            xtc-probability: Sets the chance for token removal (checked once on sampler start) (default: 0.0). XTC sampler as described in https://github.com/oobabooga/text-generation-webui/pull/6335
            xtc-threshold: Sets a minimum probability threshold for tokens to be removed (default: 0.1). XTC sampler as described in https://github.com/oobabooga/text-generation-webui/pull/6335
            dry_multiplier: Set the DRY (Don't Repeat Yourself) repetition penalty multiplier. Default: `0.0`, which is disabled.
            dry_base`: Set the DRY repetition penalty base value. Default: `1.75`
            dry_allowed_length: Tokens that extend repetition beyond this receive exponentially increasing penalty: multiplier * base ^ (length of repeating sequence before token - allowed length). Default: `2`
            dry_penalty_last_n: How many tokens to scan for repetitions. Default: `0`, where `0` is disabled and `-1` is context size.
            dry_seq_breakers: Specify an array of sequence breakers for DRY sampling. Only a JSON array of strings is accepted. Default: `['\n', ':', '"', '*']`
            model: The name to use for the model in the completion object.
            stopping_criteria: A list of stopping criteria to use.
            logit_bias: A logit bias to use.
            logits_processor: A list of logits processors to use.
            grammar: A grammar to use for constrained sampling.

        Raises:
            ValueError: If the requested tokens exceed the context window.
            RuntimeError: If the prompt fails to tokenize or the model fails to evaluate the prompt.

        Returns:
            Response object containing the generated text.
        """
        completion_or_chunks = self._create_completion(
            prompt=prompt,
            suffix=suffix,
            max_tokens=-1 if max_tokens is None else max_tokens,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            typical_p=typical_p,
            logprobs=logprobs,
            echo=echo,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            top_k=top_k,
            top_n_sigma=top_n_sigma,
            stream=stream,
            seed=seed,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            xtc_threshold=xtc_threshold,
            xtc_probability=xtc_probability,
            dry_multiplier=dry_multiplier,
            dry_base=dry_base,
            dry_allowed_length=dry_allowed_length,
            dry_penalty_last_n=dry_penalty_last_n,
            dry_seq_breakers=dry_seq_breakers,
            model=model,
            stopping_criteria=stopping_criteria,
            logit_bias=logit_bias,
            logits_processor=logits_processor,
            grammar=grammar,
            detokenize_special=detokenize_special,
        )
        if stream:
            chunks: Iterator[CreateCompletionStreamResponse] = completion_or_chunks
            return chunks
        completion: Completion = next(completion_or_chunks)  # type: ignore
        return completion


        

    def __call__(
        self,
        prompt: str,
        suffix: Optional[str] = None,
        max_tokens: Optional[int] = 16,
        temperature: float = 0.8,
        top_p: float = 0.95,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        logprobs: Optional[int] = None,
        echo: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.0,
        top_k: int = 40,
        top_n_sigma: float = -1.00,
        stream: bool = False,
        seed: Optional[int] = None,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        xtc_threshold: float = 0.1,
        xtc_probability: float = 0.0,
        dry_multiplier: float = 0.0,
        dry_base: float = 1.75,
        dry_allowed_length: int = 2,
        dry_penalty_last_n:int = 0,
        dry_seq_breakers: list[str] = ["\n", ":", "\"", "*"],
        model: Optional[str] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logit_bias: Optional[Dict[int, float]] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        grammar: Optional[LlamaGrammar] = None,
        detokenize_special: bool = True,
    ) -> Union[CreateCompletionResponse, Iterator[CreateCompletionStreamResponse]]:
        """Generate text from a prompt.

        Args:
            prompt: The prompt to generate text from.
            suffix: A suffix to append to the generated text. If None, no suffix is appended.
            max_tokens: The maximum number of tokens to generate. If max_tokens <= 0 or None, the maximum number of tokens to generate is unlimited and depends on n_ctx.
            temperature: The temperature to use for sampling.
            top_p: The top-p value to use for nucleus sampling. Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
            min_p: The min-p value to use for minimum p sampling. Minimum P sampling as described in https://github.com/ggml-org/llama.cpp/pull/3841
            typical_p: The typical-p value to use for sampling. Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
            logprobs: The number of logprobs to return. If None, no logprobs are returned.
            echo: Whether to echo the prompt.
            stop: A list of strings to stop generation when encountered.
            frequency_penalty: The penalty to apply to tokens based on their frequency in the prompt.
            presence_penalty: The penalty to apply to tokens based on their presence in the prompt.
            repeat_penalty: The penalty to apply to repeated tokens.
            top_k: The top-k value to use for sampling. Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
            top_n_sigma: Limit the next token selection to a subset of tokens with pre-softmax logits that are within n * σ less than the max logit (default: -1.00, -1.00 = disabled).
            stream: Whether to stream the results.
            seed: The seed to use for sampling.
            mirostat_mode: The mirostat sampling mode.
            mirostat_tau: The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
            mirostat_eta: The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
            xtc-probability: Sets the chance for token removal (checked once on sampler start) (default: 0.0). XTC sampler as described in https://github.com/oobabooga/text-generation-webui/pull/6335
            xtc-threshold: Sets a minimum probability threshold for tokens to be removed (default: 0.1). XTC sampler as described in https://github.com/oobabooga/text-generation-webui/pull/6335
            dry_multiplier: Set the DRY (Don't Repeat Yourself) repetition penalty multiplier. Default: `0.0`, which is disabled.
            dry_base`: Set the DRY repetition penalty base value. Default: `1.75`
            dry_allowed_length: Tokens that extend repetition beyond this receive exponentially increasing penalty: multiplier * base ^ (length of repeating sequence before token - allowed length). Default: `2`
            dry_penalty_last_n: How many tokens to scan for repetitions. Default: `0`, where `0` is disabled and `-1` is context size.
            dry_seq_breakers: Specify an array of sequence breakers for DRY sampling. Only a JSON array of strings is accepted. Default: `['\n', ':', '"', '*']`
            model: The name to use for the model in the completion object.
            stopping_criteria: A list of stopping criteria to use.
            logit_bias: A logit bias to use.
            logits_processor: A list of logits processors to use.
            grammar: A grammar to use for constrained sampling.

        Raises:
            ValueError: If the requested tokens exceed the context window.
            RuntimeError: If the prompt fails to tokenize or the model fails to evaluate the prompt.

        Returns:
            Response object containing the generated text.
        """
        return self.create_completion(
            prompt=prompt,
            suffix=suffix,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            typical_p=typical_p,
            logprobs=logprobs,
            echo=echo,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            top_k=top_k,
            top_n_sigma=top_n_sigma,
            stream=stream,
            seed=seed,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            xtc_threshold=xtc_threshold,
            xtc_probability=xtc_probability,
            dry_multiplier=dry_multiplier,
            dry_base=dry_base,
            dry_allowed_length=dry_allowed_length,
            dry_penalty_last_n=dry_penalty_last_n,
            dry_seq_breakers=dry_seq_breakers,
            model=model,
            stopping_criteria=stopping_criteria,
            logit_bias=logit_bias,
            logits_processor=logits_processor,
            grammar=grammar,
            detokenize_special=detokenize_special,
        )








# MULTI MODAL

IMAGE_URL_TAGS = ("<|image_url_starts|>", "<|image_url_ends|>")

proj_types = [
    "PROJECTOR_TYPE_PIXTRAL",
    "PROJECTOR_TYPE_QWEN2VL",
    "PROJECTOR_TYPE_QWEN25VL",
    "PROJECTOR_TYPE_QWEN3VL",
    "PROJECTOR_TYPE_LLAMA4",
    "PROJECTOR_TYPE_INTERNVL",
    "PROJECTOR_TYPE_LIGHTONOCR",
    "PROJECTOR_TYPE_LFM2",
    "PROJECTOR_TYPE_GLM4V",
]

def get_boi_eoi(proj_type: str):
    """
    Reference:
        https://github.com/ggml-org/llama.cpp/blob/a52dc60ba3ae0ef1e941ce9a4585672cc335a175/tools/mtmd/mtmd.cpp#L293
    """
    return {
        "PROJECTOR_TYPE_GEMMA3": ("<start_of_image>", "<end_of_image>"),
        #"PROJECTOR_TYPE_IDEFICS3": ("", ""),
        "PROJECTOR_TYPE_PIXTRAL": ("", "[IMG_END]"),
        "PROJECTOR_TYPE_QWEN2VL": ("<|vision_start|>", "<|vision_end|>"),
        "PROJECTOR_TYPE_QWEN25VL": ("<|vision_start|>", "<|vision_end|>"),
        "PROJECTOR_TYPE_QWEN3VL": ("<|vision_start|>", "<|vision_end|>"),
        "PROJECTOR_TYPE_LLAMA4": ("<|image_start|>", "<|image_end|>"),
        "PROJECTOR_TYPE_INTERNVL": ("<img>", "</img>"),
        "PROJECTOR_TYPE_LIGHTONOCR": ("<|im_start|>", "<|im_end|>"),
        "PROJECTOR_TYPE_LFM2": ("<|image_start|>", "<|image_end|>"),
        "PROJECTOR_TYPE_GLM4V": ("<|begin_of_image|>", "<|end_of_image|>"),
    }[proj_type]



class MTMDCpp(Llava15ChatHandler):
    """
    llama.cpp or JamesPeng's llama-cpp-python provides more utility functions related to mtmd.
    It might be better to implement methods based on them.
    """

    
    
    @staticmethod
    def _load_file(path: str) -> bytes:
        with open(path, "rb") as f:
            return f.read()
            
    @staticmethod
    def _load_image(path: str) -> bytes:
        return MTMDCpp._load_file(path)

    @staticmethod
    def hash_file(path, length: int = 8) -> str:
        import hashlib
        hash = hashlib.sha256(MTMDCpp._load_file(path)).hexdigest()
        return hash[:length]

    @staticmethod
    def remove_prefix(text, prefix) -> str:
        if not text.startswith(prefix):
            message = f"""The text does not start with the given prefix.
Given text: '''{text}'''


Given prefix: '''{prefix}'''"""
            raise ValueError(message)
        return text[len(prefix):]
            
    @staticmethod
    def get_image_urls(prompt):
        import re
        image_urls: List[str] = []
        starts, ends = IMAGE_URL_TAGS[0], IMAGE_URL_TAGS[1]
        return re.findall(f"{starts.replace("|", "\\|")}(.+?){ends.replace("|", "\\|")}", prompt)

    @staticmethod
    def input_text(llama, n_tokens: Optional[int] = None) -> str:
        if n_tokens is None:
            n_tokens = llama.n_tokens
        elif n_tokens == -1:
            n_tokens = len(llama.input_ids)
        input_ids = llama.input_ids[:n_tokens]
        input_text = llama.detokenize(input_ids, special=True).decode("utf-8")
        return input_text
        
    @staticmethod
    def shared_prefix(t1: str, t2: str, llama) -> str:
        """ 
        Returns shared initial part of the paired texts given. 
        This method is desinged not to cut in the middle of token like;
        "<|im_start|>" and "<|im_end|>" -> "<|im_"
        """
        prefix = []
        for t, s in zip(
            llama.tokenize(t1.encode(), add_bos=False, special=True), 
            llama.tokenize(t2.encode(), add_bos=False, special=True)
        ):
            if t != s:
                break
            prefix.append(t)
        return llama.detokenize(prefix, special=True).decode("utf-8")

    @staticmethod
    def insert_hash_before_image_tags(
        text: str,
        *,
        skip_if_already_hashed: bool = True,
        on_missing_file: str = "raise",  # "raise" | "keep" | "empty"
    ) -> str:
        """
        Insert: <|hash_starts|>HASH<|hash_ends|> immediately before each image tag block:
          <|image_url_starts|>/path/to/img.jpg<|image_url_ends|>
    
        Example output:
          ...<|hash_starts|>abc123<|hash_ends|><|image_url_starts|>/path...<|image_url_ends|>...
    
        Parameters
        ----------
        text:
            Input text containing 0+ image tag blocks.
        skip_if_already_hashed:
            If True, won’t insert another <hash>...</hash> if one is already
            immediately before the image tag (allowing whitespace).
        on_missing_file:
            What to do if the path doesn’t exist:
            - "raise": raise FileNotFoundError
            - "keep": leave that tag untouched (no insertion)
            - "empty": insert <|hash_starts|><|hash_ends|> (empty hash)
        """
        import re
        from pathlib import Path
        from typing import Callable, Dict, Optional
        
        start_tag, end_tag = IMAGE_URL_TAGS
        pattern = re.compile(
            re.escape(start_tag) + r"(.*?)" + re.escape(end_tag),
            flags=re.DOTALL,
        )
        already_hash_pat = re.compile(r"<\|hash_starts\|>.*?<\|hash_ends\|>\s*$", flags=re.DOTALL)
    
        cache: Dict[str, str] = {}
    
        def repl(m: re.Match) -> str:
            nonlocal text
            start_idx = m.start()
    
            if skip_if_already_hashed:
                prefix = text[:start_idx]
                if already_hash_pat.search(prefix) is not None:
                    return m.group(0)
    
            raw_path = m.group(1).strip()
            # Keep original raw_path inside the tag; we only need it for hashing.
            norm_key = str(Path(raw_path))
    
            if norm_key in cache:
                h = cache[norm_key]
            else:
                p = Path(raw_path)
                if not p.exists():
                    if on_missing_file == "raise":
                        raise FileNotFoundError(f"Image path not found: {raw_path}")
                    if on_missing_file == "keep":
                        return m.group(0)
                    if on_missing_file == "empty":
                        h = ""
                    else:
                        raise ValueError(f"Unknown on_missing_file={on_missing_file!r}")
                else:
                    h = MTMDCpp.hash_file(raw_path)
                cache[norm_key] = h
    
            return f"<|hash_starts|>{h}<|hash_ends|>{m.group(0)}"
    
        # Note: repl() reads `text` to check idempotency; so we run sub on the original.
        return pattern.sub(repl, text)
    


    def token_count(
        self, 
        text, 
        llama: LlamaCppSpecial,
        add_bos=False, 
        special=True,
        hash_insertion=True,
    ) -> int:
        # Initialize mtmd context
        self._init_mtmd_context(llama)
        assert self.mtmd_ctx is not None


        if hash_insertion:
            text = self.insert_hash_before_image_tags(text)

        image_urls = self.get_image_urls(text)

        # Get the default media marker
        media_marker = self._mtmd_cpp.mtmd_default_marker().decode('utf-8')

        # Replace image URLs in text with media markers
        for image_url in image_urls:
            text = text.replace(image_url, media_marker)
            
        # Remove tags
        url_start, url_end = IMAGE_URL_TAGS
        text = text.replace(url_start, "").replace(url_end, "")

        # Create bitmaps from images
        bitmaps = []
        bitmap_cleanup = []
        try:
            for image_url in image_urls:
                image_bytes = self.load_image(image_url)
                bitmap = self._create_bitmap_from_bytes(image_bytes)
                bitmaps.append(bitmap)
                bitmap_cleanup.append(bitmap)

            # Create input text structure
            bos = llama.detokenize([llama.token_bos()], special=True).decode("utf-8")
            text = (bos if add_bos else "") + text
            input_text = self._mtmd_cpp.mtmd_input_text()
            input_text.text = text.encode('utf-8')
            input_text.add_special = False
            input_text.parse_special = special

            # Create input chunks
            chunks = self._mtmd_cpp.mtmd_input_chunks_init()
            if chunks is None:
                raise ValueError("Failed to create input chunks")

            try:
                # Tokenize text and images together
                bitmap_array = (self._mtmd_cpp.mtmd_bitmap_p_ctypes * len(bitmaps))(*bitmaps)
                result = self._mtmd_cpp.mtmd_tokenize(
                    self.mtmd_ctx,
                    chunks,
                    ctypes.byref(input_text),
                    bitmap_array,
                    len(bitmaps)
                )
                
                # Process each chunk
                n_chunks = self._mtmd_cpp.mtmd_input_chunks_size(chunks)

                n_tokens: int = 0
                for i in range(n_chunks):
                    chunk = self._mtmd_cpp.mtmd_input_chunks_get(chunks, i)
                    if chunk is None: continue

                    chunk_type = self._mtmd_cpp.mtmd_input_chunk_get_type(chunk)

                    if chunk_type == self._mtmd_cpp.mtmd_input_chunk_type.MTMD_INPUT_CHUNK_TYPE_TEXT:
                        n_tokens_out = ctypes.c_size_t()
                        tokens_ptr = self._mtmd_cpp.mtmd_input_chunk_get_tokens_text(chunk, ctypes.byref(n_tokens_out))
                        n_tokens += n_tokens_out.value
                    elif chunk_type in [
                        self._mtmd_cpp.mtmd_input_chunk_type.MTMD_INPUT_CHUNK_TYPE_IMAGE, 
                        self._mtmd_cpp.mtmd_input_chunk_type.MTMD_INPUT_CHUNK_TYPE_AUDIO
                    ]:
                        n_pos = self._mtmd_cpp.mtmd_input_chunk_get_n_pos(chunk)
                        n_tokens += n_pos
            finally:
                self._mtmd_cpp.mtmd_input_chunks_free(chunks)
        finally:
            for bitmap in bitmap_cleanup:
                self._mtmd_cpp.mtmd_bitmap_free(bitmap)
            return n_tokens


    def create_completion(
        self,
        *,
        llama: LlamaCppSpecial,
        prompt: str,
        prefix_to_skip_reeval: Optional[str] = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 40,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        seed: Optional[int] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        top_n_sigma: float = -1.00,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        xtc_threshold: float = 0.1,
        xtc_probability: float = 0.0,
        dry_multiplier: float = 0.0,
        dry_base: float = 1.75,
        dry_allowed_length: int = 2,
        dry_penalty_last_n:int = 0,
        dry_seq_breakers: list[str] = ["\n", ":", "\"", "*"],
        model: Optional[str] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        grammar: Optional[LlamaGrammar] = None,
        logit_bias: Optional[Dict[str, float]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        detokenize_special = True,
        **kwargs,  # type: ignore
    ) -> Union[llama_types.CreateChatCompletionResponse, Iterator[llama_types.CreateChatCompletionStreamResponse], ]:
        # Initialize mtmd context
        self._init_mtmd_context(llama)
        assert self.mtmd_ctx is not None

        
        # Preprocess prompt
        if prefix_to_skip_reeval is not None:
            try:
                prefix_to_skip_reeval = self.insert_hash_before_image_tags(prefix_to_skip_reeval)
                prompt = self.insert_hash_before_image_tags(prompt)
                
                prefix_to_skip_reeval = self.shared_prefix(prompt, prefix_to_skip_reeval, llama)
                prompt = self.remove_prefix(prompt, prefix_to_skip_reeval)
            except BaseException as e:
                print(e)
                prefix_to_skip_reeval = None
        
        image_urls = self.get_image_urls(prompt)
        
        # Remove tags and Replace image URLs in text with media markers
        # Before: <|image_url_starts|>/path/to/image.jpg<|image_url_ends|>
        url_start, url_end = IMAGE_URL_TAGS
        text = prompt.replace(url_start, "").replace(url_end, "")
        media_marker = self._mtmd_cpp.mtmd_default_marker().decode('utf-8')
        for image_url in image_urls:
            text = text.replace(image_url, media_marker)
            
        # At this point, the prompt became like;
        # From: <|image_url_starts|>/path/to/image.jpg<|image_url_ends|>
        # From: <|hash_starts|>12345678<|hash_ends|><__media__>
        
        if self.verbose:
            print(text, file=sys.stderr)
        
        # Create bitmaps from images
        bitmaps = []
        bitmap_cleanup = []
        try:
            for image_url in image_urls:
                image_bytes = self.load_image(image_url)
                bitmap = self._create_bitmap_from_bytes(image_bytes)
                bitmaps.append(bitmap)
                bitmap_cleanup.append(bitmap)
            
            # Create input text structure
            input_text = self._mtmd_cpp.mtmd_input_text()
            input_text.text = text.encode('utf-8')
            input_text.add_special = True
            input_text.parse_special = True

            # Create input chunks
            chunks = self._mtmd_cpp.mtmd_input_chunks_init()
            if chunks is None:
                raise ValueError("Failed to create input chunks")

            try:
                # Tokenize text and images together
                bitmap_array = (self._mtmd_cpp.mtmd_bitmap_p_ctypes * len(bitmaps))(*bitmaps)
                result = self._mtmd_cpp.mtmd_tokenize(
                    self.mtmd_ctx,
                    chunks,
                    ctypes.byref(input_text),
                    bitmap_array,
                    len(bitmaps)
                )

                if result != 0:
                    raise ValueError(f"Failed to tokenize input: error code {result}")

                # Reset llama context
                if prefix_to_skip_reeval is not None:
                    n_prefix_tokens = self.token_count(prefix_to_skip_reeval, llama, special=True, hash_insertion=False)
                    llama.n_tokens = n_prefix_tokens
                    n_past = llama.n_tokens
                    print(f"Prefix matches and skipped reevaluation of {n_prefix_tokens} tokens.")
                else:
                    llama.reset()
                    llama._ctx.memory_clear(True)
                    llama.n_tokens = 0
                    llama.input_ids = np.zeros(len(llama.input_ids), dtype=llama.input_ids.dtype)
                    n_past = 0

                # Process each chunk
                n_chunks = self._mtmd_cpp.mtmd_input_chunks_size(chunks)

                for i in range(n_chunks):
                    chunk = self._mtmd_cpp.mtmd_input_chunks_get(chunks, i)
                    if chunk is None: continue

                    chunk_type = self._mtmd_cpp.mtmd_input_chunk_get_type(chunk)

                    # The first and last chunk must be TEXT.
                    if i == 0 or i == n_chunks - 1:
                        assert chunk_type == self._mtmd_cpp.mtmd_input_chunk_type.MTMD_INPUT_CHUNK_TYPE_TEXT

                    if chunk_type == self._mtmd_cpp.mtmd_input_chunk_type.MTMD_INPUT_CHUNK_TYPE_TEXT:
                        # Handle text chunk
                        n_tokens_out = ctypes.c_size_t()
                        tokens_ptr = self._mtmd_cpp.mtmd_input_chunk_get_tokens_text(chunk, ctypes.byref(n_tokens_out))
                        chunk_n_tokens = n_tokens_out.value

                        if tokens_ptr and n_tokens_out.value > 0:
                            # Convert ctypes array to Python list
                            tokens = [tokens_ptr[j] for j in range(n_tokens_out.value)]

                            if llama.n_tokens + len(tokens) > llama.n_ctx():
                                raise ValueError(
                                    f"Prompt exceeds n_ctx: {llama.n_tokens + len(tokens)} > {llama.n_ctx()}"
                                )
                            #llama.n_tokens = n_past
                            llama.eval(tokens)
                            n_past = llama.n_tokens

                    elif chunk_type in [
                        self._mtmd_cpp.mtmd_input_chunk_type.MTMD_INPUT_CHUNK_TYPE_IMAGE, 
                        self._mtmd_cpp.mtmd_input_chunk_type.MTMD_INPUT_CHUNK_TYPE_AUDIO
                    ]:
                        # Handle image/audio chunk using helper
                        chunk_n_tokens = self._mtmd_cpp.mtmd_input_chunk_get_n_tokens(chunk)

                        if n_past + chunk_n_tokens > llama.n_ctx():
                            raise ValueError(
                                f"Prompt exceeds n_ctx: {n_past + chunk_n_tokens} > {llama.n_ctx()}"
                            )

                        new_n_past = llama_cpp.llama_pos(0)
                        result = self._mtmd_cpp.mtmd_helper_eval_chunk_single(
                            self.mtmd_ctx,
                            llama._ctx.ctx,
                            chunk,
                            llama_cpp.llama_pos(llama.n_tokens),
                            llama_cpp.llama_seq_id(0),
                            llama.n_batch,
                            False,  # logits_last
                            ctypes.byref(new_n_past)
                        )
                        for k in range(llama.n_tokens, new_n_past.value):
                            llama.input_ids[k] = 0

                        if result != 0:
                            raise ValueError(f"Failed to evaluate chunk: error code {result}")

                        # Update llama's token count
                        # `llama.eval` automatically increases `llama.n_tokens` but `mtmd_helper_eval_chunk_single` doesn't.
                        llama.n_tokens = new_n_past.value

                # Get prompt tokens to avoid a cache miss
                prompt = llama.input_ids[: llama.n_tokens].tolist()

            finally:
                self._mtmd_cpp.mtmd_input_chunks_free(chunks)

        finally:
            # Cleanup bitmaps
            for bitmap in bitmap_cleanup:
                self._mtmd_cpp.mtmd_bitmap_free(bitmap)


        return llama.create_completion(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            typical_p=typical_p,
            logprobs=top_logprobs if logprobs else None,
            stream=stream,
            stop=stop,
            seed=seed,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repeat_penalty=repeat_penalty,
            top_n_sigma=top_n_sigma,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            xtc_threshold=xtc_threshold,
            xtc_probability=xtc_probability,
            dry_multiplier=dry_multiplier,
            dry_base=dry_base,
            dry_allowed_length=dry_allowed_length,
            dry_penalty_last_n=dry_penalty_last_n,
            dry_seq_breakers=dry_seq_breakers,
            model=model,
            logits_processor=logits_processor,
            grammar=grammar,
            logit_bias=logit_bias,
            detokenize_special=detokenize_special,
        )

    def __call__(self, *args, **kwargs):
        return self.create_completion(*args, **kwargs)
        
    def mtmd_free(self):
        if self.mtmd_ctx is not None:
            self._mtmd_cpp.mtmd_free(self.mtmd_ctx)
            self.mtmd_ctx = None



class MTMDCppAutoPrefix(MTMDCpp):
    boi_eoi: Tuple[str, str]
    def __init__(self, clip_model_path, boi_eoi: Tuple[str, str], *args, **kwargs) -> None:
        super().__init__(clip_model_path, *args, **kwargs)
        self.boi_eoi = boi_eoi
        
    def extract_prefix(self, prompt: str, llama: LlamaCppSpecial) -> str:
        import re
        from lib.text_utils import indexed_placeholders
        
        boi, eoi = self.boi_eoi
        url_starts, url_ends = IMAGE_URL_TAGS
        boi_escaped = boi.replace("|", r"\|").replace("[", r"\[").replace("]", r"\]")
        eoi_escaped = eoi.replace("|", r"\|").replace("[", r"\[").replace("]", r"\]")
        input_ids = llama.input_ids[:llama.n_tokens]
        input_text = llama.detokenize(input_ids, special=True).decode("utf-8")

        input_text = indexed_placeholders(
            target=input_text, 
            pattern=boi_escaped + ".+" + eoi_escaped,
            placeholder=f"{url_starts}{{num}}{url_ends}"
        )
        
        image_urls = self.get_image_urls(prompt)
        for i in range(len(image_urls)):
            input_text = input_text.replace(
                f"{url_starts}{i}{url_ends}", 
                f"{url_starts}{image_urls[i]}{url_ends}"
            )
        # Remove hash tags AFTER slicing.
        # The order matters, because, otherwise, it doesn't know the images are identical or not without hash value.
        input_text = self.shared_prefix(
            input_text, 
            self.insert_hash_before_image_tags(prompt),
            llama,
        )
        input_text = re.sub(r"<\|hash_starts\|>.+?<\|hash_ends\|>", "", input_text)
        input_text = input_text.split("<|hash_starts|>")[0]
        return input_text

    def create_completion(self, prompt, llama, *args, **kwargs):
        prefix = self.extract_prefix(prompt, llama)
        return super().create_completion(
            prompt=prompt,
            llama=llama,
            prefix_to_skip_reeval=prefix,
            *args, **kwargs
        )