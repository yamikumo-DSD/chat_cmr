from collections import deque

class ForgetableContext:
    __history: deque
    __maxlen: int

    def __init__(self, maxlen: int) -> None:
        """ maxlen<0 for infinite length"""
        self.__history = deque([])
        self.__maxlen = maxlen

    def reset(self) -> None:
        self.__history = deque([])
    def push(self, item: str|dict) -> None:
        if isinstance(item, dict):
            try:
                item['role']
                item['content']
            except KeyError:
                raise ValueError('Item must have both role and content.')
            self.__history.append(item)
        elif isinstance(item, str):
            self.__history.append({'role': '', 'content': item})
        else:
            raise ValueError('The item is neither str nor dict object.')
    def push_message(self, role: str, content: str) -> None:
        self.__history.append({'role': role, 'content': content})
    def force_pop_front(self, stringize: bool = True) -> str|dict:
        item: dict = self.__history.pop()
        return f'{item["role"]}:{item["content"]}' if stringize else item
    def history(self, stringize: bool = False) -> list:
        if not stringize:
            return list(self.__history)
            
        stringized = []
        for item in list(self.__history):
            if item["role"] == '': 
                stringized.append(item["content"])
            else:
                stringized.append(f'{item["role"]}:{item["content"]}')
        return stringized
        
    def context(self, stringize: bool = False) -> list:
        # Slice if maxlen set, otherwise return as it is.
        return self.history(stringize)[-self.__maxlen:] if self.__maxlen >= 0 else self.history(stringize)
        
    def __str__(self) -> str:
        return '\n'.join(self.context(stringize=True))
        
    def __len__(self) -> int:
        return len(self.__history)

    def __iter__(self):
        self.__index = 0
        return self
    
    def __next__(self):
        if self.__index < len(self.__history):
            result = self.__history[self.__index]
            self.__index += 1
            return result
        else:
            raise StopIteration


class LlamaCppWrapper:
    def __init__(self, *args, **kwargs) -> None:
        from llama_cpp import Llama
        self.model = Llama(*args, **kwargs)
        
    def __call__(self, *args, **kwargs) -> str:
        stream = kwargs['stream'] if 'stream' in kwargs.keys() else False
        
        if not stream:
            out = self.model(*args, **kwargs)
            return out['choices'][0]['text']

        def wrapper():
            generator = self.model(*args, **kwargs)
            for out in generator:
                yield out['choices'][0]['text']

        return wrapper()

    def create_embedding(self, text: list[str]):
        """llama-cpp-python compatible API."""
        return self.model.create_embedding(text)

    def embed_query(self, query: str) -> list:
        """langchain-community compatible API for one query."""
        return self.create_embedding(query)['data'][0]['embedding']
    def embed_documents(self, documents: str) -> list[list]:
        """langchain-community compatible API for list or queries."""
        embeddings = self.create_embedding(documents)['data']
        return [embedding['embedding'] for embedding in embeddings]

from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from typing import Any
from pydantic import Field


class LlamaCpp(LLM):
    model: LlamaCppWrapper = Field(description='Model')

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            model=LlamaCppWrapper(*args, **kwargs)
        )
    
    @property
    def _llm_type(self) -> str:
        return 'custom'

    @property
    def llama(self):
        return self.model.model
        
    def tokenize(self, text: str):
        """ This function doesn't add bos token. """
        return self.llama.tokenize(text.encode("utf-8"), add_bos=False, special=True)
    
    def detokenize(self, tokens) -> str:
        return self.llama.detokenize(tokens).decode("utf-8")
    
    def token_count(self, text: str):
        return len(self.tokenize(text))
    
    def _call(
       self,
       prompt: str,
       max_tokens: int = 1024,
       stop: list[str]|None = None,
       run_manager: CallbackManagerForLLMRun|None = None,
       **kwargs,
    ) -> str:
        output = self.model(
            prompt=prompt, 
            stop=stop, 
            max_tokens=max_tokens, 
            stream=False,
            **kwargs
        )
        return output

    def _stream(
       self,
       prompt: str,
       max_tokens: int = 1024,
       stop: list[str]|None = None,
       run_manager: CallbackManagerForLLMRun|None = None,
       **kwargs,
    ):
        
        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk
        
        tokens = self.model(
            prompt=prompt, 
            stream=True, 
            stop=stop, 
            max_tokens=max_tokens, 
            **kwargs
        )
        
        for token in tokens:
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))

            if run_manager:
                run_manager.on_llm_new_token(token, chunk=chunk)

            yield chunk
        

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Get the identifying parameters."""
        return {"model": self.model}




from typing import Any, Iterator, List, Mapping, Optional
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.pydantic_v1 import Extra

#DEFAULT_MODEL_ID = "mlx-community/quantized-gemma-2b"
#
#class MLXPipeline(LLM):
#    model_id: str = DEFAULT_MODEL_ID
#    """Model name to use."""
#    model: Any  #: :meta private:
#    """Model."""
#    tokenizer: Any  #: :meta private:
#    """Tokenizer."""
#    tokenizer_config: Optional[dict] = None
#    """
#        Configuration parameters specifically for the tokenizer.
#        Defaults to an empty dictionary.
#    """
#    adapter_file: Optional[str] = None
#    """
#        Path to the adapter file. If provided, applies LoRA layers to the model.
#        Defaults to None.
#    """
#    lazy: bool = False
#    """
#        If False eval the model parameters to make sure they are
#        loaded in memory before returning, otherwise they will be loaded
#        when needed. Default: ``False``
#    """
#
#    class Config:
#        """Configuration for this pydantic object."""
#
#        extra = Extra.forbid
#
#    @property
#    def _identifying_params(self) -> Mapping[str, Any]:
#        """Get the identifying parameters."""
#        return {
#            "model_id": self.model_id,
#            "tokenizer_config": self.tokenizer_config,
#            "adapter_file": self.adapter_file,
#            "lazy": self.lazy,
#        }
#
#    @property
#    def _llm_type(self) -> str:
#        return "mlx_pipeline"
#
#    def _stream(
#        self,
#        prompt: str,
#        stop: Optional[List[str]] = None,
#        temperature: Optional[float] = None,
#        run_manager: Optional[CallbackManagerForLLMRun] = None,
#        **kwargs: Any,
#    ) -> Iterator[GenerationChunk]:
#        try:
#            import mlx.core as mx
#            from mlx_lm.utils import generate_step
#
#        except ImportError:
#            raise ValueError(
#                "Could not import mlx_lm python package. "
#                "Please install it with `pip install mlx_lm`."
#            )
#
#        pipeline_kwargs = kwargs.get("pipeline_kwargs", {})
#
#        temp: float = temperature if temperature is not None else pipeline_kwargs.get("temp", 0.0)
#
#        max_new_tokens: int = pipeline_kwargs.get("max_tokens", 100)
#        repetition_penalty: Optional[float] = pipeline_kwargs.get(
#            "repetition_penalty", None
#        )
#        repetition_context_size: Optional[int] = pipeline_kwargs.get(
#            "repetition_context_size", None
#        )
#
#        prompt = self.tokenizer.encode(prompt, return_tensors="np")
#
#        prompt_tokens = mx.array(prompt[0])
#
#        eos_token_id = self.tokenizer.eos_token_id
#
#        total_output = ''
#
#        for (token, prob), n in zip(
#            generate_step(
#                prompt_tokens,
#                self.model,
#                temp,
#                repetition_penalty,
#                repetition_context_size,
#            ),
#            range(max_new_tokens),
#        ):
#            # identify text to yield
#            text: Optional[str] = None
#            text = self.tokenizer.decode(token)
#            total_output += text
#
#            # yield text, if any
#            if text:
#                chunk = GenerationChunk(text=text)
#                yield chunk
#                if run_manager:
#                    run_manager.on_llm_new_token(chunk.text)
#
#            # break if stop sequence found
#            if token == eos_token_id:
#                break
#            if stop:
#                for stop_token in stop:
#                    if stop_token in total_output:
#                        return
#                        
#
#    def _call(
#        self,
#        prompt: str,
#        stop: Optional[List[str]] = None,
#        temperature: Optional[float] = None,
#        run_manager: Optional[CallbackManagerForLLMRun] = None,
#        **kwargs: Any,
#    ) -> str:
#        streamer = self._stream(prompt, stop, temperature, run_manager, **kwargs)
#        output = ""
#        for chunk in streamer:
#            output += chunk.text
#        for stop_token in stop:
#            if stop_token in output:
#                output = output.split(stop_token)[0]
#        return output



from typing import List
import numpy as np
from collections.abc import Iterable
import llama_cpp

class LlamaInputIDManager:
    model: llama_cpp.Llama
    n_ctx: int
    def __init__(self, model: llama_cpp.Llama) -> None:
        self.model = model
        self.n_ctx = model.context_params.n_ctx
    def reset(self) -> None:
        self.model.input_ids = np.zeros(self.n_ctx, dtype=int)
    def decode(self, encoding: str = "utf-8") -> str:
        return self.model.detokenize(self.model.input_ids).decode(encoding)
    def set(self, input_: List[int]|bytes|str, encoding: str = "utf-8") -> None:
        """
        Tokens(List[int]), encoded prompt(bytes), or prompt(str).
        Argument "encoding" is used only when str object is given as input.
        """
        tokens: List[int]
        if isinstance(input_, Iterable) and isinstance(input_[0], int):
            tokens = input_
        elif isinstance(input_, bytes):
            tokens = self.model.tokenize(input_)
        elif isinstance(input_, str):
            tokens = self.model.tokenize(input_.encode(encoding))
        else:
            raise ValueError("Check type of input.")
        
        if len(tokens) > self.n_ctx:
            raise ValueError(f"Length of tokens passed ({len(tokens)}) > n_ctx ({self.n_ctx})")
        self.model.input_ids = np.pad(tokens, (0, self.n_ctx - len(tokens)), "constant")
    def __len__(self) -> int:
        return len(self.model.input_ids)
    def pos(self) -> int:
        return list(self.model.input_ids).index(0) if 0 in self.model.input_ids else self.n_ctx - 1



def temporal_llama_cache(llama_model, llama_cache):
    import functools
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from llama_cpp import Llama
            if not isinstance(llama_model, Llama):
                raise ValueError("Model must be llama_cpp.Llama model.")
            temp_cache = llama_model.cache
            llama_model.set_cache(llama_cache)
            result = func(*args, **kwargs)
            llama_model.cache = temp_cache # Return
            return result
        return wrapper
    return decorator



def kv_cache_seq_ltrim(
    model: llama_cpp.Llama, 
    n_keep: int, 
    n_discard: int,
):
    """
    Implementation comes from this GitHub repository:
        https://github.com/Limour-dev/llama-python-streamingllm/blob/main/llama_cpp_python_streamingllm.py
    
    Args:
        n_keep(int): number of first tokens to keep.
        n_discard(int) number of tokens to discard.
    Returns:
        None
    Schema:
          n_keep(3)  n_keep(3)+n_discard(3)
              |        |
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # Initial state.
    [0, 1, 2, -, -, -, 6, 7, 8, 9] # kv_cache_seq_rm
    [0, 1, 2, 6, 7, 8, 9] # kv_cache_seq_shift
    
    """
    n_tokens = model.n_tokens
    
    model._ctx.kv_cache_seq_rm(-1, n_keep, n_keep + n_discard)
    model._ctx.kv_cache_seq_shift(0, n_keep + n_discard, n_tokens, -n_discard)

    model.input_ids[n_keep:n_tokens - n_discard] = model.input_ids[n_keep + n_discard:n_tokens]
    model.n_tokens = n_tokens - n_discard