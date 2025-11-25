from typing import Iterable
from typing import Any, Callable
from lib.text_utils import *
from warnings import simplefilter

ignore_warnings = lambda : simplefilter('ignore')

    
def change_directory(new_directory):
    import os
    import functools
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            prev_directory = os.getcwd()
            os.chdir(new_directory)
            try:
                result = func(*args, **kwargs)
            finally:
                os.chdir(prev_directory)
            return result
        return wrapper
    return decorator

    
def get(x: dict, key: str):
    return x.get(key)


def slice_true(x: Iterable, truth_values: Iterable[bool]):
    assert len(x) == len(truth_values), f"Length of x({len(x)}) and truth_values({len(truth_values)}) must match."
    return [item for item, truth_value in zip(x, truth_values) if truth_value]
    

class Observable:
    _value: Any|Callable[[], Any]
    _unobserve: bool = True
    _is_nullary_function: bool
    def __init__(
        self, 
        value: Any|Callable[[], Any],
        is_nullary_function: bool = False,
    ) -> None:
        """
        Args:
            value: initial value or function w/o arguments.
            is_nullary_function: if True, it observes changes in `value.__call__()` instead of the value itself.
        """
        self._value = value
        self._is_nullary_function = is_nullary_function

    @property
    def value(self) -> Any:
        return self._value() if self._is_nullary_function else self._value

    @value.setter
    def value(self, new_value) -> None:
        assert not self._is_nullary_function, "Nullary function value cannot be changed."
        self._value = new_value

    def unobserve(self) -> None:
        self._unobserve = True

    def observe(
        self, 
        on_change: Callable, 
        args: tuple = (), 
        interval: float = 1.,
    ) -> None:
        """
        on_change: callback
        args: callback arguments
        interval: sec
        """
        import asyncio
        from nest_asyncio import apply
        apply()

        self._unobserve = False

        async def monitor() -> None:
            while not self._unobserve:
                prev = self.value
                await asyncio.sleep(interval)
                if prev != self.value:
                    on_change(*args)

        asyncio.create_task(monitor())




def split_list(input_list, n):
    """
    Split a list into approximately equal sized chunks
    
    Args:
        input_list (list): The list to be split
        n (int): Number of desired chunks
        
    Returns:
        list: List of chunks
    """
    total_length = len(input_list)
    base_size = total_length // n
    remainder = total_length % n
    
    chunks = []
    start = 0
    
    for i in range(n):
        end = start + base_size + (1 if i < remainder else 0)
        chunks.append(input_list[start:end])
        start = end
        
    return chunks



    
def is_valid_path(text):
    """
    Check if the text is a valid path

    Args:
        text (str): Text to check

    Returns:
        dict: {
            'is_valid': bool,  # Whether the format is valid as a path
            'exists': bool,     # Whether it actually exists on filesystem
            'absolute': bool,   # Whether it's an absolute path
            'message': str       # Detailed message
        }
    """
    import os
    from pathlib import Path

    if not isinstance(text, str) or not text.strip():
        return {
            "is_valid": False,
            "exists": False,
            "absolute": False,
            "message": "Empty string or non-string input",
        }

    # Check path format based on OS (simplified version)
    is_valid_format = True
    message_parts = []

    if os.name == "nt":  # Windows
        if len(text) > 1 and text[1] == ":":
            # Drive letter (C:, D: etc.)
            if not text[0].isalpha():
                is_valid_format = False
                message_parts.append("Invalid drive letter")
        elif "\\" in text:
            # Contains backslash
            pass
    else:  # Unix-like systems
        if text.startswith("/"):
            # Absolute path
            pass
        elif "/" in text:
            # Relative path
            pass

    # Invalid path format
    if not is_valid_format:
        return {
            "is_valid": False,
            "exists": False,
            "absolute": False,
            "message": " ".join(message_parts) or "Invalid path format",
        }

    # Check if it's an absolute path
    is_absolute = os.path.isabs(text)

    try:
        # Check if the path exists
        path_exists = os.path.exists(text)

        if path_exists:
            message_parts.append("Exists")
        else:
            message_parts.append("Does not exist")

    except (OSError, TypeError) as e:
        path_exists = False
        message_parts.append(f"Cannot access: {str(e)}")

    return {
        "is_valid": True,
        "exists": path_exists,
        "absolute": is_absolute,
        "message": " ".join(message_parts),
    }