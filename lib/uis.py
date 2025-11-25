import IPython

import ipywidgets as widgets
from ipywidgets import Layout

def button_method(button: widgets.Button):
    def decorator(func):
        from functools import wraps
        button.on_click(func)
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator
    
def value_change_method(widget):
    def decorator(func):
        from functools import wraps
        widget.observe(func, "value")
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator
    
def activate_cancel_ui(wait_sec: int) -> bool:
    import time
    from ipywidgets import Button
    from jupyter_ui_poll import ui_events

    def on_click(btn):
        btn.description = 'Canceled'

    btn = Button(description=f'Cancel')
    btn.on_click(on_click)
    display(btn)

    # Wait for user to press the button
    with ui_events() as poll:
        count = 0
        while not btn.description == 'Canceled' and count < (wait_sec * 10):
            poll(3)          # React to UI events (upto 10 at a time)
            time.sleep(0.1)
            count = count + 1

    return btn.description == 'Canceled'


def wait_for_change(widget, value):
    """https://ipywidgets.readthedocs.io/en/7.x/examples/Widget%20Asynchronous.html"""
    import asyncio
    future = asyncio.Future()
    def getvalue(change):
        # make the new value available
        future.set_result(change.new)
        widget.unobserve(getvalue, value)
    widget.observe(getvalue, value)
    return future


class Switchable(widgets.Output):
    __switch: widgets.ToggleButton
    __output_keeper: tuple
    def __init__(self, **kwargs) -> None:
        """
        You should instantiate Switchable via factory method create().
        """
        super().__init__()
        def toggle(args) -> None:
            from copy import copy
            if self.__switch.value:
                # Restore outputs.
                self.outputs = self.__output_keeper
            else:
                self.__output_keeper = copy(self.outputs)
                self.clear_output()
        self.__switch = widgets.ToggleButton(value=True, **kwargs)
        self.__switch.observe(toggle, "value")
    def get_switch(self) -> widgets.ToggleButton:
        return self.__switch
    @staticmethod
    def create(**kwargs):
        """
        Returns:
            tuple[ToggleButton, Output]
        """
        obj = Switchable(**kwargs)
        return obj.get_switch(), obj



class Checkboxes:
    _items: list[widgets.Checkbox]
    _disabled: bool
    def __init__(
        self, 
        descriptions: list[str],
        values: list[bool]|None = None,
    ) -> None:
        if not values:
            values = [True for _ in range(len(descriptions))]
        if values and len(descriptions) != len(values):
            raise ValueError(f"Length of values({len(values)}) and descriptions({len(descriptions)}) doesn't match.")
            
        self._items = []
        self._disabled = False
        for i, description in enumerate(descriptions):
           self._items.append(widgets.Checkbox(
               description=description, 
               value=values[i],
               layout=Layout(width="auto"),
               style={"description_width": 'initial'},
           ))
    def display(self) -> None:
        return widgets.HBox(self._items)
    @property
    def value(self) -> list[bool]:
        return [item.value for item in self._items]
    @property
    def disabled(self) -> bool:
        return self._disabled
    @disabled.setter
    def disabled(self, value: bool) -> None:
        self._disabled = value
        if self._disabled:
            for item in self._items:
                item.disabled = True
        else:
            for item in self._items:
                item.disabled = False
    def __iter__(self):
        return iter(self._items)
    
    def __len__(self):
        return len(self._items)
    
    def __getitem__(self, index):
        return self._items[index]



def display_center_aligned(widget) -> None:
    from ipywidgets import HBox
    from IPython.display import display
    display(HBox(
        [widget], 
        layout = widgets.Layout(
            display="flex",
            justify_content="center",
        )
    ))


class ThreadSafeStdOutCapture:
    """
    A wrapper to port stdout/stderr into Output widget.
    Reason why this wrapper is needed:
        Normal context manager `with output:` doesn't actually capture outputs in thread or async method.
        ipywidgets privides workarounds to avoid it, which is "append"-family;
        
        `output.append_stdout`
        `output.append_stderr`
        `output.append_display_data`.

        This context managing class automatically capture stdout/stderr and send them to the `append_stdout`.
    """
    def __init__(self, output_widget: widgets.Output) -> None:
        from io import StringIO
        self._output = output_widget
        self._buf = StringIO()
        self._is_entering = False
        
    def __enter__(self):
        from contextlib import redirect_stdout, redirect_stderr
        if self._is_entering:
            raise RuntimeError("with statement of same instances of ThreadSaveStdOutCapture should not be nested or similteneously executed.")
        self._is_entering = True
        self._stdout_redirect = redirect_stdout(self._buf)
        self._stderr_redirect = redirect_stderr(self._buf)
        self._stdout_redirect.__enter__()
        self._stderr_redirect.__enter__()
        return self._buf
        
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._is_entering = False
        self._stdout_redirect.__exit__(exc_type, exc_value, traceback)
        self._stderr_redirect.__exit__(exc_type, exc_value, traceback)
        self._output.append_stdout(self._buf.getvalue())

    def clear_output(self) -> None:
        from io import StringIO
        self._output.outputs = ()
        self._buf = StringIO()


