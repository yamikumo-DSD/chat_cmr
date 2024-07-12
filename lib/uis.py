import IPython

import ipywidgets as widgets

def ipyplayaudio(x, show: bool = True) -> None|IPython.display.Audio:
    rate = 48000
    audio = IPython.display.Audio(x, rate=rate, autoplay=True)
    if show:
        display(audio)
    else:
        return audio
    
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
    ) -> None:
        self._items = []
        self._disabled = False
        for description in descriptions:
           self._items.append(widgets.Checkbox(
               description=description, 
               value=True,
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