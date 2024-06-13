import IPython
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