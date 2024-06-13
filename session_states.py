import asyncio
import dataclasses
import os
from global_settings import *
from lib.tools import MyPythonREPL
from lib.infrastructure import ForgetableContext, LlamaCpp
from style_bert_vits2.tts_model import TTSModel
from ipywidgets import widgets



assistant_name = DEFAULT_ASSISTANT_NAME
user_nickname = DEFAULT_USER_NICKNAME
context = ForgetableContext(-1)
context_window = 0
py = MyPythonREPL(replace_nl=False, temporal_working_directory=AGENT_WORKING_DIR)
login_time_stamp: str = ""
loop = asyncio.get_event_loop()
tts_model: TTSModel|None = None
prompt_builder = None
model: LlamaCpp|None = None
n_ctx: int = 8192


# Tools.
@dataclasses.dataclass
class Tool:
    name: str
    description: str
    example: str

tools = [
    Tool(
        name="google",
        description="Use this tool for Google searches with a query as an argument. This tool must be used to obtain real-time information. You NEVER obtain search results without calling this tool.",
        example="""** Call web search tool **:
```google
Here is search query.
```"""
    ),
    Tool(
        name="python",
        description="Describes and executes Python code. The code should be indented appropriately, and variables and functions will be shared within the session. Always output results using print or plt.show().", 
        example="""** Call python execution **:
```python
Here is python code.
```"""
    )
]


# Widgets.
guessing_image: widgets.Image|None = None
out = widgets.Output()
debug = widgets.Output(layout=widgets.Layout(width='800px', height='250px', overflow='scroll'))
field = widgets.Textarea(placeholder="User:", layout=widgets.Layout(width='700px', height='auto'))
user_nickname_field = widgets.Text(description='You', value=user_nickname, placeholder='Your nickname', layout=widgets.Layout(width='250px'))
assistant_name_field = widgets.Text(description='AI', value=assistant_name, placeholder='Assistant name', layout=widgets.Layout(width='250px'))
button = widgets.Button(description='📤', button_style='success', layout=widgets.Layout(width='50px', height='50px'))
reset_button = widgets.Button(description='Reset', layout=widgets.Layout(width='120px'))
retrieve = widgets.Button(description='Undo', layout=widgets.Layout(width='120px'))
create_voice = widgets.Button(description='Synthesize voice', layout=widgets.Layout(width='120px'))
voice_player = widgets.Output()
dropdown: widgets.Dropdown|None = None
voice_length = widgets.FloatSlider(value=1.0, min=0.5, max=2.0, step=0.05)
user_preamble = widgets.Textarea(placeholder=f'System prompt', value=DEFAULT_USER_PREAMBLE, layout=widgets.Layout(width='800px', height='100px'))
template_selector = widgets.Dropdown(
    options=["Command R Template", "Llama-3 Template", "ChatML Template"],
    value="Llama-3 Template",
    description="Prompt template",
)
streamingllm = widgets.Checkbox(
    value=True,
    description='StreamingLLM',
    disabled=True,
    indent=False
)

# GGUF loading options.
ggufs = [item for item in os.listdir(GGUF_DIR) if item.endswith(".gguf")]
load_button = widgets.Button(description="Load", layout=widgets.Layout(width='80px'))
unload_button = widgets.Button(description="Unload", layout=widgets.Layout(width='80px'))
@dataclasses.dataclass
class LlamaCppOptions:
    gguf_selector = widgets.Dropdown(
        description="GGUFs", 
        options=ggufs if len(ggufs) > 0 else ["No gguf in dir"], 
        value=ggufs[0] if len(ggufs) > 0 else "No gguf in dir"
    )
    define_n_ctx = widgets.BoundedIntText(
        value=n_ctx,
        min=0, max=10**10, step=1,
        description='Context size',
        layout=widgets.Layout(width='170px'),
    )
    n_gpu_layers = widgets.BoundedIntText(
        value=-1,
        min=-1, max=10**10, step=1,
        description='# layers on VRAM',
        layout=widgets.Layout(width='170px'),
    )
    flash_attention = widgets.Checkbox(
        value=True,
        description='Flash attention',
        disabled=False,
        indent=False
    )
    
llama_cpp_options = LlamaCppOptions()
buttons = [button, reset_button, retrieve, create_voice, load_button, unload_button]


@dataclasses.dataclass
class GenerationParams:
    temperature = widgets.FloatSlider(description="temp", value=0.2, min=0, max=1.0, step=0.05)
    top_p = widgets.FloatSlider(description="top_p", value=0.95, min=0, max=2.0, step=0.05)
    top_k = widgets.IntSlider(description="top_k", value=40, min=0, max=100)
    repeat_penalty = widgets.FloatSlider(description="repeat_penalty", value=1.1, min=1.0, max=1.2, step=0.005)
    frequency_penalty = widgets.FloatSlider(description="frequency_penalty", value=0, min=0, max=2, step=0.05)
    presence_penalty = widgets.FloatSlider(description="presence_penalty", value=0, min=0, max=2, step=0.05)

generation_params = GenerationParams()