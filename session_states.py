import asyncio
import dataclasses
import os
from global_settings import *
from lib.tools import MyPythonREPL
from lib.infrastructure import ForgetableContext, LlamaCpp
from style_bert_vits2.tts_model import TTSModel
from ipywidgets import widgets
from agent_tools import tools
from IPython.display import clear_output



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
max_gen_tokens: int = 512
initialized: bool = False




# Widgets.
guessing_image: widgets.Image|None = None
out = widgets.Output()
debug = widgets.Output(layout=widgets.Layout(width='800px', height='250px', overflow='scroll'))
field = widgets.Textarea(placeholder="User:", layout=widgets.Layout(max_width='700px', width="100%", height='auto'))
user_nickname_field = widgets.Text(description='You', value=user_nickname, placeholder='Your nickname', layout=widgets.Layout(width='250px'))
assistant_name_field = widgets.Text(description='AI', value=assistant_name, placeholder='Assistant name', layout=widgets.Layout(width='250px'))
button = widgets.Button(description='📤', button_style='success', layout=widgets.Layout(width='50px', height='50px'))
reset_button = widgets.Button(description='Reset', layout=widgets.Layout(width='120px'))
retrieve = widgets.Button(description='Undo', layout=widgets.Layout(width='120px'))
upload_file = widgets.FileUpload(
    description="Image",
    accept=".png,.jpg,.jpeg,.gif,.bmp",
    multiple=False,
    layout=widgets.Layout(width='120px'),
)
create_voice = widgets.Button(button_style='success', description='Synthesize', layout=widgets.Layout(width='120px'))
voice_player = widgets.Output()
dropdown: widgets.Dropdown|None = None
active_gguf: str = ""
voice_length = widgets.FloatSlider(
    description="Relative duration", 
    value=1.0, min=0.5, max=2.0, step=0.05, 
    style={"description_width": "initial"}, 
    layout=widgets.Layout(max_width="300px", width="100%")
)
user_preamble = widgets.Textarea(placeholder=f'System prompt', value=DEFAULT_USER_PREAMBLE, layout=widgets.Layout(max_width="800px", width='100%', height='100px'))

template_selector = widgets.Dropdown(
    options=[
        "Command R",
        "Llama-3 Instruct",
        "ChatML",
        "Llama-2 Instruct",
        "Llama-2 Instruct JA",
        "Gemma 2 Instruct",
    ],
    value="Llama-3 Instruct",
    description="Prompt template", 
    style={"description_width": 'initial'},
)

streamingllm = widgets.Checkbox(
    value=True,
    description='StreamingLLM',
    disabled=True,
    indent=False, style={"description_width": 'initial'},
)

# GGUF loading options.
ggufs = [item for item in os.listdir(GGUF_DIR) if item.endswith(".gguf")]
active_gguf_viewer = widgets.Output()
load_button = widgets.Button(button_style='success', description="Load", layout=widgets.Layout(width='80px'))
unload_button = widgets.Button(description="Unload", layout=widgets.Layout(width='80px'))
reflesh_list_button = widgets.Button(description="Scan files", layout=widgets.Layout(width='80px'))

@active_gguf_viewer.capture()
def set_gguf_viewer(name: str) -> None:
    clear_output()
    print(f"Active GGUF: {name}")
        
@active_gguf_viewer.capture()
def unset_gguf_viewer() -> None:
    clear_output()
    print(f"No GGUF loaded yet.")

unset_gguf_viewer()

@dataclasses.dataclass
class LlamaCppOptions:
    gguf_selector = widgets.Dropdown(
        description="GGUFs", style={"description_width": 'initial'},
        options=ggufs if len(ggufs) > 0 else ["No gguf in dir"], 
        value=ggufs[0] if len(ggufs) > 0 else "No gguf in dir"
    )
    define_n_ctx = widgets.BoundedIntText(
        value=n_ctx, style={"description_width": 'initial'},
        min=0, max=10**10, step=1,
        description='Context size',
        layout=widgets.Layout(width='170px'),
    )
    define_max_gen_tokens = widgets.BoundedIntText(
        value=max_gen_tokens, style={"description_width": 'initial'},
        min=1, max=10**10, step=1,
        description='Max generation tokens',
        layout=widgets.Layout(width='210px'),
    )
    n_gpu_layers = widgets.BoundedIntText(
        value=-1, style={"description_width": 'initial'},
        min=-1, max=10**10, step=1,
        description='# layers on VRAM',
        layout=widgets.Layout(width='170px'),
    )
    flash_attention = widgets.Checkbox(
        value=True, style={"description_width": 'initial'},
        description='Flash attention',
        disabled=False,
        indent=True,
    )
    quantize_kv = widgets.Checkbox(
        value=False, style={"description_width": 'initial'},
        description='8-bit KV',
        disabled=False,
        indent=True,
    )

    
llama_cpp_options = LlamaCppOptions()
buttons = [button, reset_button, retrieve, upload_file, 
           create_voice, load_button, unload_button, reflesh_list_button]


@dataclasses.dataclass
class GenerationParams:
    temperature = widgets.FloatSlider(
        description="temp", 
        value=0.2, min=0, max=1.0, step=0.05, 
        style={"description_width": 'initial'}
    )
    top_p = widgets.FloatSlider(
        description="top_p", 
        value=0.95, min=0, max=2.0, step=0.05, 
        style={"description_width": 'initial'}
    )
    top_k = widgets.IntSlider(
        description="top_k", 
        value=40, min=0, max=100, 
        style={"description_width": 'initial'}
    )
    repeat_penalty = widgets.FloatSlider(
        description="repeat_penalty", 
        value=1.1, min=1.0, max=1.2, step=0.005, 
        style={"description_width": 'initial'}
    )
    frequency_penalty = widgets.FloatSlider(
        description="frequency_penalty", 
        value=0, min=0, max=2, step=0.05, 
        style={"description_width": 'initial'}
    )
    presence_penalty = widgets.FloatSlider(
        description="presence_penalty", 
        value=0, min=0, max=2, step=0.05, 
        style={"description_width": 'initial'}
    )
    prefix = widgets.Text(
        description='Prefix', 
        layout=widgets.Layout(width='400px'),
        style={"description_width": 'initial'}
    )
    append_prefix = widgets.Checkbox(
        value=True, style={"description_width": 'initial'},
        description='Append prefix',
        disabled=False,
        indent=True,
    )

generation_params = GenerationParams()