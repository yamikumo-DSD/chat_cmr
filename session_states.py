import asyncio
import dataclasses
import os
from global_settings import *
from lib.infrastructure import ForgetableContext
from style_bert_vits2.tts_model import TTSModel
from ipywidgets import widgets
from IPython.display import clear_output
from lib.uis import ThreadSafeStdOutCapture



assistant_name = DEFAULT_ASSISTANT_NAME
user_nickname = DEFAULT_USER_NICKNAME
context = ForgetableContext(-1)
context_window = 0
login_time_stamp: str = ""
loop = asyncio.get_event_loop()
tts_model: TTSModel|None = None
prompt_builder = None
model = None
n_ctx: int = DEFAULT_N_CTX
max_gen_tokens: int = DEFAULT_MAX_GEN_TOKENS
session_id: str
initialized: bool = False
states_loaded: bool = False
interrupt_signal: bool = False
generating: bool = False
max_user_tokens: int = 0



# Widgets.
guessing_image: widgets.Image|None = None
out = widgets.Output()
debug = widgets.Output(layout=widgets.Layout(width='750px', height='300px', overflow='scroll'))
debug_stdout_capture = ThreadSafeStdOutCapture(debug)
field = widgets.Textarea(placeholder="User:", layout=widgets.Layout(max_width='700px', width="100%", height='auto'))
user_nickname_field = widgets.Text(description='You', value=user_nickname, placeholder='Your nickname', layout=widgets.Layout(width='250px'))
assistant_name_field = widgets.Text(description='AI', value=assistant_name, placeholder='Assistant name', layout=widgets.Layout(width='250px'))
button = widgets.Button(description='📤', button_style='success', layout=widgets.Layout(width='50px', height='50px'))
interrupt = widgets.Button(description='◼︎', layout=widgets.Layout(width='40px'), disabled=True)
reset_button = widgets.Button(description='New Session', layout=widgets.Layout(width='120px'))
retrieve = widgets.Button(description='Undo', layout=widgets.Layout(width='120px'))
load_session_button = widgets.Button(description="Load", button_style='success', layout=widgets.Layout(width='80px'))
upload_file = widgets.FileUpload(
    description="Upload",
    #accept=".png,.jpg,.jpeg,.gif,.bmp,.pdf",
    multiple=False,
    layout=widgets.Layout(width='120px'),
)
screen_shot = widgets.Button(description="Screen Shot", layout=widgets.Layout(width="120px"))
hide_thoughts = widgets.ToggleButton(value=False, description="Hide thoughts")
create_voice = widgets.Button(button_style='success', description='Synthesize', layout=widgets.Layout(width='120px'))
voice_player = widgets.Output()
dropdown: widgets.Dropdown|None = None
select_captioner = widgets.Dropdown(
    description="Caption generator", 
    options=[
        "Huihui-Qwen3-VL-2B-Instruct-abliterated",
        "Florence-2-large", 
        "Florence-2-base-PromptGen", 
        "ljnlonoljpiljm/florence-2-large-nsfw-pretrain"
    ], 
    value="Huihui-Qwen3-VL-2B-Instruct-abliterated",
    style={"description_width": 'initial'},
)
active_gguf: str = ""
voice_length = widgets.FloatSlider(
    description="Relative duration", 
    value=1.0, min=0.5, max=2.0, step=0.05, 
    style={"description_width": "initial"}, 
    layout=widgets.Layout(max_width="300px", width="100%")
)
auto_speak = widgets.Checkbox(value=False, description="Auto speak", style={"description_width": 'initial'})
user_preamble = widgets.Textarea(placeholder=f'System prompt', value=DEFAULT_USER_PREAMBLE, layout=widgets.Layout(max_width="800px", width='100%', height='100px'))

streamingllm = widgets.Checkbox(
    value=True,
    description='StreamingLLM',
    disabled=True,
    indent=False, style={"description_width": 'initial'},
)
thinking = widgets.Checkbox(
    value=False, 
    description="Thinking mode", 
    style={"description_width": 'initial'}
)
max_steps = widgets.IntSlider(
    value=3, min=1, max=10, 
    description="Max steps",
    style={"description_width": 'initial'}
)

# GGUF loading options.
ggufs = list(filter(lambda x: x.endswith(".gguf"), os.listdir(GGUF_DIR))); ggufs.sort()
active_gguf_viewer = widgets.Output()
load_button = widgets.Button(button_style='success', description="Load", layout=widgets.Layout(width='80px'))
unload_button = widgets.Button(description="Unload", layout=widgets.Layout(width='80px'))
reflesh_list_button = widgets.Button(description="Scan files", layout=widgets.Layout(width='80px'))

@active_gguf_viewer.capture()
def set_gguf_viewer(name: str) -> None:
    """ You can use HTML as name """
    from IPython.display import HTML
    clear_output()
    display(HTML(f"Active GGUF: {name}"))
        
@active_gguf_viewer.capture()
def unset_gguf_viewer() -> None:
    from IPython.display import HTML
    clear_output()
    display(HTML("No GGUF loaded yet."))

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
        value=True, style={"description_width": 'initial'},
        description='8-bit KV',
        disabled=False,
        indent=True,
    )
    

    
llama_cpp_options = LlamaCppOptions()
buttons = [
    button, 
    reset_button, 
    retrieve, 
    upload_file, 
    create_voice, 
    load_button, 
    unload_button, 
    reflesh_list_button,
    load_session_button,
    screen_shot,
]


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
        description='Prefix output  ', 
        layout=widgets.Layout(width='400px'),
        style={"description_width": 'initial'}
    )
    append_prefix = widgets.Checkbox(
        value=True, style={"description_width": 'initial'},
        description='Append prefix',
        disabled=False,
        indent=True,
    )
    prefix_thoughts = widgets.Text(
        description='Prefix thoughts', 
        layout=widgets.Layout(width='400px'),
        style={"description_width": 'initial'}
    )

generation_params = GenerationParams()


def generate_id() -> str:
    from uuid import uuid4
    return uuid4().hex[:8]


@dataclasses.dataclass
class States:
    session_id: str
    time_stamp: str
    context: ForgetableContext
    user_preamble: str
    user_nickname: str
    assistant_name: str
    active_tools: list[str]

    @staticmethod
    def load(states_file_path: str):
        """ Load States object from serialized file. """
        import pickle
        with open(states_file_path, "rb") as f:
            return pickle.load(f)
            
    def save(self, states_file_path: str) -> None:
        """ Save States object. """
        import pickle
        with open(states_file_path, "wb") as f:
            pickle.dump(self, f)
            
    @staticmethod
    def load_bytes(states_bytes: bytes):
        """ Load States object from binary representation of the states. """
        import pickle
        return pickle.loads(states_bytes)

            
    def save_bytes(self) -> bytes:
        """ Save States object as binary representation. """
        import pickle
        return pickle.dumps(self)


def get_states() -> States:
    from agent_tools import tools
    return States(
        session_id,
        login_time_stamp,
        context,
        user_preamble.value,
        user_nickname,
        assistant_name,
        [tool.name for tool in tools()],
    )


def save_states(states_file_path: str) -> None:
    get_states().save(states_file_path)
    
def save_states_bytes() -> bytes:
    return get_states().save_bytes()

def load_states(states_file_path: str) -> None:
    """
    Raises:
        raises RuntimeError when loading fails
    """
    global session_id, login_time_stamp
    global context, context_window
    global user_preamble, user_nickname, assistant_name
    
    states: States = States.load(states_file_path)

    session_id = states.session_id
    login_time_stamp = states.time_stamp

    # Restore context.
    context.reset()
    for item in states.context:
        context.push(item)
    context_window = len(context)

    # Restore system prompt.
    user_preamble.value = states.user_preamble
    user_nickname = states.user_nickname; user_nickname_field.value = states.user_nickname
    assistant_name = states.assistant_name; assistant_name_field.value = states.assistant_name

    # Restore tool selection.
    try:
        tool_displayed_names = [tools.find_tool(tool_name).displayed_name for tool_name in states.active_tools]
    except KeyError as e:
        raise RuntimeError("Failed to load session. Unknown tool found in the context.")
    for checkbox in tools.tool_selector:
        checkbox.value = checkbox.description in tool_displayed_names

def load_states_bytes(states_bytes) -> None:
    import agent_tools as tools
    global session_id, login_time_stamp
    global context, context_window
    global user_preamble, user_nickname, assistant_name
    
    states: States = States.load_bytes(states_bytes)

    session_id = states.session_id
    login_time_stamp = states.time_stamp

    # Restore context.
    context.reset()
    for item in states.context:
        context.push(item)
    context_window = len(context)

    # Restore system prompt.
    user_preamble.value = states.user_preamble
    user_nickname = states.user_nickname; user_nickname_field.value = states.user_nickname
    assistant_name = states.assistant_name; assistant_name_field.value = states.assistant_name

    # Restore tool selection.
    try:
        tool_displayed_names = [tools.find_tool(tool_name).displayed_name for tool_name in states.active_tools]
    except KeyError as e:
        raise RuntimeError("Failed to load session. Unknown tool found in the context.")
    for checkbox in tools.tool_selector:
        checkbox.value = checkbox.description in tool_displayed_names