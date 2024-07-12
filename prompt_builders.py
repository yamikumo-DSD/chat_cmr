from abc import ABC, abstractmethod
import session_states as session
from lib.infrastructure import ForgetableContext
import agent_tools as tools


class PromptBuilderBase(ABC):
    @abstractmethod
    def render_item(self, item) -> str:
        pass
    def history_in_window(self) -> list:
        return self.context.history()[-session.context_window:] if session.context_window > 0 else []
    def render_context(self) -> str:
        text = ""
        for item in self.history_in_window():
            text += self.render_item(item)
        return text
    @abstractmethod
    def render_instruction(self) -> str:
        pass
    def stops(self) -> list[str]:
        return []
    @abstractmethod
    def render_complete_prompt(self) -> str:
        pass
    def build(self, prefix: str = "") -> str:
        """
        Args:
            prefix (str): a piece of string added at the last of complete prompt, which means it forces LLM to start output with the string. This may be helpful when you want to JB the LLM.
        Returns:
            str: render_complete_prompt() + prefix
        """
        return self.render_complete_prompt() + prefix


def _add_tool_use_code(tool_name, tool_input) -> str:
    return f"<tool>{tool_name}</tool><tool_input>{tool_input}</tool_input>"
    
def _add_tool_result_code(tool_name, tool_result) -> str:
    return f"<tool>{tool_name}</tool><tool_output>{tool_result}</tool_output>"

def _embed_caption(caption) -> str:
    return f"""** An image uploaded **
<caption>{caption}</caption>"""   
    
def _render_user_item(item, token_renderer) -> str:
    content = item["content"]
    return token_renderer(content)
    
def _render_assistant_item(item, token_renderer) -> str:
    content = item["content"]
    tool = item.get("tool")
    
    if not tool:
        return token_renderer(content)
        
    tool_name = tool.get("name")
    tool_action = tool.get("action")
    tool_input = tool.get("input")

    if tool_action == "call":
        return token_renderer(_add_tool_use_code(tool_name, tool_input))
    else:
        raise RuntimeError(f"Assistant is not allowed to \"{tool_action}\" tool.")
    
def _render_tool_agent_item(item, token_renderer) -> str:
    tool = item.get("tool")
    if not tool or not tool.get("action") == "return":
        raise RuntimeError(f"{TOOL_AGENT_NAME} item must contain tool with returned value.")
        
    tool_name = tool.get("name")
    tool_action = tool.get("action")
    tool_output = tool.get("output")
    
    if tool_name == tools.web_search.name:
        references = tool_output.get("references")
        search_result = tool_output.get("search_result")
        return token_renderer(_add_tool_result_code(tools.web_search.name, search_result))
    elif tool_name == tools.exec_python.name:
        stdout = tool_output.get("stdout")
        image = tool_output.get("image")
        caption = tool_output.get("caption")
        return token_renderer(
            _add_tool_result_code(tools.exec_python.name, stdout) + (_embed_caption(caption) if caption else "")
        )
        
def _render_file_uploader_item(item, token_renderer) -> str:
    image_caption = item.get('caption')
    if image_caption:
        return token_renderer(_embed_caption(image_caption))
    else:
        return token_renderer("")





class CommandRPromptBuilder(PromptBuilderBase):
    instructions: str
    context: ForgetableContext

    def __init__(self, instructions, context) -> None:
        self.instructions = instructions
        self.context = context
        
    def assistant_token(self, text: str) -> str:
        return f"<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{text}<|END_OF_TURN_TOKEN|>"
    def user_token(self, text: str) -> str:
        return f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{text}<|END_OF_TURN_TOKEN|>"
    def system_token(self, text: str) -> str:
        return f"<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{text}<|END_OF_TURN_TOKEN|>"
    def default_token(self, text: str) -> str:
        return f"<|START_OF_TURN_TOKEN|>{text}<|END_OF_TURN_TOKEN|>"

    def render_item(self, item) -> str:
        from global_settings import TOOL_AGENT_NAME, FILE_UPLOADER_NAME

        role = item["role"]
        if role == "User": return _render_user_item(item, self.user_token)
        elif role == session.assistant_name: return _render_assistant_item(item, self.assistant_token)
        elif role == TOOL_AGENT_NAME: return _render_tool_agent_item(item, self.system_token)
        elif role == FILE_UPLOADER_NAME: return _render_file_uploader_item(item, self.system_token)
        else: raise RuntimeError(f"Unknown role \"{role}\".")

    def render_instruction(self) -> str:
        return self.instructions.format(
            now=session.login_time_stamp, 
            assistant=session.assistant_name, 
            user_nickname=session.user_nickname,
            tool_names=[tool.name for tool in session.tools()],
            tool_description='\n'.join([f"\"{tool.name}\": {tool.description}\n" for tool in session.tools()]),
            user_preamble=session.user_preamble.value,
        )
        
    def render_complete_prompt(self) -> str:
        tools = session.tools()
        
        examples = "\n".join([tool.example for tool in tools])
        return self.render_instruction() + \
               self.render_context() + \
               self.system_token(f"""You should always use tools {[tool.name for tool in tools]} to provide high quality response to user's last input. If the information doesn't need evidence or accuracy, you can skip tools and directly answer.
{examples}""") + \
               "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"







class Llama3PromptBuilder(PromptBuilderBase):
    instructions: str
    context: ForgetableContext

    def __init__(self, instructions, context) -> None:
        self.instructions = instructions
        self.context = context
        
    def assistant_token(self, text: str) -> str:
        return f"""<|start_header_id|>assistant<|end_header_id|>

{text}<|eot_id|>"""
    def user_token(self, text: str) -> str:
        return f"""<|start_header_id|>user<|end_header_id|>

{text}<|eot_id|>"""
    def system_token(self, text: str) -> str:
        return f"""<|start_header_id|>system<|end_header_id|>

{text}<|eot_id|>"""
    def tool_token(self, text: str) -> str:
        """
        It's ambiguous Llama-3 allows this.    
        Related discussion:
            https://www.reddit.com/r/LocalLLaMA/comments/1cdotw1/llama_3_tool_use_how_does_groq_do_it/
        """
        return f"""<|start_header_id|>tool<|end_header_id|>

{text}<|eot_id|>"""

    def render_item(self, item) -> str:
        from global_settings import TOOL_AGENT_NAME, FILE_UPLOADER_NAME

        role = item["role"]
        if role == "User": return _render_user_item(item, self.user_token)
        elif role == session.assistant_name: return _render_assistant_item(item, self.assistant_token)
        elif role == TOOL_AGENT_NAME: return _render_tool_agent_item(item, self.system_token)
        elif role == FILE_UPLOADER_NAME: return _render_file_uploader_item(item, self.system_token)
        else: raise RuntimeError(f"Unknown role \"{role}\".")
    
    def render_instruction(self) -> str:
        examples = "\n".join([tool.example for tool in session.tools()])
        
        return self.instructions.format(
            now=session.login_time_stamp, 
            assistant=session.assistant_name, 
            user_nickname=session.user_nickname,
            tool_names=[tool.name for tool in session.tools()],
            tool_description='\n'.join([f"{tool.name}: {tool.description}" for tool in session.tools()]),
            user_preamble=session.user_preamble.value,
            examples=examples,
        )
        
    def render_complete_prompt(self) -> str:
        examples = "\n".join([tool.example for tool in session.tools()])
        return self.render_instruction() + \
               self.render_context() + \
               """<|start_header_id|>assistant<|end_header_id|>

"""





class ChatMLPromptBuilder(PromptBuilderBase):
    instructions: str
    context: ForgetableContext

    def __init__(self, instructions, context) -> None:
        self.instructions = instructions
        self.context = context
        
    def assistant_token(self, text: str) -> str:
        return f"""<|im_start|>assistant
{text}
<|im_end|>
"""
        
    def user_token(self, text: str) -> str:
        return f"""<|im_start|>user
{text}
<|im_end|>
"""
        
    def system_token(self, text: str) -> str:
        return f"""<|im_start|>system
{text}
<|im_end|>
"""

    def render_item(self, item) -> str:
        from global_settings import TOOL_AGENT_NAME, FILE_UPLOADER_NAME

        role = item["role"]
        if role == "User": return _render_user_item(item, self.user_token)
        elif role == session.assistant_name: return _render_assistant_item(item, self.assistant_token)
        elif role == TOOL_AGENT_NAME: return _render_tool_agent_item(item, self.system_token)
        elif role == FILE_UPLOADER_NAME: return _render_file_uploader_item(item, self.system_token)
        else: raise RuntimeError(f"Unknown role \"{role}\".")

    def render_instruction(self) -> str:
        tools = session.tools()
        
        examples = "\n".join([tool.example for tool in tools])
        
        return self.instructions.format(
            now=session.login_time_stamp, 
            assistant=session.assistant_name, 
            user_nickname=session.user_nickname,
            tool_names=[tool.name for tool in tools],
            tool_description='\n'.join([f"{tool.name}: {tool.description}" for tool in tools]),
            user_preamble=session.user_preamble.value,
            examples=examples,
        )
        
    def render_complete_prompt(self) -> str:
        examples = "\n".join([tool.example for tool in session.tools()])
        return self.render_instruction() + \
               self.render_context() + \
               """<|im_start|>assistant
"""



class Llama2PromptBuilder(PromptBuilderBase):
    instructions: str
    context: ForgetableContext

    def __init__(self, instructions, context) -> None:
        self.instructions = instructions
        self.context = context
        
    def assistant_token(self, text: str) -> str:
        return f""" Assistant: {text} </s>"""
        
    def user_token(self, text: str) -> str:
        return f"""<s>[INST] User: {text} [/INST]"""

    def render_item(self, item) -> str:
        from global_settings import TOOL_AGENT_NAME, FILE_UPLOADER_NAME

        role = item["role"]
        if role == "User": return _render_user_item(item, self.user_token)
        elif role == session.assistant_name: return _render_assistant_item(item, self.assistant_token)
        elif role == TOOL_AGENT_NAME: return _render_tool_agent_item(item, self.assistant_token)
        elif role == FILE_UPLOADER_NAME: return _render_file_uploader_item(item, self.user_token)
        else: raise RuntimeError(f"Unknown role \"{role}\".")

    def render_context(self) -> str:
        text = ""
        for i, item in enumerate(self.history_in_window()):
            rendered = self.render_item(item)
            if i == 0:
                rendered = "\n\n" + rendered.lstrip(" <s>[INST]")
            text += rendered
        return text

    def render_instruction(self) -> str:
        tools = session.tools()
        
        examples = "\n".join([tool.example for tool in tools])
        
        return self.instructions.format(
            now=session.login_time_stamp, 
            assistant=session.assistant_name, 
            user_nickname=session.user_nickname,
            tool_names=[tool.name for tool in tools],
            tool_description='\n'.join([f"{tool.name}: {tool.description}" for tool in tools]),
            user_preamble=session.user_preamble.value,
            examples=examples,
        )
        
    def render_complete_prompt(self) -> str:
        examples = "\n".join([tool.example for tool in session.tools()])
        return self.render_instruction() + \
               self.render_context() + \
               f""" Assistant:"""

    def stops(self) -> list[str]:
        return ["Assistant:", "User:"]

JaCommMSPromptBuilder = Llama2PromptBuilder



class Gemma2Instruct(PromptBuilderBase):
    instructions: str
    context: ForgetableContext

    def __init__(self, instructions, context) -> None:
        self.instructions = instructions
        self.context = context
        
    def assistant_token(self, text: str) -> str:
        return f"""<start_of_turn>model
{text}<end_of_turn>"""
        
    def user_token(self, text: str) -> str:
        return f"""<|start_of_turn|>user
{text}<end_of_turn>"""
        

    def render_item(self, item) -> str:
        from global_settings import TOOL_AGENT_NAME, FILE_UPLOADER_NAME

        role = item["role"]
        if role == "User": return _render_user_item(item, self.user_token)
        elif role == session.assistant_name: return _render_assistant_item(item, self.assistant_token)
        elif role == TOOL_AGENT_NAME: return _render_tool_agent_item(item, self.assistant_token)
        elif role == FILE_UPLOADER_NAME: return _render_file_uploader_item(item, self.user_token)
        else: raise RuntimeError(f"Unknown role \"{role}\".")


    def render_instruction(self) -> str:
        tools = session.tools()
        
        examples = "\n".join([tool.example for tool in tools])
        
        return self.instructions.format(
            now=session.login_time_stamp, 
            assistant=session.assistant_name, 
            user_nickname=session.user_nickname,
            tool_names=[tool.name for tool in tools],
            tool_description='\n'.join([f"{tool.name}: {tool.description}" for tool in tools]),
            user_preamble=session.user_preamble.value,
            examples=examples,
        )
        
    def render_complete_prompt(self) -> str:
        examples = "\n".join([tool.example for tool in session.tools()])
        return self.render_instruction() + \
               self.render_context() + \
               """<start_of_turn>model
"""