from lib.infrastructure import ForgetableContext
from abc import ABC, abstractmethod
from typing import Iterable
from agent_tools import BaseTool


class PromptBuilderBase(ABC):
    displayed_name: str

    def __init__(self, displayed_name: str) -> None:
        self.displayed_name = displayed_name
    
    @abstractmethod
    def render_item(self, item) -> str:
        pass
        
    def crop_context(
        self, 
        context: ForgetableContext,
        window: int,
    ) -> list:
        return context.history()[-window:] if window > 0 else []
        
    def render_context(
        self, 
        context: ForgetableContext, 
        window: int
    ) -> str:
        return "".join(
            map(self.render_item, self.crop_context(context, window))
        )
        
    @abstractmethod
    def render_instruction(
        self,
        login_time_stamp: str,
        assistant_name: str, 
        user_nickname: str,
        tools: Iterable[BaseTool],
        user_preamble: str,
    ) -> str:
        pass

    def think_tokens(self) -> tuple[str, str]|None:
        """
        Returns when,
        Thinking prompt: a pair of thinking start/close tokens like `("<think>", "</think>")`,
        Otherwise, None.
        """
        return None
        
    def stops(self) -> list[str]:
        return []
        
    @abstractmethod
    def render_complete_prompt(
        self, 
        context: ForgetableContext, 
        window: int,
        login_time_stamp: str,
        assistant_name: str, 
        user_nickname: str,
        tools: Iterable[BaseTool],
        user_preamble: str,
    ) -> str:
        pass
        
    def build(
        self, 
        context: ForgetableContext, 
        window: int,
        login_time_stamp: str,
        assistant_name: str, 
        user_nickname: str,
        tools: Iterable[BaseTool],
        user_preamble: str,
        prefix: str = "",
    ) -> str:
        """
        Args:
            prefix (str): a piece of string added at the last of complete prompt, which means it forces LLM to start output with the string. This may be helpful when you want to JB the LLM.
        Returns:
            str: render_complete_prompt() + prefix
        """
        return self.render_complete_prompt(
            context,
            window,
            login_time_stamp,
            assistant_name,
            user_nickname,
            tools,
            user_preamble,
        ) + prefix




def _add_tool_use_code(tool_name, tool_input) -> str:
    if isinstance(tool_input, dict):
        tool_input_xml = ""
        for arg_name, arg_value in tool_input.items():
            tool_input_xml += f"<{arg_name}>{arg_value}</{arg_name}>"
        tool_input = tool_input_xml
    return f"<tool>{tool_name}</tool><tool_input>{tool_input}</tool_input>"
    
def _add_tool_result_code(tool_name, tool_result) -> str:
    return f"<tool>{tool_name}</tool><tool_output>{tool_result}</tool_output>"
    
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
    from lib.dicttoxml import dicttoxml
    from global_settings import TOOL_AGENT_NAME
    
    tool = item.get("tool")
    if not tool or not (tool.get("action") == "return" or tool.get("action") == "error"):
        raise RuntimeError(f"{TOOL_AGENT_NAME} item must contain tool with returned value.")
        
    tool_name = tool.get("name")
    tool_action = tool.get("action")
    tool_output = tool.get("output")
    
    try:
        output = dicttoxml(
            tool_output, 
            root=False, xml_declaration=False, return_bytes=False, attr_type=False
        )
    except Exception as e:
        output = "Failed to convert tool output into text format. You need to handwrite the displaying pipeline."
    return token_renderer(_add_tool_result_code(tool_name, output))
    
def _render_file_uploader_item(item, token_renderer) -> str:
    caption = item.get('caption')
    file_type = item.get('file_type') if item.get('file_type') else "image" # Assume file_type="image".
    file_name = item.get('file_name') if item.get('file_name') else "unknown_file_name"
    if caption:
        return token_renderer(f"""** File uploaded into the current directory **
<file name="{file_name}" type="{file_type}"><caption>{caption}</caption></file>""")   
    else:
        return token_renderer("")

def _load_text_file(path: str):
    text = ""
    with open(path, 'r') as f:
        lines = f.readlines()
        text = ''.join(lines)
    return text










class CommandRPromptBuilder(PromptBuilderBase):
    def __init__(self) -> None:
        super().__init__(displayed_name="Command R")
        self.instructions = _load_text_file("system_prompt_template/cmdr_sys_ppt_gen_16.txt")
        
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
        elif role == "Assistant": return _render_assistant_item(item, self.assistant_token)
        elif role == TOOL_AGENT_NAME: return _render_tool_agent_item(item, self.system_token)
        elif role == FILE_UPLOADER_NAME: return _render_file_uploader_item(item, self.system_token)
        else: raise RuntimeError(f"Unknown role \"{role}\".")

    def render_instruction(
        self,
        login_time_stamp: str,
        assistant_name: str, 
        user_nickname: str,
        tools: Iterable[BaseTool],
        user_preamble: str,
    ) -> str:
        return self.instructions.format(
            now=login_time_stamp, 
            assistant=assistant_name, 
            user_nickname=user_nickname,
            tool_names=[tool.name for tool in tools],
            tool_description='\n'.join([f"\"{tool.name}\": {tool.description}\n" for tool in tools]),
            user_preamble=user_preamble,
        )
        
    def render_complete_prompt(
        self, 
        context: ForgetableContext, 
        window: int,
        login_time_stamp: str,
        assistant_name: str, 
        user_nickname: str,
        tools: Iterable[BaseTool],
        user_preamble: str,
    ) -> str:
        examples = "\n".join([tool.example for tool in tools])
        return self.render_instruction(login_time_stamp, assistant_name, user_nickname, tools, user_preamble) + \
               self.render_context(context, window) + \
               self.system_token(f"""You should always use tools {[tool.name for tool in tools]} to provide high quality response to user's last input. If the information doesn't need evidence or accuracy, you can skip tools and directly answer.
{examples}""") + \
               "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"



class CommandRThinkingPromptBuilder(PromptBuilderBase):
    def __init__(self) -> None:
        super().__init__(displayed_name="Command R Thinking (Command A Reasoning-like style)")
        self.instructions = """<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|># System Preamble
## Basic Rules
Your role is to respond when user send messages. 
You don't have any real-time, up-to-date, or uncommon information until you invoke tools.
When using code blocks or indenting, it's generally recommended to use four half-width spaces.

Your output must fit following format;
<|START_THINKING|>draft your thoughts here before answering<|END_THINKING|><tool>tool_name</tool><tool_input>input</tool_input>
, where "tool_name" and "input" must be replaced by those of actual tools.

After tool use, the results will be given in following format;
<tool>tool_name</tool><tool_output>results</tool_output>

# User Preamble
## Task and Context
Your name is {assistant}, and user's nickname is {user_nickname}.
{user_preamble}

## Available Tools
Here is a list of tools available to you:

{tool_description}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>User logged in:{now}<|END_OF_TURN_TOKEN|>"""

    def think_tokens(self) -> tuple[str, str]:
        return ("<|START_THINKING|>", "<|END_THINKING|>")
        
    def assistant_token(self, text: str) -> str:
        return f"<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{text}<|END_OF_TURN_TOKEN|>"
    def user_token(self, text: str) -> str:
        return f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{text}<|END_OF_TURN_TOKEN|>"
    def system_token(self, text: str) -> str:
        return f"<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{text}<|END_OF_TURN_TOKEN|>"
    def tool_token(self, text: str) -> str:
        return f"<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><|START_TOOL_RESULT|>{text}<|END_TOOL_RESULT|><|END_OF_TURN_TOKEN|>"
    def default_token(self, text: str) -> str:
        return f"<|START_OF_TURN_TOKEN|>{text}<|END_OF_TURN_TOKEN|>"

    def render_item(self, item) -> str:
        from global_settings import TOOL_AGENT_NAME, FILE_UPLOADER_NAME

        role = item["role"]
        if role == "User": return _render_user_item(item, self.user_token)
        elif role == "Assistant": return _render_assistant_item(item, self.assistant_token)
        elif role == TOOL_AGENT_NAME: return _render_tool_agent_item(item, self.tool_token)
        elif role == FILE_UPLOADER_NAME: return _render_file_uploader_item(item, self.system_token)
        else: raise RuntimeError(f"Unknown role \"{role}\".")

    def render_instruction(
        self,
        login_time_stamp: str,
        assistant_name: str, 
        user_nickname: str,
        tools: Iterable[BaseTool],
        user_preamble: str,
    ) -> str:
        return self.instructions.format(
            now=login_time_stamp, 
            assistant=assistant_name, 
            user_nickname=user_nickname,
            tool_names=[tool.name for tool in tools],
            tool_description='\n'.join([f"\"{tool.name}\": {tool.description}\n" for tool in tools]),
            user_preamble=user_preamble,
        )
        
    def render_complete_prompt(
        self, 
        context: ForgetableContext, 
        window: int,
        login_time_stamp: str,
        assistant_name: str, 
        user_nickname: str,
        tools: Iterable[BaseTool],
        user_preamble: str,
    ) -> str:
        examples = "\n".join([tool.example for tool in tools])
        return self.render_instruction(login_time_stamp, assistant_name, user_nickname, tools, user_preamble) + \
               self.render_context(context, window) + \
               self.system_token(f"""You should always use tools {[tool.name for tool in tools]} to provide high quality response to user's last input. If the information doesn't need evidence or accuracy, you can skip tools and directly answer.
{examples}""") + \
               "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"






class Llama3PromptBuilder(PromptBuilderBase):
    def __init__(self) -> None:
        super().__init__(displayed_name="Llama-3 Instruct")
        self.instructions = _load_text_file("system_prompt_template/l3_sys_ppt_gen_16.txt")
        
    def assistant_token(self, text: str) -> str:
        return f"""<|start_header_id|>assistant<|end_header_id|>

{text}<|eot_id|>"""
    def user_token(self, text: str) -> str:
        return f"""<|start_header_id|>user<|end_header_id|>

{text}<|eot_id|>"""
    def system_token(self, text: str) -> str:
        return f"""<|start_header_id|>system<|end_header_id|>

{text}<|eot_id|>"""

    def render_item(self, item) -> str:
        from global_settings import TOOL_AGENT_NAME, FILE_UPLOADER_NAME

        role = item["role"]
        if role == "User": return _render_user_item(item, self.user_token)
        elif role == "Assistant": return _render_assistant_item(item, self.assistant_token)
        elif role == TOOL_AGENT_NAME: return _render_tool_agent_item(item, self.system_token)
        elif role == FILE_UPLOADER_NAME: return _render_file_uploader_item(item, self.system_token)
        else: raise RuntimeError(f"Unknown role \"{role}\".")
    
    def render_instruction(
        self,
        login_time_stamp: str,
        assistant_name: str, 
        user_nickname: str,
        tools: Iterable[BaseTool],
        user_preamble: str,
    ) -> str:
        examples = "\n".join([tool.example for tool in tools])
        
        return self.instructions.format(
            now=login_time_stamp, 
            assistant=assistant_name, 
            user_nickname=user_nickname,
            tool_names=[tool.name for tool in tools],
            tool_description='\n'.join([f"\"{tool.name}\": {tool.description}\n" for tool in tools]),
            user_preamble=user_preamble,
            examples=examples,
        )
        
    def render_complete_prompt(
        self, 
        context: ForgetableContext, 
        window: int,
        login_time_stamp: str,
        assistant_name: str, 
        user_nickname: str,
        tools: Iterable[BaseTool],
        user_preamble: str,
    ) -> str:
        return self.render_instruction(login_time_stamp, assistant_name, user_nickname, tools, user_preamble) + \
               self.render_context(context, window) + \
               """<|start_header_id|>assistant<|end_header_id|>

"""





class ChatMLThinkingPromptBuilder(PromptBuilderBase):
    def __init__(self) -> None:
        super().__init__(displayed_name="ChatMLThinking")
        self.instructions = """<|im_start|>system
# Basic Rules
Your role is to respond when user send messages. 
For real-time, up-to-date, or uncommon information, you MUST use search tool accurately following grammer described below.
When using code blocks or indenting, it's generally recommended to use four half-width spaces. 

Your output must fit following XML format;
<think>draft your thoughts here before answering</think><tool>tool_name</tool><tool_input>input</tool_input>
, where "tool_name" and "input" must be replaced by those of actual tools.

After tool use, the results will be given in following format;
<tool>tool_name</tool><tool_output>results</tool_output>

# User's Rules
Your name is {assistant}, and user's nickname is {user_nickname}.
{user_preamble}

# Tools Usage
You can always use tools {tool_names} by inserting following commands to provide high quality response to user's last input. If tools are unnecessary, simply reply to the user.
Here is some examples. Be sure to replace placeholders so that it matches actual user's requests.
{examples}

# Detailed Tool Descriptions
{tool_description}
<|im_end|>
<|im_start|>system
User logged in:{now}
<|im_end|>"""

    def think_tokens(self) -> tuple[str, str]:
        return ("<think>", "</think>")
        
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
        elif role == "Assistant": return _render_assistant_item(item, self.assistant_token)
        elif role == TOOL_AGENT_NAME: return _render_tool_agent_item(item, self.system_token)
        elif role == FILE_UPLOADER_NAME: return _render_file_uploader_item(item, self.system_token)
        else: raise RuntimeError(f"Unknown role \"{role}\".")

    def render_instruction(
        self,
        login_time_stamp: str,
        assistant_name: str, 
        user_nickname: str,
        tools: Iterable[BaseTool],
        user_preamble: str,
    ) -> str:
        examples = "\n".join([tool.example for tool in tools])
        
        return self.instructions.format(
            now=login_time_stamp, 
            assistant=assistant_name, 
            user_nickname=user_nickname,
            tool_names=[tool.name for tool in tools],
            tool_description='\n'.join([f"\"{tool.name}\": {tool.description}\n" for tool in tools]),
            user_preamble=user_preamble,
            examples=examples,
        )
        
    def render_complete_prompt(
        self, 
        context: ForgetableContext, 
        window: int,
        login_time_stamp: str,
        assistant_name: str, 
        user_nickname: str,
        tools: Iterable[BaseTool],
        user_preamble: str,
    ) -> str:
        return self.render_instruction(login_time_stamp, assistant_name, user_nickname, tools, user_preamble) + \
               self.render_context(context, window) + \
               """<|im_start|>assistant
"""


        
class ChatMLPromptBuilder(PromptBuilderBase):
    def __init__(self) -> None:
        super().__init__(displayed_name="ChatML")
        self.instructions = """<|endoftext|><|im_start|>system
# Basic Rules
Your role is to respond when user send messages. 
For real-time, up-to-date, or uncommon information, you MUST use search tool accurately following grammer described below.
When using code blocks or indenting, it's generally recommended to use four half-width spaces. 

Your output must fit following XML format;
<tool>tool_name</tool><tool_input>input</tool_input>
, where "tool_name" and "input" must be replaced by those of actual tools.

After tool use, the results will be given in following format;
<tool>tool_name</tool><tool_output>results</tool_output>

# User's Rules
Your name is {assistant}, and user's nickname is {user_nickname}.
{user_preamble}

# Tools Usage
You can always use tools {tool_names} by inserting following commands to provide high quality response to user's last input. If tools are unnecessary, simply reply to the user.
Here is some examples. Be sure to replace placeholders so that it matches actual user's requests.
{examples}

# Detailed Tool Descriptions
{tool_description}
<|im_end|>
<|im_start|>system
User logged in:{now}
<|im_end|>"""
        
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
        elif role == "Assistant": return _render_assistant_item(item, self.assistant_token)
        elif role == TOOL_AGENT_NAME: return _render_tool_agent_item(item, self.system_token)
        elif role == FILE_UPLOADER_NAME: return _render_file_uploader_item(item, self.system_token)
        else: raise RuntimeError(f"Unknown role \"{role}\".")

    def render_instruction(
        self,
        login_time_stamp: str,
        assistant_name: str, 
        user_nickname: str,
        tools: Iterable[BaseTool],
        user_preamble: str,
    ) -> str:
        examples = "\n".join([tool.example for tool in tools])
        
        return self.instructions.format(
            now=login_time_stamp, 
            assistant=assistant_name, 
            user_nickname=user_nickname,
            tool_names=[tool.name for tool in tools],
            tool_description='\n'.join([f"\"{tool.name}\": {tool.description}\n" for tool in tools]),
            user_preamble=user_preamble,
            examples=examples,
        )
        
    def render_complete_prompt(
        self, 
        context: ForgetableContext, 
        window: int,
        login_time_stamp: str,
        assistant_name: str, 
        user_nickname: str,
        tools: Iterable[BaseTool],
        user_preamble: str,
    ) -> str:
        return self.render_instruction(login_time_stamp, assistant_name, user_nickname, tools, user_preamble) + \
               self.render_context(context, window) + \
               """<|im_start|>assistant
"""




class Llama2PromptBuilder(PromptBuilderBase):
    def __init__(self) -> None:
        super().__init__(displayed_name="Llama-2 Instruct")
        self.instructions = _load_text_file("system_prompt_template/l2_sys_ppt_gen_16.txt")
        
    def assistant_token(self, text: str) -> str:
        return f""" Assistant: {text} </s>"""
        
    def user_token(self, text: str) -> str:
        return f"""<s>[INST] User: {text} [/INST]"""

    def render_item(self, item) -> str:
        from global_settings import TOOL_AGENT_NAME, FILE_UPLOADER_NAME

        role = item["role"]
        if role == "User": return _render_user_item(item, self.user_token)
        elif role == "Assistant": return _render_assistant_item(item, self.assistant_token)
        elif role == TOOL_AGENT_NAME: return _render_tool_agent_item(item, self.assistant_token)
        elif role == FILE_UPLOADER_NAME: return _render_file_uploader_item(item, self.user_token)
        else: raise RuntimeError(f"Unknown role \"{role}\".")

    def render_context(
        self, 
        context: ForgetableContext, 
        window: int
    ) -> str:
        text = ""
        for i, item in enumerate(self.crop_context(context, window)):
            rendered = self.render_item(item)
            if i == 0:
                rendered = "\n\n" + rendered.lstrip(" <s>[INST]")
            text += rendered
        return text

    def render_instruction(
        self,
        login_time_stamp: str,
        assistant_name: str, 
        user_nickname: str,
        tools: Iterable[BaseTool],
        user_preamble: str,
    ) -> str:
        examples = "\n".join([tool.example for tool in tools])
        
        return self.instructions.format(
            now=login_time_stamp, 
            assistant=assistant_name, 
            user_nickname=user_nickname,
            tool_names=[tool.name for tool in tools],
            tool_description='\n'.join([f"\"{tool.name}\": {tool.description}\n" for tool in tools]),
            user_preamble=user_preamble,
            examples=examples,
        )
        
    def render_complete_prompt(
        self, 
        context: ForgetableContext, 
        window: int,
        login_time_stamp: str,
        assistant_name: str, 
        user_nickname: str,
        tools: Iterable[BaseTool],
        user_preamble: str,
    ) -> str:
        return self.render_instruction(login_time_stamp, assistant_name, user_nickname, tools, user_preamble) + \
               self.render_context(context, window) + \
               f""" Assistant:"""

    def stops(self) -> list[str]:
        return ["Assistant:", "User:"]


class JaCommMSPromptBuilder(PromptBuilderBase):
    def __init__(self) -> None:
        super().__init__(displayed_name="Llama-2 Instruct JA")
        self.instructions = _load_text_file("system_prompt_template/ja_community_ms_sys_ppt_gen_16.txt")
        
    def assistant_token(self, text: str) -> str:
        return f""" Assistant: {text} </s>"""
        
    def user_token(self, text: str) -> str:
        return f"""<s>[INST] User: {text} [/INST]"""

    def render_item(self, item) -> str:
        from global_settings import TOOL_AGENT_NAME, FILE_UPLOADER_NAME

        role = item["role"]
        if role == "User": return _render_user_item(item, self.user_token)
        elif role == "Assistant": return _render_assistant_item(item, self.assistant_token)
        elif role == TOOL_AGENT_NAME: return _render_tool_agent_item(item, self.assistant_token)
        elif role == FILE_UPLOADER_NAME: return _render_file_uploader_item(item, self.user_token)
        else: raise RuntimeError(f"Unknown role \"{role}\".")

    def render_context(
        self, 
        context: ForgetableContext, 
        window: int
    ) -> str:
        text = ""
        for i, item in enumerate(self.crop_context(context, window)):
            rendered = self.render_item(item)
            if i == 0:
                rendered = "\n\n" + rendered.lstrip(" <s>[INST]")
            text += rendered
        return text

    def render_instruction(
        self,
        login_time_stamp: str,
        assistant_name: str, 
        user_nickname: str,
        tools: Iterable[BaseTool],
        user_preamble: str,
    ) -> str:
        examples = "\n".join([tool.example for tool in tools])
        
        return self.instructions.format(
            now=login_time_stamp, 
            assistant=assistant_name, 
            user_nickname=user_nickname,
            tool_names=[tool.name for tool in tools],
            tool_description='\n'.join([f"\"{tool.name}\": {tool.description}\n" for tool in tools]),
            user_preamble=user_preamble,
            examples=examples,
        )
        
    def render_complete_prompt(
        self, 
        context: ForgetableContext, 
        window: int,
        login_time_stamp: str,
        assistant_name: str, 
        user_nickname: str,
        tools: Iterable[BaseTool],
        user_preamble: str,
    ) -> str:
        return self.render_instruction(login_time_stamp, assistant_name, user_nickname, tools, user_preamble) + \
               self.render_context(context, window) + \
               f""" Assistant:"""

    def stops(self) -> list[str]:
        return ["Assistant:", "User:"]




class Gemma2InstructPromptBuilder(PromptBuilderBase):
    def __init__(self) -> None:
        super().__init__(displayed_name="Gemma-2 Instruct")
        self.instructions = _load_text_file("system_prompt_template/gemma_2_it_sys_ppt_gen_16.txt")
        
    def assistant_token(self, text: str) -> str:
        return f"""<start_of_turn>model
{text}<end_of_turn>
"""
        
    def user_token(self, text: str) -> str:
        return f"""<start_of_turn>user
{text}<end_of_turn>
"""
        

    def render_item(self, item) -> str:
        from global_settings import TOOL_AGENT_NAME, FILE_UPLOADER_NAME

        role = item["role"]
        if role == "User": return _render_user_item(item, self.user_token)
        elif role == "Assistant": return _render_assistant_item(item, self.assistant_token)
        elif role == TOOL_AGENT_NAME: return _render_tool_agent_item(item, self.assistant_token)
        elif role == FILE_UPLOADER_NAME: return _render_file_uploader_item(item, self.user_token)
        else: raise RuntimeError(f"Unknown role \"{role}\".")


    def render_instruction(
        self,
        login_time_stamp: str,
        assistant_name: str, 
        user_nickname: str,
        tools: Iterable[BaseTool],
        user_preamble
    ) -> str:
        examples = "\n".join([tool.example for tool in tools])
        
        return self.instructions.format(
            now=login_time_stamp, 
            assistant=assistant_name, 
            user_nickname=user_nickname,
            tool_names=[tool.name for tool in tools],
            tool_description='\n'.join([f"\"{tool.name}\": {tool.description}\n" for tool in tools]),
            user_preamble=user_preamble,
            examples=examples,
        )
        
    def render_complete_prompt(
        self, 
        context: ForgetableContext, 
        window: int,
        login_time_stamp: str,
        assistant_name: str, 
        user_nickname: str,
        tools: Iterable[BaseTool],
        user_preamble: str,
    ) -> str:
        return self.render_instruction(login_time_stamp, assistant_name, user_nickname, tools, user_preamble) + \
               self.render_context(context, window) + \
               """<start_of_turn>model
"""






class DeepSeekV2PromptBuilder(PromptBuilderBase):
    def __init__(self) -> None:
        super().__init__(displayed_name="DeepSeek V2")
        self.instructions = _load_text_file("system_prompt_template/deepseek_v2_sys_ppt.txt")
        
    def assistant_token(self, text: str) -> str:
        return f"""Assistant: {text}<｜end▁of▁sentence｜>""" # use "｜" not "|"
    def user_token(self, text: str) -> str:
        return f"""User: {text}"""

    def render_item(self, item) -> str:
        from global_settings import TOOL_AGENT_NAME, FILE_UPLOADER_NAME

        role = item["role"]
        if role == "User": return _render_user_item(item, self.user_token)
        elif role == "Assistant": return _render_assistant_item(item, self.assistant_token)
        elif role == TOOL_AGENT_NAME: return _render_tool_agent_item(item, self.assistant_token)
        elif role == FILE_UPLOADER_NAME: return _render_file_uploader_item(item, self.user_token)
        else: raise RuntimeError(f"Unknown role \"{role}\".")
    
    def render_instruction(
        self,
        login_time_stamp: str,
        assistant_name: str, 
        user_nickname: str,
        tools: Iterable[BaseTool],
        user_preamble: str,
    ) -> str:
        examples = "\n".join([tool.example for tool in tools])
        
        return self.instructions.format(
            now=login_time_stamp, 
            assistant=assistant_name, 
            user_nickname=user_nickname,
            tool_names=[tool.name for tool in tools],
            tool_description='\n'.join([f"\"{tool.name}\": {tool.description}\n" for tool in tools]),
            user_preamble=user_preamble,
            examples=examples,
        )
        
    def render_complete_prompt(
        self, 
        context: ForgetableContext, 
        window: int,
        login_time_stamp: str,
        assistant_name: str, 
        user_nickname: str,
        tools: Iterable[BaseTool],
        user_preamble: str,
    ) -> str:
        return self.render_instruction(login_time_stamp, assistant_name, user_nickname, tools, user_preamble) + \
               self.render_context(context, window) + \
               """Assistant: """
        
    def stops(self) -> list[str]:
        return ["Assistant:", "User:"]



class MistralPromptBuilder(PromptBuilderBase):
    """
    Based on this yaml:
        https://github.com/oobabooga/text-generation-webui/blob/af839d20acee2710022d735d2d1dc9cb48696f73/instruction-templates/Mistral.yaml#L7
    This prompt template is somewhat different from the original template packed in Mistral config.json.
    """
    def __init__(self) -> None:
        super().__init__(displayed_name="Mistral")
        self.instructions = """# System Instructions
## Basic Rules
Your role is to respond as an assistant named "{assistant}" when user send messages. 
For real-time, up-to-date, or uncommon information, you MUST use search tool accurately following grammer described below.
When using code blocks or indenting, it's generally recommended to use four half-width spaces. 

Your output must fit following XML format;
<tool>tool_name</tool><tool_input>input</tool_input>
, where "tool_name" and "input" must be replaced by those of actual tools.

After tool use, the results will be given in following format;
<tool>tool_name</tool><tool_output>results</tool_output>

## User's Rules
Your name is {assistant}, and user's nickname is {user_nickname}.
{user_preamble}

## Tools Usage
You can always use tools {tool_names} by inserting following commands to provide high quality response to user's last input. If tools are unnecessary, simply reply to the user.
{examples}

## Detailed Tool Descriptions
{tool_description}

## Time zone
{now}
"""
        
    def assistant_token(self, text: str) -> str:
        return f"""{text}</s>"""
        
    def user_token(self, text: str) -> str:
        return f"""[INST] {text} [/INST]"""

    def render_item(self, item) -> str:
        from global_settings import TOOL_AGENT_NAME, FILE_UPLOADER_NAME

        role = item["role"]
        if role == "User": return _render_user_item(item, self.user_token)
        elif role == "Assistant": return _render_assistant_item(item, self.assistant_token)
        elif role == TOOL_AGENT_NAME: return _render_tool_agent_item(item, self.assistant_token)
        elif role == FILE_UPLOADER_NAME: return _render_file_uploader_item(item, self.user_token)
        else: raise RuntimeError(f"Unknown role \"{role}\".")

    def render_instruction(
        self,
        login_time_stamp: str,
        assistant_name: str, 
        user_nickname: str,
        tools: Iterable[BaseTool],
        user_preamble
    ) -> str:
        return self.instructions.format(
            now=login_time_stamp, 
            assistant=assistant_name, 
            user_nickname=user_nickname,
            tool_names=[tool.name for tool in tools],
            tool_description='\n'.join([f"\"{tool.name}\": {tool.description}\n" for tool in tools]),
            user_preamble=user_preamble,
            examples="\n".join([tool.example for tool in tools]),
        ).replace("\n\n\n", "\n\n")
        
    def render_complete_prompt(
        self, 
        context: ForgetableContext, 
        window: int,
        login_time_stamp: str,
        assistant_name: str, 
        user_nickname: str,
        tools: Iterable[BaseTool],
        user_preamble: str,
    ) -> str:
        return self.render_instruction(login_time_stamp, assistant_name, user_nickname, tools, user_preamble) + \
               self.render_context(context, window)
               # Mistral doesn't need any generation prompt unlike Llama-2.



        
class MagistralPromptBuilder(PromptBuilderBase):
    """
    This one:
        https://huggingface.co/mistralai/Magistral-Small-2506
    """
    def __init__(self) -> None:
        super().__init__(displayed_name="Magistral")
        self.instructions = """<s>[SYSTEM_PROMPT]
## Basic Rules
Your role is to respond as an assistant named "{assistant}" when user send messages. 
For real-time, up-to-date, or uncommon information, you MUST use search tool accurately following grammer described below.
When using code blocks or indenting, it's generally recommended to use four half-width spaces. 

Your output must fit following XML format;
<tool>tool_name</tool><tool_input>input</tool_input>
, where "tool_name" and "input" must be replaced by those of actual tools.

After tool use, the results will be given in following format;
<tool>tool_name</tool><tool_output>results</tool_output>

## User's Rules
Your name is {assistant}, and user's nickname is {user_nickname}.
{user_preamble}

## Tools Usage
You can always use tools {tool_names} by inserting following commands to provide high quality response to user's last input. If tools are unnecessary, simply reply to the user.
{examples}

## Detailed Tool Descriptions
{tool_description}

## Time zone
{now}[/SYSTEM_PROMPT]
"""
        
    def assistant_token(self, text: str) -> str:
        return f"""{text}</s>"""
        
    def user_token(self, text: str) -> str:
        return f"""[INST] {text} [/INST]"""

    def render_item(self, item) -> str:
        from global_settings import TOOL_AGENT_NAME, FILE_UPLOADER_NAME

        role = item["role"]
        if role == "User": return _render_user_item(item, self.user_token)
        elif role == "Assistant": return _render_assistant_item(item, self.assistant_token)
        elif role == TOOL_AGENT_NAME: return _render_tool_agent_item(item, self.assistant_token)
        elif role == FILE_UPLOADER_NAME: return _render_file_uploader_item(item, self.user_token)
        else: raise RuntimeError(f"Unknown role \"{role}\".")

    def render_instruction(
        self,
        login_time_stamp: str,
        assistant_name: str, 
        user_nickname: str,
        tools: Iterable[BaseTool],
        user_preamble
    ) -> str:
        return self.instructions.format(
            now=login_time_stamp, 
            assistant=assistant_name, 
            user_nickname=user_nickname,
            tool_names=[tool.name for tool in tools],
            tool_description='\n'.join([f"\"{tool.name}\": {tool.description}\n" for tool in tools]),
            user_preamble=user_preamble,
            examples="\n".join([tool.example for tool in tools]),
        ).replace("\n\n\n", "\n\n")
        
    def render_complete_prompt(
        self, 
        context: ForgetableContext, 
        window: int,
        login_time_stamp: str,
        assistant_name: str, 
        user_nickname: str,
        tools: Iterable[BaseTool],
        user_preamble: str,
    ) -> str:
        return self.render_instruction(login_time_stamp, assistant_name, user_nickname, tools, user_preamble) + \
               self.render_context(context, window)
               # Mistral doesn't need any generation prompt unlike Llama-2.





class OpenAIHarmonyPromptBuilder(PromptBuilderBase):
    """
    This one:
        https://cookbook.openai.com/articles/openai-harmony
    """
    def __init__(self, reasoning_level: str = "medium") -> None:
        if reasoning_level not in ["low", "medium", "high"]:
            raise ValueError("reasoning_level must be one of ['low', 'medium', 'high'].")
            
        super().__init__(displayed_name=f"OpenAI Harmony (Reasoning: {reasoning_level})")
        self.instructions = """<|startoftext|><|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: {now}

Reasoning: __gpt_oss_reasoning_level__

# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>developer<|message|># Instruction
## Basic Rules
Your role is to respond as an assistant named "{assistant}" when user send messages. 
You have access to tools to enhance your responses.

## Format
After drafting your thoughts in commentary channel, you must output XML in the format;
<tool>tool_name</tool><tool_input>input</tool_input>
`<tool>...</tool>` here choose actual tool name.
`<tool_input>...</tool_input>` here write down inputs for the tool you selected.

## User's Rules
Your name is {assistant}, and user's nickname is {user_nickname}.
{user_preamble}

## Tools Usage
You can always use tools {tool_names} by inserting following commands to provide high quality response to user's last input. If tools are unnecessary, simply reply to the user.
{examples}

## Detailed Tool Descriptions
{tool_description}<|end|>""".replace("__gpt_oss_reasoning_level__", reasoning_level)
        
    def render_instruction(
        self,
        login_time_stamp: str,
        assistant_name: str, 
        user_nickname: str,
        tools: Iterable[BaseTool],
        user_preamble
    ) -> str:
        return self.instructions.format(
            now=login_time_stamp, 
            assistant=assistant_name, 
            user_nickname=user_nickname,
            tool_names=[tool.name for tool in tools],
            tool_description='\n'.join([f"\"{tool.name}\": {tool.description}\n" for tool in tools]),
            user_preamble=user_preamble,
            examples="\n".join([tool.example for tool in tools]),
        ).replace("\n\n\n", "\n\n")
        
    def assistant_token(self, text: str) -> str:
        return f"<|start|>assistant<|channel|>final<|message|>{text}<|end|>"
        
    def user_token(self, text: str) -> str:
        return f"<|start|>user<|message|>{text}<|end|>"

    def tool_token(self, text: str) -> str:
        return f"<|start|><|channel|>commentary<|message|>{text}<|end|>"

    def render_item(self, item) -> str:
        from global_settings import TOOL_AGENT_NAME, FILE_UPLOADER_NAME

        role = item["role"]
        if role == "User": return _render_user_item(item, self.user_token)
        elif role == "Assistant": return _render_assistant_item(item, self.assistant_token)
        elif role == TOOL_AGENT_NAME: return _render_tool_agent_item(item, self.tool_token)
        elif role == FILE_UPLOADER_NAME: return _render_file_uploader_item(item, self.tool_token)
        else: raise RuntimeError(f"Unknown role \"{role}\".")

    def think_tokens(self) -> tuple[str, str]|None:
        return ("<|channel|>analysis<|message|>", "<|end|>")
        
    def render_complete_prompt(
        self, 
        context: ForgetableContext, 
        window: int,
        login_time_stamp: str,
        assistant_name: str, 
        user_nickname: str,
        tools: Iterable[BaseTool],
        user_preamble: str,
    ) -> str:
        return self.render_instruction(login_time_stamp, assistant_name, user_nickname, tools, user_preamble) + \
               self.render_context(context, window) + "<|start|>assistant"

    def stops(self) -> list[str]:
        return ["<|end|>"]





class MagistralThinkingPromptBuilder(PromptBuilderBase):
    """
    This one:
        https://huggingface.co/mistralai/Magistral-Small-2506
    """
    def __init__(self) -> None:
        super().__init__(displayed_name="Magistral-Thinking")
        self.instructions = """<s>[SYSTEM_PROMPT]
## Basic Rules
Your role is to respond as an assistant named "{assistant}" when user send messages. 
For real-time, up-to-date, or uncommon information, you MUST use search tool accurately following grammer described below.
When using code blocks or indenting, it's generally recommended to use four half-width spaces. 

Your output must fit following format, containing three tags;
[THINK]draft your thoughts before answering[/THINK]<tool>tool_name</tool><tool_input>input</tool_input>
`[THINK]...[/THINK]` Write your thoughts here before actual outputs for better responses.
`<tool>...</tool>` Here choose actual tool name.
`<tool_input>...</tool_input>` Here write down inputs for the tool you selected.

After tool use, the results will be given in following format;
<tool>tool_name</tool><tool_output>results</tool_output>

## User's Rules
Your name is {assistant}, and user's nickname is {user_nickname}.
{user_preamble}

## Tools Usage
You can always use tools {tool_names} by inserting following commands to provide high quality response to user's last input. If tools are unnecessary, simply reply to the user.
{examples}

## Detailed Tool Descriptions
{tool_description}

## Time zone
{now}[/SYSTEM_PROMPT]
"""
        
    def assistant_token(self, text: str) -> str:
        return f"""{text}</s>"""
        
    def user_token(self, text: str) -> str:
        return f"""[INST] {text} [/INST]"""

    def tool_token(self, text: str) -> str:
        return f"""[TOOL_RESULTS] {text} [/TOOL_RESULTS]"""

    def render_item(self, item) -> str:
        from global_settings import TOOL_AGENT_NAME, FILE_UPLOADER_NAME

        role = item["role"]
        if role == "User": return _render_user_item(item, self.user_token)
        elif role == "Assistant": return _render_assistant_item(item, self.assistant_token)
        elif role == TOOL_AGENT_NAME: return _render_tool_agent_item(item, self.tool_token)
        elif role == FILE_UPLOADER_NAME: return _render_file_uploader_item(item, self.tool_token)
        else: raise RuntimeError(f"Unknown role \"{role}\".")

    def think_tokens(self) -> tuple[str, str]|None:
        return ("[THINK]", "[/THINK]")

    def render_instruction(
        self,
        login_time_stamp: str,
        assistant_name: str, 
        user_nickname: str,
        tools: Iterable[BaseTool],
        user_preamble
    ) -> str:
        return self.instructions.format(
            now=login_time_stamp, 
            assistant=assistant_name, 
            user_nickname=user_nickname,
            tool_names=[tool.name for tool in tools],
            tool_description='\n'.join([f"\"{tool.name}\": {tool.description}\n" for tool in tools]),
            user_preamble=user_preamble,
            examples="\n".join([tool.example for tool in tools]),
        ).replace("\n\n\n", "\n\n")
        
    def render_complete_prompt(
        self, 
        context: ForgetableContext, 
        window: int,
        login_time_stamp: str,
        assistant_name: str, 
        user_nickname: str,
        tools: Iterable[BaseTool],
        user_preamble: str,
    ) -> str:
        return self.render_instruction(login_time_stamp, assistant_name, user_nickname, tools, user_preamble) + \
               self.render_context(context, window)
               # Mistral doesn't need any generation prompt unlike Llama-2.






class Vicuna1_1JaPromptBuilder(PromptBuilderBase):
    def __init__(self) -> None:
        super().__init__(displayed_name="Vicuna 1.1 JA")
        self.instructions = """# 基本ルール
あなたは"{assistant}"という日本語を話すアシスタントとして、ユーザーのメッセージに返事をしてください。
リアルタイム・最新情報や特殊な知識を必要とする質問には、下記に示す記法に従い検索ツールを使用してください。
コードブロックでインデントを行う場合は、半角スペース4個を使用します。

あなたの出力は下記XMLの形式に従う必要があります。
<tool>tool_name</tool><tool_input>input</tool_input>
("tool_name"と"input"は実際に提供されているツールに沿ったもので置き換えてください。)

ツール使用後は、下記形式で結果が返ってきます。
<tool>tool_name</tool><tool_output>results</tool_output>

# ユーザー追加ルール
あなたの名前は{assistant}で、ユーザーのニックネームは{user_nickname}です。
{user_preamble}

# ツール使用方法
ユーザのリクエストに対して高品質な回答を行うために、下記の文法に従いコマンドを入力することでツール{tool_names}を使用できます。ツールが不要の場合は通常通り返信を行なってください。
{examples}

# 各ツール説明
{tool_description}

# 現在日時
{now}
----------------
"""
        
    def assistant_token(self, text: str) -> str:
        return f"""ASSISTANT: {text}
"""
        
    def user_token(self, text: str) -> str:
        return f"""USER: {text}
"""

    def render_item(self, item) -> str:
        from global_settings import TOOL_AGENT_NAME, FILE_UPLOADER_NAME

        role = item["role"]
        if role == "User": return _render_user_item(item, self.user_token)
        elif role == "Assistant": return _render_assistant_item(item, self.assistant_token)
        elif role == TOOL_AGENT_NAME: return _render_tool_agent_item(item, self.assistant_token)
        elif role == FILE_UPLOADER_NAME: return _render_file_uploader_item(item, self.user_token)
        else: raise RuntimeError(f"Unknown role \"{role}\".")

    def render_instruction(
        self,
        login_time_stamp: str,
        assistant_name: str, 
        user_nickname: str,
        tools: Iterable[BaseTool],
        user_preamble
    ) -> str:
        return self.instructions.format(
            now=login_time_stamp, 
            assistant=assistant_name, 
            user_nickname=user_nickname,
            tool_names=[tool.name for tool in tools],
            tool_description='\n'.join([f"\"{tool.name}\": {tool.description}\n" for tool in tools]),
            user_preamble=user_preamble,
            examples="\n".join([tool.example for tool in tools]),
        ).replace("\n\n\n", "\n\n")
        
    def render_complete_prompt(
        self, 
        context: ForgetableContext, 
        window: int,
        login_time_stamp: str,
        assistant_name: str, 
        user_nickname: str,
        tools: Iterable[BaseTool],
        user_preamble: str,
    ) -> str:
        return self.render_instruction(login_time_stamp, assistant_name, user_nickname, tools, user_preamble) + \
               self.render_context(context, window) + \
               "ASSISTANT: "

    def stops(self) -> list[str]:
        return ["ASSISTANT:", "USER:"]




class CommandAReasoningPromptBuilder(PromptBuilderBase):
    def __init__(self) -> None:
        super().__init__(displayed_name="Command A Reasoning")
        self.instructions = """<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|># System Preamble
## Basic Rules
Your role is to respond when user send messages. 
You don't have any real-time, up-to-date, or uncommon information until you invoke tools.
When using code blocks or indenting, it's generally recommended to use four half-width spaces.

Your output must fit following format;
<|START_THINKING|>draft your thoughts here before answering for better response<|END_THINKING|><|START_ACTION|><tool>tool_name</tool><tool_input>input</tool_input><|END_ACTION|>
, where "tool_name" and "input" must be replaced by those of actual tools.
The draft of your thoughts should contain step-by-step explanation to enhance your response.

After tool use, the results will be given in following format;
<tool>tool_name</tool><tool_output>results</tool_output>

# User Preamble
## Task and Context
Your name is {assistant}, and user's nickname is {user_nickname}.
{user_preamble}

## Available Tools
Here is a list of tools available to you:

{tool_description}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>User logged in:{now}<|END_OF_TURN_TOKEN|>"""

    def think_tokens(self) -> tuple[str, str]:
        return ("<|START_THINKING|>", "<|END_THINKING|>")
        
    def assistant_token(self, text: str) -> str:
        return f"<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_ACTION|>{text}<|END_ACTION|><|END_OF_TURN_TOKEN|>"
    def user_token(self, text: str) -> str:
        return f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{text}<|END_OF_TURN_TOKEN|>"
    def system_token(self, text: str) -> str:
        return f"<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{text}<|END_OF_TURN_TOKEN|>"
    def tool_token(self, text: str) -> str:
        return f"<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><|START_TOOL_RESULT|>{text}<|END_TOOL_RESULT|><|END_OF_TURN_TOKEN|>"
    def default_token(self, text: str) -> str:
        return f"<|START_OF_TURN_TOKEN|>{text}<|END_OF_TURN_TOKEN|>"
    def stops(self) -> list[str]:
        return ["<|END_RESPONSE|>"]

    def render_item(self, item) -> str:
        from global_settings import TOOL_AGENT_NAME, FILE_UPLOADER_NAME

        role = item["role"]
        if role == "User": return _render_user_item(item, self.user_token)
        elif role == "Assistant": return _render_assistant_item(item, self.assistant_token)
        elif role == TOOL_AGENT_NAME: return _render_tool_agent_item(item, self.tool_token)
        elif role == FILE_UPLOADER_NAME: return _render_file_uploader_item(item, self.system_token)
        else: raise RuntimeError(f"Unknown role \"{role}\".")

    def render_instruction(
        self,
        login_time_stamp: str,
        assistant_name: str, 
        user_nickname: str,
        tools: Iterable[BaseTool],
        user_preamble: str,
    ) -> str:
        return self.instructions.format(
            now=login_time_stamp, 
            assistant=assistant_name, 
            user_nickname=user_nickname,
            tool_names=[tool.name for tool in tools],
            tool_description='\n'.join([f"\"{tool.name}\": {tool.description}\n" for tool in tools]),
            user_preamble=user_preamble,
        )
        
    def render_complete_prompt(
        self, 
        context: ForgetableContext, 
        window: int,
        login_time_stamp: str,
        assistant_name: str, 
        user_nickname: str,
        tools: Iterable[BaseTool],
        user_preamble: str,
    ) -> str:
        examples = "\n".join([tool.example for tool in tools])
        return self.render_instruction(login_time_stamp, assistant_name, user_nickname, tools, user_preamble) + \
               self.render_context(context, window) + \
               self.system_token(f"""You should always use tools {[tool.name for tool in tools]} to provide high quality response to user's last input. If the information doesn't need evidence or accuracy, you can skip tools and directly answer.
{examples}""") + \
               "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_ACTION|>"





class GLM4PromptBuilder(PromptBuilderBase):
    """https://www.reddit.com/r/LocalLLaMA/comments/1no3qka/glm_45_air_template_breaking_llamacpp_prompt/"""
    def __init__(self) -> None:
        super().__init__(displayed_name="GLM4(Think)")
        # "gMASK" is the BOS token.
        self.instructions = """[gMASK]<sop>
<|system|>
# Basic Rules
Your role is to respond when user send messages. 
You don't have any real-time, up-to-date, or uncommon information until you invoke tools.
When using code blocks or indenting, it's generally recommended to use four half-width spaces.

Your output must fit following format;
<think>draft your thoughts here before answering for better response</think><tool>tool_name</tool><tool_input>input</tool_input>
, where "tool_name" and "input" must be replaced by those of actual tools.
The draft of your thoughts should contain step-by-step explanation to enhance your response.

After tool use, the results will be given in following format;
<tool>tool_name</tool><tool_output>results</tool_output>

# Additional Rules
Your name is {assistant}, and user's nickname is {user_nickname}.
{user_preamble}

# Tools
Here is a list of tools available to you:
<tools>
{tool_description}
</tools>
<|system|>
User logged in:{now}
"""

    def think_tokens(self) -> tuple[str, str]:
        return ("<think>", "</think>")
        
    def assistant_token(self, text: str) -> str:
        return f"""<|assistant|>
{text}
"""
    def user_token(self, text: str) -> str:
        return f"""<|user|>
{text}
"""
    def system_token(self, text: str) -> str:
        return f"""<|system|>
{text}
"""
    def tool_token(self, text: str) -> str:
        return f"""<|system|>
{text}
"""
    def default_token(self, text: str) -> str:
        return f"{text}"
    def stops(self) -> list[str]:
        return ["<|user|>", "<|system|>"]

    def render_item(self, item) -> str:
        from global_settings import TOOL_AGENT_NAME, FILE_UPLOADER_NAME

        role = item["role"]
        if role == "User": return _render_user_item(item, self.user_token)
        elif role == "Assistant": return _render_assistant_item(item, self.assistant_token)
        elif role == TOOL_AGENT_NAME: return _render_tool_agent_item(item, self.tool_token)
        elif role == FILE_UPLOADER_NAME: return _render_file_uploader_item(item, self.system_token)
        else: raise RuntimeError(f"Unknown role \"{role}\".")

    def render_instruction(
        self,
        login_time_stamp: str,
        assistant_name: str, 
        user_nickname: str,
        tools: Iterable[BaseTool],
        user_preamble: str,
    ) -> str:
        return self.instructions.format(
            now=login_time_stamp, 
            assistant=assistant_name, 
            user_nickname=user_nickname,
            tool_names=[tool.name for tool in tools],
            tool_description='\n'.join([f"\"{tool.name}\": {tool.description}\n" for tool in tools]),
            user_preamble=user_preamble,
        )
        
    def render_complete_prompt(
        self, 
        context: ForgetableContext, 
        window: int,
        login_time_stamp: str,
        assistant_name: str, 
        user_nickname: str,
        tools: Iterable[BaseTool],
        user_preamble: str,
    ) -> str:
        examples = "\n".join([tool.example for tool in tools])
        return self.render_instruction(login_time_stamp, assistant_name, user_nickname, tools, user_preamble) + \
               self.render_context(context, window) + \
               """<|assistant|>
"""






prompt_builders = [
    CommandRPromptBuilder(),
    CommandRThinkingPromptBuilder(),
    CommandAReasoningPromptBuilder(),
    Llama3PromptBuilder(),
    ChatMLPromptBuilder(),
    ChatMLThinkingPromptBuilder(),
    Llama2PromptBuilder(),
    JaCommMSPromptBuilder(),
    Gemma2InstructPromptBuilder(),
    MistralPromptBuilder(),
    MagistralPromptBuilder(),
    MagistralThinkingPromptBuilder(),
    DeepSeekV2PromptBuilder(),
    Vicuna1_1JaPromptBuilder(),
    GLM4PromptBuilder(),
    OpenAIHarmonyPromptBuilder(reasoning_level="low"),
    OpenAIHarmonyPromptBuilder(reasoning_level="medium"),
    OpenAIHarmonyPromptBuilder(reasoning_level="high"),
]