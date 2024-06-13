from abc import ABC, abstractmethod
import session_states as session


class PromptBuilderBase(ABC):
    @abstractmethod
    def render_item(self, item) -> str:
        pass
    @abstractmethod
    def render_context(self) -> str:
        pass
    @abstractmethod
    def render_instruction(self) -> str:
        pass
    @abstractmethod
    def render_complete_prompt(self) -> str:
        pass
    def build(self) -> str:
        """ Just an alias of render_complete_prompt. """
        return self.render_complete_prompt()





from lib.infrastructure import ForgetableContext

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
        from lib.utils import get
        from global_settings import PYTHON_RUNTIME_NAME, SEARCH_AGENT_NAME

        
        role = get(item, 'role')
        content = get(item, 'content')
        code = get(item, 'code')
        code_output = get(item, 'code_output')
        search_query = get(item, 'search_query')
        search_result = get(item, 'search_result')
        references = get(item, 'references')
    
        item_text = content
    
        def add_block(block_name, block_content) -> str:
            return f"""
```{block_name}
{block_content}
```
"""
        
        if code: item_text += add_block("python", code)
        if code_output: item_text += add_block("output", code_output)
        if search_query: item_text += add_block("google", search_query)
        if search_result: item_text += add_block("result", search_result)
    
        if role == session.assistant_name: return self.assistant_token(item_text)
        elif role == "User": return self.user_token(item_text)
        elif role == "System": return self.system_token(item_text)
        elif role == PYTHON_RUNTIME_NAME: return self.system_token(item_text)
        elif role == SEARCH_AGENT_NAME: return self.system_token(item_text)
        else: return self.default_token(item_text)

    def render_context(self) -> str:
        text = ""
        for item in self.context.history()[-session.context_window:]:
            text += self.render_item(item)
        return text

    def render_instruction(self) -> str:
        return self.instructions.format(
            now=session.login_time_stamp, 
            assistant=session.assistant_name, 
            user_nickname=session.user_nickname,
            tool_names=[tool.name for tool in session.tools],
            tool_description='\n'.join([f"{tool.name}: {tool.description}" for tool in session.tools]),
            user_preamble=session.user_preamble.value,
        )
        
    def render_complete_prompt(self) -> str:
        tools = session.tools
        
        examples = "\n".join([tool.example for tool in tools])
        return self.render_instruction() + \
               self.render_context() + \
               self.system_token(f"""You can always use tools{[tool.name for tool in tools]} by inserting following commands to provide high quality response to user's last input. If tools are unnecessary, simply reply to the user.
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

    def render_item(self, item) -> str:
        from lib.utils import get
        from global_settings import PYTHON_RUNTIME_NAME, SEARCH_AGENT_NAME

        
        role = get(item, 'role')
        content = get(item, 'content')
        code = get(item, 'code')
        code_output = get(item, 'code_output')
        search_query = get(item, 'search_query')
        search_result = get(item, 'search_result')
        references = get(item, 'references')
    
        item_text = content
    
        def add_block(block_name, block_content) -> str:
            return f"""
```{block_name}
{block_content}
```
"""
        
        if code: item_text += add_block("python", code)
        if code_output: item_text += add_block("output", code_output)
        if search_query: item_text += add_block("google", search_query)
        if search_result: item_text += add_block("result", search_result)
    
        if role == session.assistant_name: return self.assistant_token(item_text)
        elif role == "User": return self.user_token(item_text)
        elif role == "System": return self.system_token(item_text)
        elif role == PYTHON_RUNTIME_NAME: return self.assistant_token(item_text)
        elif role == SEARCH_AGENT_NAME: return self.assistant_token(item_text)
        else: raise RuntimeError("Default token not set.")

    def render_context(self) -> str:
        text = ""
        for item in self.context.history()[-session.context_window:]:
            text += self.render_item(item)
        return text

    def render_instruction(self) -> str:
        examples = "\n".join([tool.example for tool in session.tools])
        
        return self.instructions.format(
            now=session.login_time_stamp, 
            assistant=session.assistant_name, 
            user_nickname=session.user_nickname,
            tool_names=[tool.name for tool in session.tools],
            tool_description='\n'.join([f"{tool.name}: {tool.description}" for tool in session.tools]),
            user_preamble=session.user_preamble.value,
            examples=examples,
        )
        
    def render_complete_prompt(self) -> str:
        examples = "\n".join([tool.example for tool in session.tools])
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
        from lib.utils import get
        from global_settings import PYTHON_RUNTIME_NAME, SEARCH_AGENT_NAME

        
        role = get(item, 'role')
        content = get(item, 'content')
        code = get(item, 'code')
        code_output = get(item, 'code_output')
        search_query = get(item, 'search_query')
        search_result = get(item, 'search_result')
        references = get(item, 'references')
    
        item_text = content
    
        def add_block(block_name, block_content) -> str:
            return f"""
```{block_name}
{block_content}
```
"""
        
        if code: item_text += add_block("python", code)
        if code_output: item_text += add_block("output", code_output)
        if search_query: item_text += add_block("google", search_query)
        if search_result: item_text += add_block("result", search_result)
    
        if role == session.assistant_name: return self.assistant_token(item_text)
        elif role == "User": return self.user_token(item_text)
        elif role == "System": return self.system_token(item_text)
        elif role == PYTHON_RUNTIME_NAME: return self.assistant_token(item_text)
        elif role == SEARCH_AGENT_NAME: return self.assistant_token(item_text)
        else: raise RuntimeError("Default token not set.")

    def render_context(self) -> str:
        text = ""
        for item in self.context.history()[-session.context_window:]:
            text += self.render_item(item)
        return text

    def render_instruction(self) -> str:
        tools = session.tools
        
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
        examples = "\n".join([tool.example for tool in session.tools])
        return self.render_instruction() + \
               self.render_context() + \
               """
<|im_start|>assistant
"""