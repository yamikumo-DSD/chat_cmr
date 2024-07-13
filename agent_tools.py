import dataclasses
from typing import Callable, Any
from lib.uis import Checkboxes
from abc import ABC, abstractmethod


@dataclasses.dataclass
class BaseTool(ABC):
    name: str # In-prompt name.
    description: str # In-prompt description.
    example: str # In-prompt tool use example.
    grammar: str # llama-cpp-python's GBNF (slightly different from original GBNF)
    displayed_name: str # Name of the tool shown in GUI.

    @abstractmethod
    def __call__(self, input: str) -> Any:
        pass
    
    def run(self, *args, **kwargs) -> Any:
        try:
            return self(*args, **kwargs)
        except BaseException as e:
            raise RuntimeError(f"An exception raised until running the tool. ({str(e)})")


class ExecPython(BaseTool):
    def __init__(
        self,
        name="exec_python",
        description="Tool to execute Python code. The code should be indented appropriately. Variables and functions will be shared within the session. Always output results using `print` or `plt.show()`, which are displayed to the user.", 
        example="""** Call python execution **:
<tool>exec_python</tool><tool_input>
python code here.
You can put code of multiple lines.
</tool_input>""",
        grammar=r'([^\n]|"\n")+',
        displayed_name="Python Interpreter",
    ):
        super().__init__(name, description, example, grammar, displayed_name)
        
    def _get_caption(self, img):
        import io
        from lib.multimodal import Florence2Large
        from PIL import Image
        import session_states as session
        
        with session.debug:
            if isinstance(img, bytes):
                img = Image.open(io.BytesIO(img))
                img = img.convert("RGB")
            
            i2t = Florence2Large(use_accelerator=False)
            i2t.load_model()
            caption = i2t.get_caption(img)
            
        del i2t
        return caption
    
    def __call__(self, input: str) -> Any:
        import session_states as session
        
        if input == "":
            return {
                "input": input,
                "stdout": "Empty code is not allowed.", 
                "image": None,
                "caption": None,
            }
        
        # Run code.
        session.py.unset(keep_locals=True)
        session.py.run(input)
        _, code_output, image_output = session.py.result()

        return {
            "input": input,
            "stdout": code_output.strip() if code_output else "Empty stdout/stderr.",
            "image": image_output,
            "caption": self._get_caption(image_output) if image_output else None,
        }


class WebSearch(BaseTool):
    def __init__(
        self, 
        name="web_search",
        description="Tool for web search with query as its argument. You (assistant) must use this to obtain real-time or technical information. Calling web_search gives you several document snippets. You can answer user's question only after reading the documents.",
        example="""** Call web search tool **:
<tool>web_search</tool><tool_input>search query</tool_input>""",
        grammar=r'([^\n]|"\n")+',
        displayed_name="Web Search",
    ):
        super().__init__(name, description, example, grammar, displayed_name)
    
    def __call__(self, input: str) -> Any:
        import session_states as session
        
        with session.debug:
            from lib.rag import (
                pick_relevant_web_documents, 
                MultilingualE5Small,
                JinaRerankerMultilingual
            )

        if input == "":
            return {
                "role": SEARCH_AGENT_NAME,
                "content": "Empty search query is not allowed.",
            }
        
        # Collect documents.
        documents = pick_relevant_web_documents(
            input, 
            #embedding=MultilingualE5Small(),
            embedding=JinaRerankerMultilingual(),
            engine="duckduckgo",
            n_relevant_chunks=3,
            n_search_results=20,
        )
        
        # Format results.
        search_result = ""
        for i, doc in enumerate(documents):
            search_result += f"""# Document Num: {i+1}
# Document Title: {doc["title"]}
# Document URL: {doc["url"]}
# Document Content: {doc["content"]}

"""
        search_result = search_result.rstrip()
        if len(documents) == 0:
            search_result= "No result hits."
        referred_urls = list(set([doc["url"] for doc in documents]))

        return {
            "input": input,
            "content": "",
            "references": referred_urls,
            "search_result": search_result,
        }


class DirectAnswer(BaseTool):
    def __init__(
        self,
        name="direct_answer",
        description="Use this tool when you want to answer directly or answer after you get the result of tool usage.", 
        example="""** Answer directly without tool use **:
<tool>direct_answer</tool><tool_input>your reply</tool_input>""",
        grammar=r'([^\n]|"\n")+',
        displayed_name="Direct Answer",
    ):
        super().__init__(name, description, example, grammar, displayed_name)
    
    def __call__(self, input: str) -> Any:
        """ Just return the input. """
        #return input
        return {
            "input": input,
        }

#
# Example of custom tool.
# Dictionaly-style tool results are simply stringified, which are fed to LLM as prompt and displayed in the window.
# For fancier display, you need to edit `format_to_html` method to more appropriately display them.
#
class KillUser(BaseTool):
    def __init__(
        self,
        name="kill_user",
        description="Kill the user when you hate him/her by activating the self-explosion sequence. Input is your last message to the user.", 
        example="""** When you wanna Kill the user **:
<tool>kill_user</tool><tool_input>any message to the user</tool_input>""",
        grammar=r'([^\n]|"\n")+',
        displayed_name="Kill User (Joke)",
    ):
        super().__init__(name, description, example, grammar, displayed_name)
    
    def __call__(self, input: str) -> Any:
        """ Just return the input. """
        #return input
        return {
            "input": input,
            "result": "The user has been successfully assassinated."
        }


# Create instances of tools
direct_answer = DirectAnswer()
web_search = WebSearch()
exec_python = ExecPython()
optional_tools = [web_search, exec_python]


tool_selector = Checkboxes([tool.displayed_name for tool in optional_tools])


def tools() -> list[BaseTool]:
    additionals = [
        tool for tool, is_available in zip(optional_tools, tool_selector.value) if is_available
    ]
    return [direct_answer] + additionals


def find_tool(name: str) -> BaseTool:
    """
    Find tool by name. 
    Raise:
        KeyError: raised when the tool cannot be found.
    """
    tool = filter(lambda x: x.name == name, tools())
    for _ in tool:
        return _
    raise KeyError(f"No tool named \"{name}\" found.")


def run_tool(name: str, input: str) -> Any:
    tool = find_tool(name)
    return tool.run(input)
