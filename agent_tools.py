"""
Custom tool recipes. Refer `URLFetcher` for the simplest case.

Steps to create tools:
    1. Define a tool inheriting `BaseTool`.
    2. Fill in fields (name, description, etc.) in `__init__`.
    3. Override `__call__` to define behavior, which must return a `dict` or `list` object.

Fields to be filled:
    name(str): In-prompt name.
    description(str): In-prompt description.
    displayed_name(str): Name of the tool shown in GUI.
    args: list[ToolArgument]: Ar
    enabled_by_default(bool): If True, the tool is enabled by default.

Tool interpretation:
    Dictionary-style tool results are stringified in display and prompts.
    When the return value of `__call__` is:
    ```dict
    {
        "result_1": "value 1",
        "result_2": IgnoredKey("value 2"), # This will be ignored.
        "result_3": "value 3",
    }
    ```
    It will be interpreted as a string like:
    ```xml
    <result_1>value 1</result_1>
    <result_3>value 3</result_3>
    ```
    in displays and prompts.

Important notes:
    * When you wrap values with `IgnoredKey`, they are ignored in prompt and display.
    * For fancier display, edit `format_to_html` method in the main notebook code.
    * Objects of `bytes` or other non str-castable types are replaced by corresponding placeholder texts.
    * Refer to predefined tools `WebSearch` and `ExecPython` for fancier display.
    * llama-cpp-python's GBNF doesn't allow repetition tokens like `{m}`, `{m,n}` (https://github.com/abetlen/llama-cpp-python/issues/1547).
"""
    
import dataclasses
from typing import Any
from lib.uis import Checkboxes
from abc import ABC, abstractmethod
from lib.dicttoxml import IgnoredKey
from global_settings import AGENT_WORKING_DIR, KNOWLEDGE_DB_DIR


@dataclasses.dataclass
class ToolArgument:
    name: str
    # Only str is allowed for now.
    #type_: type
    description: str # In-prompt description.
    grammar: str # llama-cpp-python's GBNF (slightly different from original GBNF)


@dataclasses.dataclass
class BaseTool(ABC):
    name: str # In-prompt name.
    description: str # In-prompt description.
    displayed_name: str # Name of the tool shown in GUI.
    args: list[ToolArgument]
    enabled_by_default: bool = True # If True, the tool is enabled by default.

    @abstractmethod
    def __call__(self, input: str) -> Any:
        pass
    
    def run(self, *args, **kwargs) -> Any:
        try:
            return self(*args, **kwargs)
        except BaseException as e:
            raise RuntimeError(f"An exception raised until running the tool. ({str(e)})")

    @property
    def example(self) -> str:
        """ Render example string """
        n_args = len(self.args)
        if n_args == 0:
            return f"""** Use `{self.name}` **
<tool>{self.name}</tool>"""
        elif n_args == 1:
            return f"""** Use `{self.name}` **
<tool>{self.name}</tool>
<tool_input>\n{self.args[0].description}\n</tool_input>"""
        else:
            text = f"""** Use `{self.name}` **
<tool>{self.name}</tool>
"""
            text += "<tool_input>"
            text += "\n".join(
                map(lambda x: f"<{x.name}>\n{x.description}\n</{x.name}>", self.args)
            )
            text += "</tool_input>"
            return text

    @property
    def grammar(self) -> str:
        """ Render GBNF string """
        n_args = len(self.args)
        if n_args == 0:
            return f""
        elif n_args == 1:
            return self.args[0].grammar
        else:
            return "".join(
                # GBNF: "<arg_name>"[^<]"</arg_name>"
                # This has a potential problem where `x.grammar` can be ANY string that won't stop at closing tag "</".
                map(lambda x: '"<' + x.name + '>"' + x.grammar + '"</' + x.name + '>"', self.args)
            )


class ExecPython(BaseTool):
    def __init__(
        self,
        name="exec_python",
        description="Tool to execute Python code. The code should be indented appropriately. Variables and functions will be shared within the session. Always output results using `print` or `plt.show()`, which are displayed to the user.", 
        args=[ToolArgument(
            name="code", 
            description="Executable Python code.\nCode of multiple lines is granted.", 
            grammar=r'([^\n]|"\n")+'
        )],
        displayed_name="Python Interpreter",
        enabled_by_default=False,
    ):
        from global_settings import AGENT_WORKING_DIR
        from lib.python_repl import PythonREPL
        
        super().__init__(name, description, displayed_name, args)
        self._py = PythonREPL(replace_nl=False, temporal_working_directory=AGENT_WORKING_DIR)
        
    def _get_caption(self, img):
        import io
        from lib.multimodal import Florence2Large, HuihuiQwen3VLNSFWQA
        from PIL import Image
        
        if isinstance(img, bytes):
            img = Image.open(io.BytesIO(img))
            img = img.convert("RGB")
        
        i2t = HuihuiQwen3VLNSFWQA(use_accelerator=True)
        #i2t.load_model()
        caption = i2t.get_caption(img)
            
        del i2t
        return caption

    def reset_state(self) -> None:
        self._py.unset()
    
    def __call__(self, input: str) -> Any:
        if input == "":
            return {
                "stdout": "Empty code is not allowed.", 
                "image": None,
                "caption": None,
            }
        
        # Run code.
        self._py.unset(keep_locals=True)
        self._py.run(input)
        _, code_output, image_output = self._py.result()

        return {
            "stdout": code_output.strip() if code_output else "Empty stdout/stderr.",
            "image": image_output,
            "caption": self._get_caption(image_output) if image_output else None,
        }


class WebSearch(BaseTool):
    def __init__(
        self, 
        name="web_search",
        description="Tool for web search with query as its argument. You (assistant) must use this to obtain real-time or technical information. Calling web_search gives you several document snippets. When the topic needs multiple information (e.g A and B), you should split queries into simpler steps (e.g 'A'->'B') to avoid no result.",
        args=[ToolArgument(
            name="query", 
            description="Search query.", 
            grammar=r'([^\n]|"\n")+'
        )],
        displayed_name="Web Search",
    ):
        super().__init__(name, description, displayed_name, args)
    
    def __call__(self, input: str) -> Any:
        from lib.rag import (
            pick_relevant_web_documents, 
            MultilingualE5Small,
            JinaRerankerMultilingual,
            JinaRerankerV3,
        )

        if input == "":
            return [{
                "title": "No input",
                "url": "No input",
                "content": "No input",
                "score": IgnoredKey(0),
            }]
        
        # Collect documents.
        documents = pick_relevant_web_documents(
            input, 
            #embedding=MultilingualE5Small(),
            embedding=JinaRerankerMultilingual(),
            #embedding=JinaRerankerV3(),
            engine="duckduckgo",
            n_relevant_chunks=3,
            n_search_results=20,
            score_thresh=0.5,
        )
        for doc in documents:
            doc["score"] = IgnoredKey(doc["score"])

        if len(documents) == 0:
            return [{
                "title": "No result.",
                "url": "No result.",
                "content": "No result.",
                "score": IgnoredKey(0),
            }]

        return documents
        


class DirectAnswer(BaseTool):
    def __init__(
        self,
        name="direct_answer",
        description="Use this tool when you want to answer directly or answer after you get the result of tool usage.", 
        displayed_name="Direct Answer",
        args=[ToolArgument(
            name="reply", 
            description="Your reply.", 
            grammar=r'([^\n]|"\n")+'
        )],
    ):
        super().__init__(name, description, displayed_name, args)
    
    def __call__(self, input: str) -> Any:
        """ Just return the input. """
        #return input
        return {
            "echo": input,
        }


class URLFetcher(BaseTool):
    def __init__(
        self,
        name="url_fetcher",
        description="Fetch contents of web site of the URL given. The contents will be formatted in markdown style.", 
        displayed_name="URL Fetcher",
        args=[ToolArgument(
            name="url", 
            description="https://example.com", 
            grammar=r'"http" "s"? "://" ([^\n])+',
        )],
        cutoff: int = 1500,
        enabled_by_default: bool = True,
    ):
        self.cutoff = cutoff
        super().__init__(name, description, displayed_name, args, enabled_by_default)
    
    def __call__(self, input: str) -> Any:
        import requests
        from lib.utils import replace_text
        from lib.scraping import extract_html_elements, stringify_html_elements
        from lib.scraping import is_youtube_url, fetch_youtube_transcript
        
        try:
            response = requests.get(input)
        except Exception as e:
            return {
            "result": f"`requests` failed to fetching contents.",
        }

        # Format content (markdown).
        html = response.content
        elems = extract_html_elements(html)
        result = stringify_html_elements(elems)
        result = result if len(result) <= self.cutoff else result[:self.cutoff]+"...(truncated)"
        result = replace_text(result, {"<": "&lt;", ">": "&gt;"})

        if is_youtube_url(input):
            subtitle = fetch_youtube_transcript(input)
            subtitle = subtitle if len(subtitle) <= self.cutoff-len(result) else subtitle[:self.cutoff-len(result)]+"...(truncated)"
            return {
                "result": result,
                "subtitle": subtitle[:self.cutoff-len(result)]
            }

        return {
            "result": result,
        }



        
class RunZshSandbox(BaseTool):
    """
    Experimental.
    https://blog.syum.ai/entry/2025/04/27/232946
    """
    def __init__(
        self,
        name="zsh",
        description="Run shell code with zsh (MacOS).", 
        displayed_name="Run Zsh in Sandbox",
        args=[ToolArgument(
            name="code", 
            description="shell code", 
            grammar=r'([^\n]|"\n")+',
        )],
        enabled_by_default=False,
        timeout: int|None = 5,
        cutoff: int = 1500,
    ):
        super().__init__(name, description, displayed_name, args, enabled_by_default)
        
        self._cutoff = cutoff
        self._timeout = timeout

    def __call__(self, input: str) -> Any:
        import os
        from lib.utils import change_directory
        from global_settings import AGENT_WORKING_DIR
        
        # Create sandbox profile.
        PROFILE_TEMP_FILE = "temp_sandbox_profile.sb"
        profile = f"""
(version 1)
(allow default)

(deny file-write* 
    (require-not
        (require-any
            (subpath "/private/tmp/")
            (subpath "{os.path.join(os.getcwd(), AGENT_WORKING_DIR)}/")
            (literal "/dev/null")
        )
    )
)
"""
        with open(PROFILE_TEMP_FILE, "w") as file:
            file.write(profile)

        @change_directory(AGENT_WORKING_DIR)
        def run_code(code: str):
            import subprocess
            
            try:
                process = subprocess.run(
                    ["sandbox-exec", "-f", f"../{PROFILE_TEMP_FILE}", "zsh", "-c", input],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    timeout=self._timeout,
                )
            except subprocess.TimeoutExpired as e:
                return {
                    "result": f"Execution timeout ({str(e)})."
                }
    
            stdout: None|str = None
            stderr: None|str = None
            if process.stdout is not None:
                stdout = process.stdout if len(process.stdout) <= self._cutoff else process.stdout[:self._cutoff]+"...(truncated)"
            if process.stderr is not None:
                stderr = process.stderr if len(process.stderr) <= self._cutoff else process.stderr[:self._cutoff]+"...(truncated)"
                
            if process.returncode != 0:
                return {
                    "result": f"An error occured while running the script (returncode={process.returncode}).",
                    "stdout": stdout,
                    "stderr": stderr,
                }
            
            return {
                "result": "The code was successfully executed.",
                "stdout": stdout
            }

        return run_code(input)




# Test tool w/ multiple arguments.
class ImageQA(BaseTool):
    def __init__(
        self,
        name="image_qa",
        description="Returns answer for query about given image.",
        displayed_name="ImageQA",
        args=[
            ToolArgument(name="image_path", description="Path for an image.", grammar=r'[^<]+'),
            ToolArgument(name="query", description="Question about the image in natural language of English.(e.g. 'Describe the pose of XXX.' or something).", grammar=r"[a-zA-Z.,?!;:#$%&' ]+"),
        ]
    ):
        super().__init__(name, description, displayed_name, args)
        
    def __call__(self, image_path, query):
        from lib.utils import change_directory
        from lib.multimodal import HuihuiQwen3VLNSFWQA

        @change_directory(AGENT_WORKING_DIR)
        def get_caption(*args, **kwargs):
            captioner = HuihuiQwen3VLNSFWQA()
            return captioner.get_caption(image_path, query)
            
        return {
            "answer": get_caption(image_path, query)
        }


class RunZsh(BaseTool):
    def __init__(
        self,
        name="zsh",
        description="Run shell code with zsh (MacOS).", 
        displayed_name="Run Zsh",
        args=[ToolArgument(
            name="code", 
            description="shell code", 
            grammar=r'([^\n]|"\n")+',
        )],
        enabled_by_default=False,
        timeout: int|None = 5,
        cutoff: int = 1500
    ):
        super().__init__(name, description, displayed_name, args, enabled_by_default)
        
        self._cutoff = cutoff
        self._timeout = timeout

    def __call__(self, input: str) -> Any:
        from lib.utils import change_directory
        from global_settings import AGENT_WORKING_DIR

        @change_directory(AGENT_WORKING_DIR)
        def run_code(code: str):
            import subprocess
            
            try:
                process = subprocess.run(
                    ["zsh", "-c", input],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    timeout=self._timeout,
                )
            except subprocess.TimeoutExpired as e:
                return {
                    "result": f"Execution timeout ({str(e)})."
                }
    
            stdout: None|str = None
            stderr: None|str = None
            if process.stdout is not None:
                stdout = process.stdout if len(process.stdout) <= self._cutoff else process.stdout[:self._cutoff]+"...(truncated)"
            if process.stderr is not None:
                stderr = process.stderr if len(process.stderr) <= self._cutoff else process.stderr[:self._cutoff]+"...(truncated)"
                
            if process.returncode != 0:
                return {
                    "result": f"An error occured while running the script (returncode={process.returncode}).",
                    "stdout": stdout,
                    "stderr": stderr,
                }
            
            return {
                "result": "The code was successfully executed.",
                "stdout": stdout
            }

        return run_code(input)


        
class RunBash(BaseTool):
    def __init__(
        self,
        name="bash",
        description="Run shell code with bash.", 
        displayed_name="Run Bash",
        args=[ToolArgument(
            name="code", 
            description="shell code", 
            grammar=r'([^\n]|"\n")+',
        )],
        enabled_by_default=False,
        timeout: int|None = 5,
        cutoff: int = 1500
    ):
        super().__init__(name, description, displayed_name, args, enabled_by_default)
        
        self._cutoff = cutoff
        self._timeout = timeout

    def __call__(self, input: str) -> Any:
        from lib.utils import change_directory
        from global_settings import AGENT_WORKING_DIR

        @change_directory(AGENT_WORKING_DIR)
        def run_code(code: str):
            import subprocess
            
            try:
                process = subprocess.run(
                    ["bash", "-c", input],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    timeout=self._timeout,
                )
            except subprocess.TimeoutExpired as e:
                return {
                    "result": f"Execution timeout ({str(e)})."
                }
    
            stdout: None|str = None
            stderr: None|str = None
            if process.stdout is not None:
                stdout = process.stdout if len(process.stdout) <= self._cutoff else process.stdout[:self._cutoff]+"...(truncated)"
            if process.stderr is not None:
                stderr = process.stderr if len(process.stderr) <= self._cutoff else process.stderr[:self._cutoff]+"...(truncated)"
                
            if process.returncode != 0:
                return {
                    "result": f"An error occured while running the script (returncode={process.returncode}).",
                    "stdout": stdout,
                    "stderr": stderr,
                }
            
            return {
                "result": "The code was successfully executed.",
                "stdout": stdout
            }

        return run_code(input)



        
class DocSearch(BaseTool):
    def __init__(
        self, 
        name="doc_search",
        description="Tool to search local document with query as its argument. Compatible with semantic search like web search tool. Calling this tool gives you several document snippets.",
        args=[ToolArgument(
            name="query", 
            description="Search query.", 
            grammar=r'([^\n]|"\n")+'
        )],
        displayed_name="Document Search",
        db_path=KNOWLEDGE_DB_DIR,
        enabled_by_default=False,
    ):
        super().__init__(name, description, displayed_name, args, enabled_by_default)
        self.db_path = db_path
    
    def __call__(self, input: str) -> Any:
        from lib.knowledge_db import pick_relevant_local_documents
        from lib.rag import (
            MultilingualE5Small,
            JinaRerankerMultilingual,
            JinaRerankerV3,
        )

        if input == "":
            return [{
                "file_name": "No input",
                "content": "No input",
                "score": IgnoredKey(0),
            }]
        
        # Collect documents.
        documents = pick_relevant_local_documents(
            input, 
            db_path=self.db_path,
            reranker=JinaRerankerMultilingual(),
            #reranker=JinaRerankerV3(),
            n_relevant_chunks=3,
            n_search_results=40,
            score_thresh=0.1,
        )
        for doc in documents:
            doc["score"] = IgnoredKey(doc["score"])

        if len(documents) == 0:
            return [{
                "file_name": "No result.",
                "content": "No result.",
                "score": IgnoredKey(0),
            }]

        return documents

    def build_db(
        self,
        directory_path: str, 
        chunk_size: int = 100
    ):
        from lib.knowledge_db import build_knowledge_db
        build_knowledge_db(directory_path, self.db_path, chunk_size)


class Click(BaseTool):
    def __init__(
        self,
        name="click",
        description="Click particular location of the display. The coordinate must be relative coordinate (x,y)=(0~999, 0~999), obtained from `image_qa` tool given a screenshot provided by the user.",
        displayed_name="Click",
        enabled_by_default=False,
        args=[
            ToolArgument(name="x_coord", description="x coordinate (from `image_qa`)", grammar=r'[0-9]+'),
            ToolArgument(name="y_coord", description="y coordinate (from `image_qa`)", grammar=r'[0-9]+'),
        ]
    ):
        super().__init__(name, description, displayed_name, args)
        
    def __call__(self, x_coord, y_coord):
        import pyautogui
        import time
        x_coord, y_coord = int(x_coord), int(y_coord)

        if x_coord < 0 or x_coord > 1000 or y_coord < 0 or y_coord > 1000:
            return {
                "result": f"Error: the coordinate ({x_coord, y_coord}) is out of the border (0~1000, 0~1000). Is it really the value obtained by `image_qa`?",
            }

        # relative (0-1000) to absolute coordinate.
        screen_width, screen_height = pyautogui.size()
        absolute_x = int(screen_width * x_coord / 1000)
        absolute_y = int(screen_height * y_coord / 1000)
        
        # action
        time.sleep(2)
        pyautogui.click(absolute_x, absolute_y)
        time.sleep(1)
        pyautogui.click(absolute_x, absolute_y)
            
        return {
            "result": "Action was successfully completed.",
        }



class Screenshot(BaseTool):
    def __init__(
        self,
        name="take_screenshot",
        description="Capture screenshot to see what is happening on the screen. Useful to confirm the result or your actions. No arguments.", 
        args=[],
        displayed_name="Screenshot",
    ):
        
        super().__init__(name, description, displayed_name, args)
        
    def _get_caption(self, img):
        import io
        from lib.multimodal import HuihuiQwen3VLNSFWQA
        from PIL import Image
        
        if isinstance(img, bytes):
            img = Image.open(io.BytesIO(img))
            img = img.convert("RGB")
        
        i2t = HuihuiQwen3VLNSFWQA(use_accelerator=True)
        caption = i2t.get_caption(img, "Describe the what is happening on the screen.")
            
        return caption
    
    def __call__(self) -> Any:
        from global_settings import AGENT_WORKING_DIR

        from PIL import ImageGrab
        import io
        import time
        import os
        
        filename = time.strftime("screenshot_%Y-%m-%d-%H-%M-%S.png")
        img = ImageGrab.grab()
        
        buffer = io.BytesIO()
        img = img.convert("RGB")
        img.save(buffer, format="JPEG")
        img.save(os.path.join(AGENT_WORKING_DIR, filename), format="PNG")
        img_binary = buffer.getvalue()
        
        return {
            "file_name": filename,
            "image": img_binary,
            "caption": self._get_caption(img),
        }















# Create instances of tools
direct_answer = DirectAnswer()
web_search = WebSearch()
doc_search = DocSearch()
exec_python = ExecPython()
url_fetcher = URLFetcher(cutoff=5000)
run_zsh = RunZsh(timeout=30, cutoff=1500)
run_zsh_sandbox = RunZshSandbox(timeout=30, cutoff=1500)
run_bash = RunBash(timeout=30, cutoff=1500)
image_qa = ImageQA()
click = Click()
screenshot = Screenshot()

optional_tools = [
    web_search, 
    doc_search,
    exec_python, 
    url_fetcher,
    #run_zsh,
    run_zsh_sandbox,
    run_bash,
    image_qa,
    click,
    screenshot,
]


tool_selector = Checkboxes(
    [tool.displayed_name for tool in optional_tools],
    [tool.enabled_by_default for tool in optional_tools]
)


def tools() -> list[BaseTool]:
    from lib.utils import slice_true
    return [direct_answer] + slice_true(optional_tools, tool_selector.value)


def find_tool(name: str) -> BaseTool:
    """
    Find tool by name. 
    Raise:
        KeyError: raised when the tool cannot be found.
    """
    tool = filter(lambda x: x.name == name, [direct_answer] + optional_tools)
    for _ in tool:
        return _
    raise KeyError(f"No tool named \"{name}\" found.")


def run_tool(name: str, *args, **kwargs) -> Any:
    tool = find_tool(name)
    return tool.run(*args, **kwargs)
