from lib.utils import change_directory

class MyPythonREPL:
    __code: str|None = None
    __output: str|None = None
    __output_len_cutoff: int = None
    __image_buffer: bytes|None = None  # Store the image as bytes
    __local_env: dict
    __replace_nl: bool
    __temporal_working_directory: str
    __predefined: dict

    def __init__(self, output_len_cutoff: int = 1024, replace_nl: bool = False, predefined: dict = {}, temporal_working_directory: str = '.'):
        self.__output_len_cutoff = output_len_cutoff
        self.__predefined = predefined
        self.__local_env = predefined
        self.__replace_nl = replace_nl
        self.__temporal_working_directory = temporal_working_directory

    def run(self, code):
        import io
        import contextlib
        import matplotlib.pyplot as plt
        import re
        import textwrap
        from matplotlib.backend_bases import FigureCanvasBase
        
        @change_directory(self.__temporal_working_directory)
        def _exec(*args, **kwargs):
            return exec(*args, **kwargs)

        # Fix common inappropriate accessories.
        self.__code = re.sub(r"(?s)```(python|py)?(.+)```", r"\2", code)
        if self.__replace_nl:
            self.__code = self.__code.replace('\\n', '\n')
        
        with io.StringIO() as buf, contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf), io.BytesIO() as img_buf:
            # Override plt.show to save the figure to the buffer
            def show_override(*args, **kwargs):
                for fig_num in plt.get_fignums():
                    fig = plt.figure(fig_num)
                    # Ensure the figure renders in the background before saving
                    FigureCanvasBase(fig).draw()
                    fig.savefig(img_buf, format='png')
                    img_buf.seek(0)
                print("runtime.message: plt.show() wrapper called. The images are transferred.")

            # Setup a controlled execution environment
            self.__local_env['plt'] = plt
            self.__local_env['np'] = __import__('numpy')
            self.__local_env['plt'].show = show_override
            self.__local_env['__name__'] = '__main__'

            try:
                _exec(
                """import traceback
try:
{code}
except:
    print(traceback.format_exc(limit=2))
""".format(code=textwrap.indent(self.__code, '    ')), 
                self.__local_env, self.__local_env
                )
            except SyntaxError as e:
                print(e)
            
            self.__output = buf.getvalue()
            
            # Check if the image was saved
            if img_buf.getbuffer().nbytes > 0:
                self.__image_buffer = img_buf.getvalue()

        if len(self.__output) > self.__output_len_cutoff:
            self.__output = self.__output[:self.__output_len_cutoff // 2] + '\n...(output is too long and omitted)...\n' + self.__output[-self.__output_len_cutoff // 2:]

        return self.__output

    def result(self) -> tuple[str, str, bytes|None]:
        """Return tuple of executed code, output, and image buffer (if any)."""
        return self.__code, self.__output, self.__image_buffer

    def unset(self, keep_locals: bool = False):
        self.__code = None
        self.__output = None
        self.__image_buffer = None
        if not keep_locals:
            self.__local_env = self.predefined



