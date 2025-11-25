import llama_cpp
from packaging.version import Version

if Version(llama_cpp.__version__) > Version("0.3.16"):
    from lib.llama_cpp_wrapper_v2 import *
else:
    from lib.llama_cpp_wrapper_v1 import *