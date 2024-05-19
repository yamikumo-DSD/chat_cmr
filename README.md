<h2>LLM inference GUI for Jupyter notebook</h2>
<br>
This is LLM inference GUI for Jupyter notebook, with default prompt for Cohere Command R/Command R+.<br>
<br>
<h3>Features</h3>
1. Streaming output.<br>
2. Context shifting manipulating KV-Cache significantly reducing evaluation time.<br>
3. Web searching agent equipped by langchain.agent (needs google search API key).<br>
4. Automatic execution of Python code created by LLM.<br>
5. Integration to Japanese text-to-speech model called style-bert-vits2.<br>
6. Easy to integrate any tools.<br>
7. Configurable UIs.<br>
<br>
<h3>Prerequisite</h3>
1. Jupyter must be installed.<br>
2. Activate ipywidgets by one of following commands:<br>
# jupyter-lab<br>
<code>jupyter labextension install @jupyter-widgets/jupyterlab-manager</code><br>
# jupyter notebook<br>
<code>jupyter nbextension enable --py widgetsnbextension</code><br>
3. Download GGUF from HuggingFace (https://huggingface.co/models?search=command-r)
