<h2>LLM inference GUI for Jupyter notebook</h2>
<br>
This is an LLM-powered chat interface integrated with a lot of useful tools.<br>
For better results, I strongly recommend you to select a model large enough or trained for tool use.<br>
<br>
<h3>Features</h3>
1. Multi Modal. <br>
2. Streaming output. <br>
3. StreamingLLM (aka Context shifting).<br>
4. Multi-step tool use.<br>
5. Integration to Japanese text-to-speech model called style-bert-vits2.<br>
6. Easy to integrate any User-defined tools.<br>
7. Configurable UIs.<br>
<br>
<h3>Built-in Tools</h3>
1. Web Search <br>
2. Local Document Search <br>
3. Python Executor <br>
4. URL Fetcher <br>
5. Shell Executor (Sandbox is available in macOS) <br>
6. Image Recognition <br>
7. Mouse Click <br>
8. Screenshot <br>
<br>
<h2>Prerequisite</h2>
1. Install the latest llama-cpp-python.<br>
<code>pip install llama-cpp-python</code><br>
2. Jupyter must be installed.<br>
<code>$ pip install jupyterlab</code><br>
3. Activate ipywidgets by one of following commands:<br>
<pre><code># jupyter-lab
$ jupyter labextension install @jupyter-widgets/jupyterlab-manager
# jupyter notebook / Google Colab<br>
$ jupyter nbextension enable --py widgetsnbextension</code></pre>
4. Install dependencies.<br>
<code>$ pip install -r requirements.txt</code><br>
<h2>Screen Shots</h2>
<h3>Python execution</h3>
<img src=https://github.com/yamikumo-DSD/chat_cmr/blob/main/SS1.png>
<h3>Web search</h3>
<img src=https://github.com/yamikumo-DSD/chat_cmr/blob/main/SS2.png>
<h3>Understading what you are seeing</h3>
<img src=https://github.com/yamikumo-DSD/chat_cmr/blob/main/SS3.png>
