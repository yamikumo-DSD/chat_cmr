<h2>LLM inference GUI for Jupyter notebook</h2>
<br>
This is an LLM-powered chat interface integrated with voice synthesis model, web search-based RAG, and python environment.<br>
<b>Everything is running locally</b> in your computer, so it doesn't require any API keys.<br>
For better results, I strongly recommend you to select a model large enough or trained for tool use.<br>
<hr>
<h2>Features</h2>
1. Streaming output.<br>
2. Context shifting manipulating KV-Cache significantly reducing evaluation time (called StreamingLLM).<br>
3. Automatic tool use (aka. function calling) involving "Web Search", "Python Interpreter".<br>
4. Integration to Japanese text-to-speech model called style-bert-vits2.<br>
5. Image recognition utilizing Image-to-Text model.<br>
<hr>
<h2>Installation/Prerequisites</h2>
1. Jupyter must be installed.<br>
<code>$ pip install jupyterlab</code><br>
<br>
2. Activate ipywidgets by one of following commands:<br>
&nbsp;&nbsp;&nbsp;&nbsp;2-1. jupyter-lab<br>
&nbsp;&nbsp;&nbsp;&nbsp;<code>$ jupyter labextension install @jupyter-widgets/jupyterlab-manager</code><br>
&nbsp;&nbsp;&nbsp;&nbsp;2-2. jupyter notebook / Google Colab<br>
&nbsp;&nbsp;&nbsp;&nbsp;<code>$ jupyter nbextension enable --py widgetsnbextension</code><br>
<br>
3. Install dependencies.<br>
<code>$ pip install -r requirements.txt</code><br>
<br>
4. <i>(Optional)</i> You might want to unlock websocket message size limit to send large files or to play longer sounds (the limit is 10MB by default).<br>
<code>$ jupyter notebook --generate-config</code><br>
After running the command, edit following line of the config file:<br>
<code>c.ServerApp.tornado_settings = {"websocket_max_message_size":100*1024*1024}</code><br>
<hr>
<h2>Screen Shots</h2>
<div style="display: flex;">
  <img src="https://github.com/yamikumo-DSD/chat_cmr/blob/main/SS1.png" width="300px">
  <img src="https://github.com/yamikumo-DSD/chat_cmr/blob/main/SS2.png" width="300px">
</div>
