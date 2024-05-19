from typing import List
from langchain.agents import Tool
from langchain.prompts import BaseChatPromptTemplate
from pydantic import Field

DEFAULT_TEMPLATE = """<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|># System Preamble
## Basic Rules
与えられたToolを使ってQuestionに対するFinal Answerを作成してください。

## Basic Format
Thought: Actionに移る前に常に何をすべきかを考えるようにしてください。
Action: [{tool_names}]からActionを1つ選択し、その名前だけ記入します(その他の情報は書いてはいけない)。
Action Input: Actionに対する入力。選択したActionに合う形にしてください。
Observation: Actionの結果得られた情報。
...(Thought/Action/Action Input/Observationを答えが出るまで繰り返す。)
Thought: あなたが最終的に答えるべき回答を最後に考えます。
Final Answer: Questionに対する最終的な答えです。かならずこのステップで回答を終了させてください。

# User Preamble
## Available Tools
{tools}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Question: {input}<|END_OF_TURN_TOKEN|>{agent_scratchpad}"""

class CustomPromptTemplate(BaseChatPromptTemplate):
    template: str
    tools: List[Tool]
    max_iterations: int|None = Field(description=('max iteration length'), default=None)
  
    def format_messages(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += f"<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{action.log}<|END_OF_TURN_TOKEN|>"
            thoughts += f"<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>Observation: {observation}<|END_OF_TURN_TOKEN|>"
            
        current_iterations = len(intermediate_steps)
        if (self.max_iterations != None) and (current_iterations >= self.max_iterations - 2):
            thoughts += "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>繰り返し回数が上限に近づいています。そろそろFinal Answerを生成してください。<|END_OF_TURN_TOKEN|>"
            
        thoughts += "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>Thought: "
            
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ",".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        #print(formatted)
        return [HumanMessage(content=formatted)]


from langchain.agents import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from typing import Union
import re
from langchain.schema.output_parser import OutputParserException

class CustomOutputParser(AgentOutputParser):
    allowed_tools: list[str] = Field(description=('list of names of tools'))
    def parse(self, llm_output:str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(
              return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
              log=llm_output,
            )
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise OutputParserException(llm_output)
        else:
            action = match.group(1).strip().lower()
            action_input = match.group(2)
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'),log=llm_output)

handle_parsing_errors = '形式に問題があります。少なくとも"Action: "を指定して継続するか、"Final Answer: "を生成して終了する必要があります。'

def create_agent_executor(
    model, 
    tools, 
    verbose=True, 
    handle_parsing_errors=handle_parsing_errors,
    model_kwargs: dict|None = None,
    max_iterations: int = 10,
    **kwargs
):
    from langchain import LLMChain
    from langchain.agents.agent import LLMSingleActionAgent
    from langchain.agents import AgentExecutor, create_react_agent

    prompt = CustomPromptTemplate(
        template = DEFAULT_TEMPLATE, 
        tools=tools, 
        input_variables=["input","intermediate_steps"]
    )
    prompt.max_iterations = max_iterations
    
    llm_chain = LLMChain(
        llm = model.bind(**model_kwargs) if model_kwargs else model, 
        prompt=prompt,
    )
    
    parser = CustomOutputParser()
    parser.allowed_tools = [tool.name for tool in tools]
    
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=parser,
        stop=["\nObservation:"], # This is important cuz LLMs try to produce Observation on their own.
        allowed_tools=[tool.name for tool in tools]
    )
    
    return AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=handle_parsing_errors,
        verbose=verbose,
        max_iterations=max_iterations,
        **kwargs
    )
