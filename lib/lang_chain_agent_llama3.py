from typing import List
from langchain.agents import Tool
from langchain.prompts import BaseChatPromptTemplate
from lib.utils import now
from pydantic import Field

DEFAULT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

与えられたToolを使ってユーザーのQuestionに対するFinal Answerを作成してください。<|eot_id|><|start_header_id|>system<|end_header_id|>

## Basic Format
Thought: Actionに移る前に常に何をすべきかを考えるようにしてください。
Action: [{tool_names}]から1つだけActionとして選んでください。Actionの名前以外はここには絶対に含めてはいけません。
Action Input: Actionに対する入力。選択したActionに合う形にしてください。
Observation: Actionの結果得られた情報。
...(Thought/Action/Action Input/Observationを答えが出るまで繰り返す。)
Thought: あなたが最終的に答えるべき回答を最後に考えます。
Final Answer: Questionに対する最終的な答えです。かならずこのステップで回答を終了させてください。<|eot_id|><|start_header_id|>system<|end_header_id|>

## Available Tools
{tools}<|eot_id|><|start_header_id|>user<|end_header_id|>Question: {input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Thought: {agent_scratchpad}"""

DEFAULT_TEMPLATE = DEFAULT_TEMPLATE.replace('{now}', now())

class CustomPromptTemplate(BaseChatPromptTemplate):
    template: str
    tools: List[Tool]
    max_iterations: int|None = Field(description=('max iteration length'), default=None)
  
    def format_messages(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            # We don't want error messages piling up.
            if action.tool == '_Exception': continue
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
            
        current_iterations = len(intermediate_steps)
        if current_iterations == self.max_iterations // 2:
            input = kwargs['input']
            thoughts += f'ここまでの情報で、"{input}"に対するFinal Answerが生成できるか考えてみよう。'
        if (self.max_iterations != None) and (current_iterations >= self.max_iterations - 2):
            thoughts += '以上の情報で、Final Answerを生成しよう。'
            
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ",".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
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
            raise OutputParserException(f'Could not parse LLM output. ({llm_output})')
        else:
            action = match.group(1).strip().lower()
            for tool_name in self.allowed_tools:
               if tool_name in action:
                   action = tool_name
            action_input = match.group(2)
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'),log=llm_output)

handle_parsing_errors = 'あなたの出力は"Action: "あるいは"Final Answer: "で始まらなければなりません。'

def create_agent_executor(
    model, 
    tools, 
    verbose=True, 
    handle_parsing_errors=handle_parsing_errors,
    model_kwargs: dict|None = None,
    max_iterations: int = 10,
    **kwargs
):
    from lib.lang_chain_agent import CustomOutputParser, CustomPromptTemplate, DEFAULT_TEMPLATE
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
        stop=["\nObservation:", "--", "***", '\n\n\n'],
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
