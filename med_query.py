from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.agents import load_tools, AgentExecutor, ZeroShotAgent
from langchain.chains.llm import LLMChain

from langchain.schema import HumanMessage, SystemMessage
from langchain import LLMChain
from graphql import GraphQLError


def initialize_llm(
    model_name: str = "gpt-3.5-turbo-16k",
    temperature: float = 0.0,
) -> ChatOpenAI:
    return ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
    )

def initialize_memory() -> ConversationBufferMemory:
    return ConversationBufferMemory(
        memory_key="chat_history",
        input_key="input",
        output_key="output",
        return_messages=True,
    )

def initialize_opentargets_access_tool(
    llm: ChatOpenAI,
    graphql_endpoint: str = "https://api.platform.opentargets.org/api/v4/graphql",
) -> List[BaseTool]:
    return load_tools(
        ["graphql"],
        llm=llm,
        graphql_endpoint=graphql_endpoint,
    )

def initialize_zero_shot_agent_prompt(
    tools: List[BaseTool],
    prefix: str,
    suffix: str,
    input_variables: List[str],
) -> PromptTemplate:
    return ZeroShotAgent.create_prompt(
        tools=tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=input_variables
    )

def initialize_llm_chain(
    llm: ChatOpenAI,
    prompt: PromptTemplate,
) -> LLMChain:
    return LLMChain(
        llm=llm,
        prompt=prompt,
    )

def initialize_agent(
    llm_chain: LLMChain,
    tools: List[BaseTool],
) -> ZeroShotAgent:
    return ZeroShotAgent(
        llm_chain=llm_chain,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )

def initialize_agent_executor(
    agent: ZeroShotAgent,
    memory: ConversationBufferMemory,
    tools: List[BaseTool]
) -> AgentExecutor:
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        memory=memory,
        tools=tools,
        return_intermediate_steps=True,
        verbose=True,
    )