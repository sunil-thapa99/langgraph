from typing import TypedDict, List, Union, Sequence, Annotated

from langchain_core.messages import BaseMessage # The foundational class for all message types in LangGraph
from langchain_core.messages import ToolMessage # Passes data back to LLM after it calls a tool such as the content and the tool_call_id
from langchain_core.messages import SystemMessage # Message for providing instructions to the LLM
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from openai import OpenAI

# LM Studio OpenAI-compatible client
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

'''
Annotated - provides additional context without affecting the type itself
Example:
email = Annotated[str, "This has to be a valid email format."]
print(email.__metadata__) -> "This has to be a valid email format."

Sequence - To automatically handle the state updates for sequences such as by adding new messages to a chat history
'''

class AgentState(TypedDict):
    messages: List[BaseMessage]
    tools: List[ToolNode]
