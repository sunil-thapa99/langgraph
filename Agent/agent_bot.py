from typing import TypedDict, List
from langchain_core.messages import HumanMessage

from langgraph.graph import StateGraph, START, END

from openai import OpenAI

class AgentState(TypedDict):
    messages: List[HumanMessage]

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")


def process(state: AgentState) -> AgentState:
    response = client.chat.completions.create(
        model="model-identifier",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            *[{"role": "user", "content": m.content} for m in state["messages"]]
        ],
    )
    # response = llm.invoke(state["messages"])
    # print(f"\nAI: {response.content}")
    print(f"\nAI: {response.choices[0].message.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

# user_input  = input("User: ")
# agent.invoke({"messages": [HumanMessage(content=user_input)]})

user_input = input("User: ")
while user_input != "exit":
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("User: ")
