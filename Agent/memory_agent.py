import os
from typing import TypedDict, List, Union

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from openai import OpenAI

# LM Studio OpenAI-compatible client
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Define state type
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]

MAX_MESSAGES = 5

# Processing node
def process(state: AgentState) -> AgentState:
    """This node will solve the request input"""
    
    trimmed_messages = state["messages"][-MAX_MESSAGES:]

    # Convert LangChain messages to OpenAI format with roles
    formatted_messages = [
        {"role": "user", "content": m.content} if isinstance(m, HumanMessage)
        else {"role": "assistant", "content": m.content} if isinstance(m, AIMessage)
        else {"role": "system", "content": m.content}
        for m in trimmed_messages
    ]

    # Send to LM Studio
    print(f"\nFormated messages: {formatted_messages}")
    response = client.chat.completions.create(
        model="qwen2.5-coder-14b-instruct",  # <-- Replace with your LM Studio model name
        messages=formatted_messages,
        temperature=0.7,
    )

    # Extract and append assistant response
    ai_response = response.choices[0].message.content
    state["messages"].append(AIMessage(content=ai_response))

    print(f"\nAI: {ai_response}")
    return state

# LangGraph setup
graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

# Main chat loop with memory
conversation_history: List[Union[HumanMessage, AIMessage, SystemMessage]] = [
    SystemMessage(content="You are a helpful assistant.")  # Or use SystemMessage
]


user_input = input("User: ")
while user_input.strip().lower() != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]
    user_input = input("User: ")



with open("chat_log.txt", "w") as f:
    f.write("Your Conversation Log:\n")
    for msg in conversation_history:
        role = "You" if isinstance(msg, HumanMessage) else "AI" if isinstance(msg, AIMessage) else "System"
        f.write(f"{role}: {msg.content}\n")
    
    f.write("End of Conversation")

print("Conversation saved to logging.txt")