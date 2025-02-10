import streamlit as st
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
import os
from PIL import Image
from io import BytesIO

# Load environment variables
from dotenv import load_dotenv
_ = load_dotenv()

# Initialize search tool
tool = TavilySearchResults(max_results=4)

# Define AgentState
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

# Define AI Agent
class Agent:
    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            st.info(f"Calling tool: {t['name']} with arguments: {t['args']}")
            if not t['name'] in self.tools:
                result = "Invalid tool call, retrying..."
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        st.info("Returning to LLM with tool results...")
        return {'messages': results}

# Define system prompt
system_prompt = """You are an intelligent assistant that retrieves relevant information. \
Analyze the question carefully and use external knowledge when needed. \
You may issue multiple information requests when necessary. \
Ensure responses are well-structured and accurate.
"""

# Initialize agent
model = ChatOpenAI(model="gpt-4o")
abot = Agent(model, [tool], system=system_prompt)

# Streamlit UI
st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.title("AI Research Assistant")

# Sidebar layout with instructions and execution graph
with st.sidebar:
    st.header("Instructions")
    st.write(
        "Enter a question below. The AI will analyze your request and determine the best approach "
        "to process and retrieve relevant information efficiently."
    )

    # Display execution flow in the sidebar
    st.subheader("Execution Flow")
    graph_image = abot.graph.get_graph().draw_png()
    image = Image.open(BytesIO(graph_image))
    st.image(image, caption="Graph Structure", use_container_width=True)

# User input
user_query = st.text_area("Enter your question:", "")

# Button to submit query
if st.button("Get Answer"):
    if user_query.strip():
        with st.spinner("Processing..."):
            messages = [HumanMessage(content=user_query)]
            result = abot.graph.invoke({"messages": messages})
            st.success("Response received!")

            # Display response
            st.subheader("AI Response:")
            st.write(result['messages'][-1].content)
            st.success("Final answer displayed.")
    else:
        st.warning("Please enter a question before submitting.")
