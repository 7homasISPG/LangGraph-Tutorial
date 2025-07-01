# agentic_rag.py
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from langchain_google_genai import ChatGoogleGenerativeAI
from cdb import query_collection
import os
from dotenv import load_dotenv

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Google Generative AI
chat = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=google_api_key
)

# Define the state
class GraphState(dict): pass

# Node: Retrieve documents
def retrieve(state: GraphState):
    query = state['query']
    results = query_collection(query_texts=[query], n_results=3)
    context = "\n".join([doc for result in results['documents'] for doc in result])
    state['context'] = context
    return state

# Node: Generate response using Gemini
def generate(state: GraphState):
    context = state.get("context", "No relevant context found.")
    query = state['query']
    
    prompt = f"""Using the contexts below, answer the query.
Contexts:
{context}

Query: {query}
"""
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=prompt)
    ]
    response = chat.invoke(messages)
    state['response'] = response.content
    return state

# Build the LangGraph
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile
agentic_rag = workflow.compile()

