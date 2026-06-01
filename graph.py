from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from state import DigestState
import nodes
from langgraph.checkpoint.memory import InMemorySaver

def check_results(state: DigestState) -> str:
    # returns a node name
    if len(state["articles"]) == 0 and state["tries"] < state["max_tries"]:
        return "expand_search"
    return "filter"

graph_builder = StateGraph(DigestState)
graph_builder.add_node("fetch", nodes.fetch_articles)
graph_builder.add_node("expand_search", nodes.expand_search)
graph_builder.add_edge("expand_search", "fetch")
graph_builder.add_node("filter", nodes.fillter_articles)
graph_builder.add_node("summarize", nodes.summraize)
graph_builder.add_node("send_email", nodes.send_email)
graph_builder.add_edge(START, "fetch")
graph_builder.add_conditional_edges("fetch", check_results) 
graph_builder.add_edge("filter", "summarize")
graph_builder.add_edge("summarize", "send_email")
graph_builder.add_edge("send_email", END)
graph = graph_builder.compile(checkpointer=InMemorySaver())
