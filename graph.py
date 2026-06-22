from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from state import DigestState
import nodes


def check_fetch(state: DigestState) -> str:
    """
    Controls the fetch → expand_search loop.
    - articles found          → extract
    - days_back >= max_days_back → no_content (time window exhausted)
    - tries >= max_tries      → no_content (too many transient failures)
    - otherwise               → expand_search (widen the window by one day)
    """
    if len(state["articles"]) > 0:
        return "extract"
    if state["days_back"] >= state["max_days_back"]:
        return "no_content"
    if state["tries"] >= state["max_tries"]:
        return "no_content"
    return "expand_search"


def check_filter(state: DigestState) -> str:
    """Exit early if the filter node produced no relevant articles."""
    if len(state["top_articles"]) == 0:
        return "no_content"
    return "dedupe"


def check_dedupe(state: DigestState) -> str:
    """Exit early if dedupe removed all candidates."""
    if len(state["top_articles"]) == 0:
        return "no_content"
    return "summarize"


graph_builder = StateGraph(DigestState)
graph_builder.add_node("fetch", nodes.fetch_articles)
graph_builder.add_node("extract", nodes.extract_sub_articles)
graph_builder.add_node("expand_search", nodes.expand_search)
graph_builder.add_node("filter", nodes.fillter_articles)
graph_builder.add_node("dedupe", nodes.dedupe_articles)
graph_builder.add_node("summarize", nodes.summraize)
graph_builder.add_node("send_email", nodes.send_email)
graph_builder.add_node("no_content", nodes.no_content)

graph_builder.add_edge(START, "fetch")
graph_builder.add_conditional_edges("fetch", check_fetch, ["extract", "expand_search", "no_content"])
graph_builder.add_edge("extract", "filter")
graph_builder.add_edge("expand_search", "fetch")
graph_builder.add_conditional_edges("filter", check_filter, ["dedupe", "no_content"])
graph_builder.add_conditional_edges("dedupe", check_dedupe, ["summarize", "no_content"])
graph_builder.add_edge("summarize", "send_email")
graph_builder.add_edge("send_email", END)
graph_builder.add_edge("no_content", END)

graph = graph_builder.compile(checkpointer=InMemorySaver())
