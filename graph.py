from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from state import DigestState
import nodes


def check_fetch(state: DigestState) -> str:
    """
    Controls the fetch → expand_search loop. Exits early on three conditions:
    - articles found → proceed to filter
    - max_tries reached → no_content (hard bound)
    - stagnation: two consecutive fetches returned the same count → no_content (convergence)
    """
    articles = state["articles"]
    tries = state["tries"]
    max_tries = state["max_tries"]
    last_count = state["article_count_last"]

    if len(articles) > 0:
        return "filter"
    if tries >= max_tries:
        return "no_content"
    if tries > 0 and len(articles) == last_count:
        return "no_content"
    return "expand_search"


def check_filter(state: DigestState) -> str:
    """Exit early if the filter node produced no relevant articles."""
    if len(state["top_articles"]) == 0:
        return "no_content"
    return "summarize"


graph_builder = StateGraph(DigestState)
graph_builder.add_node("fetch", nodes.fetch_articles)
graph_builder.add_node("expand_search", nodes.expand_search)
graph_builder.add_node("filter", nodes.fillter_articles)
graph_builder.add_node("summarize", nodes.summraize)
graph_builder.add_node("send_email", nodes.send_email)
graph_builder.add_node("no_content", nodes.no_content)

graph_builder.add_edge(START, "fetch")
graph_builder.add_conditional_edges("fetch", check_fetch, ["filter", "expand_search", "no_content"])
graph_builder.add_edge("expand_search", "fetch")
graph_builder.add_conditional_edges("filter", check_filter, ["summarize", "no_content"])
graph_builder.add_edge("summarize", "send_email")
graph_builder.add_edge("send_email", END)
graph_builder.add_edge("no_content", END)

graph = graph_builder.compile(checkpointer=InMemorySaver())
