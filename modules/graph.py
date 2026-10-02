from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from modules.state import DigestState
from modules import nodes


def check_fetch(state: DigestState) -> str:
    """
    Controls the inbox branch's fetch → expand_search loop.
    - articles found             → extract
    - days_back >= max_days_back → extract (window exhausted; nothing from the inbox,
                                   but the other collectors may still have content)
    - tries >= max_tries         → extract (too many transient failures; same reasoning)
    - otherwise                  → expand_search (widen the window by one day)
    The inbox branch always ends in extract so that merge can join all collectors.
    """
    if len(state["articles"]) > 0:
        return "extract"
    if state["days_back"] >= state["max_days_back"]:
        return "extract"
    if state["tries"] >= state["max_tries"]:
        return "extract"
    return "expand_search"


def check_merge(state: DigestState) -> str:
    """Exit early if no collector found anything."""
    if len(state["articles"]) == 0:
        return "no_content"
    return "dedupe"


def check_dedupe(state: DigestState) -> str:
    """Exit early if everything collected was already sent."""
    if len(state["articles"]) == 0:
        return "no_content"
    return "shortlist"


def check_rank(state: DigestState) -> str:
    """Exit early if the ranker found nothing worth sending."""
    if len(state["top_articles"]) == 0:
        return "no_content"
    return "summarize"


graph_builder = StateGraph(DigestState)
graph_builder.add_node("fetch", nodes.fetch_articles)
graph_builder.add_node("extract", nodes.extract_sub_articles)
graph_builder.add_node("collect_medium", nodes.collect_medium)
graph_builder.add_node("collect_hackernews", nodes.collect_hackernews)
graph_builder.add_node("merge", nodes.merge_sources)
graph_builder.add_node("expand_search", nodes.expand_search)
graph_builder.add_node("shortlist", nodes.shortlist_articles)
graph_builder.add_node("reader", nodes.read_articles)
graph_builder.add_node("rank", nodes.rank_articles)
graph_builder.add_node("dedupe", nodes.dedupe_articles)
graph_builder.add_node("summarize", nodes.summraize)
graph_builder.add_node("send_email", nodes.send_email)
graph_builder.add_node("no_content", nodes.no_content)

# Collectors run in parallel from START and join in `merge`.
graph_builder.add_edge(START, "fetch")
graph_builder.add_edge(START, "collect_medium")
graph_builder.add_edge(START, "collect_hackernews")
graph_builder.add_conditional_edges("fetch", check_fetch, ["extract", "expand_search"])
graph_builder.add_edge(["extract", "collect_medium", "collect_hackernews"], "merge")
graph_builder.add_conditional_edges("merge", check_merge, ["dedupe", "no_content"])
graph_builder.add_edge("expand_search", "fetch")
graph_builder.add_conditional_edges("dedupe", check_dedupe, ["shortlist", "no_content"])
graph_builder.add_edge("shortlist", "reader")
graph_builder.add_edge("reader", "rank")
graph_builder.add_conditional_edges("rank", check_rank, ["summarize", "no_content"])
graph_builder.add_edge("summarize", "send_email")
graph_builder.add_edge("send_email", END)
graph_builder.add_edge("no_content", END)

graph = graph_builder.compile(checkpointer=InMemorySaver())
