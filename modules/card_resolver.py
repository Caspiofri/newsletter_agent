from modules.models import Article, BriefItem, NewsletterData


def resolve_cards(data: NewsletterData, articles: list[Article]) -> NewsletterData:
    """Map summarizer ids to article URLs; drop bad/duplicate ids; demote external snippet cards to brief items."""
    seen: set[int] = set()

    def claim(item_id):
        if item_id is None or not 0 <= item_id < len(articles) or item_id in seen:
            return None
        seen.add(item_id)
        return articles[item_id]

    cards, demoted, briefs = [], [], []
    for card in data.cards:
        art = claim(card.id)
        if art is None:
            continue
        url = art.url or ""
        if art.origin != "inbox" and art.read_status == "snippet":
            demoted.append(BriefItem(id=card.id, summary=card.brief, url=url))
        else:
            cards.append(card.model_copy(update={"url": url}))
    for item in data.brief_items:
        art = claim(item.id)
        if art is not None:
            briefs.append(item.model_copy(update={"url": art.url or ""}))
    return NewsletterData(cards=cards, brief_items=briefs + demoted, actions=list(data.actions))
