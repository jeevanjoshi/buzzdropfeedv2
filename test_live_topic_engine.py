import datetime
from src.agents.fact_retriever import FactRetrieverAgent
from src.schemas.state import GlobalState
from src.engine.rss_ingestion import fetch_live_rss_feeds


def run_live_test(region: str = "all"):
    print("=" * 80)
    print(f"🚀 RUNNING LIVE REAL-TIME TOPIC SELECTION ENGINE TEST (Region: {region.upper()})")
    print("=" * 80)

    agent = FactRetrieverAgent()
    state = GlobalState(
        pipeline_id=f"csvg-live-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    print("\nFetching Live Real-Time RSS Feeds (Global & India)...")
    live_feeds = fetch_live_rss_feeds(region=region)
    print(f"Ingested {len(live_feeds)} live articles from real-time news feeds.")

    if not live_feeds:
        print("Warning: No live feeds returned (network offline or feeds blocked). Using fallback feed items.")

    print("\nExecuting TOPSIS Multi-Criteria Decision Analysis (7 Criteria)...")
    msg = agent.process(state, use_live_rss=True, region=region)

    selected = state.selected_topic

    if selected:
        print("\n" + "=" * 80)
        print("WINNING MATHEMATICALLY SELECTED TOPIC:")
        print("=" * 80)
        print(f"Headline      : {selected.headline}")
        print(f"Summary       : {selected.summary[:150]}...")
        print(f"Source URL    : {selected.source_url}")
        print(f"Keywords      : {', '.join(selected.keywords)}")
        print("-" * 80)
        print(f"TOPSIS Score  : {selected.topsis_score:.4f} / 1.0000")
        print(f"Trend Velocity: {selected.tvs_score:.2f} (EMA Acceleration)")
        print(f"Advertiser RPM: {selected.rpm_score:.2f} (CPM Cosine Similarity)")
        print(f"Novelty Index : {selected.idi_score:.2f} (Information Density)")
        print(f"Conflict Index: {selected.sdi_score:.2f} (Sentiment Variance)")
        print(f"Social Hype   : {selected.shm_score:.2f} (X/Reddit Multiplier)")
        print(f"YouTube VPH   : {selected.vph_score:.2f} (Competitor Views/Hr)")
        print(f"Saturation    : {selected.sat_score:.2f} (Competition Penalty)")
        print("=" * 80)
    else:
        print("Topic selection failed.")


if __name__ == "__main__":
    run_live_test(region="all")
