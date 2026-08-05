"""
Hermetic unit tests for the promotional/listicle/affiliate content filter
in the RSS ingestion pipeline. Ensures SEO spam and advertorials never pollute
the candidate pool or the RAG corpus, while genuine news passes through.
"""
from src.engine.rss_ingestion import _is_promotional_listicle


def _check(headline, summary="", expect_block=True):
    result = _is_promotional_listicle(headline, summary)
    assert result is expect_block, f"headline={headline!r} expected block={expect_block}, got {result}"


def test_blocks_top_n_listicle():
    _check("Top 10 AI Tools That Will Transform Your Content Creation in 2026", "Looking to level up your content creation game?")
    _check("Top 5 Ways to Grow Your Channel Fast", "")


def test_blocks_how_i_make_style():
    _check("How I Get Free Traffic from ChatGPT in 2026 (AIO vs SEO)", "I tested something that changed organic traffic.")


def test_blocks_review_affiliate():
    _check("LimeWire AI Studio Review 2026: Details, Pricing & Features", "LimeWire emerges as a unique generative AI platform.")
    _check("Best New Tool Review: Features, Pricing and Alternatives", "")


def test_blocks_advertorial_growth_hype():
    _check("Ready for Growth? Take These Strategic Next Steps for the Fastest, Lowest-Risk ROI", "")
    _check("Transform Your Business With These 7 Secrets Today", "")


def test_blocks_direct_cta():
    _check("Sign Up Now and Get Started Free", "")
    _check("Buy Now: Limited Time Offer Ends Today", "")


def test_allows_real_news():
    _check("In Lawsuit, NJ Accuses Amazon of Suppressing Pay for Delivery Drivers",
           "The state attorney general filed an antitrust lawsuit on Tuesday.", expect_block=False)
    _check("Senators demand crackdown on wildfire prediction market bets", "", expect_block=False)
    _check("SpaceX, in First Earnings After IPO, Reports Soaring AI Spending", "", expect_block=False)
    _check("The Invisible Tax Every Black Woman Leader Is Paying — and How to Stop", "", expect_block=False)
