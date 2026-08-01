from src.engine.gif_retriever import gif_retriever


def test_giphy_reaction_search():
    clips = gif_retriever.search_giphy_reaction("shocked reaction", limit=2)
    assert isinstance(clips, list)
    if clips:
        assert "title" in clips[0]
        assert "mp4_url" in clips[0]
        assert clips[0]["source"] == "GIPHY"
