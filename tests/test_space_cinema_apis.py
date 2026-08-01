from src.engine.space_cinema_apis import space_cinema_api_manager


def test_nasa_image_library_search():
    assets = space_cinema_api_manager.fetch_nasa_space_imagery("mars rover", limit=2)
    assert isinstance(assets, list)
    if assets:
        assert "title" in assets[0]
        assert "image_url" in assets[0]


def test_tmdb_movie_fallback_data():
    movie = space_cinema_api_manager.fetch_tmdb_movie_box_office("Avatar")
    assert "title" in movie
    assert "revenue" in movie


def test_wikipedia_on_this_day_history():
    history_facts = space_cinema_api_manager.fetch_on_this_day_history()
    assert isinstance(history_facts, list)
    assert len(history_facts) >= 1
    assert "On This Day" in history_facts[0].headline
