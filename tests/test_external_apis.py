from src.engine.external_apis import external_api_manager


def test_world_bank_gdp_inflation_api():
    wb_data = external_api_manager.fetch_world_bank_gdp_inflation(country_code="IND")
    assert "country" in wb_data
    assert "gdp_growth" in wb_data
    assert "inflation" in wb_data
    assert wb_data["country"] == "IND"


def test_alpha_vantage_fallback_quote():
    quote = external_api_manager.fetch_alpha_vantage_stock_quote("NVDA")
    assert "symbol" in quote
    assert quote["symbol"] == "NVDA"
    assert "price" in quote
