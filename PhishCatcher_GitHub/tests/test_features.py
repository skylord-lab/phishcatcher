from src.feature_extraction import extract_features

def test_feature_extraction():
    f = extract_features("https://example.com/login")
    assert f["has_https"] == 1
    assert f["dot_count"] >= 1
