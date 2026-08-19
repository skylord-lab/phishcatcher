"""
PhishCatcher - cleaned URL feature extraction starter.
"""

import re
from urllib.parse import urlsplit


def extract_features(url: str) -> dict:
    p = urlsplit(url if "://" in url else "http://" + url)
    host = p.netloc
    path = p.path
    query = p.query

    return {
        "url_length": len(url),
        "hostname_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "dot_count": url.count("."),
        "hyphen_count": url.count("-"),
        "slash_count": url.count("/"),
        "question_count": url.count("?"),
        "equal_count": url.count("="),
        "at_count": url.count("@"),
        "ampersand_count": url.count("&"),
        "percent_count": url.count("%"),
        "has_ip_address": int(bool(
            re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host.split(":")[0])
        )),
        "has_https": int(p.scheme.lower() == "https"),
    }


if __name__ == "__main__":
    for url in ["https://www.google.com", "http://example.com/login"]:
        print(url, "=>", extract_features(url))
