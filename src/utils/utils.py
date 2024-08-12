import json
from urllib.parse import urlparse
from typing import Dict


def now() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d%H%M")[:-1]


def is_url(url_or_filename) -> bool:
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def load_json(filename) -> Dict:
    with open(filename, "r") as f:
        return json.load(f)
