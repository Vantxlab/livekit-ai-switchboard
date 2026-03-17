from __future__ import annotations

import os

import pytest

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _has_key(name: str) -> bool:
    return bool(os.environ.get(name))


@pytest.fixture
def fast_llm():
    """Real fast LLM via OpenAI gpt-4o-mini."""
    from livekit.plugins.openai import LLM

    return LLM(model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])


@pytest.fixture
def smart_llm():
    """Real smart LLM via OpenAI gpt-4o."""
    from livekit.plugins.openai import LLM

    return LLM(model="gpt-4o", api_key=os.environ["OPENAI_API_KEY"])


skip_no_openai = pytest.mark.skipif(
    not _has_key("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
