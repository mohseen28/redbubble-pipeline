from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from pipeline.llm.base import LLMError, load_prompt
from pipeline.llm.ollama_client import OllamaClient
from pipeline.schemas import BriefBatch, Listing


def _resp(content: str | None = None, status: int = 200, payload: dict | None = None) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload if payload is not None else {"message": {"content": content}}
    r.text = "err"
    return r


def _client(posts, models=("qwen2.5:14b",)) -> tuple[OllamaClient, MagicMock]:
    session = MagicMock()
    session.post.side_effect = posts
    tags = MagicMock()
    tags.json.return_value = {"models": [{"name": m} for m in models]}
    session.get.return_value = tags
    return OllamaClient(session=session, max_retries=2, fallback_model="qwen2.5:7b"), session


GOOD = json.dumps({"title": "Cozy cottage", "tags": ["cozy"], "description": "A cottage."})


def test_schema_pass_through_and_payload():
    client, session = _client([_resp(GOOD)])
    out = client.complete_json("sys", "user", Listing)
    assert out.title == "Cozy cottage"
    url, = session.post.call_args.args
    payload = session.post.call_args.kwargs["json"]
    assert url.endswith("/api/chat")
    assert payload["stream"] is False
    assert payload["format"] == Listing.model_json_schema()
    assert payload["options"] == {"temperature": 0.8, "num_ctx": 4096}
    assert payload["model"] == "qwen2.5:14b" and payload["keep_alive"] == "10m"
    assert payload["messages"][0] == {"role": "system", "content": "sys"}


def test_retry_on_invalid_json_then_success():
    client, session = _client([_resp("not json"), _resp('{"title": 1}'), _resp(GOOD)])
    out = client.complete_json("s", "u", Listing)
    assert out.tags == ["cozy"] and session.post.call_count == 3
    last_msgs = session.post.call_args.kwargs["json"]["messages"]
    assert last_msgs[-1]["role"] == "user" and "did not match" in last_msgs[-1]["content"]


def test_gives_up_after_retries():
    client, session = _client([_resp("x")] * 3)
    with pytest.raises(LLMError, match="3 attempts"):
        client.complete_json("s", "u", Listing)
    assert session.post.call_count == 3


def test_code_fences_stripped_and_images_encoded():
    client, session = _client([_resp("```json\n" + GOOD + "\n```")])
    client.complete_json("s", "u", Listing, images=[b"abc"])
    assert session.post.call_args.kwargs["json"]["messages"][1]["images"] == ["YWJj"]


def test_fallback_model_and_missing_model():
    client, session = _client([_resp(GOOD)], models=("qwen2.5:7b",))
    client.complete_json("s", "u", Listing)
    assert session.post.call_args.kwargs["json"]["model"] == "qwen2.5:7b"
    client2, _ = _client([], models=())
    with pytest.raises(LLMError, match="ollama pull"):
        client2.complete_json("s", "u", Listing)


def test_http_error_and_unload():
    client, _ = _client([_resp(status=500, payload={})])
    with pytest.raises(LLMError, match="HTTP 500"):
        client.complete_json("s", "u", Listing)
    client2, s2 = _client([_resp(payload={})])
    client2.unload()
    assert s2.post.call_args.kwargs["json"] == {"model": "qwen2.5:14b", "keep_alive": 0}


def test_prompt_files_load():
    for name in ("concept_system", "niche_generate", "trademark_judge", "qa_vision_rubric", "metadata_system"):
        assert load_prompt(name, text_rule="x", text_field_rule="y", count=3, today="2026-11-07",
                           title_max=100, tags_max=15, tag_max_len=30, desc_max=1000)
    assert "plain solid" in load_prompt("concept_system", text_rule="x", text_field_rule="y").lower()
    assert BriefBatch.model_json_schema()["properties"]["briefs"]
