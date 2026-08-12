"""Tests for parse_transcript_tokens — Claude Code and Pi JSONL transcript formats."""

import json
from pathlib import Path

from alignsim.src.persistence.run_logger import _pick_transcript, parse_transcript_tokens


def _write_jsonl(path: Path, events: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return path


def test_parse_transcript_tokens_claude_format(tmp_path: Path) -> None:
    """Claude Code stream-json with multiple assistant messages across two models."""
    transcript = _write_jsonl(tmp_path / "transcript.json", [
        {"type": "assistant", "message": {
            "model": "claude-sonnet-4-6-20250514",
            "usage": {
                "input_tokens": 1000,
                "output_tokens": 200,
                "cache_creation_input_tokens": 500,
                "cache_read_input_tokens": 300,
            },
        }},
        {"type": "assistant", "message": {
            "model": "claude-sonnet-4-6-20250514",
            "usage": {
                "input_tokens": 1500,
                "output_tokens": 400,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 800,
            },
        }},
        {"type": "assistant", "message": {
            "model": "claude-opus-4-20250514",
            "usage": {
                "input_tokens": 2000,
                "output_tokens": 600,
                "cache_creation_input_tokens": 100,
                "cache_read_input_tokens": 0,
            },
        }},
        {"type": "tool_use", "content": "some tool output"},
    ])

    result = parse_transcript_tokens(transcript)

    assert result == {
        "claude-sonnet-4-6-20250514": {
            "input_tokens": 2500,
            "output_tokens": 600,
            "cache_creation_input_tokens": 500,
            "cache_read_input_tokens": 1100,
        },
        "claude-opus-4-20250514": {
            "input_tokens": 2000,
            "output_tokens": 600,
            "cache_creation_input_tokens": 100,
            "cache_read_input_tokens": 0,
        },
    }


def test_parse_transcript_tokens_claude_dedups_by_message_id(tmp_path: Path) -> None:
    """The claude-code session log repeats each assistant message; identical ids count once."""
    dup = {"type": "assistant", "message": {
        "id": "msg_1", "model": "claude-sonnet-4-6",
        "usage": {"input_tokens": 10, "output_tokens": 5,
                  "cache_creation_input_tokens": 2, "cache_read_input_tokens": 100},
    }}
    transcript = _write_jsonl(tmp_path / "transcript.jsonl", [
        dup, dup, dup,  # same message.id logged three times
        {"type": "assistant", "message": {
            "id": "msg_2", "model": "claude-sonnet-4-6",
            "usage": {"input_tokens": 20, "output_tokens": 7,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 200},
        }},
    ])

    result = parse_transcript_tokens(transcript)

    assert result == {
        "claude-sonnet-4-6": {
            "input_tokens": 30, "output_tokens": 12,
            "cache_creation_input_tokens": 2, "cache_read_input_tokens": 300,
        },
    }


def test_parse_transcript_tokens_claude_prefers_result_modelusage(tmp_path: Path) -> None:
    """A result event (autonomous stream-json) is authoritative: its camelCase modelUsage wins,
    and the per-assistant output_tokens (start-of-message placeholders) are ignored."""
    transcript = _write_jsonl(tmp_path / "transcript.jsonl", [
        {"type": "assistant", "message": {
            "id": "msg_1", "model": "claude-sonnet-4-6",
            "usage": {"input_tokens": 5, "output_tokens": 1,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 999},
        }},
        {"type": "result", "modelUsage": {
            "claude-sonnet-4-6": {
                "inputTokens": 1200, "outputTokens": 34000,
                "cacheCreationInputTokens": 500, "cacheReadInputTokens": 90000,
            },
        }},
    ])

    result = parse_transcript_tokens(transcript)

    assert result == {
        "claude-sonnet-4-6": {
            "input_tokens": 1200, "output_tokens": 34000,
            "cache_creation_input_tokens": 500, "cache_read_input_tokens": 90000,
        },
    }


def test_parse_transcript_tokens_pi_format(tmp_path: Path) -> None:
    """Pi JSONL with session/model_change events (skipped) plus assistant messages."""
    transcript = _write_jsonl(tmp_path / "transcript.json", [
        {"type": "session", "version": 3, "id": "abc123"},
        {"type": "model_change", "id": "x", "provider": "llama-server",
         "modelId": "gemma-4-26B-A4B-it-Q8_0.gguf"},
        {"type": "thinking_level_change", "level": "none"},
        {"type": "message", "id": "u1", "message": {
            "role": "user",
            "content": [{"type": "text", "text": "hello"}],
        }},
        {"type": "message", "id": "a1", "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}],
            "model": "gemma-4-26B-A4B-it-Q8_0.gguf",
            "usage": {"input": 100, "output": 50, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 150},
            "stopReason": "stop",
        }},
        {"type": "message", "id": "a2", "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "done"}],
            "model": "gemma-4-26B-A4B-it-Q8_0.gguf",
            "usage": {"input": 200, "output": 75, "cacheRead": 10, "cacheWrite": 20, "totalTokens": 275},
            "stopReason": "stop",
        }},
    ])

    result = parse_transcript_tokens(transcript)

    assert result == {
        "gemma-4-26B-A4B-it-Q8_0.gguf": {
            "input_tokens": 300,
            "output_tokens": 125,
            "cache_creation_input_tokens": 20,
            "cache_read_input_tokens": 10,
        },
    }


def test_parse_transcript_tokens_pi_skips_user_messages(tmp_path: Path) -> None:
    """Pi transcript with user messages interleaved — only assistant messages counted."""
    transcript = _write_jsonl(tmp_path / "transcript.json", [
        {"type": "message", "id": "u1", "message": {
            "role": "user",
            "content": [{"type": "text", "text": "do something"}],
        }},
        {"type": "message", "id": "a1", "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "model": "gemma-4-26B-A4B-it-Q8_0.gguf",
            "usage": {"input": 50, "output": 25, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 75},
            "stopReason": "stop",
        }},
        {"type": "message", "id": "u2", "message": {
            "role": "user",
            "content": [{"type": "text", "text": "now what"}],
        }},
    ])

    result = parse_transcript_tokens(transcript)

    assert result == {
        "gemma-4-26B-A4B-it-Q8_0.gguf": {
            "input_tokens": 50,
            "output_tokens": 25,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


def test_parse_transcript_tokens_pi_skips_aborted(tmp_path: Path) -> None:
    """Pi assistant message with stopReason=aborted and zero usage doesn't crash."""
    transcript = _write_jsonl(tmp_path / "transcript.json", [
        {"type": "message", "id": "a1", "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "starting..."}],
            "model": "gemma-4-26B-A4B-it-Q8_0.gguf",
            "usage": {"input": 100, "output": 30, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 130},
            "stopReason": "stop",
        }},
        {"type": "message", "id": "a2", "message": {
            "role": "assistant",
            "content": [],
            "model": "gemma-4-26B-A4B-it-Q8_0.gguf",
            "usage": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0},
            "stopReason": "aborted",
        }},
    ])

    result = parse_transcript_tokens(transcript)

    assert result == {
        "gemma-4-26B-A4B-it-Q8_0.gguf": {
            "input_tokens": 100,
            "output_tokens": 30,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


def test_pick_transcript_prefers_session_log(tmp_path: Path) -> None:
    """When several transcripts coexist in a dir, the claude session log wins, then pi-session, then the
    stream-json stdout. This is what keeps a claude run reading its authoritative session.jsonl (complete
    output) instead of the placeholder-output transcript.jsonl."""
    for name in ("transcript.json", "transcript.jsonl", "pi-session.jsonl", "session.jsonl"):
        (tmp_path / name).write_text("{}\n")
    assert _pick_transcript(tmp_path).name == "session.jsonl"

    (tmp_path / "session.jsonl").unlink()
    assert _pick_transcript(tmp_path).name == "pi-session.jsonl"

    (tmp_path / "pi-session.jsonl").unlink()
    assert _pick_transcript(tmp_path).name == "transcript.jsonl"


def test_pick_transcript_none_when_empty(tmp_path: Path) -> None:
    """A directory with no recognised transcript yields None."""
    assert _pick_transcript(tmp_path) is None


def test_parse_transcript_tokens_empty_file(tmp_path: Path) -> None:
    """Empty transcript file returns empty dict."""
    transcript = tmp_path / "transcript.json"
    transcript.write_text("")

    result = parse_transcript_tokens(transcript)

    assert result == {}


def test_parse_transcript_tokens_malformed_lines(tmp_path: Path) -> None:
    """File with invalid JSON lines mixed with valid ones — valid lines parsed, invalid skipped."""
    lines = [
        "not valid json at all",
        json.dumps({"type": "assistant", "message": {
            "model": "claude-sonnet-4-6-20250514",
            "usage": {"input_tokens": 500, "output_tokens": 100, "cache_creation_input_tokens": 0,
                      "cache_read_input_tokens": 0},
        }}),
        "{truncated json",
        "",
        json.dumps({"type": "message", "id": "a1", "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "model": "gemma-4-26B-A4B-it-Q8_0.gguf",
            "usage": {"input": 200, "output": 80, "cacheRead": 5, "cacheWrite": 10, "totalTokens": 280},
            "stopReason": "stop",
        }}),
    ]
    transcript = tmp_path / "transcript.json"
    transcript.write_text("\n".join(lines) + "\n")

    result = parse_transcript_tokens(transcript)

    assert result == {
        "claude-sonnet-4-6-20250514": {
            "input_tokens": 500,
            "output_tokens": 100,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
        "gemma-4-26B-A4B-it-Q8_0.gguf": {
            "input_tokens": 200,
            "output_tokens": 80,
            "cache_creation_input_tokens": 10,
            "cache_read_input_tokens": 5,
        },
    }
