"""Principia subagent helper for text LLM reasoning.

Replaces direct anthropic.Anthropic().messages.create() calls by spawning
a principia CLI subprocess.  The response is wrapped to match Claude's
response format (response.content[0].text) for drop-in compatibility.
"""
import json
import os
import subprocess
import sys

# Subagent config directory — empty MCP settings so we don't spawn scene-gen servers.
_SUBAGENT_CONFIG_DIR = "/tmp/principia-subagent"


def _ensure_subagent_config():
    """Create a lightweight principia config dir for subagents (no MCP servers)."""
    data_dir = os.path.join(_SUBAGENT_CONFIG_DIR, "data")
    settings_dir = os.path.join(data_dir, "settings")
    os.makedirs(settings_dir, exist_ok=True)

    mcp_path = os.path.join(settings_dir, "principia_mcp_settings.json")
    if not os.path.exists(mcp_path):
        with open(mcp_path, "w") as f:
            json.dump({"mcpServers": {}}, f)

    # Symlink secrets so the subagent can authenticate
    main_data = os.path.expanduser("~/.principia/data")
    for fname in ("secrets.json", "globalState.json"):
        link = os.path.join(data_dir, fname)
        target = os.path.join(main_data, fname)
        if os.path.exists(target) and not os.path.exists(link):
            try:
                os.symlink(target, link)
            except OSError:
                pass


class _ContentItem:
    """Mimics anthropic's ContentBlock so existing code can do response.content[0].text."""
    def __init__(self, text: str):
        self.text = text
        self.type = "text"


class SubagentResponse:
    """Drop-in replacement for an anthropic Message object."""
    def __init__(self, text: str):
        self.content = [_ContentItem(text)]
        self.id = "subagent"
        self.model = "principia-subagent"
        self.usage = None


def call_llm_via_subagent(prompt: str, timeout: int = 300) -> SubagentResponse:
    """Spawn principia CLI subagent for text LLM reasoning.

    Returns a SubagentResponse whose .content[0].text holds the model output,
    matching the shape of an anthropic Message.
    """
    try:
        _ensure_subagent_config()
        env = os.environ.copy()
        env["PRINCIPIA_DIR"] = _SUBAGENT_CONFIG_DIR
        env.pop("CLAUDECODE", None)  # Avoid nested-session error
        result = subprocess.run(
            ["principia", prompt, "--json", "-y"],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        output = result.stdout.strip()
        if not output:
            output = result.stderr.strip() or "(empty subagent response)"

        # principia --json wraps output in a JSON envelope; extract the text
        text = _parse_subagent_output(output)

        # Debug: log raw output and parsed result for diagnosis
        try:
            _debug_path = "/tmp/principia-subagent-debug.log"
            with open(_debug_path, "a") as _df:
                _df.write(f"\n{'='*60}\n")
                _df.write(f"PROMPT (first 200 chars): {prompt[:200]}\n")
                _df.write(f"RAW OUTPUT LINES: {len(output.splitlines())}\n")
                for _i, _line in enumerate(output.splitlines()[-20:]):
                    _df.write(f"  LINE[-{min(20, len(output.splitlines()))-_i}]: {_line[:300]}\n")
                _df.write(f"PARSED TEXT (first 500 chars): {text[:500]}\n")
                _df.write(f"CONTAINS_JSON: {_contains_json(text)}\n")
        except Exception:
            pass

        return SubagentResponse(text)

    except subprocess.TimeoutExpired:
        print(f"[subagent] timeout after {timeout}s", file=sys.stderr)
        return SubagentResponse(f"(subagent timed out after {timeout}s)")
    except FileNotFoundError:
        print("[subagent] 'principia' CLI not found on PATH", file=sys.stderr)
        return SubagentResponse("(principia CLI not found)")
    except Exception as e:
        print(f"[subagent] error: {e}", file=sys.stderr)
        return SubagentResponse(f"(subagent error: {e})")


def _parse_subagent_output(raw: str) -> str:
    """Extract text from principia --json output.

    The CLI emits NDJSON (one JSON event per line).  Each line has the form:
        {"ts":..., "type":"say", "say":"<kind>", "text":"..."}

    We look for the last "completion_result" or "text" event and return its text.
    When the caller expects JSON (e.g. object recommendations), the completion_result
    is often a human-readable summary while the actual JSON lives in an earlier
    "text" event.  We therefore prefer content that looks like JSON.

    Falls back to the last non-empty text from any event.
    """
    all_texts = []        # all "text" events, in order
    last_completion = ""
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
            if not isinstance(event, dict):
                continue
            say = event.get("say", "")
            text = event.get("text", "")
            if say == "completion_result" and text:
                last_completion = text
            elif say == "text" and text:
                all_texts.append(text)
        except (json.JSONDecodeError, ValueError):
            # Not JSON line — accumulate as fallback
            if line:
                all_texts.append(line)

    # If completion_result already contains JSON, use it directly.
    if last_completion and _contains_json(last_completion):
        return last_completion

    # Otherwise scan all text events (most recent first) for JSON content.
    for text in reversed(all_texts):
        if _contains_json(text):
            return text

    # JSON may be split across multiple text events — try concatenating them all.
    if len(all_texts) > 1:
        combined = "\n".join(all_texts)
        if _contains_json(combined):
            return combined

    # Fall back to original priority: completion_result > last text > raw
    last_text = all_texts[-1] if all_texts else ""
    return last_completion or last_text or raw


def _contains_json(text: str) -> bool:
    """Quick heuristic: does *text* contain extractable JSON?

    Checks for ```json code blocks or bare JSON objects (``{ ... }``).
    Does NOT do full validation — just enough to steer the parser.
    """
    import re

    # Markdown code-fence with JSON
    if re.search(r'```json\s*\n', text, re.IGNORECASE):
        return True

    # Bare JSON object spanning a good chunk of the text
    stripped = text.strip()
    if stripped.startswith('{') and stripped.endswith('}'):
        return True

    # JSON object embedded somewhere in the text
    if re.search(r'\{\s*"[^"]+"\s*:', text):
        return True

    return False
