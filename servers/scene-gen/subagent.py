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
    Falls back to the last non-empty text from any event.
    """
    last_text = ""
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
                last_text = text
        except (json.JSONDecodeError, ValueError):
            # Not JSON line — accumulate as fallback
            if line:
                last_text = line

    return last_completion or last_text or raw
