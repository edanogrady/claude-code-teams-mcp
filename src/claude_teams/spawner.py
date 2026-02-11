from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from claude_teams import messaging, teams
from claude_teams.models import COLOR_PALETTE, InboxMessage, TeammateMember
from claude_teams.teams import _VALID_NAME_RE
from claude_teams import opencode_client

logger = logging.getLogger(__name__)


_OPENCODE_PROMPT_WRAPPER = """\
You are team member '{name}' on team '{team_name}'.

You have MCP tools from the claude-teams server for team coordination:
- poll_inbox(team_name="{team_name}", agent_name="{name}") - Check for new messages
- send_message(team_name="{team_name}", type="message", sender="{name}", recipient="team-lead", content="...", summary="...") - Message teammates
- task_list(team_name="{team_name}") - View team tasks
- task_update(team_name="{team_name}", task_id="...", status="...") - Update task status
- task_get(team_name="{team_name}", task_id="...") - Get task details

Start by reading your inbox for instructions.

---

{prompt}"""

_MCP_PROMPT_WRAPPER = """\
You are team member '{name}' on team '{team_name}'.

You have MCP tools from the claude-teams server for team coordination:
- poll_inbox(team_name="{team_name}", agent_name="{name}") - Check for new messages
- send_message(team_name="{team_name}", type="message", sender="{name}", recipient="team-lead", content="...", summary="...") - Message teammates
- task_list(team_name="{team_name}") - View team tasks
- task_update(team_name="{team_name}", task_id="...", status="...") - Update task status
- task_get(team_name="{team_name}", task_id="...") - Get task details

IMPORTANT: You must actively poll your inbox for messages from the orchestrator.
Call poll_inbox periodically — after completing each task, before claiming the next,
and whenever waiting for instructions.

Start by reading your inbox for instructions.

---

{prompt}"""

_PROMPT_DIR = Path("/tmp")
_DEFAULT_MODELS_BY_BACKEND = {
    "claude": "sonnet",
    "opencode": "sonnet",
    "codex": "gpt-5.3-codex",
    "gemini": "gemini-3-pro-preview",
}


def discover_harness_binary(name: str) -> str | None:
    return shutil.which(name)


def use_tmux_windows() -> bool:
    """Return True when teammate processes should be spawned in tmux windows."""
    val = os.environ.get("USE_TMUX_WINDOWS", "").strip().lower()
    return val in ("1", "true", "yes")


def build_tmux_spawn_args(command: str, name: str) -> list[str]:
    """Build the tmux command used to spawn a teammate process."""
    if use_tmux_windows():
        return [
            "tmux",
            "new-window",
            "-dP",
            "-F",
            "#{window_id}",
            "-n",
            f"@claude-team | {name}",
            command,
        ]
    return ["tmux", "split-window", "-dP", "-F", "#{pane_id}", command]


def _ensure_tmux_available() -> None:
    tmux_binary = shutil.which("tmux")
    if not tmux_binary:
        raise ValueError(
            "Cannot spawn teammate: 'tmux' binary not found on PATH. "
            "Install tmux and ensure it is in your PATH."
        )

    result = subprocess.run(
        [tmux_binary, "start-server"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        details = (result.stderr or result.stdout).strip()
        suffix = f" tmux output: {details}" if details else ""
        raise ValueError(
            "Cannot spawn teammate: tmux server is unavailable and could not be started."
            f"{suffix}"
        )


def discover_opencode_models(opencode_binary: str) -> list[str]:
    """Run ``opencode models --refresh`` and return available model names."""
    try:
        result = subprocess.run(
            [opencode_binary, "models", "--refresh"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return []
        lines = result.stdout.strip().splitlines()
        # First line is status message, rest are model names
        return [line.strip() for line in lines[1:] if line.strip()]
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return []


def assign_color(team_name: str, base_dir: Path | None = None) -> str:
    config = teams.read_config(team_name, base_dir)
    count = sum(1 for m in config.members if isinstance(m, TeammateMember))
    return COLOR_PALETTE[count % len(COLOR_PALETTE)]


def build_spawn_command(
    member: TeammateMember,
    claude_binary: str,
    lead_session_id: str,
) -> str:
    team_name = member.agent_id.split("@", 1)[1]
    cmd = (
        f"cd {shlex.quote(member.cwd)} && "
        f"CLAUDECODE=1 CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1 "
        f"{shlex.quote(claude_binary)} "
        f"--agent-id {shlex.quote(member.agent_id)} "
        f"--agent-name {shlex.quote(member.name)} "
        f"--team-name {shlex.quote(team_name)} "
        f"--agent-color {shlex.quote(member.color)} "
        f"--parent-session-id {shlex.quote(lead_session_id)} "
        f"--agent-type {shlex.quote(member.agent_type)} "
        f"--model {shlex.quote(member.model)}"
    )
    if member.plan_mode_required:
        cmd += " --plan-mode-required"
    return cmd


def build_opencode_attach_command(
    opencode_binary: str,
    server_url: str,
    session_id: str,
    cwd: str,
) -> str:
    return (
        f"{shlex.quote(opencode_binary)} attach "
        f"{shlex.quote(server_url)} "
        f"-s {shlex.quote(session_id)} "
        f"--dir {shlex.quote(cwd)}"
    )


def _write_cli_launcher(
    team_name: str,
    name: str,
    wrapped_prompt: str,
    cwd: str,
    binary: str,
    extra_args: list[str],
) -> str:
    """Write prompt file + bash launcher script. Returns shell command to run."""
    spawn_dir = Path(tempfile.mkdtemp(prefix="claude-teams-", dir=str(_PROMPT_DIR)))
    spawn_dir.chmod(0o700)

    prompt_path = spawn_dir / "prompt.txt"
    prompt_flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        prompt_flags |= os.O_NOFOLLOW
    prompt_fd = os.open(prompt_path, prompt_flags, 0o600)
    with os.fdopen(prompt_fd, "w", encoding="utf-8") as prompt_file:
        prompt_file.write(wrapped_prompt)

    cli_cmd = " ".join(shlex.quote(a) for a in [binary, *extra_args])
    launcher_path = spawn_dir / "launcher.sh"
    launcher_path.write_text(
        f"#!/usr/bin/env bash\n"
        f"cd {shlex.quote(cwd)}\n"
        f"prompt=$(<{shlex.quote(str(prompt_path))})\n"
        f'exec {cli_cmd} -- "$prompt"\n',
        encoding="utf-8",
    )
    launcher_path.chmod(0o700)
    return f"bash {shlex.quote(str(launcher_path))}"


def spawn_teammate(
    team_name: str,
    name: str,
    prompt: str,
    claude_binary: str,
    lead_session_id: str,
    *,
    model: str = "",
    subagent_type: str = "general-purpose",
    cwd: str | None = None,
    plan_mode_required: bool = False,
    base_dir: Path | None = None,
    backend_type: str = "claude",
    opencode_binary: str | None = None,
    opencode_server_url: str | None = None,
    opencode_agent: str | None = None,
    codex_binary: str | None = None,
    gemini_binary: str | None = None,
) -> TeammateMember:
    if not _VALID_NAME_RE.match(name):
        raise ValueError(
            f"Invalid agent name: {name!r}. Use only letters, numbers, hyphens, underscores."
        )
    if len(name) > 64:
        raise ValueError(f"Agent name too long ({len(name)} chars, max 64)")
    if name == "team-lead":
        raise ValueError("Agent name 'team-lead' is reserved")
    if backend_type == "opencode" and not opencode_binary:
        raise ValueError(
            "Cannot spawn opencode teammate: 'opencode' binary not found on PATH. "
            "Install OpenCode or ensure it is in your PATH."
        )
    if backend_type == "opencode" and not opencode_server_url:
        raise ValueError(
            "Cannot spawn opencode teammate: OPENCODE_SERVER_URL is not set. "
            "Start 'opencode serve' and set the environment variable."
        )
    if backend_type == "claude" and not claude_binary:
        raise ValueError(
            "Cannot spawn claude teammate: 'claude' binary not found on PATH. "
            "Install Claude Code or ensure it is in your PATH."
        )
    if backend_type == "codex" and not codex_binary:
        raise ValueError(
            "Cannot spawn codex teammate: 'codex' binary not found on PATH. "
            "Install Codex CLI or ensure it is in your PATH."
        )
    if backend_type == "gemini" and not gemini_binary:
        raise ValueError(
            "Cannot spawn gemini teammate: 'gemini' binary not found on PATH. "
            "Install Gemini CLI or ensure it is in your PATH."
        )
    if backend_type in {"codex", "gemini"}:
        logger.warning(
            "Spawning %s teammate %s. Ensure claude-teams is configured as an MCP "
            "server in your %s config; otherwise poll_inbox/send_message/task_* "
            "tools will be unavailable.",
            backend_type,
            name,
            backend_type,
        )
    _ensure_tmux_available()

    normalized_model = (model or "").strip()
    resolved_model = normalized_model or _DEFAULT_MODELS_BY_BACKEND.get(
        backend_type, "sonnet"
    )

    resolved_cwd = cwd or str(Path.cwd())
    opencode_session_id: str | None = None

    if backend_type == "opencode":
        opencode_client.verify_mcp_configured(opencode_server_url)
        opencode_session_id = opencode_client.create_session(
            opencode_server_url,
            title=f"{name}@{team_name}",
            permissions=[{"permission": "*", "pattern": "*", "action": "allow"}],
        )

    color = assign_color(team_name, base_dir)
    now_ms = int(time.time() * 1000)

    member = TeammateMember(
        agent_id=f"{name}@{team_name}",
        name=name,
        agent_type=subagent_type,
        model=resolved_model,
        prompt=prompt,
        color=color,
        plan_mode_required=plan_mode_required,
        joined_at=now_ms,
        tmux_pane_id="",
        cwd=resolved_cwd,
        backend_type=backend_type,
        opencode_session_id=opencode_session_id,
        is_active=False,
    )

    member_added = False
    try:
        teams.add_member(team_name, member, base_dir)
        member_added = True

        messaging.ensure_inbox(team_name, name, base_dir)
        initial_msg = InboxMessage(
            from_="team-lead",
            text=prompt,
            timestamp=messaging.now_iso(),
            read=False,
        )
        messaging.append_message(team_name, name, initial_msg, base_dir)

        if backend_type == "opencode":
            wrapped = _OPENCODE_PROMPT_WRAPPER.format(
                name=name,
                team_name=team_name,
                prompt=prompt,
            )
            opencode_client.send_prompt_async(
                opencode_server_url,
                opencode_session_id,
                wrapped,
                agent=opencode_agent or "build",
            )
            cmd = build_opencode_attach_command(
                opencode_binary,
                opencode_server_url,
                opencode_session_id,
                resolved_cwd,
            )
        elif backend_type == "codex":
            wrapped = _MCP_PROMPT_WRAPPER.format(
                name=name, team_name=team_name, prompt=prompt,
            )
            cmd = _write_cli_launcher(
                team_name, name, wrapped, resolved_cwd,
                codex_binary, ["--full-auto", "-m", resolved_model],
            )
        elif backend_type == "gemini":
            wrapped = _MCP_PROMPT_WRAPPER.format(
                name=name, team_name=team_name, prompt=prompt,
            )
            cmd = _write_cli_launcher(
                team_name, name, wrapped, resolved_cwd,
                gemini_binary, ["--yolo", "-m", resolved_model],
            )
        else:
            cmd = build_spawn_command(member, claude_binary, lead_session_id)

        result = subprocess.run(
            build_tmux_spawn_args(cmd, name),
            capture_output=True,
            text=True,
            check=True,
        )
        pane_id = result.stdout.strip()

        # Auto-tile panes so all workers are visible side by side
        if not use_tmux_windows():
            subprocess.run(
                ["tmux", "select-layout", "tiled"],
                capture_output=True,
                check=False,
            )

        try:
            teams.set_member_tmux_pane(team_name, name, pane_id, base_dir=base_dir)
        except Exception:
            subprocess.run(["tmux", "kill-pane", "-t", pane_id], check=False)
            raise
    except Exception:
        if member_added:
            try:
                teams.remove_member(team_name, name, base_dir)
            except Exception:
                pass
        if backend_type == "opencode" and opencode_server_url and opencode_session_id:
            try:
                opencode_client.abort_session(opencode_server_url, opencode_session_id)
            except Exception:
                pass
            try:
                opencode_client.delete_session(opencode_server_url, opencode_session_id)
            except Exception:
                pass
        raise

    member.tmux_pane_id = pane_id
    return member


def kill_tmux_pane(pane_id: str) -> None:
    if pane_id.startswith("@"):
        subprocess.run(["tmux", "kill-window", "-t", pane_id], check=False)
        return
    subprocess.run(["tmux", "kill-pane", "-t", pane_id], check=False)
