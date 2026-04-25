"""Shared subprocess execution with retry and exponential backoff.

All CLI agents (Claude, Codex, Gemini) delegate to this so retry
logic and timeout handling live in one place.
"""

from __future__ import annotations

import asyncio

from src.agents.base import AgentResponse
from src.config import DEFAULT_CONFIG, OrchestratorConfig


async def _terminate_process(proc: asyncio.subprocess.Process) -> None:
    """Terminate a subprocess cleanly, then kill it if it refuses to exit."""
    if proc.returncode is not None:
        return

    proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=2.0)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()


async def run_cli(
    agent_name: str,
    args: list[str],
    *,
    config: OrchestratorConfig = DEFAULT_CONFIG,
    metadata: dict | None = None,
) -> AgentResponse:
    """Run a CLI subprocess with retry on transient failures.

    Retries on non-zero exit codes and timeouts with exponential backoff.
    Timeout handling explicitly terminates the child process before retrying
    so failed agent calls do not leave orphaned Claude, Codex or Gemini
    processes running in the background.
    """
    last_error = ""

    for attempt in range(1 + config.agent_retries):
        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=config.agent_timeout
            )

            if proc.returncode == 0:
                return AgentResponse(
                    agent_name=agent_name,
                    content=stdout.decode(errors="replace").strip(),
                    metadata=metadata or {},
                )

            last_error = stderr.decode(errors="replace").strip() or f"exit code {proc.returncode}"

        except asyncio.TimeoutError:
            if proc is not None:
                await _terminate_process(proc)
            last_error = f"CLI timed out after {config.agent_timeout:.0f}s"
        except FileNotFoundError as e:
            return AgentResponse(
                agent_name=agent_name, content="",
                success=False, error=f"CLI not found: {e}",
            )
        except Exception as e:
            last_error = str(e)

        if attempt < config.agent_retries:
            delay = config.retry_base_delay * (2 ** attempt)
            await asyncio.sleep(delay)

    return AgentResponse(
        agent_name=agent_name, content="",
        success=False,
        error=f"{last_error} (after {1 + config.agent_retries} attempts)",
    )
