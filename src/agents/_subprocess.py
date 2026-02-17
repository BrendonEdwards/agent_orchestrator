"""Shared subprocess execution with retry and exponential backoff.

All CLI agents (Claude, Codex, Gemini) delegate to this so retry
logic and timeout handling live in one place.
"""

from __future__ import annotations

import asyncio

from src.agents.base import AgentResponse
from src.config import DEFAULT_CONFIG, OrchestratorConfig


async def run_cli(
    agent_name: str,
    args: list[str],
    *,
    config: OrchestratorConfig = DEFAULT_CONFIG,
    metadata: dict | None = None,
) -> AgentResponse:
    """Run a CLI subprocess with retry on transient failures.

    Retries on non-zero exit codes and timeouts with exponential backoff.
    Returns an AgentResponse whether it succeeds or exhausts retries.
    """
    last_error = ""

    for attempt in range(1 + config.agent_retries):
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
                    content=stdout.decode().strip(),
                    metadata=metadata or {},
                )

            last_error = stderr.decode().strip() or f"exit code {proc.returncode}"

        except asyncio.TimeoutError:
            last_error = f"CLI timed out after {config.agent_timeout:.0f}s"
        except FileNotFoundError as e:
            # Binary not found - no point retrying
            return AgentResponse(
                agent_name=agent_name, content="",
                success=False, error=f"CLI not found: {e}",
            )
        except Exception as e:
            last_error = str(e)

        # Exponential backoff before next retry
        if attempt < config.agent_retries:
            delay = config.retry_base_delay * (2 ** attempt)
            await asyncio.sleep(delay)

    return AgentResponse(
        agent_name=agent_name, content="",
        success=False,
        error=f"{last_error} (after {1 + config.agent_retries} attempts)",
    )
