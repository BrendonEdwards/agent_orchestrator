"""Claude agent - the central hub of the orchestrator.

Claude serves as the primary coordinator, delegating tasks to specialized
agents and synthesizing their results. It has bidirectional connections
to Codex, Gemini, and Llama Local.
"""

from __future__ import annotations

import os
from typing import Any

from src.agents.base import AgentCapability, AgentMessage, AgentResponse, BaseAgent


class ClaudeAgent(BaseAgent):
    """Claude agent acting as the central orchestration hub.

    Claude is the primary reasoning engine that:
    - Analyzes incoming tasks and determines which agents to delegate to
    - Synthesizes responses from multiple agents
    - Maintains the high-level conversation context
    - Coordinates inter-agent communication
    """

    def __init__(
        self,
        model_id: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
    ):
        super().__init__(
            name="claude",
            model_id=model_id,
            capabilities=[
                AgentCapability.TEXT_GENERATION,
                AgentCapability.CODE_GENERATION,
                AgentCapability.CODE_REVIEW,
                AgentCapability.REASONING,
                AgentCapability.MATH,
                AgentCapability.SUMMARIZATION,
                AgentCapability.CONVERSATION,
                AgentCapability.FUNCTION_CALLING,
            ],
        )
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        return self._client

    async def send(self, message: AgentMessage) -> AgentResponse:
        """Send a message to Claude and return the response."""
        try:
            client = self._get_client()

            system_prompt = (
                "You are the central coordinator of a multi-agent AI system. "
                "You analyze tasks, delegate to specialized agents (Codex for code, "
                "Gemini for multimodal tasks, Llama for local/private processing), "
                "and synthesize their outputs into coherent responses."
            )

            messages = []
            for ctx in self._context:
                messages.append({"role": ctx["role"], "content": ctx["content"]})
            messages.append({"role": "user", "content": message.content})

            response = await client.messages.create(
                model=self.model_id,
                max_tokens=4096,
                system=system_prompt,
                messages=messages,
            )

            result_text = response.content[0].text
            self.add_to_context("user", message.content)
            self.add_to_context("assistant", result_text)

            return AgentResponse(
                agent_name=self.name,
                content=result_text,
                token_usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                metadata={"model": self.model_id, "source_message_id": message.id},
            )
        except Exception as e:
            return AgentResponse(
                agent_name=self.name,
                content="",
                success=False,
                error=str(e),
            )

    async def analyze_task(self, task: str) -> dict[str, Any]:
        """Analyze a task and determine which agents should handle it.

        Returns a routing plan with agent assignments and subtasks.
        """
        analysis_prompt = (
            f"Analyze this task and determine which AI agents should handle it.\n\n"
            f"Available agents:\n"
            f"- claude: reasoning, conversation, code review, synthesis\n"
            f"- codex: code generation, code completion, technical translation\n"
            f"- gemini: multimodal (images, audio, video), language translation\n"
            f"- llama: local execution, privacy-sensitive tasks, fast inference\n\n"
            f"Task: {task}\n\n"
            f"Respond with a JSON object containing:\n"
            f"- primary_agent: the main agent for this task\n"
            f"- supporting_agents: list of agents that should assist\n"
            f"- subtasks: list of {{agent, description}} objects\n"
            f"- reasoning: brief explanation of the routing decision"
        )

        msg = AgentMessage(source="orchestrator", target="claude", content=analysis_prompt)
        response = await self.send(msg)

        return {
            "raw_analysis": response.content,
            "success": response.success,
            "error": response.error,
        }

    async def synthesize(self, results: list[AgentResponse]) -> AgentResponse:
        """Synthesize results from multiple agents into a coherent response."""
        parts = []
        for r in results:
            if r.success:
                parts.append(f"[{r.agent_name}]: {r.content}")
            else:
                parts.append(f"[{r.agent_name}]: (failed) {r.error}")

        synthesis_prompt = (
            "Synthesize the following agent responses into a single coherent answer.\n\n"
            + "\n\n".join(parts)
        )

        msg = AgentMessage(source="orchestrator", target="claude", content=synthesis_prompt)
        return await self.send(msg)

    async def health_check(self) -> bool:
        """Check if the Anthropic API is reachable."""
        try:
            client = self._get_client()
            response = await client.messages.create(
                model=self.model_id,
                max_tokens=10,
                messages=[{"role": "user", "content": "ping"}],
            )
            return len(response.content) > 0
        except Exception:
            return False
