"""Message router for inter-agent communication.

Handles routing messages between agents based on the orchestrator's
topology: Claude as the hub with bidirectional connections to Codex,
Gemini, and Llama Local (and between those agents as well).
"""

from __future__ import annotations

import asyncio
from typing import Any

from src.agents.base import AgentCapability, AgentMessage, AgentResponse, BaseAgent
from src.memento.memento import Memento
from src.rules.loader import RulesLoader


class MessageRouter:
    """Routes messages between agents in the orchestration network.

    Supports the full mesh topology shown on the whiteboard where every
    agent can communicate with every other agent, with Claude as the
    preferred hub for coordination.
    """

    def __init__(
        self,
        agents: dict[str, BaseAgent],
        rules_loader: RulesLoader,
        memento: Memento,
        codex_agent: BaseAgent | None = None,
    ):
        self._agents = agents
        self._rules = rules_loader
        self._memento = memento
        self._codex = codex_agent  # Used for context compression

    async def route(self, message: AgentMessage) -> AgentResponse:
        """Route a message to its target agent."""
        target = self._agents.get(message.target)
        if not target:
            return AgentResponse(
                agent_name=message.target,
                content="",
                success=False,
                error=f"Unknown agent: {message.target}",
            )

        self._memento.record(
            event_type="routing",
            source=message.source,
            target=message.target,
            content=f"Routing message: {message.content[:100]}...",
            metadata={"message_id": message.id},
        )

        # Apply context compression if routing between non-Claude agents
        content = message.content
        if (
            self._codex
            and message.source not in ("orchestrator", "claude")
            and message.target != "codex"
            and len(content) > 500
        ):
            from src.agents.codex_agent import CodexAgent

            if isinstance(self._codex, CodexAgent):
                content = await self._codex.translate_for_context(content)
                message = AgentMessage(
                    source=message.source,
                    target=message.target,
                    content=content,
                    metadata={**message.metadata, "compressed": True},
                    parent_message_id=message.id,
                )
                self._memento.record(
                    event_type="context_compression",
                    source="codex",
                    target=message.target,
                    content=f"Compressed message from {len(message.content)} chars",
                )

        response = await target.send(message)

        self._memento.record(
            event_type="response",
            source=message.target,
            target=message.source,
            content=f"Response (success={response.success}): {response.content[:100]}",
            metadata={"token_usage": response.token_usage},
        )

        return response

    async def broadcast(
        self, message: AgentMessage, targets: list[str] | None = None
    ) -> dict[str, AgentResponse]:
        """Send a message to multiple agents in parallel."""
        if targets is None:
            targets = list(self._agents.keys())

        self._memento.record(
            event_type="broadcast",
            source=message.source,
            content=f"Broadcasting to: {', '.join(targets)}",
            metadata={"targets": targets},
        )

        tasks = {}
        for target_name in targets:
            if target_name in self._agents:
                target_msg = AgentMessage(
                    source=message.source,
                    target=target_name,
                    content=message.content,
                    metadata=message.metadata,
                    parent_message_id=message.id,
                )
                tasks[target_name] = self.route(target_msg)

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        responses: dict[str, AgentResponse] = {}
        for name, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                responses[name] = AgentResponse(
                    agent_name=name,
                    content="",
                    success=False,
                    error=str(result),
                )
            else:
                responses[name] = result

        return responses

    async def route_by_capability(
        self, message: AgentMessage, capability: AgentCapability
    ) -> AgentResponse:
        """Route a message to the best agent for a given capability."""
        best_agent = self._rules.get_best_agent_for(capability.value)
        if best_agent and best_agent in self._agents:
            message_copy = AgentMessage(
                source=message.source,
                target=best_agent,
                content=message.content,
                metadata=message.metadata,
                parent_message_id=message.id,
            )
            return await self.route(message_copy)

        # Fallback: find any agent with the capability
        for name, agent in self._agents.items():
            if agent.supports(capability):
                message_copy = AgentMessage(
                    source=message.source,
                    target=name,
                    content=message.content,
                    metadata=message.metadata,
                    parent_message_id=message.id,
                )
                return await self.route(message_copy)

        return AgentResponse(
            agent_name="router",
            content="",
            success=False,
            error=f"No agent found with capability: {capability.value}",
        )

    async def health_check_all(self) -> dict[str, bool]:
        """Check health of all registered agents."""
        tasks = {name: agent.health_check() for name, agent in self._agents.items()}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        return {
            name: (result if isinstance(result, bool) else False)
            for name, result in zip(tasks.keys(), results)
        }
