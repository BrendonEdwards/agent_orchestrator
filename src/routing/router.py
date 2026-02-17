"""Message router for inter-agent communication.

Every routed message gets memento survival notes injected. Agents are
stateless - the memento briefing is their only context. This is the
core anti-context-rot mechanism: fresh agents + concise notes.
"""

from __future__ import annotations

import asyncio

from src.agents.base import AgentCapability, AgentMessage, AgentResponse, BaseAgent
from src.memento.memento import Memento
from src.routing import protocol
from src.rules.loader import RulesLoader


class MessageRouter:
    """Routes messages between stateless agents.

    Every message gets the current memento briefing stamped onto it.
    Agents receive: their task + survival notes. Nothing else.
    """

    def __init__(
        self,
        agents: dict[str, BaseAgent],
        rules_loader: RulesLoader,
        memento: Memento,
    ):
        self._agents = agents
        self._rules = rules_loader
        self._memento = memento

    async def route(self, message: AgentMessage) -> AgentResponse:
        """Route a message to its target agent with memento notes injected."""
        target = self._agents.get(message.target)
        if not target:
            return AgentResponse(
                agent_name=message.target,
                content="",
                success=False,
                error=f"Unknown agent: {message.target}",
            )

        # Stamp memento briefing onto the message - this is the only
        # context the fresh agent will have
        briefing = self._memento.briefing()
        msg = AgentMessage(
            source=message.source,
            target=message.target,
            content=message.content,
            memento=briefing,
            metadata=message.metadata,
        )

        response = await target.send(msg)

        # Record key outcomes as memento notes for future agents
        if response.success and response.content:
            self._memento.note(
                f"r:{message.target}",
                response.content[:150],
                priority=1,
            )

        return response

    async def route_compact(
        self,
        source: str,
        target: str,
        task_type: str,
        description: str,
        **kwargs: str,
    ) -> AgentResponse:
        """Route using compact protocol format instead of English."""
        compact = protocol.task_message(
            task_type=task_type,
            description=description,
            **kwargs,
        )
        msg = AgentMessage(
            source=source,
            target=target,
            content=compact,
            metadata={"protocol": "compact"},
        )
        return await self.route(msg)

    async def delegate_grunt_work(self, task: str) -> AgentResponse:
        """Send simple work to Gemini. No memento needed."""
        target = self._agents.get("gemini")
        if not target:
            return AgentResponse(
                agent_name="gemini", content="", success=False,
                error="Gemini agent not available",
            )
        # Grunt work goes direct - no memento, no context, just the task
        msg = AgentMessage(source="orchestrator", target="gemini", content=task)
        return await target.send(msg)

    async def broadcast(
        self, message: AgentMessage, targets: list[str] | None = None
    ) -> dict[str, AgentResponse]:
        """Send a message to multiple agents in parallel.

        Each agent gets a fresh call with memento notes.
        """
        if targets is None:
            targets = list(self._agents.keys())

        tasks = {}
        for target_name in targets:
            if target_name in self._agents:
                target_msg = AgentMessage(
                    source=message.source,
                    target=target_name,
                    content=message.content,
                    metadata=message.metadata,
                )
                tasks[target_name] = self.route(target_msg)

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        responses: dict[str, AgentResponse] = {}
        for name, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                responses[name] = AgentResponse(
                    agent_name=name, content="", success=False, error=str(result),
                )
            else:
                responses[name] = result

        return responses

    async def route_by_capability(
        self, message: AgentMessage, capability: AgentCapability
    ) -> AgentResponse:
        """Route to the best agent for a given capability."""
        best_agent = self._rules.get_best_agent_for(capability.value)
        if best_agent and best_agent in self._agents:
            msg = AgentMessage(
                source=message.source,
                target=best_agent,
                content=message.content,
                metadata=message.metadata,
            )
            return await self.route(msg)

        for name, agent in self._agents.items():
            if agent.supports(capability):
                msg = AgentMessage(
                    source=message.source,
                    target=name,
                    content=message.content,
                    metadata=message.metadata,
                )
                return await self.route(msg)

        return AgentResponse(
            agent_name="router", content="", success=False,
            error=f"No agent with capability: {capability.value}",
        )

    async def health_check_all(self) -> dict[str, bool]:
        """Check health of all registered agents."""
        tasks = {name: agent.health_check() for name, agent in self._agents.items()}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        return {
            name: (result if isinstance(result, bool) else False)
            for name, result in zip(tasks.keys(), results)
        }
