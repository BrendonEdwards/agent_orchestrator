"""Message router for inter-agent communication.

Routes messages between agents using a compact protocol format
instead of English. Agents talk to each other in terse notation
to minimize token waste. The Memento system provides survival
notes to fight context rot on long-running tasks.
"""

from __future__ import annotations

import asyncio

from src.agents.base import AgentCapability, AgentMessage, AgentResponse, BaseAgent
from src.memento.memento import Memento
from src.routing import protocol
from src.rules.loader import RulesLoader


class MessageRouter:
    """Routes messages between agents using compact protocol format.

    Key behaviors:
    - Inter-agent messages are encoded in compact protocol format
      (not English) to minimize token usage
    - Memento briefings are injected into messages to fight context rot
    - Llama gets simple tasks routed to it as grunt work
    - Claude, Codex, Gemini get the complex work
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
        """Route a message to its target agent."""
        target = self._agents.get(message.target)
        if not target:
            return AgentResponse(
                agent_name=message.target,
                content="",
                success=False,
                error=f"Unknown agent: {message.target}",
            )

        # Inject memento briefing into the message metadata so the
        # agent has survival notes even if earlier context is lost
        briefing = self._memento.briefing()
        if briefing:
            message = AgentMessage(
                source=message.source,
                target=message.target,
                content=message.content,
                metadata={**message.metadata, "memento": briefing},
                parent_message_id=message.parent_message_id,
            )

        response = await target.send(message)

        # Record key outcomes as memento notes
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
        """Route using compact protocol format instead of English.

        This is the preferred way for agents to talk to each other.
        The message is encoded as terse key-value pairs, not prose.
        """
        # Build compact message
        compact = protocol.task_message(
            task_type=task_type,
            description=description,
            context=self._memento.briefing(),
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
        """Send simple work to Llama. No thinking required."""
        msg = AgentMessage(source="orchestrator", target="llama", content=task)
        return await self.route(msg)

    async def broadcast(
        self, message: AgentMessage, targets: list[str] | None = None
    ) -> dict[str, AgentResponse]:
        """Send a message to multiple agents in parallel."""
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
            msg = AgentMessage(
                source=message.source,
                target=best_agent,
                content=message.content,
                metadata=message.metadata,
                parent_message_id=message.id,
            )
            return await self.route(msg)

        # Fallback: find any agent with the capability
        for name, agent in self._agents.items():
            if agent.supports(capability):
                msg = AgentMessage(
                    source=message.source,
                    target=name,
                    content=message.content,
                    metadata=message.metadata,
                    parent_message_id=message.id,
                )
                return await self.route(msg)

        return AgentResponse(
            agent_name="router",
            content="",
            success=False,
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
