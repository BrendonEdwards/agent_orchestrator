"""Tests for the message router."""

import pytest

from src.agents.base import AgentCapability, AgentMessage
from src.memento.memento import Memento
from src.routing.router import MessageRouter
from src.rules.loader import RulesLoader


@pytest.mark.asyncio
async def test_route_to_known_agent(fake_team):
    router = MessageRouter(
        agents=fake_team,
        rules_loader=RulesLoader(),
        memento=Memento(),
    )
    msg = AgentMessage(source="user", target="claude", content="hello")
    resp = await router.route(msg)
    assert resp.success
    assert resp.content == "synthesized result"


@pytest.mark.asyncio
async def test_route_to_unknown_agent(fake_team):
    router = MessageRouter(
        agents=fake_team,
        rules_loader=RulesLoader(),
        memento=Memento(),
    )
    msg = AgentMessage(source="user", target="nonexistent", content="hello")
    resp = await router.route(msg)
    assert not resp.success
    assert "Unknown agent" in resp.error


@pytest.mark.asyncio
async def test_memento_injected(fake_team):
    mem = Memento()
    mem.note("goal", "test injection")
    router = MessageRouter(
        agents=fake_team,
        rules_loader=RulesLoader(),
        memento=mem,
    )
    msg = AgentMessage(source="user", target="claude", content="hello")
    await router.route(msg)
    # The agent should have received a message with memento set
    assert len(fake_team["claude"].calls) == 1
    sent_msg = fake_team["claude"].calls[0]
    assert "goal=test injection" in sent_msg.memento


@pytest.mark.asyncio
async def test_broadcast(fake_team):
    router = MessageRouter(
        agents=fake_team,
        rules_loader=RulesLoader(),
        memento=Memento(),
    )
    msg = AgentMessage(source="user", target="all", content="hello everyone")
    results = await router.broadcast(msg)
    assert len(results) == 3
    assert all(r.success for r in results.values())


@pytest.mark.asyncio
async def test_delegate_grunt_work(fake_team):
    router = MessageRouter(
        agents=fake_team,
        rules_loader=RulesLoader(),
        memento=Memento(),
    )
    resp = await router.delegate_grunt_work("format this json")
    assert resp.success
    assert resp.content == "formatted output"


@pytest.mark.asyncio
async def test_health_check_all(fake_team):
    router = MessageRouter(
        agents=fake_team,
        rules_loader=RulesLoader(),
        memento=Memento(),
    )
    results = await router.health_check_all()
    assert len(results) == 3
    assert all(results.values())
