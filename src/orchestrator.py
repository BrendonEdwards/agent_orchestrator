"""Fractal multi-agent orchestrator with cross-model QA.

Same pattern at every scale: decompose, delegate, QA, synthesize.
Each sub-swarm composes its own agent team. A different model type
reviews every piece of work against a strict checklist. Work gets
sent back if it doesn't pass.

Agents run via CLI subprocesses - uses your existing Pro subscriptions:
- Claude: `claude -p "prompt"` (Claude Pro subscription)
- Codex: `codex -q "prompt"` (ChatGPT Plus subscription)
- Gemini: `gemini -p "prompt"` (Gemini Advanced subscription)
- Llama: Groq API (free tier, the one exception)

Max depth capped at 3 by default. Each depth level adds ~20-30s of
latency from CLI subprocess overhead, so 3 levels keeps wall-clock
time reasonable while still allowing meaningful decomposition.
"""

from __future__ import annotations

import asyncio
from typing import Any

from src.agents.base import AgentMessage, AgentResponse, BaseAgent
from src.agents.claude_agent import ClaudeAgent
from src.agents.codex_agent import CodexAgent
from src.agents.gemini_agent import GeminiAgent
from src.agents.llama_agent import LlamaAgent
from src.memento.memento import Memento
from src.routing.router import MessageRouter
from src.rules.loader import RulesLoader


# Simple tasks that should go to Llama
_GRUNT_KEYWORDS = [
    "format", "convert", "list", "sort", "extract", "template",
    "boilerplate", "rename", "reorder", "cleanup", "prettify",
]

# Keywords that signal which agent type a subtask needs
_AGENT_SIGNALS = {
    "codex": ["code", "program", "function", "implement", "debug", "algorithm",
              "class", "module", "api", "endpoint", "test", "refactor"],
    "gemini": ["image", "picture", "photo", "audio", "sound", "video",
               "visual", "diagram", "describe image", "transcribe"],
    "llama": ["format", "convert", "sort", "list", "template", "boilerplate",
              "cleanup", "extract", "rename"],
    "claude": ["design", "architect", "plan", "analyze", "review", "explain",
               "reason", "compare", "evaluate", "synthesize"],
}

# Cross-model QA pairings: worker -> reviewer (different model = different blind spots)
_QA_PAIRINGS = {
    "claude": "codex",    # Claude's work reviewed by Codex
    "codex": "claude",    # Codex's work reviewed by Claude
    "gemini": "claude",   # Gemini's work reviewed by Claude
    "llama": None,        # Grunt work doesn't need QA
}

# Max times work can be sent back before accepting
_MAX_QA_RETRIES = 2

# QA pass threshold (out of 10)
_QA_PASS_SCORE = 7


class QAResult:
    """Result of a cross-model QA review."""
    __slots__ = ("passed", "score", "feedback", "checklist_scores", "reviewer")

    def __init__(
        self,
        passed: bool,
        score: int,
        feedback: str,
        checklist_scores: dict[str, bool],
        reviewer: str,
    ):
        self.passed = passed
        self.score = score
        self.feedback = feedback
        self.checklist_scores = checklist_scores
        self.reviewer = reviewer


class Orchestrator:
    """Fractal multi-agent orchestrator with cross-model QA.

    Flow at every scale:
    1. Task comes in
    2. Orchestrator generates a strict checklist
    3. Agent does the work
    4. A DIFFERENT model reviews against the checklist
    5. If it fails QA -> work goes back with feedback
    6. If it passes -> result bubbles up

    Sub-swarms compose their own agent teams:

        "build a web app with image upload"
                    |
            Orchestrator (d=0)
                    |
            Claude decomposes + generates checklists:
            ┌───────┼──────────┐
            |       |          |
         code     images    boilerplate
            |       |          |
        Codex     Gemini      Llama
        does it   does it     (no QA)
            |       |
        Claude    Codex       <- different model reviews
        reviews   reviews
            |       |
        pass/fail pass/fail
            |       |
        retry or  retry or
        accept    accept
    """

    def __init__(
        self,
        rules_path: str | None = None,
        memento_path: str | None = None,
        claude_cli: str | None = None,
        codex_cli: str | None = None,
        gemini_cli: str | None = None,
        llama_provider: str = "groq",
        depth: int = 0,
        max_depth: int = 3,
        agents: dict[str, BaseAgent] | None = None,
        qa_enabled: bool = True,
        max_qa_retries: int = _MAX_QA_RETRIES,
        qa_pass_score: int = _QA_PASS_SCORE,
    ):
        self.depth = depth
        self.max_depth = max_depth
        self.qa_enabled = qa_enabled
        self.max_qa_retries = max_qa_retries
        self.qa_pass_score = qa_pass_score

        # Config for spawning children
        self._config = {
            "rules_path": rules_path,
            "claude_cli": claude_cli,
            "codex_cli": codex_cli,
            "gemini_cli": gemini_cli,
            "llama_provider": llama_provider,
        }

        self.memento = Memento(persist_path=memento_path)
        self.rules = RulesLoader(rules_path=rules_path)
        self.rules.load()

        # Agents: either custom (from fractal spawn) or default team
        if agents:
            self._agents = agents
        else:
            self._agents = self._build_default_team()

        # Always need a claude reference for decomposition/synthesis/checklist
        self._claude = self._find_agent(ClaudeAgent) or ClaudeAgent(cli_path=claude_cli)
        self._llama = self._find_agent(LlamaAgent)

        self.router = MessageRouter(
            agents=self._agents,
            rules_loader=self.rules,
            memento=self.memento,
        )

    def _build_default_team(self) -> dict[str, BaseAgent]:
        """Default team: one of each."""
        return {
            "claude": ClaudeAgent(cli_path=self._config["claude_cli"]),
            "codex": CodexAgent(cli_path=self._config["codex_cli"]),
            "gemini": GeminiAgent(cli_path=self._config["gemini_cli"]),
            "llama": LlamaAgent(provider=self._config["llama_provider"]),
        }

    def _find_agent(self, agent_type: type) -> BaseAgent | None:
        """Find the first agent of a given type in the team."""
        for agent in self._agents.values():
            if isinstance(agent, agent_type):
                return agent
        return None

    def _build_team_for(self, subtask: str) -> dict[str, BaseAgent]:
        """Build a custom agent team weighted for a specific subtask."""
        task_lower = subtask.lower()

        scores = {}
        for agent_name, keywords in _AGENT_SIGNALS.items():
            score = sum(1 for kw in keywords if kw in task_lower)
            scores[agent_name] = score

        team: dict[str, BaseAgent] = {
            "claude": ClaudeAgent(cli_path=self._config["claude_cli"]),
            "llama": LlamaAgent(provider=self._config["llama_provider"]),
        }

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best]

        team["codex"] = CodexAgent(cli_path=self._config["codex_cli"])
        team["gemini"] = GeminiAgent(cli_path=self._config["gemini_cli"])

        if best_score >= 2 and best not in ("claude", "llama"):
            for i in range(min(best_score - 1, 3)):
                if best == "codex":
                    team[f"{best}_{i + 2}"] = CodexAgent(cli_path=self._config["codex_cli"])
                else:
                    team[f"{best}_{i + 2}"] = GeminiAgent(cli_path=self._config["gemini_cli"])

        return team

    # ── Checklist generation ────────────────────────────────────────

    async def _generate_checklist(self, task: str) -> list[str]:
        """Claude generates a strict checklist BEFORE work begins.

        The checklist is what the agent will be scored against.
        It gets stored in memento so the worker sees it too.
        """
        msg = AgentMessage(
            source="orchestrator",
            target="claude",
            memento=self.memento.briefing(),
            content=(
                "Generate a strict QA checklist for this task. "
                "Return 3-7 concrete, verifiable criteria as a numbered list. "
                "Each item must be pass/fail - no subjective judgments. "
                "Focus on correctness, completeness, and edge cases.\n\n"
                f"Task: {task}"
            ),
        )
        response = await self._claude.send(msg)
        if not response.success:
            return [f"Task completed correctly: {task}"]

        items = []
        for line in response.content.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            for prefix in [".", ")", "- ", "* "]:
                idx = line.find(prefix)
                if idx != -1 and idx < 4:
                    line = line[idx + len(prefix):].strip()
                    break
            if line:
                items.append(line)

        return items if items else [f"Task completed correctly: {task}"]

    # ── Cross-model QA ──────────────────────────────────────────────

    def _pick_reviewer(self, worker_name: str) -> BaseAgent | None:
        """Pick a reviewer that's a DIFFERENT model type than the worker.

        Different models have different blind spots. Claude won't catch
        what another Claude would miss. But Codex might.
        """
        # Normalize: codex_2, codex_3 etc. -> codex
        base_name = worker_name.split("_")[0]
        reviewer_name = _QA_PAIRINGS.get(base_name)

        if not reviewer_name:
            return None

        # Find the reviewer in our team, or create one
        for name, agent in self._agents.items():
            if name.split("_")[0] == reviewer_name:
                return agent

        # Reviewer type not in team - create ephemeral one
        if reviewer_name == "claude":
            return ClaudeAgent(cli_path=self._config["claude_cli"])
        elif reviewer_name == "codex":
            return CodexAgent(cli_path=self._config["codex_cli"])
        elif reviewer_name == "gemini":
            return GeminiAgent(cli_path=self._config["gemini_cli"])
        return None

    async def _qa_review(
        self,
        task: str,
        work: str,
        checklist: list[str],
        worker_name: str,
    ) -> QAResult:
        """A different model reviews work against the checklist.

        Returns pass/fail, score, and specific feedback per checklist item.
        """
        reviewer = self._pick_reviewer(worker_name)
        if not reviewer:
            return QAResult(
                passed=True, score=10, feedback="No QA reviewer available",
                checklist_scores={}, reviewer="none",
            )

        checklist_text = "\n".join(f"{i+1}. {item}" for i, item in enumerate(checklist))
        msg = AgentMessage(
            source="orchestrator",
            target=reviewer.name,
            memento=self.memento.briefing(),
            content=(
                "You are a QA reviewer. A different AI model produced the work below. "
                "Score it against EACH checklist item (PASS or FAIL). "
                "Then give an overall score 1-10 and specific feedback on what to fix.\n\n"
                "IMPORTANT: Be strict. Different models have different blind spots - "
                "that's why YOU are reviewing, not the original model.\n\n"
                f"Original task: {task}\n\n"
                f"Checklist:\n{checklist_text}\n\n"
                f"Work to review:\n{work}\n\n"
                "Format your response EXACTLY as:\n"
                "ITEM_1: PASS or FAIL\n"
                "ITEM_2: PASS or FAIL\n"
                "...\n"
                "SCORE: N/10\n"
                "FEEDBACK: specific things to fix (or 'none' if all pass)"
            ),
        )
        response = await reviewer.send(msg)
        if not response.success:
            return QAResult(
                passed=True, score=10,
                feedback=f"QA reviewer failed: {response.error}",
                checklist_scores={}, reviewer=reviewer.name,
            )

        return self._parse_qa_response(response.content, checklist, reviewer.name)

    def _parse_qa_response(
        self, content: str, checklist: list[str], reviewer_name: str
    ) -> QAResult:
        """Parse the structured QA review response."""
        checklist_scores: dict[str, bool] = {}
        score = 10
        feedback = ""

        for line in content.strip().split("\n"):
            line = line.strip()
            upper = line.upper()

            # Parse ITEM_N: PASS/FAIL
            if upper.startswith("ITEM_"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    item_key = parts[0].strip()
                    passed = "PASS" in parts[1].upper()
                    # Map back to checklist item
                    try:
                        idx = int(item_key.replace("ITEM_", "")) - 1
                        if 0 <= idx < len(checklist):
                            checklist_scores[checklist[idx]] = passed
                    except ValueError:
                        pass

            # Parse SCORE: N/10
            elif upper.startswith("SCORE:"):
                try:
                    score_part = line.split(":")[1].strip()
                    score = int(score_part.split("/")[0].strip())
                except (ValueError, IndexError):
                    pass

            # Parse FEEDBACK:
            elif upper.startswith("FEEDBACK:"):
                feedback = line.split(":", 1)[1].strip()

        passed = score >= self.qa_pass_score
        return QAResult(
            passed=passed, score=score, feedback=feedback,
            checklist_scores=checklist_scores, reviewer=reviewer_name,
        )

    # ── Core execution with QA loop ─────────────────────────────────

    def spawn(self, subtask: str | None = None, inherit_notes: bool = True) -> Orchestrator:
        """Spawn a child orchestrator with a team built for the subtask."""
        if self.depth >= self.max_depth:
            raise RecursionError(
                f"Max depth ({self.max_depth}) reached. Cannot decompose further."
            )

        team = self._build_team_for(subtask) if subtask else None

        child = Orchestrator(
            **self._config,
            depth=self.depth + 1,
            max_depth=self.max_depth,
            agents=team,
            qa_enabled=self.qa_enabled,
            max_qa_retries=self.max_qa_retries,
            qa_pass_score=self.qa_pass_score,
        )

        if inherit_notes:
            briefing = self.memento.briefing()
            if briefing:
                child.memento.note("parent_ctx", briefing, priority=3)

        return child

    async def run(self, task: str) -> str:
        """Run a task. Fractal + QA at every scale.

        1. Grunt work? -> Llama (no QA)
        2. Simple? -> agent does it, different model QAs it
        3. Complex? -> decompose, spawn sub-swarms, each gets QA'd
        4. Failed QA? -> work goes back with feedback, retry
        """
        self.memento.note("goal", task[:150], priority=3)

        if self._is_grunt_work(task):
            return await self._do_grunt(task)

        subtasks = await self._decompose(task)

        if len(subtasks) > 1 and self.depth < self.max_depth:
            return await self._run_fractal(subtasks)

        return await self._run_single_with_qa(task)

    async def _do_grunt(self, task: str) -> str:
        """Grunt work -> Llama, no QA."""
        self.memento.note("route", "llama:grunt", priority=1)
        if self._llama:
            msg = AgentMessage(source="orchestrator", target="llama", content=task)
            response = await self._llama.send(msg)
            if response.success:
                self.memento.note("done", response.content[:150], priority=2)
                return response.content
        return await self._run_single_with_qa(task)

    async def _run_single_with_qa(self, task: str) -> str:
        """Single agent does the work, different model QAs it, retry if needed."""
        primary = self.rules.get_best_agent_for(task) or "claude"
        if primary not in self._agents:
            primary = "claude"
        self.memento.note("route", f"{primary}:leaf", priority=1)

        # Step 1: Generate checklist BEFORE work begins
        checklist: list[str] = []
        if self.qa_enabled:
            checklist = await self._generate_checklist(task)
            checklist_text = " | ".join(checklist[:5])
            self.memento.note("checklist", checklist_text[:150], priority=2)

        # Step 2: Agent does the work (with checklist in the brief)
        work_task = task
        if checklist:
            cl_text = "\n".join(f"  {i+1}. {item}" for i, item in enumerate(checklist))
            work_task = (
                f"{task}\n\n"
                f"You MUST satisfy this checklist (you will be scored against it):\n"
                f"{cl_text}"
            )

        result = await self._execute_agent(primary, work_task)

        # Step 3: QA loop - different model reviews, retry if failed
        if self.qa_enabled and checklist:
            result = await self._qa_loop(task, result, checklist, primary)

        self.memento.note("done", result[:150], priority=2)
        return result

    async def _execute_agent(self, agent_name: str, task: str) -> str:
        """Execute a task on a specific agent."""
        msg = AgentMessage(source="orchestrator", target=agent_name, content=task)
        response = await self.router.route(msg)
        return response.content if response.success else f"Failed: {response.error}"

    async def _qa_loop(
        self, task: str, work: str, checklist: list[str], worker_name: str
    ) -> str:
        """QA loop: review, reject, retry until pass or max retries."""
        for attempt in range(self.max_qa_retries + 1):
            qa = await self._qa_review(task, work, checklist, worker_name)

            self.memento.note(
                f"qa_d{self.depth}",
                f"r={qa.reviewer} s={qa.score}/10 {'PASS' if qa.passed else 'FAIL'} "
                f"attempt={attempt + 1}",
                priority=2,
            )

            if qa.passed:
                return work

            if attempt < self.max_qa_retries:
                # Send work back with specific feedback
                failed_items = [
                    item for item, passed in qa.checklist_scores.items() if not passed
                ]
                failed_text = "\n".join(f"  - {item}" for item in failed_items)
                retry_task = (
                    f"Your previous work was reviewed by {qa.reviewer} and scored "
                    f"{qa.score}/10. It did not pass QA.\n\n"
                    f"Original task: {task}\n\n"
                    f"Failed checklist items:\n{failed_text}\n\n"
                    f"Reviewer feedback: {qa.feedback}\n\n"
                    f"Your previous work:\n{work}\n\n"
                    f"Fix the issues and resubmit. You MUST address every failed item."
                )
                work = await self._execute_agent(worker_name, retry_task)

        # Max retries exhausted - return best effort
        self.memento.note(
            f"qa_warn_d{self.depth}",
            f"max retries ({self.max_qa_retries}) exhausted, accepting best effort",
            priority=2,
        )
        return work

    async def _decompose(self, task: str) -> list[str]:
        """Claude breaks a task into independent subtasks, or returns it as-is."""
        msg = AgentMessage(
            source="orchestrator",
            target="claude",
            memento=self.memento.briefing(),
            content=(
                "Break this task into 2-5 independent subtasks that can run in parallel. "
                "Return ONLY a numbered list, one subtask per line. "
                "If the task is simple enough for one agent, return ONLY the task itself "
                "as a single line with no numbering.\n\n"
                f"Task: {task}"
            ),
        )
        response = await self._claude.send(msg)
        if not response.success:
            return [task]

        lines = response.content.strip().split("\n")
        subtasks = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            for prefix in [".", ")", "- ", "* "]:
                idx = line.find(prefix)
                if idx != -1 and idx < 4:
                    line = line[idx + len(prefix):].strip()
                    break
            if line:
                subtasks.append(line)

        return subtasks if subtasks else [task]

    async def _run_fractal(self, subtasks: list[str]) -> str:
        """Spawn sub-swarms, run in parallel, each gets QA'd, synthesize."""
        self.memento.note(
            "fractal",
            f"d={self.depth}->d={self.depth + 1}, {len(subtasks)} branches",
            priority=2,
        )

        async def run_branch(subtask: str) -> str:
            child = self.spawn(subtask=subtask, inherit_notes=True)
            return await child.run(subtask)

        results = await asyncio.gather(
            *(run_branch(st) for st in subtasks),
            return_exceptions=True,
        )

        responses = [
            AgentResponse(
                agent_name=f"d{self.depth + 1}_{i}",
                content=str(r) if not isinstance(r, Exception) else "",
                success=not isinstance(r, Exception),
                error=str(r) if isinstance(r, Exception) else None,
            )
            for i, r in enumerate(results)
        ]

        successful = [r for r in responses if r.success]
        if not successful:
            return "All branches failed."

        if len(successful) == 1:
            return successful[0].content

        synthesis = await self._claude.synthesize(
            successful, memento=self.memento.briefing()
        )
        self.memento.note("done", synthesis.content[:150], priority=2)
        return synthesis.content

    async def send_to(self, agent_name: str, content: str) -> AgentResponse:
        """Send directly to a specific agent."""
        msg = AgentMessage(source="user", target=agent_name, content=content)
        return await self.router.route(msg)

    async def grunt(self, task: str) -> str:
        """Grunt work -> Llama."""
        if self._llama:
            msg = AgentMessage(source="orchestrator", target="llama", content=task)
            resp = await self._llama.send(msg)
            return resp.content if resp.success else ""
        return ""

    async def health_check(self) -> dict[str, bool]:
        return await self.router.health_check_all()

    def notes(self) -> str:
        return self.memento.briefing()

    @property
    def team(self) -> list[str]:
        """Who's on this orchestrator's team."""
        return list(self._agents.keys())

    def _is_grunt_work(self, task: str) -> bool:
        task_lower = task.lower()
        return any(kw in task_lower for kw in _GRUNT_KEYWORDS)
