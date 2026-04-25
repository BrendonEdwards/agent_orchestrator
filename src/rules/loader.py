"""Rules loader for model strengths and routing preferences."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field


class ModelRules(BaseModel):
    """Rules and capabilities for a specific model or agent."""

    name: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    preferred_tasks: list[str] = Field(default_factory=list)
    avoid_tasks: list[str] = Field(default_factory=list)
    context_limit: int | None = None
    notes: str = ""


class RulesLoader:
    """Loads and parses model rules from a markdown file.

    The rules file documents model strengths, weaknesses and routing
    preferences. The main runtime capability mapping currently lives in
    `src.orchestrator.CAPABILITY_PROVIDERS`.
    """

    def __init__(self, rules_path: str | None = None):
        self._rules_path = Path(rules_path) if rules_path else None
        self._models: dict[str, ModelRules] = {}

    def load(self, path: str | None = None) -> dict[str, ModelRules]:
        """Load and parse the rules markdown file."""
        file_path = Path(path) if path else self._rules_path
        if not file_path or not file_path.exists():
            return self._get_defaults()

        content = file_path.read_text()
        return self._parse_markdown(content)

    def _parse_markdown(self, content: str) -> dict[str, ModelRules]:
        """Parse a markdown file into ModelRules objects."""
        models: dict[str, ModelRules] = {}
        current_model: str | None = None
        current_section: str | None = None

        for line in content.split("\n"):
            line = line.strip()

            model_match = re.match(r"^##\s+(.+)$", line)
            if model_match:
                current_model = model_match.group(1).strip().lower()
                models[current_model] = ModelRules(name=current_model)
                current_section = None
                continue

            if not current_model:
                continue

            section_match = re.match(r"^###\s+(.+)$", line)
            if section_match:
                current_section = section_match.group(1).strip().lower()
                continue

            item_match = re.match(r"^[-*]\s+(.+)$", line)
            if item_match and current_section and current_model in models:
                item = item_match.group(1).strip()
                model = models[current_model]
                if current_section == "strengths":
                    model.strengths.append(item)
                elif current_section == "weaknesses":
                    model.weaknesses.append(item)
                elif "preferred" in current_section or "best for" in current_section:
                    model.preferred_tasks.append(item)
                elif "avoid" in current_section:
                    model.avoid_tasks.append(item)

            ctx_match = re.match(r"^[-*]\s+context.limit:\s*(\d+)", line, re.IGNORECASE)
            if ctx_match and current_model in models:
                models[current_model].context_limit = int(ctx_match.group(1))

        self._models = models
        return models

    def _get_defaults(self) -> dict[str, ModelRules]:
        """Return default three-agent rules when no rules file is available."""
        self._models = {
            "claude": ModelRules(
                name="claude",
                strengths=["reasoning", "analysis", "synthesis", "conversation", "code review"],
                weaknesses=["image generation", "audio processing"],
                preferred_tasks=["task analysis", "response synthesis", "complex reasoning"],
            ),
            "codex": ModelRules(
                name="codex",
                strengths=["code generation", "code translation", "debugging", "refactoring"],
                weaknesses=["multimodal tasks", "creative writing"],
                preferred_tasks=["code writing", "language translation", "technical tasks"],
            ),
            "gemini": ModelRules(
                name="gemini",
                strengths=["multimodal understanding", "image analysis", "audio processing", "formatting"],
                weaknesses=["specialist code generation"],
                preferred_tasks=["image tasks", "audio tasks", "translation", "fast formatting"],
            ),
        }
        return self._models

    def get_best_agent_for(self, task_description: str) -> str | None:
        """Suggest the best agent for a task based on the parsed rules."""
        if not self._models:
            self.load()

        task_lower = task_description.lower()
        scores: dict[str, int] = {}

        for name, rules in self._models.items():
            score = 0
            for strength in rules.strengths:
                if any(word in task_lower for word in strength.lower().split()):
                    score += 2
            for preferred in rules.preferred_tasks:
                if any(word in task_lower for word in preferred.lower().split()):
                    score += 3
            for weakness in rules.weaknesses:
                if any(word in task_lower for word in weakness.lower().split()):
                    score -= 2
            for avoid in rules.avoid_tasks:
                if any(word in task_lower for word in avoid.lower().split()):
                    score -= 3
            scores[name] = score

        if not scores:
            return "claude"
        return max(scores, key=scores.get)  # type: ignore[arg-type]

    def get_routing_context(self) -> str:
        """Generate a text summary of rules for inclusion in routing prompts."""
        if not self._models:
            self.load()

        lines = []
        for name, rules in self._models.items():
            strengths = ", ".join(rules.strengths[:5])
            weaknesses = ", ".join(rules.weaknesses[:3])
            lines.append(f"- {name}: strengths=[{strengths}], weaknesses=[{weaknesses}]")
        return "\n".join(lines)
