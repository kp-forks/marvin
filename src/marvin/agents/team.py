import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from pydantic_ai.result import RunResult

from marvin.agents.actor import Actor
from marvin.agents.names import TEAM_NAMES
from marvin.memory.memory import Memory
from marvin.prompts import Template

if TYPE_CHECKING:
    from marvin.engine.end_turn import EndTurn
    from marvin.engine.llm import Message
    from marvin.engine.orchestrator import Orchestrator


@dataclass(kw_only=True)
class Team(Actor):
    """A team is a container that maintains state for a group of agents."""

    members: list[Actor] = field(kw_only=False, repr=False)
    active_member: Actor = field(init=False, repr=False)

    name: str = field(
        default_factory=lambda: random.choice(TEAM_NAMES),
        metadata={"description": "Name of the team"},
    )

    prompt: str | Path = field(
        default=Path("team.jinja"),
        metadata={"description": "Template for the team's prompt"},
        repr=False,
    )

    delegates: dict[Actor, list[Actor]] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if not self.members:
            raise ValueError("Team must have at least one member")
        self.active_member = self.members[0]

    async def start_turn(self, orchestrator: "Orchestrator"):
        await self.active_member.start_turn(orchestrator=orchestrator)

    async def end_turn(
        self,
        orchestrator: "Orchestrator",
        result: RunResult,
    ):
        await self.active_member.end_turn(result=result, orchestrator=orchestrator)

    def get_prompt(self) -> str:
        return Template(source=self.prompt).render(team=self)

    def get_memories(self) -> list[Memory]:
        return self.active_member.get_memories()

    async def _run(
        self,
        messages: list["Message"],
        tools: list[Callable[..., Any]],
        end_turn_tools: list["EndTurn"],
    ) -> RunResult:
        return await self.active_member._run(
            messages=messages,
            tools=self.get_tools() + tools,
            end_turn_tools=self.get_end_turn_tools() + end_turn_tools,
        )

    def friendly_name(self, verbose: bool = True) -> str:
        return self.active_member.friendly_name(verbose=verbose)

    def get_end_turn_tools(self) -> list["EndTurn"]:
        from marvin.engine.end_turn import create_delegate_to_actor

        end_turn_tools = super().get_end_turn_tools()

        for delegate in self.delegates.get(self.active_member, []):
            end_turn_tools.append(
                create_delegate_to_actor(delegate_actor=delegate, team=self)
            )

        return end_turn_tools


@dataclass(kw_only=True)
class Swarm(Team):
    """A swarm is a team that permits all agents to delegate to each other."""

    def __post_init__(self):
        super().__post_init__()
        if not self.delegates:
            self.delegates = {
                member: [m for m in self.members if m is not member]
                for member in self.members
            }


@dataclass(kw_only=True)
class RoundRobinTeam(Team):
    description: str | None = "A team of agents that rotate turns."

    async def start_turn(self, orchestrator: "Orchestrator"):
        index = self.members.index(self.active_member)
        self.active_member = self.members[(index + 1) % len(self.members)]
        await super().start_turn(orchestrator=orchestrator)


@dataclass(kw_only=True)
class RandomTeam(Team):
    description: str | None = "A team of agents that randomly selects an agent to act."

    async def start_turn(self, orchestrator: "Orchestrator"):
        self.active_member = random.choice(self.members)
        await super().start_turn(orchestrator=orchestrator)
