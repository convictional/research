"""Base protocol for player harnesses."""

from typing import Protocol

from alignsim.src.models.actions import TurnActions
from alignsim.src.models.game_state import GameState
from alignsim.src.models.goals import GoalAttainmentScore
from alignsim.src.models.observations import TurnObservation


class PlayerHarness(Protocol):
    """Protocol defining the interface between the game engine and a player."""

    async def on_game_start(self, scenario_info: dict) -> None:
        """Called once at the start of the game with public scenario information."""
        ...

    async def decide(self, observation: TurnObservation, state_summary: dict) -> TurnActions:
        """Given the current observation and state summary, return actions for this turn."""
        ...

    async def on_game_end(self, score: GoalAttainmentScore, state: GameState) -> None:
        """Called once at the end of the game with final results."""
        ...
