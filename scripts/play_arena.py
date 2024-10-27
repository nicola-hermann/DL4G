from jass.arena.arena import Arena
from jass.agents.agent_random_schieber import AgentRandomSchieber

from dl4g.agents.baseline_agent import BaselineAgent, TrumpSelectionAgent

arena = Arena(nr_games_to_play=1000)
# Create a txt file for logs
with open("log.txt", "w") as f:
    f.write("Log file for the arena\n")


arena.set_players(
    BaselineAgent(), TrumpSelectionAgent(), TrumpSelectionAgent(), TrumpSelectionAgent()
)

arena.play_all_games()

print(arena.points_team_0.sum(), arena.points_team_1.sum())
