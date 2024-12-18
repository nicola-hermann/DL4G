from jass.arena.arena import Arena
from jass.agents.agent_random_schieber import AgentRandomSchieber

from dl4g.agents.baseline_agent import BaselineAgent
from dl4g.agents.ismcts_agent import ISMCTSAgent


# Create a txt file for logs
with open("log.txt", "w") as f:
    f.write("Log file for the arena\n")


arena = Arena(nr_games_to_play=10)
arena.set_players(
    ISMCTSAgent(0.3, 2),
    AgentRandomSchieber(),
    ISMCTSAgent(0.3, 2),
    AgentRandomSchieber(),
)

arena.play_all_games()
print(arena.points_team_0.sum(), arena.points_team_1.sum())
