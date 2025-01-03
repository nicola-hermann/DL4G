from jass.arena.arena import Arena
from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.agents.agent_by_network import AgentByNetwork

from dl4g.agents.baseline_agent import BaselineAgent
from dl4g.agents.better_baseline_agent import BetterBaselineAgent
from dl4g.agents.ismcts_agent import ISMCTSAgent


# Create a txt file for logs
with open("log.txt", "w") as f:
    f.write("Log file for the arena\n")

agent = ISMCTSAgent(1.3, 9)

arena = Arena(nr_games_to_play=10000)
arena.set_players(
    AgentRandomSchieber(),
    BetterBaselineAgent(),
    AgentRandomSchieber(),
    BetterBaselineAgent(),
)

arena.play_all_games()
print(arena.points_team_0.sum(), arena.points_team_1.sum())
