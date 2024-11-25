from jass.arena.arena import Arena
from jass.agents.agent_random_schieber import AgentRandomSchieber

from dl4g.agents.baseline_agent import BaselineAgent
from dl4g.agents.rl_agent import RLAgent
from dl4g.rl_ppo import PPOPolicy

arena = Arena(nr_games_to_play=1000)
# Create a txt file for logs
with open("log.txt", "w") as f:
    f.write("Log file for the arena\n")


arena.set_players(
    RLAgent(model=PPOPolicy(84, 36, 128), weights="ppo_model.pth"),
    BaselineAgent(),
    RLAgent(model=PPOPolicy(84, 36, 128), weights="ppo_model.pth"),
    BaselineAgent(),
)

arena.play_all_games()

print(arena.points_team_0.sum(), arena.points_team_1.sum())
