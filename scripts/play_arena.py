from jass.arena.arena import Arena
from jass.agents.agent_random_schieber import AgentRandomSchieber

from dl4g.agent import MyAgent

arena = Arena(nr_games_to_play=100)
arena.set_players(MyAgent(), AgentRandomSchieber(), MyAgent(), AgentRandomSchieber())

arena.play_all_games()

print(arena.points_team_0.sum(), arena.points_team_1.sum())
