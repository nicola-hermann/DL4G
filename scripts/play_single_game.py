from jass.arena.arena import Arena

from dl4g.agent import MyAgent
from jass.game.rule_schieber import RuleSchieber
from jass.game.game_sim import GameSim
from jass.game.const import NORTH
from jass.game.game_util import deal_random_hand


rule = RuleSchieber()
game = GameSim(rule=rule)
agent = MyAgent()
game.init_from_cards(hands=deal_random_hand(), dealer=NORTH)

obs = game.get_observation()
trump = agent.action_trump(obs)
game.action_trump(trump)

while not game.is_done():
    game.action_play_card(agent.action_play_card(game.get_observation()))

print(game.state.points)
