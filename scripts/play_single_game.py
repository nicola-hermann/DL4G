from jass.agents.agent_by_network import AgentByNetwork
from jass.game.rule_schieber import RuleSchieber
from jass.game.game_sim import GameSim
from jass.game.const import NORTH
from jass.game.game_util import deal_random_hand
from dl4g.agents.ismcts_agent import ISMCTSAgent


rule = RuleSchieber()
game = GameSim(rule=rule)
# agent = AgentByNetwork(
#     "https://dl4g-64472410636.europe-west3.run.app/differenzler", timeout=10
# )
agent = ISMCTSAgent()
game.init_from_cards(hands=deal_random_hand(), dealer=NORTH)

obs = game.get_observation()
breakpoint()
trump = agent.action_trump(obs)
game.action_trump(trump)

while not game.is_done():
    obs = game.get_observation()
    print(obs.player)
    game.action_play_card(agent.action_play_card(obs))

print(game.state.points)
