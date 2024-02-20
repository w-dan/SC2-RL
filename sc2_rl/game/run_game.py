import argparse
import sys
import time

import aioredis
import cv2
import numpy as np
from sc2 import maps  # maps method for loading maps to play in.
from sc2.data import Difficulty, Race
from sc2.main import (  # function that facilitates actually running the agents in games
    run_game,
)
from sc2.player import (  # wrapper for whether or not the agent is one of your bots, or a "computer" player
    Bot,
    Computer,
)

# from sc2_rl.bots.artanis_bot import ArtanisBot

# redis = aioredis.from_url(
#     "redis://localhost:6379", encoding="utf-8", decode_responses=True
# )

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--map", dest="map_name", required=True)
parser.add_argument("-v", "--verbose", type=int, required=True)
args = parser.parse_args()

ARTANIS = ArtanisBot(args.verbose)

map_sc2 = maps.get(args.map_name)
players = [
    Bot(ArtanisBot.RACE, ai=ARTANIS, name="Artanis"),
    Computer(Race.Zerg, Difficulty.Hard),
]

# result = run_game(map_sc2, players, realtime=False)

if str(result) == "Result.Victory":
    rwd = 500
else:
    rwd = -500

with open("results.txt", "a") as f:
    f.write(f"{result}\n")

map_state = np.zeros((224, 224, 3), dtype=np.uint8)

state = {
    "state": map_state,
    "micro-reward": rwd,
    "info": {
        "iteration": 0,
        "n_VOIDRAY": 0,
    },
}
redis.rpush("state_queue", state)

cv2.destroyAllWindows()
cv2.waitKey(1)
time.sleep(3)
sys.exit()
