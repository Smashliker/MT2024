from stable_baselines3 import PPO
from environment.federatedPPO import federatedPPO, stepPPO
from environment.env import Env
from environment.grc import GRC

from pathlib import Path
import json

import pickle

from main import DEFAULTARRIVALCONFIG, DEFAULTNETWORKPATH

DEFAULTSEED = 0 #Different from main.py since we here want determinism for testing

if __name__ == "__main__":

    networkPath = DEFAULTNETWORKPATH

    with open(Path(DEFAULTARRIVALCONFIG), "r") as file:
        arrival_config = json.load(file)
    arrival_config["seed"] = DEFAULTSEED

    env = Env(networkPath, arrival_config, takeStats=True)

    #policy = "./5MfederatedPolicy"
    policy = "grc"

    #agent = PPO.load(policy, env=env)
    agent = GRC(env, 0.5, 0.01)

    done = False
    obs = env.reset()
    while not done:
        action, _ = agent.predict(obs)
        obs, _, done, _ = env.step(action)

    with open(f"./data/{policy}StatKeeper.gpickle", 'wb') as f:
        pickle.dump(env.statKeeper, f, pickle.HIGHEST_PROTOCOL)