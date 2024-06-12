from stable_baselines3 import PPO
from environment.federatedPPO import federatedPPO, stepPPO
from environment.env import Env
from environment.grc import GRC

from pathlib import Path
import json

import pickle

from main import DEFAULTARRIVALCONFIG, DEFAULTNETWORKPATH

DEFAULTSEED = 1 #Different from main.py since we here want determinism for testing

def evalPolicy(policyName: str):
    networkPath = DEFAULTNETWORKPATH

    with open(Path(DEFAULTARRIVALCONFIG), "r") as file:
        arrival_config = json.load(file)
    arrival_config["seed"] = DEFAULTSEED

    env = Env(networkPath, arrival_config, takeStats=True)

    if "grc" not in policyName:
        agent = PPO.load(policyName, env=env)
    if "grc" in policyName:
        agent = GRC(env, 0.15, 0.00001)

    done = False
    obs = env.reset()
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)

    with open(f"./data/{policyName}StatKeeper.gpickle", 'wb') as f:
        pickle.dump(env.statKeeper, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    policyNames = [
        "./bandwidthRegularPolicy",
        "./bandwidthFederatedPolicy",
        "grc",
        
        #"./1250NewFederatedPolicy",
        #"./2500NewFederatedPolicy",
        #"./5000NewFederatedPolicy",
        #"./10000NewFederatedPolicy",
        #"./20000NewFederatedPolicy",

        #"./2500NewFederatedPolicy",
        #"./rewardFederatedPolicy",
    ]

    for policyName in policyNames:
        evalPolicy(policyName)
    