import logging
import argparse
from pathlib import Path

import stable_baselines3
from stable_baselines3.common.callbacks import EvalCallback

from environment.env import Env
from environment.arrival import *
from environment.monitor import EvalLogCallback, StatsWrapper

from stable_baselines3 import PPO
from environment.federatedPPO import federatedPPO, stepPPO

from gym.utils.env_checker import check_env

from typing import List, Union

DEFAULTTOTALTIMESTEPS = 1000000 / 2 #TODO
DEFAULTNETWORKPATH = "data/network.gpickle"
DEFAULTNEVALEPISODES = 5
DEFAULTEVALFREQ = 10000
DEFAULTOUTPUT = "output"
DEFAULTDEBUGGING = False
DEFAULTARRIVALCONFIG = "data/requests.json"
DEFAULTSEED = None
DEFAULTFEDERATED = True

def main(
        totalTimesteps: int = DEFAULTTOTALTIMESTEPS,
        networkPath: str = DEFAULTNETWORKPATH,
        nEvalEpisodes: int = DEFAULTNEVALEPISODES,
        evalFreq: int = DEFAULTEVALFREQ,
        output: str = DEFAULTOUTPUT,
        debug: bool = DEFAULTDEBUGGING,
        arrivalConfig: str = DEFAULTARRIVALCONFIG,
        seed: int = DEFAULTSEED,
        federated: bool = DEFAULTFEDERATED
) -> None:
    
    logging.basicConfig()
    debug_level = logging.DEBUG if debug else logging.INFO
    logging.getLogger().setLevel(debug_level)

    Path(f"{output}/logs").mkdir(exist_ok=True, parents=True)

    with open(Path(arrivalConfig), "r") as file:
        arrival_config = json.load(file)

    arrival_config["seed"] = seed

    groundNetworkPath = networkPath[:-8] + "_ground" + networkPath[-8:]
    if Path.exists(Path(groundNetworkPath)) and federated: #networkPath[-8:] corresponds to the .gpickle suffix
        domains = ["space", "air", "ground"]
        domainModels = []

        for _, domain in enumerate(domains):
            domainNetworkPath = networkPath[:-8] + f"_{domain}" + networkPath[-8:]

            env = Env(domainNetworkPath, arrival_config)

            domainModels.append(createPPOAgent(env, f"{output}/logs"))

    env = Env(networkPath, arrival_config)
    #check_env(env)

    evalEnv = StatsWrapper(Env(networkPath, arrival_config))
    evalLogCallback = EvalLogCallback()
    evalCallback = EvalCallback(
        evalEnv,
        n_eval_episodes=nEvalEpisodes, #TODO: Figure this out
        eval_freq=evalFreq, #Evaluation is done each evalFreq timesteps
        deterministic=False,
        render=False,
        callback_after_eval=evalLogCallback,
    )

    if federated:
        federatedAgent = createPPOAgent(env, f"{output}/logs", federated=True)

        federatedAgent.learn(
            total_timesteps=totalTimesteps,
            tb_log_name="federatedPPO",
            callback=evalCallback,
            models=domainModels
        )
    elif not federated:
        
        agent = createPPOAgent(env, f"{output}/logs", federated=False, standalone=True)

        agent.learn(
            total_timesteps=totalTimesteps,
            tb_log_name="PPO",
            callback=evalCallback
        )


def createPPOAgent(
        env: Env,
        output: str = DEFAULTOUTPUT,
        federated: bool = False,
        standalone: bool = False
) -> Union[PPO, stepPPO, federatedPPO]:

    parameters = {
                "policy": "MlpPolicy", #This policy is best suited for the continous state in this task
                "env": env,
                "verbose": 1,
                #"learning_rate": 0.003, #Value 0.005 originates from AG
                "clip_range": 0.2, #Default value, AG suggested maybe 0.3?
                "n_epochs": 200, #100 originates from AG, 200 from the enhanced paper
                "batch_size": 128, #64 is default, AG suggested trying 100 or doubling (128)
                "tensorboard_log": output
                }

    if not federated and not standalone:
        del parameters["tensorboard_log"]
        return stepPPO(**parameters)
    elif not federated and standalone:
        return stepPPO(**parameters)
    elif federated:
        return federatedPPO(**parameters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--totalTimesteps", type=int, default=DEFAULTTOTALTIMESTEPS)
    parser.add_argument("--networkPath", type=str, default=DEFAULTNETWORKPATH)
    parser.add_argument("--nEvalEpisodes", type=int, default=DEFAULTNEVALEPISODES)
    parser.add_argument("--evalFreq", type=int, default=DEFAULTEVALFREQ)
    parser.add_argument("--output", type=str, default=DEFAULTOUTPUT)
    parser.add_argument("--debug", type=bool, default=DEFAULTDEBUGGING) #This works as long as default is "False"
    parser.add_argument("--arrivalConfig", type=str, default=DEFAULTARRIVALCONFIG)
    parser.add_argument("--seed", type=int, default=DEFAULTSEED)
    parser.add_argument("--notFederated", action='store_true') #Flag

    args = parser.parse_args()

    main(
        args.totalTimesteps,
        args.networkPath,
        args.nEvalEpisodes,
        args.evalFreq,
        args.output,
        args.debug,
        args.arrivalConfig,
        args.seed,
        (not args.notFederated)
    )
