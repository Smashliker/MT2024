import logging
import argparse
from pathlib import Path

#import stable_baselines3
from stable_baselines3.common.callbacks import EvalCallback

from environment.env import Env
from environment.arrival import *
from environment.monitor import EvalLogCallback, StatsWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from environment.federatedPPO import federatedPPO, stepPPO
from stable_baselines3.common.vec_env import DummyVecEnv

#from gym.utils.env_checker import check_env

from typing import List, Union

#import multiprocessing
from threading import Thread, Lock
import sys

import time

DEFAULTTOTALTIMESTEPS = 3000000 / 2
DEFAULTNETWORKPATH = "data/network.gpickle"
DEFAULTNEVALEPISODES = 5
DEFAULTEVALFREQ = 10000
DEFAULTOUTPUT = "output"
DEFAULTDEBUGGING = False
DEFAULTARRIVALCONFIG = "data/requests.json"
DEFAULTSEED = None #NOTE: It is important that this seed is not set for the sake of the training of the federated model. 
DEFAULTFEDERATED = True
DEFAULTVERBOSE = 1

def main(
        totalTimesteps: int = DEFAULTTOTALTIMESTEPS,
        networkPath: str = DEFAULTNETWORKPATH,
        nEvalEpisodes: int = DEFAULTNEVALEPISODES,
        evalFreq: int = DEFAULTEVALFREQ,
        output: str = DEFAULTOUTPUT,
        debug: bool = DEFAULTDEBUGGING,
        arrivalConfig: str = DEFAULTARRIVALCONFIG,
        seed: int = DEFAULTSEED,
        federated: bool = DEFAULTFEDERATED,
        verbose: int = DEFAULTVERBOSE
) -> None:
    
    logging.basicConfig()
    debug_level = logging.DEBUG if debug else logging.INFO
    logging.getLogger().setLevel(debug_level)

    Path(f"{output}/logs").mkdir(exist_ok=True, parents=True)

    with open(Path(arrivalConfig), "r") as file:
        arrival_config = json.load(file)

    arrival_config["seed"] = seed

    """
    groundNetworkPath = networkPath[:-8] + "_ground" + networkPath[-8:]
    if Path.exists(Path(groundNetworkPath)) and federated: #networkPath[-8:] corresponds to the .gpickle suffix
        domains = ["space", "air", "ground"]
        domainModels = []

        for _, domain in enumerate(domains):
            domainNetworkPath = networkPath[:-8] + f"_{domain}" + networkPath[-8:]

            env = Env(domainNetworkPath, arrival_config)

            domainModels.append(createPPOAgent(env, f"{output}/logs"))
    """

    env = Env(networkPath, arrival_config)
    #check_env(env)
    if verbose >= 1:
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])

    evalEnv = StatsWrapper(Env(networkPath, arrival_config))
    evalLogCallback = EvalLogCallback()
    evalCallback = EvalCallback(
        evalEnv,
        n_eval_episodes=nEvalEpisodes,
        eval_freq=evalFreq, #Evaluation is done each evalFreq timesteps
        deterministic=False,
        render=False,
        callback_after_eval=evalLogCallback,
    )

    if federated:
        federatedAgents: List[federatedPPO] = []
        for domain in ["space", "air", "ground"]:
            federatedAgents.append(createPPOAgent(env, f"{output}/logs/{domain}", federated=True, verbose=verbose))
        
        lock = Lock()

        for federatedIndex, federatedAgent in enumerate(federatedAgents):
            federatedAgent.lock = lock
            federatedAgent.domain = federatedIndex
            federatedAgent.models = federatedAgents
        
        parameters = {
            "total_timesteps": totalTimesteps,
            "tb_log_name": "federatedPPO",
            #"callback": evalCallback,
            "models": federatedAgents
        }
        
        """
        with multiprocessing.Pool() as pool:
            for federatedAgent in federatedAgents:
                pool.apply(startLearning, [federatedAgent, parameters])
        """

        threads: List[Thread] = []
        for federatedIndex, federatedAgent in enumerate(federatedAgents):
            #processes.append(multiprocessing.Process(target=federatedAgent.learn, kwargs=parameters))
            if federatedIndex == 2:
                parameters["callback"] = evalCallback
            threads.append(Thread(target=federatedAgent.learn, kwargs=parameters))

        #https://stackoverflow.com/a/529427
        for threadIndex, thread in reversed(list(enumerate(threads))):
            if threadIndex < len(threads) - 1:
                thread.daemon = True
            thread.start()

        threads[-1].join()
        
        #federatedAgents[-1].save("./federatedPolicy", exclude=["lock"])

        federated_parameters=federatedAgents[-1].get_parameters()

        saveAgent = createPPOAgent(env, f"{output}/logs", federated=False, standalone=True, verbose=verbose)

        saveAgent.set_parameters(federated_parameters)

        saveAgent.save("./federatedPolicy")

    elif not federated:
        
        agent = createPPOAgent(env, f"{output}/logs", federated=False, standalone=True, verbose=verbose)

        agent.learn(
            total_timesteps=totalTimesteps,
            tb_log_name="PPO",
            callback=evalCallback
        )

        agent.save("./regularPolicy")


def createPPOAgent(
        env: Env,
        output: str = DEFAULTOUTPUT,
        federated: bool = False,
        standalone: bool = False,
        verbose: int = DEFAULTVERBOSE
) -> Union[PPO, stepPPO, federatedPPO]:

    parameters = {
                "policy": "MlpPolicy", #This policy is best suited for the continous state in this task
                "env": env,
                "verbose": verbose,
                "learning_rate": 0.0005,
                "clip_range": 0.3, 
                "n_epochs": 100, 
                "batch_size": 512,
                "tensorboard_log": output,
                "gamma": 0.99,
                #"n_steps": 4096
                }

    if not federated and not standalone:
        del parameters["tensorboard_log"]
        return stepPPO(**parameters)
    elif not federated and standalone:
        return PPO(**parameters)
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
    parser.add_argument("--verbose", type=int, default=DEFAULTVERBOSE)

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
        (not args.notFederated),
        args.verbose
    )
