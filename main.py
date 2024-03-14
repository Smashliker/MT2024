import logging
import argparse
from pathlib import Path

import stable_baselines3
from stable_baselines3.common.callbacks import EvalCallback

from environment.env import Env
from environment.arrival import *
from environment.monitor import EvalLogCallback, StatsWrapper

DEFAULTTOTALTIMESTEPS = 1000000
DEFAULTNETWORKPATH = "data/network.gpickle"
DEFAULTAGENTTYPE = "PPO"
DEFAULTNEVALEPISODES = 5
DEFAULTEVALFREQ = 10000
DEFAULTOUTPUT = "output"
DEFAULTDEBUGGING = False
DEFAULTARRIVALCONFIG = "data/requests.json"

def main(
        totalTimesteps: int = DEFAULTTOTALTIMESTEPS,
        networkPath: str = DEFAULTNETWORKPATH,
        agentType: str = DEFAULTAGENTTYPE, #NOTE: Assumed to be in the stable_baselines3 package
        nEvalEpisodes: int = DEFAULTNEVALEPISODES,
        evalFreq: int = DEFAULTEVALFREQ,
        output: str = DEFAULTOUTPUT,
        debug: bool = DEFAULTDEBUGGING,
        arrivalConfig: str = DEFAULTARRIVALCONFIG
) -> None:
    
    logging.basicConfig()
    debug_level = logging.DEBUG if debug else logging.INFO
    logging.getLogger().setLevel(debug_level)

    Path(f"{output}/logs").mkdir(exist_ok=True, parents=True)
    #Path(f"{output}/evaluation").mkdir(exist_ok=True, parents=True)

    with open(Path(arrivalConfig), "r") as file:
        arrival_config = json.load(file)

    env = Env(networkPath, arrival_config)

    evalEnv = StatsWrapper(Env(networkPath, arrival_config))
    evalLogCallback = EvalLogCallback()
    evalCallback = EvalCallback(
        evalEnv,
        n_eval_episodes=nEvalEpisodes,
        log_path=f"{output}/evaluation",
        eval_freq=evalFreq,
        deterministic=False,
        render=False,
        callback_after_eval=evalLogCallback,
    )

    Agent = getattr(stable_baselines3, agentType)
    agent = Agent(
        **{
            "policy": "MlpPolicy", #This policy is best suited for the contionous state in this task
            "env": env,
            "verbose": 1,
            "tensorboard_log": f"{output}/logs",
        }
    )

    agent.learn(
        total_timesteps=totalTimesteps,
        tb_log_name=agentType,
        callback=evalCallback,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--totalTimesteps", type=int, default=DEFAULTTOTALTIMESTEPS)
    parser.add_argument("--networkPath", type=str, default=DEFAULTNETWORKPATH)
    parser.add_argument("--agent", type=str, default=DEFAULTAGENTTYPE)
    parser.add_argument("--nEvalEpisodes", type=int, default=DEFAULTNEVALEPISODES)
    parser.add_argument("--evalFreq", type=int, default=DEFAULTEVALFREQ)
    parser.add_argument("--output", type=str, default=DEFAULTOUTPUT)
    parser.add_argument("--debug", type=bool, default=DEFAULTDEBUGGING)
    parser.add_argument("--arrivalConfig", type=str, default=DEFAULTARRIVALCONFIG)

    args = parser.parse_args()

    main(
        args.totalTimesteps,
        args.networkPath,
        args.agent,
        args.nEvalEpisodes,
        args.evalFreq,
        args.output,
        args.debug,
        args.arrivalConfig
    )
