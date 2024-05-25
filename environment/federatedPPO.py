import warnings
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union

import numpy as np
import torch as th
#from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, obs_as_tensor, safe_mean
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs

from stable_baselines3 import PPO
import time

from typing import List
from collections import defaultdict

import gym

import numpy.typing as npt

import random

from threading import Thread, Lock

#SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")

class stepPPO(PPO):
    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "OnPolicyAlgorithm",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "OnPolicyAlgorithm":
        
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        callback.on_training_start(locals(), globals())

        continueTraining = True

        while self.num_timesteps < total_timesteps and continueTraining:
            continueTraining = self.learnStep(callback, total_timesteps, log_interval, iteration)

        callback.on_training_end()

        return self


    def learnStep(
            self,
            callback: MaybeCallback,
            total_timesteps: int,
            log_interval: int,
            iteration: int
            ) -> bool:
        """
        The logic present in PPO's learn method, but separated into its own step function to separate code better
        """
        continueTraining = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

        if continueTraining is False:
            return False

        iteration += 1
        self._update_current_progress_remaining(self.num_timesteps, total_timesteps)  

        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.safeRMean = safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])

        # Display training infos
        if log_interval is not None and iteration % log_interval == 0:
            fps = int((self.num_timesteps - self._num_timesteps_at_start) / (time.time() - self.start_time))
            self.logger.record("time/iterations", iteration, exclude="tensorboard")
            if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                self.logger.record("rollout/ep_rew_mean", self.safeRMean)
                self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
            self.logger.record("time/fps", fps)
            self.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(step=self.num_timesteps)

        self.train()

        return True
    
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            new_obs, dones, n_steps, notPremature = self.rolloutStep(env, n_steps, callback, rollout_buffer)

            if not notPremature:
                return False

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True
    
    def rolloutStep(
            self,
            env: gym.Env,
            n_steps: int,
            callback: BaseCallback,
            rollout_buffer: RolloutBuffer
            ) -> tuple[VecEnvObs, npt.NDArray[any], int, bool]:
        
        if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
            # Sample a new noise matrix
            self.policy.reset_noise(env.num_envs)

        with th.no_grad():
            # Convert to pytorch tensor or to TensorDict
            obs_tensor = obs_as_tensor(self._last_obs, self.device)
            actions, values, log_probs = self.policy(obs_tensor)
        actions = actions.cpu().numpy()

        # Rescale and perform action
        clipped_actions = actions
        # Clip the actions to avoid out of bound error
        if isinstance(self.action_space, gym.spaces.Box):
            clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

        new_obs, rewards, dones, infos = env.step(clipped_actions)

        self.num_timesteps += env.num_envs

        # Give access to local variables
        callback.update_locals(locals())
        if callback.on_step() is False:
            return new_obs, dones, n_steps, False

        self._update_info_buffer(infos)
        n_steps += 1

        if isinstance(self.action_space, gym.spaces.Discrete):
            # Reshape in case of discrete action
            actions = actions.reshape(-1, 1)

        # Handle timeout by bootstraping with value function
        # see GitHub issue #633
        for idx, done in enumerate(dones):
            if (
                done
                and infos[idx].get("terminal_observation") is not None
                and infos[idx].get("TimeLimit.truncated", False)
            ):
                terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                with th.no_grad():
                    terminal_value = self.policy.predict_values(terminal_obs)[0]
                rewards[idx] += self.gamma * terminal_value

        rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)
        self._last_obs = new_obs
        self._last_episode_starts = dones

        return new_obs, dones, n_steps, True
    

class federatedPPO(PPO):

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "OnPolicyAlgorithm",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        models: List[stepPPO] = [],
    ) -> "OnPolicyAlgorithm":
        
        iteration = 0
        self.safeRMean = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        callback.on_training_start(locals(), globals())

        continueTraining = True

        milestoneIncrement = 5000 #NOTE: This value was found to be optimal via testing for this Thesis' problem specifically
        nextMilestone = milestoneIncrement

        self.models = models

        while self.num_timesteps < total_timesteps and continueTraining:
            continueTraining = self.learnStep(callback, total_timesteps, log_interval, iteration)

            if nextMilestone < self.num_timesteps and self.domain == 0: #NOTE: Hardcoded focus on space domain
                self.lock.acquire()

                nextMilestone += milestoneIncrement

                self.weighPolicies(self.createWeightDict())
                print("Policies uploaded and downloaded!")

                self.lock.release()

        callback.on_training_end()

        return self
        
        """
        iteration = 0

        self.models = models

        self.sumRewards = 0
        self.modelRewards: dict[stepPPO | federatedPPO, float] = defaultdict(lambda: 0)

        #NOTE: It is assumed here that getting total_timesteps from this method is superflous for the later models.
        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        for model in self.models:
            _, model.callback = model._setup_learn( #We save the callback in the model class
            total_timesteps, None, None, eval_freq, n_eval_episodes, None, reset_num_timesteps, None
        )

        callback.on_training_start(locals(), globals())

        continueTraining = True

        milestoneIncrement = 10000 
        nextMilestone = milestoneIncrement

        #The actions in this while-loop is executed every batch-size (2048 by default)
        while self.num_timesteps < total_timesteps and continueTraining:
            continueTraining = self.learnStep(callback, total_timesteps, log_interval, iteration) 
            self.modelRewards[self] = self.safeRMean

            if nextMilestone < self.num_timesteps:
                nextMilestone += milestoneIncrement

                self.weighPolicies(self.createWeightDict())
                print("Policies uploaded and downloaded!")

        callback.on_training_end()

        return self
        """
    
    
    def learnStep(
            self,
            callback: MaybeCallback,
            total_timesteps: int,
            log_interval: int,
            iteration: int
            ) -> bool:
        """
        The logic present in PPO's learn method, but separated into its own step function to separate code better
        """
        continueTraining = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

        if continueTraining is False:
            return False

        iteration += 1
        self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.safeRMean = safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])

        # Display training infos
        if log_interval is not None and iteration % log_interval == 0:
            fps = int((self.num_timesteps - self._num_timesteps_at_start) / (time.time() - self.start_time))
            self.logger.record("time/iterations", iteration, exclude="tensorboard")
            if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                self.logger.record("rollout/ep_rew_mean", self.safeRMean)
                self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
            self.logger.record("time/fps", fps)
            self.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")

            self.logger.record("federation/domain", self.domain)
            self.logger.record("federation/estimatedDomainWeight", self.createWeightDict()[self])

            self.logger.dump(step=self.num_timesteps)

        #Must lock to avoid conflicts of interest
        self.lock.acquire()
        self.train()
        self.lock.release()

        return True
    
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:

        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False) #TODO: Make for all models

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            self.lock.acquire()

            candidate = env.envs[0].unwrapped.sc.vnfs[env.envs[0].unwrapped.vnfIndex]["candidate_domain"]
            if self.domain == candidate:
                new_obs, dones, n_steps, notPremature = self.rolloutStep(env, n_steps, callback, rollout_buffer)

                if not notPremature:
                    self.lock.release()
                    return False
                
            self.lock.release()
            

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True
    
    def rolloutStep(
            self,
            env: gym.Env,
            n_steps: int,
            callback: BaseCallback,
            rollout_buffer: RolloutBuffer
            ) -> tuple[VecEnvObs, npt.NDArray[any], int, bool]:

        if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
            # Sample a new noise matrix
            self.policy.reset_noise(env.num_envs)

        with th.no_grad():
            # Convert to pytorch tensor or to TensorDict
            obs_tensor = obs_as_tensor(self._last_obs, self.device)
            actions, values, log_probs = self.policy(obs_tensor)
        actions = actions.cpu().numpy()

        # Rescale and perform action
        clipped_actions = actions
        # Clip the actions to avoid out of bound error
        if isinstance(self.action_space, gym.spaces.Box):
            clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

        env.envs[0].unwrapped.vnfBacktrack.currentDomain = self.domain

        new_obs, rewards, dones, infos = env.step(clipped_actions)

        self.num_timesteps += env.num_envs

        # Give access to local variables
        callback.update_locals(locals())
        if callback.on_step() is False:
            print(f"{self.domain} - premature")
            return new_obs, dones, n_steps, False

        self._update_info_buffer(infos)
        n_steps += 1

        if isinstance(self.action_space, gym.spaces.Discrete):
            # Reshape in case of discrete action
            actions = actions.reshape(-1, 1)

        # Handle timeout by bootstraping with value function
        # see GitHub issue #633
        for idx, done in enumerate(dones):
            if (
                done
                and infos[idx].get("terminal_observation") is not None
                and infos[idx].get("TimeLimit.truncated", False)
            ):
                terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                with th.no_grad():
                    terminal_value = self.policy.predict_values(terminal_obs)[0]
                rewards[idx] += self.gamma * terminal_value

        rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)
        self._last_obs = new_obs
        self._last_episode_starts = dones

        return new_obs, dones, n_steps, True
    
    #https://github.com/pietromosca1994/DeRL_Golem/blob/main/src/utils.py
    def weighPolicies(
            self,
            weightDict: dict[stepPPO, float]
            ) -> None:

        federated_policy=self.models[0].get_parameters()["policy"]
        for key in federated_policy.keys():
            federated_policy[key] = th.mul(federated_policy[key], weightDict[self])

            for modelIndex, model in enumerate(self.models):
                if modelIndex > 0:
                    federated_policy[key] = th.add(federated_policy[key], th.mul(model.get_parameters()['policy'][key], weightDict[model]))


        for _, model in enumerate(self.models):
            # model initialization
            federated_parameters=model.get_parameters()

            # substitute federated parameters
            federated_parameters['policy']=federated_policy

            model.set_parameters(federated_parameters)

            
    def createWeightDict(
            self,
            rewardWeights: bool = False
            ) -> dict[stepPPO, float]:
        sumR = 0
        returnDict = dict() 

        if not rewardWeights:
            domainIndexToWeight = {0: 0.1, 1: 0.3, 2: 0.6}

            for domainIndex, model in enumerate(self.models):
                returnDict[model] = domainIndexToWeight[domainIndex]

        else:
            for _, model in enumerate(self.models):
                sumR += model.safeRMean

                returnDict[model] = model.safeRMean

            for model, undividedWeight in returnDict.items():
                returnDict[model] = undividedWeight / sumR if sumR != 0 else 0

        return returnDict

    
