import warnings
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, obs_as_tensor, safe_mean

from stable_baselines3 import PPO
import time

from typing import List
from collections import defaultdict

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

        self.models = models

        self.sumRewards = 0
        self.modelRewards: dict[stepPPO | federatedPPO, float] = defaultdict(lambda: 0)

        #NOTE: It is assumed here that getting total_timesteps from this method is superflous for the later models.
        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        for model in self.models:
            _, model.callback = model._setup_learn( #We save the callback in the model class, TODO: Evaluate how good this is
            total_timesteps, None, None, eval_freq, n_eval_episodes, None, reset_num_timesteps, None
        )

        callback.on_training_start(locals(), globals())

        continueTraining = True

        milestoneIncrement = 10000 #TODO
        nextMilestone = milestoneIncrement

        #The actions in this while-loop is executed every batch-size (2048 by default)
        while self.num_timesteps < total_timesteps and continueTraining:
            continueTraining = self.learnStep(callback, total_timesteps, log_interval, iteration) 
            self.modelRewards[self] = self.safeRMean

            for _, model in enumerate(self.models):
                continueTraining = continueTraining and model.learnStep(model.callback, total_timesteps, None, iteration) #log_interval is none
                self.modelRewards[model] = model.safeRMean

            if nextMilestone < self.num_timesteps:
                nextMilestone += milestoneIncrement

                self.weighPolicies(self.createWeightDict())
                print("Policies uploaded and downloaded!")

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

            #print(self.createWeightDict())
            self.logger.record("federation/estimatedMainWeight", self.createWeightDict()[self])

            self.logger.dump(step=self.num_timesteps)

        self.train()

        return True
    
    #https://github.com/pietromosca1994/DeRL_Golem/blob/main/src/utils.py
    def weighPolicies(
            self,
            weightDict: dict[stepPPO, float]
            ) -> None:

        federated_policy=self.get_parameters()["policy"]
        for key in federated_policy.keys():
            federated_policy[key] = th.mul(federated_policy[key], weightDict[self])

            for _, model in enumerate(self.models):
                federated_policy[key] = th.add(federated_policy[key], th.mul(model.get_parameters()['policy'][key], weightDict[model]))
        
        # model initialization
        federated_parameters=self.get_parameters()

        # substitute federated parameters
        federated_parameters['policy']=federated_policy
        self.set_parameters(federated_parameters)

        for _, model in enumerate(self.models):
            model.set_parameters(federated_parameters)

        #return federated_model
            
    def createWeightDict(self) -> defaultdict[stepPPO, float]:
        sumR = 0
        returnDict = defaultdict(lambda: 0)

        for model, reward in self.modelRewards.items():
            sumR += reward

            returnDict[model] = reward

        for model, undividedWeight in returnDict.items():
            returnDict[model] = undividedWeight / sumR

        return returnDict

    
    """
    def train(self) -> None:
        ""
        Update policy using the currently gathered rollout buffer.
        ""
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    """

