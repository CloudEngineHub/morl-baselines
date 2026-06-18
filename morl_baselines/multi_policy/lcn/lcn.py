"""Lorenz Conditioned Networks. Extends Pareto Conditioned Networks (PCN) with Lorenz dominance."""

import heapq
import os
from typing import List, Optional, Type, Union

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn.functional as F
import wandb

from morl_baselines.common.evaluation import log_all_multi_policy_metrics
from morl_baselines.common.morl_algorithm import MOAgent, MOPolicy
from morl_baselines.common.pareto import get_non_dominated_inds
from morl_baselines.common.performance_indicators import hypervolume
from morl_baselines.multi_policy.pcn.pcn import (
    BasePCNModel,
    ContinuousActionsDefaultModel,
    DiscreteActionsDefaultModel,
    Transition,
    crowding_distance,
)


def lorenz_vector(points: np.ndarray, proportional: bool = False) -> np.ndarray:
    """Compute the Lorenz vector of a set of points.

    The Lorenz vector of a return is the cumulative sum of its objectives sorted in increasing order.
    Comparing Lorenz vectors (instead of raw returns) for Pareto dominance yields Lorenz dominance, which
    favors returns that distribute reward more equitably across objectives.

    Args:
        points (np.ndarray): array of points (e.g. returns), of shape (n_points, n_objectives).
        proportional (bool, optional): whether to normalize the Lorenz vector by the total sum of each
            point's objectives. Defaults to False.

    Returns:
        np.ndarray: the Lorenz vector of each point, of the same shape as points.
    """
    sorted_points = np.sort(points, axis=1)
    lv = np.cumsum(sorted_points, axis=1)
    if proportional:
        lv = lv / np.sum(points, axis=1, keepdims=True)
    return lv


class LCN(MOAgent, MOPolicy):
    """Lorenz Conditioned Networks (LCN).

    Michailidis, D., Röpke, W., Roijers, D. M., Ghebreab, S., & Santos, F. P. (2026). Scalable
    Multi-Objective Reinforcement Learning with Fairness Guarantees using Lorenz Dominance. Journal of
    Artificial Intelligence Research, 85. https://jair.org/index.php/jair/article/view/19862

    LCN extends Pareto Conditioned Networks (PCN) (Reymond et al., 2022) by selecting and conditioning on
    commands using Lorenz dominance instead of (or interpolated with) Pareto dominance. Lorenz dominance
    compares returns by the cumulative sum of their sorted objectives (their Lorenz vector), which biases
    the learned set of policies towards more equitable trade-offs across objectives. The `lcn_lambda`
    parameter interpolates between strict Lorenz dominance and ordinary dominance computed on the sorted
    returns ("λ-Lorenz dominance"), trading off fairness against raw performance.

    ## Credits

    This code extends the PCN implementation of this repository
    (`morl_baselines.multi_policy.pcn.pcn`), which is itself a refactor of the code from the authors of
    the PCN paper, available at: https://github.com/mathieu-reymond/pareto-conditioned-networks
    """

    def __init__(
        self,
        env: Optional[gym.Env],
        scaling_factor: np.ndarray,
        learning_rate: float = 1e-2,
        gamma: float = 1.0,
        batch_size: int = 32,
        hidden_dim: int = 64,
        noise: float = 0.1,
        distance_ref: str = "nondominated",
        lcn_lambda: Optional[float] = None,
        project_name: str = "MORL-Baselines",
        experiment_name: str = "LCN",
        wandb_entity: Optional[str] = None,
        log: bool = True,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        model_class: Optional[Type[BasePCNModel]] = None,
    ) -> None:
        """Initialize LCN agent.

        Args:
            env (Optional[gym.Env]): Gym environment.
            scaling_factor (np.ndarray): Scaling factor for the desired return and horizon used in the model.
            learning_rate (float, optional): Learning rate. Defaults to 1e-2.
            gamma (float, optional): Discount factor. Defaults to 1.0.
            batch_size (int, optional): Batch size. Defaults to 32.
            hidden_dim (int, optional): Hidden dimension. Defaults to 64.
            noise (float, optional): Standard deviation of the noise to add to the action in the continuous action case. Defaults to 0.1.
            distance_ref (str, optional): Dominance notion used to select and condition on episodes. One of
                "nondominated" (strict Lorenz dominance) or "lambda_lorenz" (λ-Lorenz dominance, interpolating
                between Lorenz dominance and ordinary dominance on the sorted returns, controlled by `lcn_lambda`).
                Defaults to "nondominated".
            lcn_lambda (Optional[float], optional): λ parameter used when `distance_ref="lambda_lorenz"`. Controls
                the trade-off between fairness (Lorenz dominance) and raw performance (ordinary dominance).
                Required when `distance_ref="lambda_lorenz"`. Defaults to None.
            project_name (str, optional): Name of the project for wandb. Defaults to "MORL-Baselines".
            experiment_name (str, optional): Name of the experiment for wandb. Defaults to "LCN".
            wandb_entity (Optional[str], optional): Entity for wandb. Defaults to None.
            log (bool, optional): Whether to log to wandb. Defaults to True.
            seed (Optional[int], optional): Seed for reproducibility. Defaults to None.
            device (Union[th.device, str], optional): Device to use. Defaults to "auto".
            model_class (Optional[Type[BasePCNModel]], optional): Model class to use. Defaults to None.
        """
        MOAgent.__init__(self, env, device=device, seed=seed)
        MOPolicy.__init__(self, device=device)

        self.experience_replay = []  # List of (distance, time_step, transition)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.distance_ref = distance_ref
        self.lcn_lambda = lcn_lambda
        self.scaling_factor = scaling_factor
        self.desired_return = None
        self.desired_horizon = None
        self.continuous_action = True if type(self.env.action_space) is gym.spaces.Box else False
        self.noise = noise

        if model_class and not issubclass(model_class, BasePCNModel):
            raise ValueError("model_class must be a subclass of BasePCNModel")

        if model_class is None:
            if self.continuous_action:
                model_class = ContinuousActionsDefaultModel
            else:
                model_class = DiscreteActionsDefaultModel

        self.model = model_class(
            self.observation_dim, self.action_dim, self.reward_dim, self.scaling_factor, hidden_dim=self.hidden_dim
        ).to(self.device)
        self.opt = th.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.log = log
        if log:
            experiment_name += " continuous action" if self.continuous_action else ""
            self.setup_wandb(project_name, experiment_name, wandb_entity)

    def get_config(self) -> dict:
        """Get configuration of LCN model."""
        return {
            "env_id": self.env.unwrapped.spec.id,
            "reward_dim": self.reward_dim,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "hidden_dim": self.hidden_dim,
            "scaling_factor": self.scaling_factor,
            "continuous_action": self.continuous_action,
            "noise": self.noise,
            "distance_ref": self.distance_ref,
            "lcn_lambda": self.lcn_lambda,
            "seed": self.seed,
        }

    def update(self):
        """Update LCN model."""
        batch = []
        # randomly choose episodes from experience buffer
        s_i = self.np_random.choice(np.arange(len(self.experience_replay)), size=self.batch_size, replace=True)
        for i in s_i:
            # episode is tuple (return, transitions)
            ep = self.experience_replay[i][2]
            # choose random timestep from episode,
            # use it's return and leftover timesteps as desired return and horizon
            t = self.np_random.integers(0, len(ep))
            # reward contains return until end of episode
            s_t, a_t, r_t, h_t = ep[t].observation, ep[t].action, np.float32(ep[t].reward), np.float32(len(ep) - t)
            batch.append((s_t, a_t, r_t, h_t))

        obs, actions, desired_return, desired_horizon = zip(*batch)
        prediction = self.model(
            th.tensor(np.array(obs)).to(self.device),
            th.tensor(np.array(desired_return)).to(self.device),
            th.tensor(np.array(desired_horizon)).unsqueeze(1).to(self.device),
        )

        self.opt.zero_grad()
        if self.continuous_action:
            l = F.mse_loss(th.tensor(np.array(actions)).float().to(self.device), prediction)
        else:
            # one-hot of action for CE loss
            actions = F.one_hot(th.tensor(np.array(actions)).long().to(self.device), len(prediction[0]))
            # cross-entropy loss
            l = th.sum(-actions * prediction, -1)
            l = l.mean()
        l.backward()
        self.opt.step()

        return l, prediction

    def _add_episode(self, transitions: List[Transition], max_size: int, step: int) -> None:
        # compute return
        for i in reversed(range(len(transitions) - 1)):
            transitions[i].reward += self.gamma * transitions[i + 1].reward
        # pop smallest episode of heap if full, add new episode
        # heap is sorted by negative distance, (updated in nlargest)
        # put positive number to ensure that new item stays in the heap
        if len(self.experience_replay) == max_size:
            heapq.heappushpop(self.experience_replay, (1, step, transitions))
        else:
            heapq.heappush(self.experience_replay, (1, step, transitions))

    def _nlargest(self, n: int, threshold: float = 0.2):
        """Select the n most useful episodes to condition on next, for diversity.

        Candidate episodes are ranked by their distance to the closest dominant return, where dominance is
        either strict Lorenz dominance or λ-Lorenz dominance, depending on `self.distance_ref`. See Section 4.4 of
        https://arxiv.org/pdf/2204.05036.pdf for the crowding-distance penalty (shared with PCN), and the LCN
        paper for Lorenz / λ-Lorenz dominance.
        """
        returns = np.array([e[2][0].reward for e in self.experience_replay])
        # crowding distance of each point, check ones that are too close together
        distances = crowding_distance(returns)
        sma = np.argwhere(distances <= threshold).flatten()

        if self.distance_ref == "lambda_lorenz":
            assert self.lcn_lambda is not None, "lcn_lambda must be set when using distance_ref='lambda_lorenz'"
            # Lorenz vectors are defined over sorted objective values.
            returns = np.sort(returns, axis=1)
            lv = lorenz_vector(returns)
            # the final vector is a weighted average of the Lorenz vector and the sorted returns
            fv = self.lcn_lambda * returns + (1 - self.lcn_lambda) * lv
            non_dominated_i = get_non_dominated_inds(fv)
        else:  # "nondominated": strict Lorenz dominance
            lv = lorenz_vector(returns)
            non_dominated_i = get_non_dominated_inds(lv)

        non_dominated = returns[non_dominated_i]
        # we will compute distance of each point with each non-dominated point,
        # duplicate each point with number of non_dominated to compute respective distance
        returns_exp = np.tile(np.expand_dims(returns, 1), (1, len(non_dominated), 1))
        # distance to closest non_dominated point
        l2 = np.min(np.linalg.norm(returns_exp - non_dominated, axis=-1), axis=-1) * -1

        # all points that are too close together (crowding distance < threshold) get a penalty
        non_dominated_i = np.nonzero(non_dominated_i)[0]
        _, unique_i = np.unique(non_dominated, axis=0, return_index=True)
        unique_i = non_dominated_i[unique_i]
        duplicates = np.ones(len(l2), dtype=bool)
        duplicates[unique_i] = False
        l2[duplicates] -= 1e-5
        l2[sma] *= 2

        sorted_i = np.argsort(l2)
        largest = [self.experience_replay[i] for i in sorted_i[-n:]]
        # before returning largest elements, update all distances in heap
        for i in range(len(l2)):
            self.experience_replay[i] = (l2[i], self.experience_replay[i][1], self.experience_replay[i][2])
        heapq.heapify(self.experience_replay)
        return largest

    def _choose_commands(self, num_episodes: int):
        # get best episodes, according to their crowding distance
        episodes = self._nlargest(num_episodes, self.cd_threshold)
        returns, horizons = list(zip(*[(e[2][0].reward, len(e[2])) for e in episodes]))
        # keep only Lorenz-non-dominated returns
        lv = lorenz_vector(np.array(returns))
        nd_i = get_non_dominated_inds(lv)
        returns = np.array(returns)[nd_i]
        horizons = np.array(horizons)[nd_i]
        # pick random return from random best episode
        r_i = self.np_random.integers(0, len(returns))
        desired_horizon = np.float32(horizons[r_i] - 2)
        # mean and std per objective
        _, s = np.mean(returns, axis=0), np.std(returns, axis=0)
        # desired return is sampled from [M, M+S], to try to do better than mean return
        desired_return = returns[r_i].copy()
        # random objective
        r_i = self.np_random.integers(0, len(desired_return))
        desired_return[r_i] += self.np_random.uniform(high=s[r_i])
        desired_return = np.float32(desired_return)
        return desired_return, desired_horizon

    def _act(self, obs: np.ndarray, desired_return, desired_horizon, eval_mode=False) -> int:
        prediction = self.model(
            th.tensor(np.array([obs])).float().to(self.device),
            th.tensor(np.array([desired_return])).float().to(self.device),
            th.tensor(np.array([desired_horizon])).unsqueeze(1).float().to(self.device),
        )

        if self.continuous_action:
            action = prediction.detach().cpu().numpy()[0]
            if not eval_mode:
                # Add Gaussian noise: https://arxiv.org/pdf/2204.05027.pdf
                action = action + np.random.normal(0.0, self.noise)
            return action
        else:
            log_probs = prediction.detach().cpu().numpy()[0]

            if eval_mode:
                action = np.argmax(log_probs)
            else:
                action = self.np_random.choice(np.arange(len(log_probs)), p=np.exp(log_probs))
            return action

    def _run_episode(self, env, desired_return, desired_horizon, max_return, eval_mode=False):
        transitions = []
        obs, _ = env.reset()
        done = False
        while not done:
            action = self._act(obs, desired_return, desired_horizon, eval_mode)
            n_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            transitions.append(
                Transition(
                    observation=obs,
                    action=action,
                    reward=np.float32(reward).copy(),
                    next_observation=n_obs,
                    terminal=terminated,
                )
            )

            obs = n_obs
            # clip desired return, to return-upper-bound,
            # to avoid negative returns giving impossible desired returns
            desired_return = np.clip(desired_return - reward, None, max_return, dtype=np.float32)
            # clip desired horizon to avoid negative horizons
            desired_horizon = np.float32(max(desired_horizon - 1, 1.0))
        return transitions

    def set_desired_return_and_horizon(self, desired_return: np.ndarray, desired_horizon: int):
        """Set desired return and horizon for evaluation."""
        self.desired_return = desired_return
        self.desired_horizon = desired_horizon

    def eval(self, obs, w=None):
        """Evaluate policy action for a given observation."""
        return self._act(obs, self.desired_return, self.desired_horizon, eval_mode=True)

    def evaluate(self, env, max_return, n=10):
        """Evaluate policy in the given environment."""
        n = min(n, len(self.experience_replay))
        episodes = self._nlargest(n, self.cd_threshold)
        returns, horizons = list(zip(*[(e[2][0].reward, len(e[2])) for e in episodes]))
        returns = np.float32(returns)
        horizons = np.float32(horizons)
        e_returns = []
        for i in range(n):
            transitions = self._run_episode(env, returns[i], np.float32(horizons[i]), max_return, eval_mode=True)
            # compute return
            for i in reversed(range(len(transitions) - 1)):
                transitions[i].reward += self.gamma * transitions[i + 1].reward
            e_returns.append(transitions[0].reward)

        distances = np.linalg.norm(np.array(returns) - np.array(e_returns), axis=-1)
        return e_returns, np.array(returns), distances

    def save(self, filename: str = "LCN_model", save_dir: str = "weights"):
        """Save LCN."""
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        th.save(self.model, f"{save_dir}/{filename}.pt")

    def load(self, path: str):
        """Load LCN."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file {path} does not exist.")
        self.model = th.load(path, map_location=self.device, weights_only=False)

    def train(
        self,
        total_timesteps: int,
        eval_env: gym.Env,
        ref_point: np.ndarray,
        known_pareto_front: Optional[List[np.ndarray]] = None,
        num_eval_weights_for_eval: int = 50,
        num_er_episodes: int = 500,
        num_step_episodes: int = 10,
        num_model_updates: int = 100,
        max_return: np.ndarray = None,
        max_buffer_size: int = 500,
        num_points_pf: int = 100,
        save_dir: str = "weights",
        cd_threshold: float = 0.2,
    ):
        """Train LCN.

        Args:
            total_timesteps: total number of time steps to train for
            eval_env: environment for evaluation
            ref_point: reference point for hypervolume calculation
            known_pareto_front: Optimal pareto front for metrics calculation, if known.
            num_eval_weights_for_eval (int): Number of weights use when evaluating the Pareto front, e.g., for computing expected utility.
            num_er_episodes: number of episodes to fill experience replay buffer.
            num_step_episodes: number of episodes per training step
            num_model_updates: number of model updates per episode
            max_return: maximum return for clipping desired return. When None, this will be set to 100 for all objectives.
            max_buffer_size: maximum buffer size
            num_points_pf: number of points to sample from pareto front for metrics calculation
            save_dir: directory to save model weights
            cd_threshold: threshold for the crowding distance penalty
        """
        max_return = max_return if max_return is not None else np.full(self.reward_dim, 100.0, dtype=np.float32)
        if self.log:
            self.register_additional_config(
                {
                    "total_timesteps": total_timesteps,
                    "ref_point": ref_point.tolist(),
                    "known_front": known_pareto_front,
                    "num_eval_weights_for_eval": num_eval_weights_for_eval,
                    "num_er_episodes": num_er_episodes,
                    "num_step_episodes": num_step_episodes,
                    "num_model_updates": num_model_updates,
                    "max_return": max_return.tolist(),
                    "max_buffer_size": max_buffer_size,
                    "num_points_pf": num_points_pf,
                    "save_dir": save_dir,
                    "distance_ref": self.distance_ref,
                    "lcn_lambda": self.lcn_lambda,
                    "cd_threshold": cd_threshold,
                }
            )
        self.global_step = 0
        total_episodes = num_er_episodes
        n_checkpoints = 0
        self.cd_threshold = cd_threshold

        # fill buffer with random episodes
        self.experience_replay = []
        for _ in range(num_er_episodes):
            transitions = []
            obs, _ = self.env.reset()
            done = False
            while not done:
                action = self.env.action_space.sample()
                n_obs, reward, terminated, truncated, _ = self.env.step(action)
                transitions.append(Transition(obs, action, np.float32(reward).copy(), n_obs, terminated))
                done = terminated or truncated
                obs = n_obs
                self.global_step += 1
            # add episode in-place
            self._add_episode(transitions, max_size=max_buffer_size, step=self.global_step)

        while self.global_step < total_timesteps:
            loss = []
            entropy = []
            for _ in range(num_model_updates):
                l, lp = self.update()
                loss.append(l.detach().cpu().numpy())
                if not self.continuous_action:
                    lp = lp.detach().cpu().numpy()
                    ent = np.sum(-np.exp(lp) * lp)
                    entropy.append(ent)

            desired_return, desired_horizon = self._choose_commands(num_er_episodes)

            # get all leaves, contain biggest elements, experience_replay got heapified in choose_commands
            leaves_r = np.array([e[2][0].reward for e in self.experience_replay[len(self.experience_replay) // 2 :]])

            if self.log:
                hv = hypervolume(ref_point, leaves_r)
                wandb.log(
                    {
                        "train/hypervolume": hv,
                        "train/loss": np.mean(loss),
                        "global_step": self.global_step,
                    },
                )
                if not self.continuous_action:
                    wandb.log(
                        {
                            "train/entropy": np.mean(entropy),
                            "global_step": self.global_step,
                        },
                    )

            returns = []
            horizons = []
            for _ in range(num_step_episodes):
                transitions = self._run_episode(self.env, desired_return, desired_horizon, max_return)
                self.global_step += len(transitions)
                self._add_episode(transitions, max_size=max_buffer_size, step=self.global_step)
                returns.append(transitions[0].reward)
                horizons.append(len(transitions))

            total_episodes += num_step_episodes
            if self.log:
                wandb.log(
                    {
                        "train/episode": total_episodes,
                        "train/horizon_desired": desired_horizon,
                        "train/mean_horizon_distance": np.linalg.norm(np.mean(horizons) - desired_horizon),
                        "global_step": self.global_step,
                    },
                )

                for i in range(self.reward_dim):
                    wandb.log(
                        {
                            f"train/desired_return_{i}": desired_return[i],
                            f"train/mean_return_{i}": np.mean(np.array(returns)[:, i]),
                            f"train/mean_return_distance_{i}": np.linalg.norm(
                                np.mean(np.array(returns)[:, i]) - desired_return[i]
                            ),
                            "global_step": self.global_step,
                        },
                    )
            print(
                f"step {self.global_step} \t return {np.mean(returns, axis=0)}, ({np.std(returns, axis=0)}) \t loss {np.mean(loss):.3E} \t horizons {np.mean(horizons)}"
            )

            if self.global_step >= (n_checkpoints + 1) * total_timesteps / 100:
                self.save(save_dir=save_dir, filename=f"LCN_model_{n_checkpoints}")
                n_checkpoints += 1
                e_returns, _, _ = self.evaluate(eval_env, max_return, n=num_points_pf)

                if self.log:
                    log_all_multi_policy_metrics(
                        current_front=e_returns,
                        hv_ref_point=ref_point,
                        reward_dim=self.reward_dim,
                        global_step=self.global_step,
                        n_sample_weights=num_eval_weights_for_eval,
                        ref_front=known_pareto_front,
                    )

        self.env.close()
