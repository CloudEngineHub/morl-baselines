import mo_gymnasium as mo_gym
import numpy as np
from mo_gymnasium.wrappers import MORecordEpisodeStatistics

from morl_baselines.multi_policy.lcn.lcn import LCN


def main():
    def make_env():
        env = mo_gym.make("fruit-tree-v0")
        env = MORecordEpisodeStatistics(env, gamma=1.0)
        return env

    env = make_env()

    agent = LCN(
        env,
        scaling_factor=np.array([1, 1, 1, 1, 1, 1, 0.1]),
        learning_rate=1e-2,
        batch_size=32,
        distance_ref="nondominated",
        project_name="MORL-Baselines",
        experiment_name="LCN",
        log=True,
    )

    agent.train(
        eval_env=make_env(),
        total_timesteps=int(1e5),
        ref_point=np.zeros(6),
        num_er_episodes=500,
        max_buffer_size=500,
        num_model_updates=100,
        max_return=np.full(6, 10.0, dtype=np.float32),
        known_pareto_front=env.unwrapped.pareto_front(gamma=1.0),
    )


if __name__ == "__main__":
    main()
