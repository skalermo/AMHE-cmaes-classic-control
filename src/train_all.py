import os
import sys
sys.path.append('../')
from typing import Type, Union

import gym
from stable_baselines3 import A2C, PPO
import numpy as np

from src.cmaes_nn import CMAESNN
from src.env_info import env_to_action_type
from src.log_utils import captured_output


def _create_dirs(*paths: str):
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)


MODEL_TYPE = Type[Union[A2C, PPO, CMAESNN]]


def _str_to_class(classname: str) -> MODEL_TYPE:
    return getattr(sys.modules[__name__], classname)


def _test_loop(model: MODEL_TYPE, env_id: str) -> int:
    env = gym.make(env_id)
    obs = env.reset()

    return_ = 0
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        return_ += reward
        # env.render()
    return return_


def main():
    total_timesteps = 500_000

    env_ids = env_to_action_type.keys()
    models = {
        'CMAESNN': lambda env_id, verbose: CMAESNN(env_id, verbose=verbose),
        'A2C': lambda env_id, verbose: A2C(policy='MlpPolicy', env=gym.make(env_id), verbose=verbose),
        'PPO': lambda env_id, verbose: PPO(policy='MlpPolicy', env=gym.make(env_id), verbose=verbose),
    }

    train_runs = 5
    test_runs = 10

    data_dir = '../.data'
    logs_dir = f'{data_dir}/logs'
    models_dir = f'{data_dir}/models'
    _create_dirs(logs_dir, models_dir)

    # skips if already trained
    for env_id in env_ids:
        for model_name, model_fn in models.items():
            for run in range(train_runs):
                model_path = f'{models_dir}/{env_id}_{model_name}_{run}.zip'
                print(f'Training {model_name} on {env_id} run {run}')
                if os.path.exists(model_path):
                    print(f'Model {model_path} already exists, skipping')
                    continue

                with captured_output() as (out, _):
                    model = model_fn(env_id, verbose=1)
                    model.learn(total_timesteps=total_timesteps)

                with open(f'{logs_dir}/{env_id}_{model_name}_{run}.log', 'w') as f:
                    f.write(out.getvalue())
                model.save(model_path)

    for env_id in env_ids:
        for model_name, _ in models.items():
            returns = []
            for run in range(train_runs):
                model_path = f'{models_dir}/{env_id}_{model_name}_{run}.zip'
                model = _str_to_class(model_name).load(model_path)
                for test_run in range(test_runs):
                    return_ = _test_loop(model, env_id)
                    returns.append(return_)
            print(f'{model_name} on {env_id} returns return avg: {np.mean(returns)}, std: {np.std(returns)}')


if __name__ == '__main__':
    main()
