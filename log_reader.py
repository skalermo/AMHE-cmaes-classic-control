from typing import List, Callable
from os import listdir
from os.path import isfile, join

import numpy as np

from src.env_info import env_to_action_type
from src.log_utils import process_logs, process_cmaess_nn_logs


def aggregate_and_apply(processed_datas: List[List], key: str, fn: Callable) -> list:
    return [fn([r.get(key) for r in rows]) for rows in zip(*processed_datas)]


def avg_stds(stds: List[float]) -> float:
    variations = list(map(lambda x: x ** 2, stds))
    return np.sqrt(np.mean(variations))


log_path = '.data/logs'
all_logs = [f'{log_path}/{f}' for f in listdir(log_path) if isfile(join(log_path, f))]

# models = ['A2C', 'PPO', 'CMAESNN']
models = ['CMAESNN']

for i, env_id in enumerate(env_to_action_type.keys()):
    for m in models:
        process_fn = process_cmaess_nn_logs if m == 'CMAESNN' else process_logs
        reward_key = 'population_avg_return' if m == 'CMAESNN' else 'ep_rew_mean'

        logs = [f for f in all_logs if env_id in f and m in f]
        processed_datas = []
        for log in logs:
            with open(log, 'r') as f:
                processed = process_fn(f.read())
                processed_datas.append(processed)

        max_returns = []
        stds = []
        for data in processed_datas:
            # max_return = max([row.get(reward_key) for row in data])
            max_idx = np.argmax([row.get(reward_key) for row in data])
            max_row = data[max_idx]
            max_return = max_row.get(reward_key)
            std = max_row.get('population_return_std')
            max_returns.append(max_return)
            stds.append(std)

        print(f'{env_id} {m} max return: {np.mean(max_returns)}, std: {np.std(max_returns)}+{avg_stds(stds)}')
        print(f'{env_id} {m} max return: {np.mean(max_returns)}, std: {avg_stds([np.std(max_returns), *stds])}')

