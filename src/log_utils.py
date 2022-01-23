from typing import Generator, Tuple
import sys
from contextlib import contextmanager
from io import StringIO


@contextmanager
def captured_output() -> Generator[Tuple[StringIO, StringIO], None, None]:
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def chunk_rollouts(logs: str) -> Generator[str, None, None]:
    while (start := logs.find("rollout")) != -1:
        end = logs.find("rollout", start + 1)
        if end == -1:
            end = len(logs)
        yield logs[start:end]
        logs = logs[end:]


def extract_data(chunk: str) -> dict:
    # ------------------------------------
    # | rollout / | |
    # | ep_len_mean | 16 |
    # | ep_rew_mean | 16 |
    # | time / | |
    # | fps | 918 |
    # | iterations | 4 |
    # | time_elapsed | 0 |
    # | total_timesteps | 20 |
    # | train / | |
    # | entropy_loss | -0.693 |
    # | explained_variance | -0.00313 |
    # | learning_rate | 0.0007 |
    # | n_updates | 3 |
    # | policy_loss | 2.07 |
    # | value_loss | 10.7 |
    # ------------------------------------
    data = {'ep_rew_mean': None, 'iterations': None, 'total_timesteps': None}
    for line in chunk.split('\n'):
        words = line.split()
        if len(words) >= 2 and words[1].strip() in data.keys():
            value = float(words[3])
            data[words[1].strip()] = value
    return data


def process_logs(logs: str) -> list:
    data = []
    for chunk in chunk_rollouts(logs):
        data.append(extract_data(chunk))
    return data


def process_cmaess_nn_logs(logs: str) -> list:
    data = []
    start_idx = logs.find('Start learning')
    lines = logs[start_idx:].split('\n')[1:]
    for line in lines:
        if not line:
            continue
        # episode = 1 population_best_return = 9.0 population_avg_return = 11.375 population_return_std = 2.6896793489187516 total_timesteps = 8
        _data = {'episode': None, 'population_best_return': None, 'population_avg_return': None, 'population_return_std': None, 'total_timesteps': None}
        words = line.split()
        for word in words:
            k, v = word.split('=')
            _data[k] = float(v)
        data.append(_data)
    return data
