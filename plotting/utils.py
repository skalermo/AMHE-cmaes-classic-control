from typing import Callable, List

import numpy as np


def aggregate_and_apply(processed_datas: List[List], key: str, fn: Callable) -> list:
    return [fn([r.get(key) for r in rows]) for rows in zip(*processed_datas)]


def avg_stds(stds: List[float]) -> float:
    variations = list(map(lambda x: x ** 2, stds))
    return np.sqrt(np.mean(variations))
