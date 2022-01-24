from typing import Callable, List


def aggregate_and_apply(processed_datas: List[List], key: str, fn: Callable) -> list:
    return [fn([r.get(key) for r in rows]) for rows in zip(*processed_datas)]
