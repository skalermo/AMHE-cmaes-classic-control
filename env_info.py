from enum import Enum, auto


class ActionType(Enum):
    Discrete = auto()
    Continuous = auto()


env_to_action_type = {
    'CartPole-v0': ActionType.Discrete,
    'CartPole-v1': ActionType.Discrete,
    'Acrobot-v1': ActionType.Discrete,
    'MountainCar-v0': ActionType.Discrete,
    'MountainCarContinuous-v0': ActionType.Continuous,
    'Pendulum-v1': ActionType.Continuous,
}
