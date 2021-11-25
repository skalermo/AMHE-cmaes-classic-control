from enum import Enum, auto
import pickle

import gym
import numpy as np
from cmaes import CMA


class ActionType(Enum):
    Discrete = auto()
    Continuous = auto()


from nn import NN


env_to_action_type = {
    'CartPole-v0': ActionType.Discrete,
    'CartPole-v1': ActionType.Discrete,
    'Acrobot-v1': ActionType.Discrete,
    'MountainCar-v0': ActionType.Discrete,
    'MountainCarContinuous-v0': ActionType.Continuous,
    'Pendulum-v1': ActionType.Continuous,
}


class CMAESAgent:
    def __init__(self, env_id: str, cmaes_sigma: float = 1.3, cmaes_population_size: int = 30, seed: int = 0):
        if env_to_action_type.get(env_id) is not None:
            self.action_type = env_to_action_type.get(env_id)
            self.env = gym.make(env_id)
        else:
            raise 'Provided unknown environment'

        self.seed = seed
        self.model = self._create_nn(self.env, self.action_type)
        self.optimizer = CMA(
            mean=np.zeros(self.model.parameters_count()),
            sigma=cmaes_sigma, population_size=cmaes_population_size, seed=self.seed
        )

    @staticmethod
    def _create_nn(env: gym.Env, action_type: ActionType):
        state_size = env.observation_space.shape[0]
        if action_type == ActionType.Continuous:
            actions_size = env.action_space.shape[0]
        else:
            actions_size = env.action_space.n
        return NN(state_size, actions_size, action_type)

    def learn(self, total_episodes: int = 500, episode_length: int = 1000):
        def _evaluate_model(_model: NN, env: gym.Env) -> int:
            end_reward = 0
            state = env.reset()
            for time in range(episode_length):
                action = _model.map_to_action(state)
                if self.action_type == ActionType.Continuous:
                    action = [action]
                state, reward, done, info = env.step(action)
                if not done:
                    end_reward += reward
                else:
                    break
            return end_reward

        model = self._create_nn(self.env, self.action_type)
        best_weights = None
        best_reward = -1e12

        for e in range(total_episodes):
            solutions = []
            rewards = []
            for _ in range(self.optimizer.population_size):
                w = self.optimizer.ask()
                model.set_weights(w)
                reward_obtained = _evaluate_model(model, self.env)
                rewards.append(reward_obtained)
                if reward_obtained > best_reward:
                    best_reward = reward_obtained
                    best_weights = w
                solutions.append((w, -reward_obtained))
            self.optimizer.tell(solutions)
            print(f'{e=} {best_reward=} avg_reward={sum(rewards) / len(rewards)}')
        self.model.set_weights(best_weights)

    def predict(self, observation: np.ndarray):
        return self.model.map_to_action(observation)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path):
        try:
            with open(path, 'rb') as f:
                obj = pickle.load(f)
        except FileNotFoundError:
            return None
        return obj
