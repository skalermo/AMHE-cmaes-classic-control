import pickle
from typing import Callable, Union

import gym
import numpy as np
from cmaes import CMA

from src.nn import NN
from env_info import ActionType, env_to_action_type


class CMAESAgent:
    def __init__(self, env_id: str, max_nn_params: Union[str, int] = 'standard', seed: int = 0,
                 cmaes_sigma: float = 1.3, cmaes_population_size: int = 30,
                 model: NN = None, verbose=False):
        self.env_id = env_id
        self.verbose = verbose
        if env_to_action_type.get(env_id) is not None:
            self.action_type = env_to_action_type.get(env_id)
            self.env = gym.make(env_id)
        else:
            raise 'Provided unknown environment'

        self.seed = seed
        self.create_nn: Callable = lambda: self._create_nn(self.env, self.action_type, max_nn_params)
        if model is not None:
            self.model = model
        else:
            self.model = self.create_nn()
        self.optimizer = CMA(
            mean=np.zeros(self.model.parameters_count()),
            sigma=cmaes_sigma, population_size=cmaes_population_size, seed=self.seed
        )
        if verbose:
            self.info()

    def info(self):
        print(self.env_id, self._extract_env_info(self.env, self.action_type))
        print(self.model)
        print(f'NN: hidden={self.model.hidden}, parameters={self.model.parameters_count()}')

    @staticmethod
    def _extract_env_info(env: gym.Env, action_type: ActionType) -> tuple:
        state_size = env.observation_space.shape[0]
        if action_type == ActionType.Continuous:
            actions_size = env.action_space.shape[0]
        else:
            actions_size = env.action_space.n
        return state_size, actions_size

    @staticmethod
    def _create_nn(env: gym.Env, action_type: ActionType, max_nn_parameters: Union[str, int]) -> NN:
        state_size, actions_size = CMAESAgent._extract_env_info(env, action_type)
        return NN(state_size, actions_size, action_type, max_nn_parameters)

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

        model = self.create_nn()
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
            if self.verbose:
                print(f'{e=} {best_reward=} avg_reward={sum(rewards) / len(rewards)}')
        self.model.set_weights(best_weights)

    def predict(self, observation: np.ndarray) -> Union[int, list]:
        action = self.model.map_to_action(observation)
        if self.action_type == ActionType.Continuous:
            action = [action]
        return action

    def save(self, path):
        self.create_nn = None
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
