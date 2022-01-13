import pickle
from typing import Callable, Union, Tuple

import gym
import numpy as np
from cmaes import CMA

from src.nn import NN
from src.env_info import ActionType, env_to_action_type


class CMAESAgent:
    def __init__(self, env_id: str, max_nn_params: Union[str, int] = 'standard',
                 cmaes_sigma: float = 1.3, seed: int = 0,
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
            sigma=cmaes_sigma, seed=self.seed,
        )
        if verbose:
            self.info()

    def info(self):
        print(self.env_id, self._extract_env_info(self.env, self.action_type))
        print(self.model)
        print(f'NN: hidden={self.model.hidden}, parameters={self.model.parameters_count()}')

    @staticmethod
    def _extract_env_info(env: gym.Env, action_type: ActionType) -> Tuple[int, int]:
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

    def learn(self, total_timesteps: int = 500_000, log_interval: int = 100):
        # assuming environments have episode limit of <= 1000
        episode_length = 1000

        def _evaluate_model(_model: NN, env: gym.Env) -> Tuple[int, int]:
            state = env.reset()
            cur_return = 0
            time = 0
            for time in range(episode_length):
                action = _model.map_to_action(state)
                if self.action_type == ActionType.Continuous:
                    action = [action]
                state, reward, done, info = env.step(action)
                cur_return += reward
                if done:
                    break
            return cur_return, time

        num_timesteps = 0
        model = self.create_nn()
        best_offspring = None
        best_avg_return = -np.inf
        iteration = 0

        while num_timesteps < total_timesteps:
            iteration += 1
            solutions = []
            timesteps_list = []
            offsprings_returns = []

            for _ in range(self.optimizer.population_size):
                w = self.optimizer.ask()
                model.set_weights(w)
                return_obtained, timesteps_used = _evaluate_model(model, self.env)
                solutions.append((w, -return_obtained))
                offsprings_returns.append(return_obtained)
                timesteps_list.append(timesteps_used)

            self.optimizer.tell(solutions)
            best_offspring_idx = np.argmax([-r[1] for r in solutions])
            best_offspring_in_episode = solutions[best_offspring_idx]
            num_timesteps += timesteps_list[best_offspring_idx]

            avg_return = np.average(offsprings_returns)
            if avg_return > best_avg_return:
                best_offspring = best_offspring_in_episode

            if self.verbose and iteration % log_interval == 0:
                best_return = offsprings_returns[best_offspring_idx]
                std_return = np.std(offsprings_returns)
                print(f'{iteration=} {best_return=} {avg_return=}, {std_return=}')
                print(f'Timesteps used: {num_timesteps}/{total_timesteps} ({round(num_timesteps / total_timesteps * 100, 2)}%)')

        self.model.set_weights(best_offspring[0])

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
