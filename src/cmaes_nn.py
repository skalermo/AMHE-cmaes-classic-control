import pickle
from typing import Callable, Union, Tuple, Optional

import gym
import numpy as np
from cmaes import CMA

from src.nn import NN
from src.env_info import ActionType, env_to_action_type


class CMAESNN:
    def __init__(self, env_id: str, max_nn_params: Union[str, int] = 'standard',
                 cmaes_sigma: float = 1.3, seed: int = 0, pop_size: Optional[int] = None,
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
            population_size=pop_size,
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
        state_size, actions_size = CMAESNN._extract_env_info(env, action_type)
        return NN(state_size, actions_size, action_type, max_nn_parameters)

    def learn(self, total_timesteps: int = 500_000, log_interval: int = 100):
        if self.verbose:
            print('Start learning')
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
        episode = 0

        while num_timesteps < total_timesteps:
            episode += 1
            solutions = []
            timesteps_list = []
            population_returns = []

            for _ in range(self.optimizer.population_size):
                w = self.optimizer.ask()
                model.set_weights(w)
                return_obtained, timesteps_used = _evaluate_model(model, self.env)
                solutions.append((w, -return_obtained))
                population_returns.append(return_obtained)
                timesteps_list.append(timesteps_used)

            self.optimizer.tell(solutions)
            best_offspring_idx_in_episode = np.argmax([-r[1] for r in solutions])
            best_offspring_in_episode = solutions[best_offspring_idx_in_episode]
            num_timesteps += timesteps_list[best_offspring_idx_in_episode]

            population_avg_return = np.average(population_returns)
            if population_avg_return > best_avg_return:
                best_offspring = best_offspring_in_episode

            if self.verbose and episode % log_interval == 0:
                population_best_return = population_returns[best_offspring_idx_in_episode]
                population_return_std = np.std(population_returns)
                print(f'{episode=} {population_best_return=} {population_avg_return=} {population_return_std=} total_timesteps={num_timesteps}')

        self.model.set_weights(best_offspring[0])

    def predict(self, observation: np.ndarray) -> [Union[int, list], None]:
        action = self.model.map_to_action(observation)
        if self.action_type == ActionType.Continuous:
            action = [action]
        return action, None

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
