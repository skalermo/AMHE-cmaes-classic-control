import unittest
import os

import gym

from src.cmaes_agent import CMAESAgent
from env_info import env_to_action_type


class E2E(unittest.TestCase):
    def test_runs_ok_in_every_env(self):
        def _run_in_env(env_id):
            model = CMAESAgent(env_id, cmaes_population_size=5, max_nn_params='minimal')
            model.learn(total_episodes=1)
            env = gym.make(env_id)
            obs = env.reset()
            action = model.predict(obs)
            env.step(action)

        for env_id in env_to_action_type.keys():
            try:
                _run_in_env(env_id)
            except Exception as e:
                self.fail(f'{e=}')

    def test_reaches_end_of_env(self):
        env_id = 'CartPole-v0'
        model = CMAESAgent(env_id, cmaes_population_size=5, max_nn_params='minimal')
        model.learn(total_episodes=1)
        env = gym.make(env_id)
        obs = env.reset()
        done = False
        for _ in range(100):
            action = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            if done:
                break
        self.assertTrue(done)

    def test_can_save_load_in_every_env(self):
        model_path = '.test_model'

        def _train_save_load_run(env_id):
            model = CMAESAgent(env_id, cmaes_population_size=5, max_nn_params='minimal')
            model.learn(total_episodes=1)
            model.save(model_path)
            del model
            model = CMAESAgent.load(model_path)
            env = gym.make(env_id)
            obs = env.reset()
            action = model.predict(obs)
            env.step(action)

        for env_id in env_to_action_type.keys():
            try:
                _train_save_load_run(env_id)
            except Exception as e:
                self.fail(f'{e=}')
        try:
            os.remove(model_path)
        except FileNotFoundError:
            pass


if __name__ == '__main__':
    unittest.main()
