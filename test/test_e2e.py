import unittest
import os

import gym

from src.cmaes_nn import CMAESNN
from src.env_info import env_to_action_type


class E2E(unittest.TestCase):
    def test_runs_ok_in_every_env(self):
        def _run_in_env(env_id):
            model = CMAESNN(env_id, max_nn_params='minimal')
            model.learn(total_timesteps=1)
            env = gym.make(env_id)
            obs = env.reset()
            action = model.predict(obs)
            env.step(action)

        for env_id in env_to_action_type.keys():
            with self.subTest(f'in env {env_id}'):
                try:
                    _run_in_env(env_id)
                except Exception as e:
                    self.fail(f'{e=}')

    def test_reaches_end_of_env(self):
        env_id = 'CartPole-v0'
        model = CMAESNN(env_id, max_nn_params='minimal')
        model.learn(total_timesteps=1)
        env = gym.make(env_id)
        obs = env.reset()
        done = False
        i = 0
        for i in range(100):
            action = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            if done:
                break
        self.assertTrue(done or i == 100 - 1)

    def test_can_save_load_in_every_env_for_every_param_config(self):
        model_path = '.test_model'
        params_configs = ['minimal', 'standard', 100]

        def _train_save_load_run(_env_id, _config):
            model = CMAESNN(_env_id, max_nn_params=_config)
            model.learn(total_timesteps=1)
            model.save(model_path)
            del model
            model = CMAESNN.load(model_path)
            env = gym.make(_env_id)
            obs = env.reset()
            action = model.predict(obs)
            env.step(action)

        for env_id in env_to_action_type.keys():
            for config in params_configs:
                with self.subTest(f'in env {env_id} for param config {config}'):
                    try:
                        _train_save_load_run(env_id, config)
                    except Exception as e:
                        self.fail(f'{e=}')
        try:
            os.remove(model_path)
        except FileNotFoundError:
            pass


if __name__ == '__main__':
    unittest.main()
