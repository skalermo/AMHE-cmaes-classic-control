import gym

from src.cmaes_nn import CMAESNN


def main():
    # 'CartPole-v0', 'CartPole-v1', 'Acrobot-v1', 'MountainCar-v0', 'MountainCarContinuous-v0', 'Pendulum-v1'
    env_id = 'MountainCar-v0'
    env = gym.make(env_id)
    obs = env.reset()

    saved_model_path = f'./.data/models/{env_id}_CMAESNN_0.zip'
    if (model := CMAESNN.load(saved_model_path)) is None:
        print('Model not found. Please train the model first.')
        model = CMAESNN(env_id, verbose=True)
        model.learn(total_timesteps=1000, log_interval=100)
        model.save(saved_model_path)

    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()


if __name__ == '__main__':
    main()
