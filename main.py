import gym

from src.cmaes_nn import CMAESNN


def main():
    env_id = 'CartPole-v0'
    model_name = '_'.join([CMAESNN.__name__, env_id])
    cmaes_nn = CMAESNN.load(model_name)
    if cmaes_nn is None:
        cmaes_nn = CMAESNN(env_id, max_nn_params='standard', verbose=True)
        cmaes_nn.learn(total_timesteps=123_456)
        cmaes_nn.save(model_name)

    env = gym.make(env_id)
    obs = env.reset()

    while True:
        action = cmaes_nn.predict(obs)
        obs, reward, done, info = env.step(action)
        if done:
            break
        env.render()


if __name__ == '__main__':
    main()
