from os import listdir
from os.path import isfile, join

import numpy as np
import matplotlib.pyplot as plt

from src.env_info import env_to_action_type
from src.log_utils import process_logs, process_cmaess_nn_logs
from plotting.utils import aggregate_and_apply


plt.style.use('ggplot')
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 10))

log_path = '../.data/logs'
all_logs = [f'{log_path}/{f}' for f in listdir(log_path) if isfile(join(log_path, f))]

envs = env_to_action_type.keys()
# envs = ['MountainCarContinuous-v0']
models = ['A2C', 'PPO', 'CMAESNN']
colors = ['#E24A33', '#348ABD', '#988ED5']

for i, env_id in enumerate(envs):
    for m, c in zip(models[::-1], colors[::-1]):
        logs = [f for f in all_logs if env_id in f and m in f and '4' in f]

        if m == 'CMAESNN':
            process_fn = process_cmaess_nn_logs
            return_key = 'population_avg_return'
        else:
            process_fn = process_logs
            return_key = 'ep_rew_mean'

        with open(logs[0], 'r') as f:
            processed = process_fn(f.read())

        return_avg = aggregate_and_apply([processed], return_key, lambda x: x[0])
        return_std = aggregate_and_apply([processed], 'population_return_std', lambda x: x[0])
        timesteps = aggregate_and_apply([processed], 'total_timesteps', lambda x: x[0])
        if m == 'CMAESNN':
            every_nth = 10
            return_avg = return_avg[::every_nth]
            return_std = return_std[::every_nth]
            timesteps = timesteps[::every_nth]
        a = ax[i // 3, i % 3]
        if env_id == 'MountainCarContinuous-v0':
            a.set_yscale('symlog')
        else:
            a.set_yscale('linear')
        a.set_xlabel('Timestep')
        a.set_ylabel('Return')
        a.set_title(env_id)

        plot = a.plot(timesteps, return_avg, label=m, color=c)
        # c = plot[-1].get_color()
        # print(c)
        if m == 'CMAESNN':
            a.fill_between(timesteps, np.asarray(return_avg) - np.asarray(return_std), np.asarray(return_avg) + np.asarray(return_std),
                            alpha=0.2, color=c)

handles, labels = ax[-1][-1].get_legend_handles_labels()
fig.tight_layout()
fig.legend(handles, labels, loc='center left')

# ax.legend()
plt.show()
fig.savefig(str(__file__).split('.')[0] + '4.pdf', bbox_inches='tight')
