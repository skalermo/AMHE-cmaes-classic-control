from os import listdir
from os.path import isfile, join

import numpy as np
import matplotlib.pyplot as plt

from src.env_info import env_to_action_type
from src.log_utils import process_cmaess_nn_logs
from plotting.utils import aggregate_and_apply


plt.style.use('ggplot')
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 10))

log_path_no_bias = '../.data/logs'
no_bias_logs = [f'{log_path_no_bias}/{f}' for f in listdir(log_path_no_bias) if isfile(join(log_path_no_bias, f))]
no_bias = list(filter(lambda x: 'CMAESNN' in x, no_bias_logs))

log_path_with_bias = '../.data_bias/logs'
bias_logs = [f'{log_path_with_bias}/{f}' for f in listdir(log_path_with_bias) if isfile(join(log_path_with_bias, f))]
bias = list(filter(lambda x: 'CMAESNN' in x, bias_logs))

envs = env_to_action_type.keys()
# colors = ['#E24A33', '#348ABD', '#988ED5']
#
for i, env_id in enumerate(envs):
    for j, log in enumerate([no_bias, bias]):
        log_path = [f for f in log if env_id in f and '4' in f]
        with open(log_path[0], 'r') as f:
            processed = process_cmaess_nn_logs(f.read())

        return_avg = aggregate_and_apply([processed], 'population_avg_return', lambda x: x[0])
        return_std = aggregate_and_apply([processed], 'population_return_std', lambda x: x[0])
        timesteps = aggregate_and_apply([processed], 'total_timesteps', lambda x: x[0])

        every_nth = 20
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

        plot = a.plot(timesteps, return_avg, label=f'CMAESNN {"no" if j == 0 else "with"} bias')
        c = plot[-1].get_color()
        a.fill_between(timesteps, np.asarray(return_avg) - np.asarray(return_std), np.asarray(return_avg) + np.asarray(return_std),
                       alpha=0.2, color=c)


handles, labels = ax[-1][-1].get_legend_handles_labels()
fig.tight_layout()
fig.legend(handles, labels, loc='center left')

# ax.legend()
plt.show()
fig.savefig(str(__file__).split('.')[0] + '4.pdf', bbox_inches='tight')
