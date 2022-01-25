from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt

from src.env_info import env_to_action_type
from src.log_utils import process_logs, process_cmaess_nn_logs
from plotting.utils import aggregate_and_apply


plt.style.use('ggplot')
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 10))

log_path = '../.data/logs'
all_logs = [f'{log_path}/{f}' for f in listdir(log_path) if isfile(join(log_path, f))]

envs = env_to_action_type.keys()
models = ['A2C', 'PPO', 'CMAESNN']
colors = ['#E24A33', '#348ABD', '#988ED5']

for i, env_id in enumerate(envs):
    for m, c in zip(models[::-1], colors[::-1]):
        logs = [f for f in all_logs if env_id in f and m in f and '0' in f]

        if m == 'CMAESNN':
            process_fn = process_cmaess_nn_logs
        else:
            process_fn = process_logs

        with open(logs[0], 'r') as f:
            processed = process_fn(f.read())

        time_elapsed = aggregate_and_apply([processed], 'time_elapsed', lambda x: x[0])
        timesteps = aggregate_and_apply([processed], 'total_timesteps', lambda x: x[0])

        if m == 'CMAESNN':
            every_nth = 10
            time_elapsed = time_elapsed[::every_nth]
            timesteps = timesteps[::every_nth]

        a = ax[i // 3, i % 3]
        a.set_xlabel('Timestep')
        a.set_ylabel('Time')
        a.set_title(env_id)

        plot = a.plot(timesteps, time_elapsed, label=m, color=c)

handles, labels = ax[-1][-1].get_legend_handles_labels()
fig.tight_layout()
fig.legend(handles, labels, loc='center left')

plt.show()
fig.savefig(str(__file__).split('.')[0] + '0.pdf', bbox_inches='tight')

