import matplotlib.pyplot as plt
import seaborn as sns
# plt.style.use(['science','ieee'])
import tensorflow as tf
import pandas as pd
import numpy as np
import stormpy
import shutil
from datetime import datetime, timedelta

import scipy.stats

import math
import time
import os
import csv
from pathlib import Path
from datetime import datetime
import json
from transitions.extensions import GraphMachine as Machine
import os
from os import walk

import utils

BASE_OUTPUT_PATH = 'data/output'
BASE_CACHE_PATH = 'data/cache'
PRISM_ENVS_PATH = 'data/input/envs/prism'
DPI = 400

def cache_pdtmc(pdtmc_str):
    """ Outputs pdtmc_str in args with filename established from current timestamp. """
    dt = datetime.now()
    dt_str = dt.strftime("%Y%m%d%H%M%S%f")
    fn = f'{BASE_CACHE_PATH}/{dt_str}pdtmc.prism'
    pdtmc_str = f'// automatically generated @ {dt}\n' + pdtmc_str
    with open(fn, 'w') as file:
        file.write(pdtmc_str)
    return fn

def clear_cache(dt = None):
    if dt is None: # delete cache created before 10 minutes ago.
        dt = datetime.now()
        dt = dt - timedelta(minutes = 10)
    dt_str = dt.strftime("%Y%m%d%H%M%S%f")
    dt_int = int(dt_str[:20])
    for (dirpath, dirnames, filenames) in walk(f'{BASE_CACHE_PATH}'):
        pass
    for f_idx, fn in enumerate(filenames):
        if not fn.endswith('.prism'):
            continue
        fn_dt_int = int(fn[:20])
        if fn_dt_int < dt_int:
            fn_whole = dirpath + '/' + fn
            os.remove(fn_whole)

class Log:

    def __init__(self, experiment):

        self.experiment = experiment
        self.cfgs = experiment.cfgs
        self.num_cfgs = len(experiment.cfgs)
        self.num_runs = experiment.num_runs

        self.base_output_dir = f'data/output/{self.experiment.name}'
        self._create_dirs()

        self.time = time.time()

    def flush(self, cfg_idx, run_idx, **kwargs):
        """ Flushes kwargs to output folder. """

        for name in kwargs:
            value = kwargs[name]
            self._to_file(cfg_idx, run_idx, name, value)
        self._to_file(cfg_idx, run_idx, 'clock', time.time() - self.time) # also output time.

    def _create_dirs(self):
        """ Creates a folder for this log. """
        for c in range(self.num_cfgs):
            for r in range(self.num_runs):
                output_dir = f'{self.base_output_dir}/{c}/{r}'
                if os.path.exists(output_dir):
                    shutil.rmtree(output_dir)
                Path(output_dir).mkdir(parents = True, exist_ok = False)
            cfg = self.cfgs[c]
            cfg_output_file = f'{self.base_output_dir}/{c}/cfg.json'
            with open(cfg_output_file, 'w') as file:
                json.dump(cfg, file, indent = 4)

    def _to_file(self, cfg_idx, run_idx, name, value):
        """ Writes or appends content of value to file. """
        output_dir = f'{self.base_output_dir}/{cfg_idx}/{run_idx}'
        if isinstance(value, float) or isinstance(value, int):
            value = str(value)
            filename = f'{output_dir}/{name}.txt'
            with open(filename, 'a') as file:
                file.write(value + '\n')
        elif isinstance(value, np.ndarray):
            value = np.array2string(value)
            filename = f'{output_dir}/{name}.txt'
            with open(filename, 'a') as file:
                file.write(value + '\n')
        elif isinstance(value, dict):
            filename = f'{output_dir}/{name}.txt'
            with open(filename, 'a') as file:
                json.dump(value, file, indent = 4)













    @staticmethod
    def plot_action_distribution(fsc, pomdp, filename):
        """
        Plots the action distribution of policy in args to filename. The pomdp is used to retrieve labeling.

        """

        sqrt_nM = np.sqrt(fsc.nM_generated)
        nrows = int(sqrt_nM)
        ncols = math.ceil(fsc.nM_generated / sqrt_nM)
        figsize = (ncols * 3.50, nrows * 2)

        nM_idx = 0
        if fsc.nM_generated == 1:
            data = fsc.action_distributions[nM_idx]
            sns.heatmap(data, vmin = 0, vmax = 1, yticklabels = pomdp.observation_labels)
        else:
            fig, axs = plt.subplots(nrows = nrows, ncols = ncols, sharex = True, sharey = True, figsize = figsize)
            for ax_idx, ax in enumerate(axs.reshape(-1)):
                    if nM_idx >= fsc.nM_generated:
                        break
                    ax.set_title(f'Memory node {nM_idx}')
                    ax.set_xlabel('Action index')
                    ax.set_ylabel('Observation')
                    data = fsc.action_distributions[nM_idx]
                    sns.heatmap(data, vmin = 0, vmax = 1, ax = ax)
                    ax.set_yticks(np.arange(len(pomdp.observation_labels)) - 0.5)
                    ax.set_yticklabels(pomdp.observation_labels)
                    nM_idx += 1

            for remaining_idx in range(ncols * nrows - nM_idx):
                np.array(axs).flatten()[remaining_idx].axis('off')

        plt.plot()
        plt.savefig(filename)
        plt.close()

        utils.inform(f'Plotted FSC action distribution to {filename}.')

        return

    @staticmethod
    def plot_memory_update(fsc, pomdp, filename):
        if fsc.is_randomized:
            raise NotImplementedError('To be implemented (or not).')

        transitions = []
        nodes = set()
        for m in fsc.M:
            nodes.add(f'm: {m}')
            for o, next_m in enumerate(fsc.next_memories[fsc.index_of_M[m]]):
                trigger_str = pomdp.observation_labels[o]
                transitions.append({'source' : f'm: {m}', 'trigger' : trigger_str, 'dest' : f'm: {next_m}'})
                nodes.add(f'm: {next_m}')
        nodes = list(nodes)

        machine = Machine(states = nodes, transitions = transitions, initial = 'm: 0', title = f'Memory update rule')
        machine.get_graph().draw(filename, prog = 'dot')

        utils.inform(f'Plotted FSC memory update to {filename}.')

        return

    def _plot_action_distribution(self, policy, filename):

        utils.inform(f'Plotting FSC action distribution to {filename}')

        sqrt_nM = np.sqrt(policy.nM_generated)
        nrows = int(sqrt_nM)
        ncols = math.ceil(policy.nM_generated / sqrt_nM)
        figsize = (nrows * 10, ncols * 1.75)

        observation_labels = [utils.labels_to_str(o) for o in self.instance.pomdp.observation_labels]
        nM_idx = 0
        if policy.nM_generated == 1:
            data = policy.action_distributions[nM_idx]
            sns.heatmap(data, vmin = 0, vmax = 1, yticklabels = observation_labels)
        else:
            fig, axs = plt.subplots(nrows = nrows, ncols = ncols, sharex = True, sharey = True, figsize = figsize)
            for ax_idx, ax in enumerate(axs.reshape(-1)):
                    if nM_idx >= policy.nM:
                        break
                    ax.set_title(f'Memory node {nM_idx}')
                    ax.set_xlabel('Action index')
                    ax.set_ylabel('Observation')
                    data = policy.action_distributions[nM_idx]
                    sns.heatmap(data, vmin = 0, vmax = 1, ax = ax)
                    ax.set_yticks(np.arange(len(observation_labels)))
                    ax.set_yticklabels(observation_labels)
                    nM_idx += 1

            for remaining_idx in range(ncols * nrows - nM_idx):
                np.array(axs).flatten()[remaining_idx].axis('off')

        plt.plot()
        plt.savefig(filename)
        plt.close()

        return

    def _plot_memory_update(self, policy, filename):

        utils.inform(f'Plotting memory update rule to {filename}.')

        if policy.is_randomized:
            filename = filename[:-4]
            for o_idx in range(self.instance.pomdp.nO):
                observation_label = self.instance.pomdp.observation_labels[o_idx]
                observation_label_str = utils.labels_to_str(observation_label)
                memory_nodes = [f'm: {m}' for m in range(self.instance.nM)]

                transitions = []
                for m, memory_node in enumerate(memory_nodes):
                    for next_m, next_memory_node in enumerate(memory_nodes):
                        trigger_str = '%.2f' % policy.next_memories[m, o_idx, next_m]
                        transitions.append({'source' : f'm: {m}', 'trigger' : trigger_str, 'dest' : f'm: {next_m}'})

                machine = Machine(states = memory_nodes, transitions = transitions, initial = 'm: 0', title = f'Randomized memory update rule for {observation_label_str}')

                observation_filename = filename + '_' + observation_label_str + '.png'
                machine.get_graph().draw(observation_filename, prog = 'dot')

        else:
            reachable = policy.M

            transitions = []
            nodes = set()
            for m in reachable:
                for o_idx in range(self.instance.pomdp.nO):
                    trigger_str = utils.labels_to_str(self.instance.pomdp.observation_labels[o_idx])
                    next_m = policy.next_memories[policy.index_of_M[m], o_idx]
                    transitions.append({'source' : f'm: {m}', 'trigger' : trigger_str, 'dest' : f'm: {next_m}'})
                    nodes.add(f'm: {m}')
            nodes = list(nodes)

            machine = Machine(states = nodes, transitions = transitions, initial = 'm: 0', title = f'Memory update rule')
            machine.get_graph().draw(filename, prog = 'dot')

    def _plot_pdtmc(self, pdtmc, filename, tol):
        """
        Plots this PDTMC in the form of a finite state machine.

        """

        if len(pdtmc.model_parameters) != 0:
            raise NotImplementedError('Plotting of parameters is not yet implemented.')

        utils.inform(f'Plotting pDTMC to {filename}')

        state_labels = self.instance.pomdp.state_labels
        observation_labels = self.instance.pomdp.observation_labels
        state_observation_labels = [observation_labels[self.instance.pomdp.O[s]] for s in range(self.instance.pomdp.nS)]

        reachable_states = pdtmc.reachable_states(tol)
        states = np.arange(pdtmc.nS)

        if not pdtmc.init_state_is_real_state:
            node_labels = []
            for m in range(pdtmc.underlying_nM):
                for s in range(self.instance.pomdp.nS):
                    check_value = pdtmc.check_values[s, m]
                    if not np.isnan(check_value):
                        node_labels.append(\
                            f'm: {m}, s: {state_labels[s]}, o: {state_observation_labels[s]} (%.2f)' % check_value)
            node_labels = np.array(node_labels)

            states = states[reachable_states]

            transitions = []
            for state in  states:
                state_label = node_labels[state]
                for next_state in states:
                    next_state_label = node_labels[next_state]
                    trigger = pdtmc.T[state, 0, next_state]
                    if trigger >= 0.01:
                        trigger_str = '%.2f' % trigger
                        transitions.append({'source' : state_label, 'dest' : next_state_label, 'trigger' : trigger_str})

            name = 'Policy-induced DTMC'
            machine = Machine(states = list(node_labels[states]), transitions = transitions, initial = node_labels[0])

        else:
            state_labels = np.array(
                [f'm: {m}, s: {utils.labels_to_str(state)}, o: {state_observation_labels[s]} (%.2f)' % pdtmc.check_values[s, m]\
                    for s, state in enumerate(state_labels) for m in range(pdtmc.underlying_nM)])
            states = states[reachable_states]

            transitions = []
            for state in  states:
                state_label = state_labels[state]
                for next_state in states:
                    next_state_label = state_labels[next_state]
                    trigger = pdtmc.T[state, 0, next_state]
                    if trigger >= 0.01:
                        trigger_str = '%.2f' % trigger
                        transitions.append({'source' : state_label, 'dest' : next_state_label, 'trigger' : trigger_str})

            name = 'Policy-induced DTMC'
            machine = Machine(states = list(state_labels[states]), transitions = transitions, initial = state_labels[0])

        plot = machine.get_graph().draw(filename, prog = 'dot')
        return plot

    @staticmethod
    def plot_pdtmc(pdtmc, filename):
        used_state_labels = set()
        transitions = []
        for state in pdtmc.model.states:
            for action in state.actions:
                for transition in action.transitions:
                    trigger_str = '%.3f' % transition.value()
                    next_state = transition.column
                    if state.id in pdtmc.reachable and next_state in pdtmc.reachable and pdtmc.T[state.id, 0, next_state] > 0:
                        state_label = f'{state.id}, {pdtmc.state_labels[state.id]}'
                        next_state_label = f'{transition.column}, {pdtmc.state_labels[transition.column]}'
                        transitions.append({'source' : state_label, 'dest' : next_state_label, 'trigger' : trigger_str})
                        used_state_labels.add(state_label)
                        used_state_labels.add(next_state_label)

        name = 'Policy-induced DTMC'
        machine = Machine(states = list(used_state_labels), transitions = transitions, initial = f'{0}, {pdtmc.state_labels[0]}')
        machine.get_graph().draw(filename, prog = 'dot')

        utils.inform(f'Plotted policy-induced DTMC to {filename}.')

        return

    @staticmethod
    def output(logs, *args):
        _dir = f'{BASE_OUTPUT_PATH}/{logs[0].experiment_name}_{logs[0].num_runs}'

        plot_dir = f'{_dir}/plots'
        txt_dir = f'{_dir}/txt'
        meta_dir = f'{_dir}/meta'
        graph_dir = f'{_dir}/graphs'

        Path(plot_dir).mkdir(parents = True, exist_ok = True)
        Path(txt_dir).mkdir(parents = True, exist_ok = True)
        Path(meta_dir).mkdir(parents = True, exist_ok = True)

        InOut.output_cfgs(logs, meta_dir)

        for arg in args:
            func_name = f'output_{arg}'
            func = getattr(InOut, func_name)
            func(logs, plot_dir)

    @staticmethod
    def output_learning_losses(logs, plot_dir):
        num_logs = len(logs)
        run_logs = np.array(logs).reshape(int(num_logs / logs[0].num_runs), logs[0].num_runs)

        fig, axs = plt.subplots(1, 3, sharex = False, sharey = False, figsize = (15, 1.75))
        axs[0].set_title('Policy loss')
        axs[1].set_title('Reconstruction error over experienced memories')
        axs[2].set_title('Reconstruction error over critical memories')

        for run_log in run_logs:

            # Policy loss.
            ax = axs[0]

            y_p = np.stack([log.policy_loss for log in run_log])
            mean_p = np.mean(y_p, axis = 0).flatten()
            std_p = np.std(y_p, axis = 0).flatten()

            g = sns.lineplot(x = range(len(mean_p)), y = mean_p, label = f'$L_p$ ({run_log[0].plot_label})', ax = ax)
            color = g.lines[-1].get_color()
            ax.fill_between(range(len(mean_p)), mean_p - std_p, mean_p + std_p, alpha = 0.2)
            ax.set_ylabel('$L_p$')
            ax.set_xlabel('Rounds times epochs')

            # Reconstruction loss over experienced memories.
            ax = axs[1]

            y_r = np.stack([log.reconstruction_error for log in run_log])
            mean_r = np.mean(y_r, axis = 0).flatten()
            std_r = np.std(y_r, axis = 0).flatten()

            sns.lineplot(x = range(len(mean_r)), y = mean_r, label = f'$L_r$ ({run_log[0].plot_label})', color = color, ax = ax)
            ax.fill_between(range(len(mean_r)), mean_r - std_r, mean_r + std_r, alpha = 0.2, color = color)

            ax.set_ylabel('$L_r$')
            ax.set_xlabel('Round')

            # Reconstruction loss over experienced memories.
            ax = axs[2]

            y_r_crit = np.stack([log.reconstruction_error_critical for log in run_log])
            mean_r_crit = np.mean(y_r_crit, axis = 0).flatten()
            std_r_crit = np.std(y_r_crit, axis = 0).flatten()

            sns.lineplot(x = range(len(mean_r_crit)), y = mean_r_crit, label = f'$L_r$ ({run_log[0].plot_label})', color = color, ax = ax)
            ax.fill_between(range(len(mean_r_crit)), mean_r_crit - std_r_crit, mean_r_crit + std_r_crit, alpha = 0.2, color = color)

            ax.set_ylabel('$L_r$')
            ax.set_xlabel('Round')

        plt.savefig(f'{plot_dir}/learning_losses.png', dpi = DPI)
        plt.close()

    @staticmethod
    def output_benchmark_table(logs, dir):

        num_logs = len(logs)
        run_logs = np.array(logs).reshape(int(num_logs / logs[0].num_runs), logs[0].num_runs)

        output = {}
        for run_log in run_logs:
            results_at_init = []
            durations = []
            values = []
            cum_rewards = []
            q_cum_rewards = []
            ks = []
            for log in run_log:
                (not_nans, ) = np.where(np.logical_not(np.isnan(log.result_at_init)))
                last_valid_index = not_nans[-1]
                results_at_init.append(log.result_at_init[last_valid_index])
                durations.append(log.duration[last_valid_index])
                values.append(log.value_at_init[last_valid_index])
                cum_rewards.append(log.cum_rewards[last_valid_index])
                q_cum_rewards.append(log.q_cum_rewards[last_valid_index])
                ks.append(log.k[last_valid_index])
            plot_label = run_log[0].plot_label
            output[plot_label] = {
                'result_at_init' : {
                    'mean' : np.mean(results_at_init),
                    'std' : np.std(results_at_init),
                    'min' : np.min(results_at_init),
                    'max' : np.max(results_at_init)
                    },
                'duration' : {
                    'mean' : np.mean(durations),
                    'std' : np.std(durations),
                    'min' : np.min(durations),
                    'max' : np.max(durations)
                    },
                'value' : {
                    'mean' : np.mean(values),
                    'std' : np.std(values),
                    'min' : np.min(values),
                    'max' : np.max(values)
                    },
                'cum_reward' : {
                    'mean' : np.mean(cum_rewards),
                    'std' : np.std(cum_rewards),
                    'min' : np.min(cum_rewards),
                    'max' : np.max(cum_rewards)
                    },
                'q_cum_reward' : {
                    'mean' : np.mean(q_cum_rewards),
                    'std' : np.std(q_cum_rewards),
                    'min' : np.min(q_cum_rewards),
                    'max' : np.max(q_cum_rewards)
                    },
                'k' : {
                    'mean' : np.mean(ks),
                    'std' : np.std(ks),
                    'min' : np.min(ks),
                    'max' : np.max(ks)
                    }
                }

        with open(f'{dir}/table.json', 'w+') as fp:
            json.dump(output, fp, indent = 4)

        return

        fig, ax = plt.subplots(figsize = (5, 1.75))
        for run_log in run_logs:
            y = np.stack([log.result_at_init for log in run_log])
            mean = np.mean(y, axis = 0)
            std = np.std(y, axis = 0)
            g = sns.lineplot(x = run_log[0].index, y = mean, label = run_log[0].plot_label, ax = ax)
            g.fill_between(run_log[0].index, mean - std, mean + std, color = plt.gca().lines[-1].get_color(), alpha = 0.2)

        if len(run_log[0].index) < 20:
            g.set_xticks(run_log[0].index)
            g.set_xticklabels(run_log[0].index + 1)

        if not 'R' in run_log[0].cfg['formula_str']:
            tenths = np.arange(11) / 10
            g.set_yticks(tenths)
            g.set_yticklabels(tenths)

        plt.ylabel('$P(\phi)$ at initial state')
        plt.xlabel('$r$')
        plt.title('$P(\phi)$ at initial state over rounds')
        plt.savefig(f'{dir}/result_at_init.png', dpi = DPI)
        plt.close()

    @staticmethod
    def output_duration(logs, num_runs, dir):

        num_logs = len(logs)
        run_logs = np.array(logs).reshape(int(num_logs / num_runs), num_runs)

        fig, ax = plt.subplots(figsize = (5, 1.75))
        for run_log in run_logs:
            y = np.stack([np.cumsum(log.duration, axis = -1) for log in run_log])
            mean = np.mean(y, axis = 0)
            std = np.std(y, axis = 0)
            g = sns.lineplot(x = run_log[0].index, y = mean, label = run_log[0].plot_label, ax = ax)
            g.fill_between(run_log[0].index, mean - std, mean + std, color = plt.gca().lines[-1].get_color(), alpha = 0.2)

        if len(run_log[0].index) < 20:
            g.set_xticks(run_log[0].index)
            g.set_xticklabels(run_log[0].index + 1)

        plt.ylabel('Total duration ($s$)')
        plt.xlabel('$r$')
        plt.title('Total duration ($s$) over rounds')
        plt.savefig(f'{dir}/time.png', dpi = DPI)
        plt.close()

    @staticmethod
    def output_memory_usage(logs, num_runs, dir):
        num_logs = len(logs)
        run_logs = np.array(logs).reshape(int(num_logs / num_runs), num_runs)

        fig, axs = plt.subplots(1, 3, sharex = True, sharey = True, figsize = (15, 1.75))

        axs[0].set_ylabel('$n_M$')
        axs[0].set_xlabel('$r$')
        axs[0].set_title('Memory nodes generated by RNN')
        axs[1].set_ylabel('$n_E$')
        axs[1].set_xlabel('$r$')
        axs[1].set_title('Memory nodes exp. in simulation')
        axs[2].set_ylabel('$n_C$')
        axs[2].set_xlabel('$r$')
        axs[2].set_title('Critical memory nodes in simulation')

        for run_log in run_logs:
            y = np.stack([log.n_q for log in run_log])
            mean = np.mean(y, axis = 0)
            std = np.std(y, axis = 0)

            sns.lineplot(run_log[0].index, mean, label = run_log[0].plot_label, ax = axs[0])
            axs[0].fill_between(run_log[0].index, mean - std, mean + std, alpha = 0.2)

            y_exp = np.stack([log.n_q_experienced for log in run_log])
            mean_exp = np.mean(y_exp, axis = 0)
            std_exp = np.std(y_exp, axis = 0)
            sns.lineplot(x = run_log[0].index, y = mean_exp, label = f'{run_log[0].plot_label}', ax = axs[1])
            axs[1].fill_between(run_log[0].index, mean_exp - std_exp, mean_exp + std_exp, alpha = 0.2)

            y_crit = np.stack([log.n_q_critical for log in run_log])
            mean_crit = np.mean(y_exp, axis = 0)
            std_crit = np.std(y_exp, axis = 0)
            sns.lineplot(x = run_log[0].index, y = mean_crit, label = f'{run_log[0].plot_label}', ax = axs[2])
            axs[2].fill_between(run_log[0].index, mean_crit - std_crit, mean_crit + std_crit, alpha = 0.2)

        if len(run_log[0].index) < 20:
            axs[0].set_xticks(run_log[0].index)
            axs[0].set_xticklabels(run_log[0].index + 1)
            axs[1].set_xticks(run_log[0].index)
            axs[1].set_xticklabels(run_log[0].index + 1)
            axs[2].set_xticks(run_log[0].index)
            axs[2].set_xticklabels(run_log[0].index + 1)

        plt.savefig(f'{dir}/memory_usage.png', dpi = DPI)
        plt.close()

    @staticmethod
    def output_entropy(logs, num_runs, dir):
        """
        Outputs a graph of mean entropy over critical state, memory pairs.

        """

        num_logs = len(logs)
        run_logs = np.array(logs).reshape(int(num_logs / num_runs), num_runs)

        fig, ax = plt.subplots(figsize = (5, 1.75))

        for run_log in run_logs:
            y = np.stack([log.mean_entropies for log in run_log])
            mean = np.mean(y, axis = 0)
            std = np.std(y, axis = 0)
            g = sns.lineplot(x = run_log[0].index, y = mean, label = run_log[0].plot_label, ax = ax)
            g.fill_between(run_log[0].index, mean - std, mean + std, color = plt.gca().lines[-1].get_color(), alpha = 0.2)

        if len(run_log[0].index) < 20:
            g.set_xticks(run_log[0].index)
            g.set_xticklabels(run_log[0].index + 1)

        plt.ylabel('Entropy')
        plt.xlabel('$r$')
        plt.title('Entropy over critical state-memory pairs')
        plt.savefig(f'{dir}/entropy.png', dpi = 500)
        plt.close()

def pdtmc_string(parametric_string, nS, nM, transitions_strings, label_strings, reward_str):
    contents = f"""
dtmc

{parametric_string}

module die

    s : [0..{nS * nM - 1}] init 0;

    {transitions_strings}

endmodule

rewards
    {reward_str}
endrewards

{label_strings}

"""

    return contents
