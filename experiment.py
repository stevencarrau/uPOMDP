import numpy as np
from copy import deepcopy

import stormpy
from scipy.stats import entropy
import tensorflow as tf
import pycarl
from joblib import Parallel, delayed
from mem_top import mem_top
import inspect

from net import Net
from instance import Instance
from check import Checker
from in_out import Log, clear_cache
import utils

import gridstorm.models as models

experiment_to_grid_model_names = {
    "avoid": models.surveillance,
    "refuel": models.refuel,
    'obstacle': models.obstacle,
    "intercept": models.intercept,
    'evade': models.evade,
    'rocks': models.rocks
}

model = experiment_to_grid_model_names["obstacle"]
model_constants = list(inspect.signature(model).parameters.keys())
constants = ["N=6"] #dict(item.split('=') for item in args.constants.split(","))
input = model(constants)
model = stormpy.parse_prism_program(input.path)
print("")

class Experiment:
    """ Represents a set of cfgs that serve an experiment. """

    def __init__(self, name, cfg, num_runs):
        self.name = name
        self.num_runs = num_runs
        self.cfgs = [cfg]
        self.cfg = cfg

    def add_cfg(self, new_cfg):
        configuration = deepcopy(self.cfg)
        for key in new_cfg:
            configuration[key] = new_cfg[key]
        self.cfgs.append(configuration)

    def execute(self, multi_thread):
        log = Log(self)
        if multi_thread:
            Parallel(n_jobs = 4)(delayed(self._run)(log, cfg_idx, run_idx) for cfg_idx in range(len(self.cfgs)) for run_idx in range(self.num_runs))
        else:
            for cfg_idx in range(len(self.cfgs)):
                for run_idx in range(self.num_runs):
                    self._run(log, cfg_idx, run_idx)
        utils.inform(f'Finished experiment {self.name}.', indent = 0, itype = 'OKGREEN')

    def _run(self, log, cfg_idx, run_idx):

        pycarl.clear_pools()
        tf.keras.backend.clear_session()

        utils.inform(f'Starting run {run_idx}.', indent = 0, itype = 'OKBLUE')

        cfg = self.cfgs[cfg_idx]
        instance = Instance(cfg)
        pomdp = instance.build_pomdp()
        ps = cfg['p_init']
        worst_ps = ps
        mdp = instance.build_mdp(ps)
        length = instance.simulation_length()
        checker = Checker(instance, cfg)
        net = Net(instance, cfg)

        for round_idx in range(cfg['rounds']):

            timesteps = np.repeat(np.expand_dims(np.arange(length), axis = 0), axis = 0, repeats = cfg['batch_dim'])

            beliefs, states, hs, observations, policies, actions, rewards = net.simulate(pomdp, mdp, greedy = False, length = length)

            num_actions = pomdp.num_choices_per_state[states]
            observation_labels = pomdp.observation_labels[observations]
            empirical_result, _ = utils.evaluate_performance(instance, states, rewards)
            valid = empirical_result < 4 * mdp.state_values[0]

            until = np.argmax(observation_labels == instance.label_to_reach, axis = -1)
            until[until == 0] = length - 1
            relevant_timesteps = timesteps < np.expand_dims(until, axis = -1)
            num_choices = pomdp.num_choices_per_state[states]
            log_policies = np.log(policies) / np.expand_dims(np.log(num_choices), axis = -1)
            log_policies[np.isinf(log_policies)] = 0
            all_entropies = np.sum(policies * - log_policies, axis = -1)
            relevant_entropies = all_entropies[np.logical_and(relevant_timesteps, num_actions > 1)]
            rnn_entropy = np.mean(relevant_entropies)

            relevant_hs = np.unique(hs[:, 1:][relevant_timesteps[:, 1:]], axis = 0)
            r_loss = net.improve_r(relevant_hs)
            utils.inform(f'{run_idx}-{round_idx}\t(RNN)\t\t\t%.4f' % empirical_result, indent = 0)
            utils.inform(f'{run_idx}-{round_idx}\t(QBN)\t\trloss \t%.4f' % r_loss[0] + '\t>>>> %3.4f' % r_loss[-1], indent = 0)

            fsc = net.extract_fsc(make_greedy = False, reshape = True)
            pdtmc = instance.instantiate_pdtmc(fsc, zero = 0)
            fsc_memories, fsc_policies = fsc.simulate(observations)
            log_fsc_policies = np.log(fsc_policies) / np.expand_dims(np.log(num_choices), axis = -1)
            log_fsc_policies[np.logical_not(np.isfinite(log_fsc_policies))] = 0
            all_entropies = np.sum(fsc_policies * - log_fsc_policies, axis = -1)
            relevant_entropies = all_entropies[np.logical_and(relevant_timesteps, num_actions > 1)]
            fsc_entropy = np.mean(relevant_entropies)

            all_cross_entropies = np.sum(fsc_policies * - log_policies, axis = -1)
            relevant_entropies = all_cross_entropies[np.logical_and(relevant_timesteps, num_actions > 1)]
            cross_entropy = np.mean(relevant_entropies) # buggy, can be > 1 for some unknown reason.

            check = checker.check_pdtmc(pdtmc)
            added = instance.add_fsc(check, fsc)
            utils.inform(f'{run_idx}-{round_idx}\t(%i-FSC)' % fsc.nM_generated + '\t\trinit \t%.4f' % check._lb_values[0] + '\t\t%.4f' % check._ub_values[0], indent = 0)

            if cfg['ctrx_gen'] == 'rnd' and pomdp.is_parametric:
                mdp, worst_value = instance.build_mdp(), 0
            elif cfg['ctrx_gen'] == 'crt' and added and pomdp.is_parametric:
                mdp, worst_ps, worst_value = instance.worst_mdp(check, fsc)
            elif cfg['ctrx_gen'] == 'crt_neg' and not added and pomdp.is_parametric:
                mdp, worst_ps, worst_value = instance.worst_mdp(check, fsc)
            elif cfg['ctrx_gen'] == 'rnd_full' and pomdp.is_parametric:
                mdp, worst_value = instance.build_mdp(), 0
            elif cfg['ctrx_gen'] == 'crt_full' and pomdp.is_parametric:
                mdp, worst_ps, worst_value = instance.worst_mdp(check, fsc)
            else:
                worst_value = -1
                worst_ps = {}

            length = instance.simulation_length()
            evalues = check.evaluate(cfg['p_evals'])

            if instance.objective == 'min':
                a_labels = utils.one_hot_encode(np.nanargmin(mdp.action_values[states], axis = -1), pomdp.nA, dtype = 'float32')
            else:
                a_labels = utils.one_hot_encode(np.nanargmax(mdp.action_values[states], axis = -1), pomdp.nA, dtype = 'float32')
            a_inputs = utils.one_hot_encode(observations, pomdp.nO, dtype = 'float32')
            a_loss = net.improve_a(a_inputs, a_labels)

            label_cross_entropies = np.sum(a_labels * - log_policies, axis = -1)
            relevant_entropies = label_cross_entropies[np.logical_and(relevant_timesteps, num_actions > 1)]
            label_cross_entropy = np.mean(label_cross_entropies)

            log.flush(cfg_idx, run_idx, nM = fsc.nM_generated, lb = check._lb_values[0], ub = check._ub_values[0],
                    ps = ps, mdp_value = mdp.state_values[0], max_distance = check.max_distance,
                    min_distance = check.min_distance,
                    evalues = evalues, worst_ps = worst_ps, added = added, slack = worst_value - check._ub_values[0],
                    empirical_result = empirical_result, front_values = np.array(instance.pareto_values),
                    mdp_policy = np.nanargmin(mdp.action_values, axis = -1), a_loss = np.array(a_loss), r_loss = np.array(r_loss),
                    fsc_entropy = fsc_entropy, rnn_entropy = rnn_entropy, cross_entropy = cross_entropy, label_cross_entropy = label_cross_entropy,
                    bounded_reach_prob = check.bounded_reach_prob)

            utils.inform(f'{run_idx}-{round_idx}\t(RNN)\t\taloss \t%.4f' % a_loss[0] + '\t>>>> %3.4f' % a_loss[-1], indent = 0)

