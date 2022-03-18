import stormpy
import stormpy.pars
import numpy as np
from sys import maxsize

import utils

class Wrapper:
    """
    Wraps a Stormpy model.

    """

    def __init__(self, model, properties, O):

        self.model = model
        self.properties = properties

        self.is_parametric = isinstance(model.transition_matrix, stormpy.ParametricSparseMatrix)

        self.all_parameters = set()
        self.reward_parameters = set()
        self.probability_parameters = set()
        self.wellformed_constrains = set()
        self.graph_preserving_constraints = set()
        self.p_names = []

        if model.has_parameters:
            self.all_parameters = model.collect_all_parameters()
            self.reward_parameters = model.collect_reward_parameters()
            self.probability_parameters = model.collect_probability_parameters()
            self.p_names = [p.name for p in self.probability_parameters]

        self.O = O
        self.T, self.C, self.A, self.S, self.P, self.D = utils.parse_transitions(model, self.p_names)
        self.transition_matrix = model.transition_matrix
        self.labeling = model.labeling

        self.nS, self.nO, self.nC = model.nr_states, len(np.unique(O)), model.nr_choices
        self.nA = max([len(state.actions) for state in self.model.states])

        self.all_labels = []
        self.observation_labels = np.full((self.nO), 'no label added no label added')
        self.state_labels = np.full((self.nS), 'no label added no label added no label')

        self.policy_mask = np.zeros((self.nO, self.nA))
        self.choice_labels = []
        self.choices_per_state = []
        self.choices_per_observation = {}
        self.labels_to_states = {}
        self.states_to_labels = []

        for state in self.model.states:
            labels = set()

            for label in state.labels:
                if label in self.labels_to_states:
                    self.labels_to_states[label].append(state.id)
                else:
                    self.labels_to_states[label] = [state.id]
            self.states_to_labels.append(list(state.labels))

            label = ' '.join(state.labels)
            self.observation_labels[self.O[state.id]] = label
            self.state_labels[state.id] = label
            self.choices_per_state.append([])

            # Establish policy mask. Assumes A(s) == A(s') if O(s) == O(s').
            for action in state.actions:
                self.policy_mask[self.O[state.id]][action.id] = 1
                observation = self.O[state.id]
                if observation in self.choices_per_observation:
                    self.choices_per_observation[observation].add(action.id)
                else:
                    self.choices_per_observation[observation] = set([action.id])
                self.choices_per_state[-1].append(action.id)

            state_labels = list(state.labels)
            self.all_labels.append(labels)

        self.num_choices_per_state = np.array([len(c) for c in self.choices_per_state])
        self.choices_per_observation = {r : list(self.choices_per_observation[r]) for r in self.choices_per_observation}
        self.choices_per_observation_label = {self.observation_labels[r] : self.choices_per_observation[r] for r in self.choices_per_observation}
        self.reward_models = model.reward_models
        self.num_reward_models = len(self.reward_models)
        self.reward_bases = {}
        self.rewards = np.zeros((self.nS, self.num_reward_models), dtype = 'float64')

        for r, reward_model_str in enumerate(self.reward_models):
            reward_model = self.reward_models[reward_model_str]
            if reward_model.has_state_rewards:
                self.reward_bases[reward_model_str] = 'state'
                for s in range(self.nS):
                    self.rewards[s, r] = utils.value_to_float(reward_model.state_rewards[s])
            else:
                raise NotImplementedError('Only state-based rewards are implemented.')

            #     self.reward_bases[reward_model_str] = 'state_action'
            #     rewards = reward_model.state_action_rewards
            #     counter = 0
            #     for s in range(self.nS):
            #         observation = self.O[s]
            #         actions = self.policy_mask[observation]
            #         for a, mask in enumerate(actions):
            #             if mask > 0:
            #                 self.rewards[s, a, :, r] = utils.value_to_float(rewards[counter])
            #                 counter += 1

            # elif reward_model.has_transition_rewards:
            #     self.reward_bases[reward_model_str] = 'transition'
            #     raise NotImplementedError('Transition based rewards are not supported yet.')

        self.initial_state = model.initial_states
        self.initial_observation = self.O[self.initial_state]
        self.initial_state_is_real_state = np.sum(np.sum(self.T, axis = 0), axis = 0)[0] > 0 # check whether state 0 is reachable after init.

    def _instantiated_transition_matrix(self, p_values):
        """
        If this Model is parametric, instantiates a transition matrix with parameter values equal
        to value of args.

        param:  model_parameter_value     :      the value that the model transition parameter needs to take.
        return: transition_matrix         :      a stormpy.SparseMatrix instance.

        """

        if not self.is_parametric:
            raise TypeError('Called to instantiate transition matrix for a model that is not parametric.')

        T = np.array(self.T)
        T[np.any(self.P != None, axis = -1)] = 0
        for p in p_values:
            value = p_values[p]
            transitions_with_p = np.where(self.P == p)
            derivatives_of_p = self.D[transitions_with_p]
            constants_of_p = self.C[transitions_with_p]
            T[transitions_with_p[:3]] += (constants_of_p + derivatives_of_p * value)

        A = np.logical_not(np.reshape(self.A, (self.nS * self.nA)))
        T = T.reshape((self.nS * self.nA, self.nS))
        T = T[A]

        transition_matrix = stormpy.build_sparse_matrix(T, row_group_indices = self.transition_matrix._row_group_indices[:-1])
        return transition_matrix

    def model_components(self, p_values = None):
        """
        Returns the components of this model with parameter instantiation equal to model_parameter_value.
        Currently only works for one model parameter.

        """

        if self.is_parametric and p_values is None:
            raise ValueError('Model is parametric, so you must specify a value to instantiate with.')
        if not self.is_parametric:
            return self.model.transition_matrix, self.model.labeling, self.model.reward_models
        else:
            transition_matrix = self._instantiated_transition_matrix(p_values)
            labeling = self.model.labeling
            reward_models = {}
            for reward_model_name in self.reward_models:
                reward_model = self.reward_models[reward_model_name]
                if reward_model.has_state_rewards:
                    state_rewards = reward_model.state_rewards
                    state_rewards_vector = [utils.value_to_float(x) for x in state_rewards]
                    reward_models[reward_model_name] = stormpy.SparseRewardModel(optional_state_reward_vector = state_rewards_vector)
                elif reward_model.has_state_action_rewards:
                    state_action_rewards = reward_model.state_action_rewards
                    state_action_rewards_vector = [utils.value_to_float(x) for x in state_action_rewards]
                    reward_models[reward_model_name] = stormpy.SparseRewardModel(optional_state_action_reward_vector = state_action_rewards_vector)
                else:
                    raise NotImplementedError('To implement.')
            return transition_matrix, labeling, reward_models

    def export(self, file):
        """
        Exports the stormpy model underlying this Wrapper to .drn

        """

        encoding_options = stormpy.DirectEncodingOptions()
        encoding_options.allow_placeholders = False
        stormpy.export_to_drn(self.model, encoding_options)


class POMDPWrapper(Wrapper):
    """
    Wraps a - possibly parametric - stormpy POMDP.

    """

    def __init__(self, model, properties):
        O = np.array(model.observations, dtype = 'int64')
        super().__init__(model, properties, O)

        # Check that only one parameter per transition can exist.
        parameterized = np.count_nonzero(self.P != None, axis = -1)
        more_than_one = np.where(parameterized > 1)
        if np.any(parameterized > 1):
            raise ValueError(f'Invalid uPOMDP passed more than 1 parameter found at {more_than_one}.')


class MDPWrapper(Wrapper):
    """
    Wraps a non-parametric stormpy MDP.

    """

    def __init__(self, model, properties):
        O = np.arange(model.nr_states, dtype = 'int64')
        super().__init__(model, properties, O)

        check = stormpy.model_checking(self.model, properties[0], extract_scheduler = True)
        # assert check.has_scheduler
        # scheduler = check.scheduler
        # assert scheduler.memoryless and scheduler.deterministic

        self.state_values = np.array(check.get_values())
        self.state_values[np.isinf(self.state_values)] = np.nan
        transitions = np.logical_not(self.A)
        self.action_values = np.full((self.nS, self.nA), np.nan)
        if self.rewards.size > 0:
            rewards = np.sum(self.T * self.rewards[:, 0], axis = -1)
        else:
            rewards = np.zeros_like((self.action_values))
        self.action_values[transitions] = np.sum(self.T[transitions] * self.state_values, axis = -1) + rewards[transitions]

        utils.inform('Synthesized MDP-policy w/ value range (%.2f' % check.min + ', %.2f)' % check.max + ' & OPT = %.2f' % self.state_values[0])


class PDTMCModelWrapper(Wrapper):
    """
    Wraps a - possible parametric - stormpy DTMC.

    """

    def __init__(self, model, pomdp, nM, p_region_dict = {}, state_labels = None, memory_labels = None):
        O = np.arange(model.nr_states, dtype = 'int64')
        super().__init__(model, pomdp.properties, O)

        self.underlying_nM = nM
        self.underlying_nS = pomdp.nS

        self.state_labels = state_labels
        self.memory_labels = memory_labels

        self.p_region_dict = p_region_dict
        self.parameter_region = stormpy.pars.ParameterRegion(p_region_dict)

# class ProductPOMDPWrapper(POMDPWrapper):
#     """
#     Wraps a - possibly parametric - stormpy POMDP computed as product of a POMDP and an FSC.

#     """

#     def __init__(self, model, pomdp, nM):
#         O = np.array(model.observations, dtype = 'int64')
#         super().__init__(model, pomdp.properties)

#         self.nM = nM
#         self.underlying_pomdp = pomdp
