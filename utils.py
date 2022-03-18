import numpy as np
from itertools import product
import tensorflow as tf
from tensorflow import cumsum
from datetime import datetime
import pycarl


class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def satisfied(instance, value):
    if instance.objective == 'min':
        return value < instance.threshold
    elif instance.objective == 'max':
        return value > instance.threshold
    else:
        raise TypeError('Objective not understood')


def dummy_observation(batch_dim, time_dim, n, dtype = 'float32', squeeze = False):
    if squeeze:
        return np.squeeze(np.ones((batch_dim, time_dim, n), dtype = dtype) * -1)
    else:
        return np.ones((batch_dim, time_dim, n), dtype = dtype) * -1

def inform(message, indent = 4, itype = None):
    indents = ''.join('\t' for i in range(indent))
    if itype is not None:
        color = getattr(BColors, itype)
        string = f'{color}{str(datetime.now())[:-3]}: {indents} {message} {BColors.ENDC}'
    else:
        string = f'{str(datetime.now())[:-3]}: {indents} {message}'
    print(string)

def calculate_memory_size(cfg):
    if cfg['method'] in ('QBNLSTM'):
        if cfg['memory_dim'] % 2 != 0:
            raise ValueError(f'You must specify even memory dim for {cfg["method"]}')
        if 'quantizer' not in cfg:
            raise ValueError(f'You must specify a quantizer for {cfg["method"]}')
        quantizer_str = cfg['quantizer']
        if quantizer_str not in quantizer_values:
            raise ValueError(f'Passed quantizer {quantizer_str} not in dict of known quantizers.')
        else:
            if 'bottleneck_dim' not in cfg:
                raise ValueError(f'You must specify a bottleneck dim for {cfg["method"]}')
            size = len(quantizer_values[quantizer_str]) ** cfg['bottleneck_dim']
            return size
    else:
        return cfg['memory_dim']

def labels_to_str(labels):
    return ''.join(str(l) + '' for l in labels)

def parse_transitions(model, p_names, check = True):

    num_ps = len(p_names)
    nA = max([len(state.actions) for state in model.states])

    T = np.zeros((model.nr_states, nA, model.nr_states)) # holds trans prob or -1 if the trans is parametric.
    C = np.zeros((model.nr_states, nA, model.nr_states, num_ps)) # specifies the constant part of the transition.
    D = np.zeros((model.nr_states, nA, model.nr_states, num_ps)) # specifies derivative values for parameter in transition.
    A = np.full((model.nr_states, nA), True) # whether there is NO transition (action) from a state.
    P = np.full((model.nr_states, nA, model.nr_states, num_ps), None) # name of the parameter for each (s, a, s') (only one is allowed)
    S = np.full((model.nr_states, nA, model.nr_states), False) # whether the action directs back to origin state.
    for state in model.states:
        for action in state.actions:
            A[state.id, action.id] = False
            for transition in action.transitions:
                next_state = transition.column
                value, variables, constants, derivative_values = parse_transition(transition)
                T[state.id, action.id, next_state] = value
                S[state.id, action.id, next_state] = state.id == next_state
                if variables is not None:
                    for v, variable in enumerate(variables):
                        index_of_variable = p_names.index(variable)
                        P[state.id, action.id, next_state, index_of_variable] = variable
                        C[state.id, action.id, next_state, index_of_variable] = constants[v]
                        D[state.id, action.id, next_state, index_of_variable] = derivative_values[v]

    if check:
        differences = np.sum(T, axis = -1) - 1
        sum_to_1 = np.isclose(differences, 0, atol = 1e-05)
        parameterized = np.any(np.any(P != None, axis = -1), axis = -1)
        no_action = A
        check1 = np.logical_or(np.logical_or(sum_to_1, parameterized), no_action)
        if not np.all(check1):
            raise ValueError(f'Transition distribution does not sum up to 1. \n{np.where(np.logical_not(check1))} \n{differences[np.where(np.logical_not(check1))]}')
        positive = T >= 0
        check2 = np.logical_or(positive, np.expand_dims(parameterized, axis = -1))
        if not np.all(check2):
            raise ValueError('Negative transition probabilities found.')
    return T, C, A, S, P, D

def normalize(a, axis = -1, raise_error = True, method = 'sum'):
    if raise_error:
        if method == 'sum':
            with np.errstate(invalid = 'raise'):
                a /= np.sum(a, axis = axis, keepdims = True)
        elif method == 'minmax':
            a = (a - np.min(a, axis = axis, keepdims = True)) / (np.max(a, axis = axis, keepdims = True) - np.min(a, axis = axis, keepdims = True))
        else:
            raise ValueError('Normalizing method not understood.')
    else:
        if method == 'sum':
            a /= np.sum(a, axis = axis, keepdims = True)
        elif method == 'minmax':
            a = (a - np.min(a, axis = axis, keepdims = True)) / (np.max(a, axis = axis, keepdims = True) - np.min(a, axis = axis, keepdims = True))
        else:
            raise ValueError('Normalizing method not understood.')
    return a

def parse_transition(transition):
    """
    Parses the transition in args.
    
    param: transition   :   a stormpy.SparseMatrixEntry

    """

    value = transition.value()
    variable = None
    c = 0
    derivative_value = 0
    if isinstance(value, float):
        return value, variable, c, derivative_value
    else:
        if value.is_constant():
            denominator = value.denominator.coefficient
            numerator = value.numerator.coefficient
            value = float(str(numerator)) / float(str(denominator))
            return value, variable, c, derivative_value
        elif hasattr(value, 'gather_variables'):
            variables = list(value.gather_variables())
            constant = value.constant_part()
            c = float(str(constant.numerator)) / float(str(constant.denominator))
            variable_names = []
            derivatives = []
            constants = []
            for variable in variables:
                variable_names.append(variable.name)
                derivative = value.derive(variable)
                derivative_value = value_to_float(derivative)
                derivatives.append(derivative_value)
                constants.append(constant)
            return -1, variable_names, constants, derivatives
        else:
            raise TypeError(f'Value type of transition {transition} not understood.')

def value_to_float(value):
    """
    Parses the value in args to float.
    
    param: value   :   a float or a stormpy.FactorizedPolynomial (must be constant.)

    """

    if isinstance(value, float):
        return value
    elif value.is_constant() or isinstance(value, pycarl.cln.cln.FactorizedRationalFunction):
        denominator = value.denominator.coefficient
        numerator = value.numerator.coefficient
        value = float(str(numerator)) / float(str(denominator))
        return value
    else:
        raise TypeError('Value type not understood')

def one_hot_encode(a, n, dtype = 'int64'):
    enc = np.array(tf.one_hot(np.arange(n, dtype = 'int64'), n), dtype = dtype)
    a_enc = enc[a]
    return a_enc

def boolean_cumsum(a, axis = -1):
    numeric = np.array(a, dtype = 'float64')
    cumsum = np.cumsum(numeric, axis = axis)
    bool_cumsum = cumsum > 0
    return bool_cumsum

def choice_from_md(a, n, mask = None):
    if mask is not None:
        a *= np.squeeze(mask)
        a /= np.sum(a, axis = -1, keepdims = True)

    index = np.random.rand(n, 1)
    cum_prob = np.cumsum(a, axis = -1)
    choices = np.argmin([index >= cum_prob], axis = -1).flatten()
    return choices.astype('int64')

def argmax_from_md(a, n = None, mask = None):
    if mask is not None:
        a *= np.squeeze(mask)

    return np.argmax(a, axis = -1)

def evaluate_performance(instance, states, rewards):
    state_labels = instance.pomdp.state_labels[states]
    batch_dim, time_dim = states.shape
    important_timesteps = np.repeat(np.expand_dims(np.arange(time_dim), axis = 0), axis = 0, repeats = batch_dim)
    if instance.kind == 'probability':
        reached_at_timesteps = np.argmax(state_labels == instance.label_to_reach, axis = -1)
        if instance.label_to_avoid != None:
            avoided_until_timesteps = np.argmax(state_labels == instance.label_to_avoid, axis = -1)
            important_timesteps = np.logical_and(
                important_timesteps < np.expand_dims(avoided_until_timesteps, axis = -1),
                important_timesteps < np.expand_dims(reached_at_timesteps, axis = -1)
            )
            success = np.count_nonzero(
                np.logical_and(
                    avoided_until_timesteps == time_dim - 1,
                    reached_at_timesteps <= time_dim)
            )
        else:
            important_timesteps = important_timesteps < np.expand_dims(reached_at_timesteps, axis = -1)
        result = success / len(states)
    else:
        cum_rewards = np.array(cumsum(rewards[:, :, 0], axis = 1, reverse = True, exclusive = True))
        important_timesteps = np.logical_and(state_labels != instance.label_to_reach, np.any(state_labels == instance.label_to_reach, axis = -1, keepdims = True))
        result = np.mean(cum_rewards[:, 0])
    return result, important_timesteps
