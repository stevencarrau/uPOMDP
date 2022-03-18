import tensorflow as tf
import larq as lq
import numpy as np
import itertools
import math

from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
import scipy.stats
from copy import deepcopy
import utils as ut
from fsc import FiniteMemoryPolicy

@tf.function
def flatter_tanh(x):
    y = 1.5 * K.tanh(x) + 0.5 * K.tanh(3 * x)
    return y

class Net(tf.keras.Model):
    def __init__(self, instance, cfg):
        super().__init__()
        self.instance, self.cfg = instance, cfg
        self.input_dim, self.output_dim = instance.input_dim, instance.output_dim
        self.memory_dim = cfg['a_memory_dim']
        self.bottleneck_dim = cfg['bottleneck_dim']
        self.qbn_gru_rnn = GRUActor(instance, cfg)
        self.actor = tf.keras.layers.Dense(self.output_dim, 'softmax')
        self.build(input_shape = (None, None, instance.input_dim))
        self.compile(loss = 'mse', optimizer = tf.keras.optimizers.Adam(learning_rate = cfg['a_lr']))

        self.hqs = np.array(list(itertools.product(*[[-1, 0, 1] for i in range(self.bottleneck_dim)])), dtype = 'float64')
        self.hqs_idxs = {np.array2string(np.squeeze(np.array(hq, dtype = 'int64'))) : idx for idx, hq in enumerate(self.hqs)}

    def call(self, o):
        h = self.qbn_gru_rnn(o)
        p = self.actor(h)
        return p

    def simulate(self, pomdp, mdp, batch_dim = None, greedy = False, length = None, quantize = False, inspect = False):
        """ Simulates an interaction of this HxQBN-GRU-RNN with a POMDP model application. """

        batch_dim = batch_dim or self.cfg['batch_dim']
        length = length or self.cfg['length']

        beliefs = np.zeros((batch_dim, length, pomdp.nS))
        states = np.zeros((batch_dim, length), dtype = 'int64')
        observations = np.zeros((batch_dim, length), dtype = 'int64')
        policies = np.zeros((batch_dim, length, pomdp.nA), dtype = 'float64')
        actions = np.zeros((batch_dim, length), dtype = 'int64')
        rewards = np.zeros((batch_dim, length, pomdp.num_reward_models), dtype = 'float64')

        hs = np.zeros((batch_dim, length, self.memory_dim), dtype = 'float32')
        if quantize:
            hxs = np.full((batch_dim, length, self.memory_dim), -2, dtype = 'float32')
            hqs = np.full((batch_dim, length, self.bottleneck_dim), -2, dtype = 'int64')

        state = np.array([np.squeeze(pomdp.initial_state) for s in range(batch_dim)], dtype = 'int64')
        observation = np.array([np.squeeze(pomdp.initial_observation) for s in range(batch_dim)], dtype = 'int64')

        belief = np.zeros((batch_dim, pomdp.nS))
        # belief[:, pomdp.initial_state] = 1

        reset = self.qbn_gru_rnn.reset(batch_dim, quantize)
        if quantize:
            h, hq, hx = reset
        else:
            h = reset

        for l in range(length):

            # beliefs[:, l] = belief

            states[:, l] = state
            observations[:, l] = observation

            hs[:, l] = h
            if quantize:
                hxs[:, l] = hx
                hqs[:, l] = hq

            x = np.reshape(ut.one_hot_encode(observation, pomdp.nO, dtype = 'float32'), (batch_dim, pomdp.nO))
            if quantize:
                a, action, h, hq, hx = self._action(x, inspect = True, greedy = greedy,  quantize = quantize, states = hx, mask = pomdp.policy_mask[observation])
            else:
                a, action, h = self._action(x, inspect = inspect, greedy = greedy, states = h, mask = pomdp.policy_mask[observation])

            policies[:, l] = a.numpy()
            actions[:, l] = action
            rewards[:, l, :] = pomdp.rewards[state, :]
            state = ut.choice_from_md(mdp.T[state, action], batch_dim)
            observation = pomdp.O[state]

            # next_belief = np.zeros((batch_dim, pomdp.nS))
            # for b in range(batch_dim):
            #     possible_states = np.where(pomdp.O == observation[b])
            #     next_belief[b, possible_states] = 1
            #     for possible_state in range(pomdp.nS):
            #         next_belief[b, possible_state] *= np.sum(belief[b] * mdp.T[:, action[b], possible_state], axis = 0)
            #     next_belief[b] = ut.normalize(next_belief[b])
            # belief = np.array(next_belief)

        if quantize:
            if inspect:
                return beliefs, states, hs, hqs, hxs, observations, policies, actions, rewards
            else:
                return beliefs, states, hs, hxs, observations, policies, actions, rewards
        else:
            return beliefs, states, hs, observations, policies, actions, rewards

    def _action(self, x, inspect = False, greedy = False, quantize = False, states = None, mask = None):
    
        if len(x.shape) > 2:
            raise ValueError('Actions can only be determined for one time-step and batch size of 1.')

        batch_dim = x.shape[0]

        self.qbn_gru_rnn.qbn_gru.set_quantize(quantize)
        self.qbn_gru_rnn.qbn_gru.set_inspect(inspect)

        # x = self.embedder(x)
        if inspect:
            h, hq, hx = self.qbn_gru_rnn.qbn_gru(x, states)
        else:
            h, hx = self.qbn_gru_rnn.qbn_gru(x, states)

        a = tf.squeeze(self.actor(h))

        if greedy:
            actions = ut.argmax_from_md(a.numpy(), batch_dim, mask = mask)
        else:
            actions = ut.choice_from_md(a.numpy(), batch_dim, mask = mask)

        self.qbn_gru_rnn.qbn_gru.set_quantize(False)
        self.qbn_gru_rnn.qbn_gru.set_inspect(False)

        if inspect:
            return a, actions, h, hq, hx
        else:
            return a, actions, h

    def improve_r(self, hs):
        train_result = self.qbn_gru_rnn.qbn_gru.hx_qbn.fit(x = hs, y = hs, batch_size = self.cfg['r_batch_size'], epochs = self.cfg['r_epochs'], verbose = 0)
        r_loss = train_result.history['loss']
        return r_loss

    def improve_a(self, inputs, labels, quantize = False, mask = None):
        """
        Trains the GRU actor on inputs and labels of args, leaving HxQBN unchanged.

        param: inputs   :   the observation inputs.
        param: labels   :   the labels.
        param: quantize :   if True, perform finetuning given the current HxQBN.
        param: mask     :   timesteps to mask when training.

        """

        batch_dim, time_dim, _ = inputs.shape

        _inputs = np.zeros((batch_dim, time_dim + 1, self.input_dim))
        dummy_observation = ut.dummy_observation(batch_dim, 1, self.input_dim, squeeze = True)
        _inputs[:, 0] = dummy_observation
        _inputs[:, 1:] = inputs

        _labels = np.zeros((batch_dim, time_dim + 1, self.output_dim))
        _labels[:, 0] = 0
        _labels[:, 1:] = labels
        _mask = np.ones((batch_dim, time_dim + 1, self.output_dim))
        _mask[:, 0] = 0

        self.qbn_gru_rnn.qbn_gru.set_quantize(False)
        self.qbn_gru_rnn.qbn_gru.set_inspect(False)

        train_result = self.fit(x = _inputs, y = _labels, batch_size = self.cfg['a_batch_size'], epochs = self.cfg['a_epochs'], verbose = 0)
        a_loss = train_result.history['loss']
        return a_loss

    def extract_fsc(self, make_greedy = True, reshape = True):

        action_distributions, next_memories = self._construct_transaction_table()
        fsc_policy = FiniteMemoryPolicy(
            action_distributions, next_memories,
            make_greedy = make_greedy, reshape = reshape,
            initial_observation = self.instance.pomdp.initial_observation)

        return fsc_policy

    def _construct_transaction_table(self):

        # First we swap indices of the first and initial hq.
        hqs = np.array(self.hqs)
        hqs_idxs = deepcopy(self.hqs_idxs)
        first_hq_str = np.array2string(np.squeeze(np.array(hqs[0], dtype = 'int64')))
        first_hq_idx = hqs_idxs[first_hq_str]
        init_h, init_hq, init_hx = self.qbn_gru_rnn.reset(batch_dim = 1, quantize = True)
        init_hq_str = np.array2string(np.squeeze(np.array(init_hq, dtype = 'int64')))
        init_hq_idx = hqs_idxs[init_hq_str]
        hqs_idxs[init_hq_str] = 0
        hqs_idxs[first_hq_str] = init_hq_idx
        hqs[[init_hq_idx, first_hq_idx]] = hqs[[first_hq_idx, init_hq_idx]]
        hxs = self.qbn_gru_rnn.qbn_gru.hx_qbn.decode(hqs).numpy()

        next_memories = np.full((len(hqs), self.input_dim), -1)
        action_distributions = np.zeros((len(hqs), self.input_dim, self.output_dim))

        self.qbn_gru_rnn.qbn_gru.set_quantize(True)
        self.qbn_gru_rnn.qbn_gru.set_inspect(True)

        for (hq_idx, hq), hx in zip(enumerate(hqs), hxs):
            hx = tf.expand_dims(hx, axis = 0)
            for o_idx, obs in enumerate(tf.one_hot(np.arange(self.input_dim), self.input_dim)):
                obs = tf.expand_dims(obs, axis = 0)
                next_h, next_hq, next_hx = self.qbn_gru_rnn.qbn_gru(obs, hx)
                next_hx = tf.reshape(next_hx, [1, self.memory_dim])
                distribution = self.actor(next_h).numpy()
                action_distributions[hq_idx, o_idx] = distribution

                next_hq_idx = hqs_idxs[np.array2string(np.array(np.squeeze(next_hq), dtype = 'int64'))]
                next_memories[hq_idx, o_idx] = next_hq_idx

        self.qbn_gru_rnn.qbn_gru.set_quantize(False)
        self.qbn_gru_rnn.qbn_gru.set_inspect(False)

        del hqs_idxs

        return action_distributions, next_memories

class GRUActor(tf.keras.layers.RNN):
    """
    Represents an GRU-RNN with quantized latent state.

    """

    def __init__(self, instance, cfg):

        self.instance = instance
        self.cfg = cfg

        self.input_dim = instance.input_dim
        self.memory_dim = cfg['a_memory_dim']
        self.output_dim = instance.output_dim

        # self.embedder = tf.keras.Sequential([
        #     tf.keras.layers.Dense(self.input_dim, activation = 'elu', name = 'emb_1'),
        #     tf.keras.layers.Dense(self.inter_dim, activation = 'relu', name = 'emb_2')
        # ], name = 'ActorEmbedder')
        # self.embedder.build(input_shape = (None, self.input_dim))

        # self.actor = tf.keras.Sequential([
        #     tf.keras.layers.Dense(self.output_dim, 'softmax', name = 'actor_1')
        # ], name = 'ActorOutput')
        # self.actor.build(input_shape = (None, self.memory_dim))

        self.qbn_gru = QBNGRU(instance, cfg)

        super().__init__(cell = self.qbn_gru, return_sequences = True, dtype = 'float32')
        # super().build((None, None, self.inter_dim))

    def call(self, x, quantize = False, mask = None):

        self.qbn_gru.set_quantize(quantize)

        # x = self.embedder(x)
        h = super().call(x, mask = mask)

        # a = self.actor(h)

        return h

    def reset(self, batch_dim, quantize):
        """
        Resets the GRU states to zero's (default) and feeds a dummy observation to fetch
        initial (quantized) latent memory node.

        """

        self.qbn_gru.reset(batch_dim)
        self.qbn_gru.set_quantize(quantize)
        if quantize:
            self.qbn_gru.set_inspect(True)

        if quantize:
            dummy_observation = ut.dummy_observation(batch_dim, 1, self.input_dim, squeeze = True)
            dummy_observation = dummy_observation.reshape((batch_dim, self.input_dim))
            h, hq, hx = self.qbn_gru(dummy_observation)
            self.qbn_gru.set_quantize(False)
            self.qbn_gru.set_inspect(False)
            return h, hq, hx
        else:
            dummy_observation = ut.dummy_observation(batch_dim, 1, self.input_dim, squeeze = True)
            h, _ = self.qbn_gru(dummy_observation)
            self.qbn_gru.set_quantize(False)
            self.qbn_gru.set_inspect(False)
            return h

class HxQBN(tf.keras.models.Model):
    def __init__(self, instance, cfg):

        super().__init__()

        self.instance = instance
        self.cfg = cfg

        self.input_dim = cfg['a_memory_dim']
        self.bottleneck_dim = cfg['bottleneck_dim']
        # self.inter_dim = self.bottleneck_dim * cfg['blow_up']
        self.inter_dim = self.bottleneck_dim * cfg['blow_up']

        self.encoder = tf.keras.models.Sequential([
            # tf.keras.layers.Dense(self.inter_dim, activation = flatter_tanh, name = 'enc_1'),
            tf.keras.layers.Dense(self.inter_dim, activation = 'tanh', name = 'enc_2'),
            tf.keras.layers.Dense(self.bottleneck_dim, activation = flatter_tanh, name = 'enc_3'),
            tf.keras.layers.Activation('ste_tern', name = 'quantizer'),
        ], name = 'HxQBN-Encoder')

        self.decoder = tf.keras.models.Sequential([
            # tf.keras.layers.Dense(self.inter_dim, activation = flatter_tanh, name = 'dec_1'),
            tf.keras.layers.Dense(self.inter_dim, activation = 'tanh', name = 'dec_2'),
            tf.keras.layers.Dense(self.input_dim, activation = 'tanh', name = 'dec_3'),
        ], name = 'HxQBN-Decoder')

        self.inspect = False

        super().build((None, self.input_dim))

        optimizer = tf.keras.optimizers.Adam(learning_rate = cfg['r_lr'])
        self.compile(optimizer = optimizer, loss = 'mse')

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def set_inspect(self, inspect):
        self.inspect = inspect
        return inspect

    def call(self, x):
        if len(x.shape) == 1:
            x = tf.reshape(x, [1, x.shape[-1]])

        x = self.encode(x)
        y = self.decode(x)

        if self.inspect:
            return x, y
        else:
            return y

class QBNGRU(tf.keras.layers.GRUCell):

    def __init__(self, instance, cfg):

        self.instance = instance
        self.cfg = cfg

        self.num_hx_qbns = cfg['num_hx_qbns']
        self.hx_qbn_idx = 0
        self.hx_qbns = [HxQBN(instance, cfg) for i in range(cfg['num_hx_qbns'])]
        self.hx_qbn = self.hx_qbns[self.hx_qbn_idx]

        self.memory_dim = cfg['a_memory_dim']
        self.input_dim = instance.input_dim

        super().__init__(self.memory_dim, name = 'qbn_gru')
        super().build(input_shape = (None, self.input_dim))

        self.quantize = False
        self.inspect = False

    def set_hx_qbn_idx(self, idx):
        self.hx_qbn_idx = idx

    def set_quantize(self, quantize):
        self.quantize = quantize

    def set_inspect(self, inspect):
        for i in range(self.num_hx_qbns):
            self.hx_qbns[i].set_inspect(inspect)
        self.inspect = inspect
        return inspect

    def reset(self, batch_dim):
        self.states = tf.zeros((batch_dim, self.memory_dim))
        return self.states

    def call(self, inputs, states = None):

        if states is None:
            batch_dim = inputs.shape[0]
            states = self.reset(batch_dim)

        [h, _] = super().call(inputs, states)

        if self.quantize:
            if self.inspect:
                hq, hx = self.hx_qbns[self.hx_qbn_idx](h)
                return h, hq, hx
            else:
                hx = self.hx_qbns[self.hx_qbn_idx](h)
                return h, hx

        return h, [h]

