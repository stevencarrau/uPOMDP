import numpy as np
import scipy.stats

from copy import deepcopy

import utils


class FiniteMemoryPolicy:

    def __init__(self, action_distributions, next_memories, is_randomized = False, reshape = True,
                 make_greedy = True, initial_observation = None):
        """
        @param: action_distributions    :   an M x O x A np.array specifying the 
                                            action distribution for each m, o pair.
        @param: next_memories           :   an M x O np.array specifying the next
                                            memory state for each m, o pair.
        @param: reshape                 :   if True, reshapes self to only memories that are reached.
        @param: make_greedy             :   if True, transforms action distributions to argmax after masking.
        @param: zero                    :   action probabilities below this number will be set to 0.

        """

        if is_randomized:
            self._check_distribution(next_memories)
            self.nM_generated = self.nM = len(next_memories)
            self.M = np.arange(self.nM)

        elif len(next_memories) > 1:

            if reshape:

                # Get rid of unreachable memories.
                self.nM = len(next_memories)
                reachable = set()
                if initial_observation is not None:
                    initial_next_memories = list(np.unique(next_memories[[0], initial_observation]))
                    new_reachable = set(initial_next_memories)
                else:
                    new_reachable = set([0])
                while len(new_reachable) > len(reachable):
                    reachable = deepcopy(new_reachable)
                    next_reachable = np.unique(next_memories[np.array(list(reachable))])
                    new_reachable = set(list(next_reachable))
                    new_reachable = new_reachable.union(reachable)
                new_reachable.add(0)

                self.M = np.array(list(new_reachable), dtype = 'int64')
                self.nM_generated = len(self.M)

                action_distributions = np.array(action_distributions[self.M])
                next_memories = np.array(next_memories[self.M], dtype = 'int64')

                self.index_of_M = np.full((self.nM), -1, dtype = 'int64')
                for m in range(self.nM):
                    index = np.where(self.M == m)[0]
                    if len(index) == 1:
                        self.index_of_M[m] = index
                    elif len(index) > 1:
                        raise ValueError(f'Found two indices for memory node {m}')

            else:
                self.nM = len(next_memories)
                self.nM_generated = self.nM
                self.M = np.arange(self.nM)
                self.index_of_M = np.arange(self.nM)

        else:
            self.M = np.arange(1)
            self.nM_generated = len(self.M)
            self.nM = self.nM_generated
            self.index_of_M = np.arange(self.nM)

        self.nM, self.nO, self.nA = action_distributions.shape

        self._check_distribution(action_distributions)

        self.action_distributions = utils.normalize(action_distributions)
        self.next_memories = next_memories

        self._next_memories = self.index_of_M[self.next_memories]

        self.memories = None

        self.make_greedy = make_greedy
        self.is_made_greedy = False
        self.is_masked = False # whether action distributions take into account (im)possible actions.
        self.is_randomized = is_randomized

    def _check_distribution(self, distributions):
        """
        Checks whether the distributions in arg sum up to 1.

        """

        sums = np.sum(distributions, axis = -1)
        if not np.all(np.isclose(sums, 1)):
            raise ValueError(f'Distributions do not sum up to (close to) 0, 1, or are NaN. Sums are: \n{sums}')
        return True

    def reset(self, batch_dim):
        """
        Resets the internal memory state, which is zero by default.

        @param batch_dim    :   N, the number of memory states to initialize.

        """

        self.batch_dim = batch_dim

        states = np.zeros((batch_dim), dtype = 'int64')
        self._update(states)

        return self.memories

    def _update(self, states):
        self.memories = states
        return self.memories

    def action_distribution(self, observations, memories = None):
        """
        Returns an array of action distributions for the input observations and current memory state.

        @param observations : an N x 1 array of observations.

        """

        if memories is None:
            self._check_initalization()
            memories = self.memories

        action_distributions = self.action_distributions[memories, observations]
        return action_distributions

    def action(self, observations, greedy = False):
        """
        Returns an array of actions for the input observations and current memory state.

        @param observations : an N x 1 array of observations.
        @param greedy       : whether actions should be greedily selected.

        """

        if len(observations) != self.batch_dim:
            raise ValueError('Shape of input observations does not match batch size.')

        action_distributions = self.action_distribution(observations)

        if greedy:
            actions = utils.argmax_from_md(action_distributions, self.batch_dim)
        else:
            actions = utils.choice_from_md(action_distributions, self.batch_dim)
        return actions

    def step(self, observations):
        self._check_initalization()
        if self.is_randomized:
            next_memories = utils.choice_from_md(self.next_memories[self.memories, observations], self.batch_dim)
        else:
            next_memories = self.index_of_M[self.next_memories[self.index_of_M[self.memories], observations]]
        return self._update(next_memories)

    def _check_initalization(self):
        if self.memories is None:
            raise ValueError('Memory state(s) have not been initialized.')

    def mask(self, mask, zero = 1e-03):
        """
        Masks the action distributions of this policy.

        """

        if not self.action_distributions.shape[-2:] == mask.shape[-2:]:
            raise ValueError('Mask is not of same shape as action distributions.')

        self.unmasked_action_distributions = np.array(self.action_distributions)

        action_distributions = self.action_distributions * mask
        invalid_entries = np.logical_and(action_distributions < zero, mask != 1)
        action_distributions[invalid_entries] = 0
        self.action_distributions = utils.normalize(action_distributions)

        if self.make_greedy:
            self.action_distributions = utils.one_hot_encode(np.argmax(self.action_distributions, axis = -1), self.nA, dtype = 'float32')
            self.is_made_greedy = True

        self._check_distribution(self.action_distributions)
        self.is_masked = True

        return self.action_distributions

    def simulate(self, observations):
        """
        Vectorized-simulation of interaction for the observations in args.

        """

        if not self.is_masked:
            raise ValueError('Generation of memory states and action distributions should be done after policy is masked.')

        batch_dim, length = observations.shape
        memories = np.zeros((batch_dim, length), dtype = 'int64')
        action_distributions = np.zeros((batch_dim, length, self.nA))

        memory = self.reset(batch_dim)

        for l in range(length):
            memories[:, l] = memory
            action_distributions[:, l] = self.action_distribution(observations[:, l], memory)
            memory = self.step(observations[:, l])

        return memories, action_distributions

    def randomized_next_memories(self, add = 1e-05):
        if self.is_randomized:
            raise ValueError('Next memories are already randomized.')
        else:
            next_memories = np.zeros((self.nM_generated, self.nO, self.nM_generated))
            one_hot = utils.one_hot_encode(np.arange(self.nM_generated), self.nM_generated, dtype = 'float32')
            next_memory_indices = self.index_of_M[self.next_memories]
            next_memories_one_hot = one_hot[next_memory_indices]
            next_memories_one_hot += add
            next_memories = utils.normalize(next_memories_one_hot)
        return next_memories
