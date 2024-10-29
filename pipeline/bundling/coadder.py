#!/bin/sh python3

import time
import numpy as np


class SignFlip:
    def __init__(self, state=None):
        self.prng = np.random.RandomState(int(1e+6*time.time()) % 2**32)
        if state:
            self.prng.set_state(state)
        self.state = self.prng.get_state()
        self.seq = None

    def gen_seq(self, obs_weights):
        self.state = self.prng.get_state()

        nums = len(obs_weights)
        obs = range(nums)
        obs_perm = self.prng.permutation(obs)

        obs_weights_perm = np.zeros_like(obs_weights)
        self.seq = np.zeros_like(obs_weights, dtype=np.bool_)

        for ob in obs_perm:
            obs_weights_perm[ob] = obs_weights[ob]

        w = obs_weights_perm
        wi = np.cumsum(w)
        wi = wi/np.max(wi)

        # no sign flip for the first half
        noflip = np.where(wi < 0.5)[0].tolist()
        # decide whether to flip the middle bundles by coin toss
        if len(w) > 1 and len(noflip) < 2 and self.prng.randint(0, 2):
            noflip.append(max(noflip)+1)

        for i in range(nums):
            if i in noflip:
                seq = False
            else:
                seq = True

            self.seq[obs_perm[i]] = seq

        return self.prng.get_state()
