from math import sqrt, log, exp
import random
import numpy as np


class Policy:
    def __init__(self, **kwargs):
        # limit of pulls of arms at once
        self.batch = kwargs.get("batch", 1)
        # the number of arms
        self.K = K = kwargs["K"]

        self.rewards = np.array([0.0] * K)
        self.ndraw = np.array([0] * K)
        self.valid_arms = set(range(K))
        self.stopped = False

        self.reward_mem = [[] for i in range(K)]
        self.t = 1

    def reset(self, **kwargs):
        self.rewards = np.array([0.0] * self.K)
        self.ndraw = np.array([0] * self.K)
        self.valid_arms = set(range(self.K))
        self.stopped = False
        self.t = 1
        self.reward_mem = [[] for i in range(self.K)]

    def update(self, ais, xs):
        for i, a in enumerate(ais):
            self.rewards[a] += xs[i]
            self.ndraw[a] += 1
            self.reward_mem[a].append(xs[i])
        self.t += 1
    
    def mean(self, i):
        if self.ndraw[i] == 0:
            return None
        return self.rewards[i] / self.ndraw[i]

    def cumulative_reward(self):
        return sum(self.rewards)
        
    def regret(self, mu_opt):
        return mu_opt * sum(self.ndraw) - sum(self.rewards)

    def select(self):
        pass


#%%
class ThompsonSamplingBernoulli(Policy):
    def __init__(self, **kwargs):
        super(ThompsonSamplingBernoulli, self).__init__(**kwargs)
        # for parameter of prior distribution
        self.ab = kwargs.get("ab", [1, 1])
        self.K = kwargs.get("K")
        self.prior_update = kwargs.get("prior_update", None)

        self.priors = [self.ab.copy() for i in range(self.K)]

    def reset(self, **kwargs):
        super(ThompsonSamplingBernoulli, self).reset(**kwargs)
        self.priors = [self.ab.copy() for i in range(self.K)]

    def select(self, ignore=None, chosen=None):
        res = []
        for i in range(self.batch):
            max_ = None
            for ai in self.valid_arms:
                if ignore and ai in ignore:
                    continue
                posterior_sample = np.random.beta(*self.priors[ai])
                if max_ is None or max_[1] < posterior_sample:
                    max_ = (ai, posterior_sample)
            if max_ is None:
                break
            res.append(max_[0])
        return res
    
    def update(self, ais, xs):
        super(ThompsonSamplingBernoulli, self).update(ais, xs)
        for i, ai in enumerate(ais):
            if self.prior_update is None:
                if xs[i] == 1:
                    self.priors[ai][0] += 1
                elif xs[i] == 0:
                    self.priors[ai][1] += 1
                else:
                    raise ValueError("invalid reward")
            elif self.prior_update == "deterministic_rounding":
                if xs[i] > 0.5:
                    self.priors[ai][0] += 1
                else:
                    self.priors[ai][1] += 1
            elif self.prior_update == "random_rounding":
                if random.random() < xs[i]:
                    self.priors[ai][0] += 1
                else:
                    self.priors[ai][1] += 1
            elif self.prior_update == "fraction":
                self.priors[ai][0] += xs[i]
                self.priors[ai][1] += 1 - xs[i]


class ThompsonSamplingNonParametric(Policy):
    def __init__(self, **kwargs):
        super(ThompsonSamplingNonParametric, self).__init__(**kwargs)
        self.K = K = kwargs.get("K")
        self.Tk = [np.array([1]) for i in range(K)]
        self.Sk = [np.array([1.0]) for i in range(K)]

    def reset(self, **kwargs):
        super(ThompsonSamplingNonParametric, self).reset(**kwargs)
        K = self.K
        self.Tk = [np.array([1]) for i in range(K)]
        self.Sk = [np.array([1.0]) for i in range(K)]

    def select(self, ignore, chosen=None):
        vals = []
        for i in range(self.K):
            if i in ignore or i not in self.valid_arms:
                vals.append(-1)
                continue
            L = np.random.dirichlet(self.Tk[i])
            V = self.Sk[i] @ L
            vals.append(V)

        vals = np.array(vals)
        if np.all(vals < 0):
            return []

        v_max = np.max(vals)
        inds = np.where(vals == v_max)[0]
        i = np.random.choice(inds)

        return [i]

    def update(self, ais, xs):
        super(ThompsonSamplingNonParametric, self).update(ais, xs)
        for i, x in zip(ais, xs):
            self.Tk[i] = np.r_[self.Tk[i], [1]]
            self.Sk[i] = np.r_[self.Sk[i], [x]]



"""

# 条件を満たすまでサンプリングする
class ThompsonSamplingBernoulliWithSelectionControl(Policy):
    def __init__(self, **kwargs):
        super(ThompsonSamplingBernoulli, self).__init__(**kwargs)
        # for parameter of prior distribution
        self.ab = kwargs.get("ab", [1, 1])
        self.K = kwargs.get("K")

        self.priors = [self.ab.copy() for i in range(self.K)]

    def reset(self, **kwargs):
        super(ThompsonSamplingBernoulli, self).reset(**kwargs)
        self.priors = [self.ab.copy() for i in range(self.K)]

    def select(self, ignore=None):
        res = []
        for i in range(self.batch):
            max_ = None
            for ai in self.valid_arms:
                if ignore and ai in ignore:
                    continue
                posterior_sample = np.random.beta(*self.priors[ai])
                if max_ is None or max_[1] < posterior_sample:
                    max_ = (ai, posterior_sample)
            if max_ is None:
                break
            res.append(max_[0])
        return res

    def update(self, ais, xs):
        super(ThompsonSamplingBernoulli, self).update(ais, xs)
"""

#%%
class TopTwoThompsonSampling(Policy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ab = kwargs.get("ab", [1, 1][:])
        self.K = kwargs.get("K")
        self.prior_update = kwargs.get("prior_update", "random_rounding")
        self.priors = [self.ab.copy() for i in range(self.K)]

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self.priors = [self.ab.copy() for i in range(self.K)]
        self.phistory = []

    def select(self, ignore=None, chosen=None):
        beta = 0.5
        res = []

        if len(ignore) == len(self.valid_arms):
            return res
        while True:
            skip = set()
            if ignore is not None:
                skip |= ignore
            # if chosen is not None:
            #     skip |= set(chosen)
            remain = [i for i in self.valid_arms if i not in skip]
            if len(remain) == 0:
                return res
            elif len(remain) == 1:
                return remain

            psample = []
            for i in range(self.K):
                psample.append(np.random.beta(self.priors[i][0], self.priors[i][1]))

            remain = [i for i in self.valid_arms if i not in skip]
            if len(remain) == 0:
                break
            i_opt = max([(psample[i], i) for i in remain])[1]
            if len(remain) == 1:
                return i_opt
            r = np.random.random()
            if r < beta:
                if len(remain) == 2:
                    remain.remove(i_opt)
                    return remain
                while True:
                    psample = []
                    for i in range(self.K):
                        psample.append(np.random.beta(self.priors[i][0], self.priors[i][1]))
                    skip = set()
                    if ignore is not None:
                        skip |= set(ignore)
                    # if chosen is not None:
                    #     skip |= set(chosen)
                    i_opt2 = max([(psample[i], i) for i in self.valid_arms if i not in skip])[1]
                    if i_opt != i_opt2:
                        return [i_opt2]
            else:
                return [i_opt]

    def update(self, ais, xs):
        super().update(ais, xs)
        for i, ai in enumerate(ais):
            if self.prior_update is None:
                if xs[i] == 1:
                    self.priors[ai][0] += 1
                elif xs[i] == 0:
                    self.priors[ai][1] += 1
                else:
                    raise ValueError("invalid reward")
            elif self.prior_update == "deterministic_rounding":
                if xs[i] > 0.5:
                    self.priors[ai][0] += 1
                else:
                    self.priors[ai][1] += 1
            elif self.prior_update == "random_rounding":
                if random.random() < xs[i]:
                    self.priors[ai][0] += 1
                else:
                    self.priors[ai][1] += 1
            elif self.prior_update == "fraction":
                self.priors[ai][0] += xs[i]
                self.priors[ai][1] += 1 - xs[i]



class SuccessiveElimination(Policy):
    def __init__(self, **kwargs):
        super(SuccessiveElimination, self).__init__(**kwargs)
        self.K = kwargs.get("K")
        self.pending = []

    def reset(self, **kwargs):
        super(SuccessiveElimination, self).reset(**kwargs)
        self.pending = list(range(self.K))

    def select(self, ignore=None, chosen=None):
        # 選択回数が一番少ない腕を選択
        res = []
        skipped_arms = set()
        while len(res) < self.batch:
            if len(skipped_arms) == self.K - len(ignore):
                break
            if len(self.pending) == 0:
                ndraw_valid = {a: self.ndraw[a] + chosen.count(a) for a in range(self.K) if a in self.valid_arms and a not in skipped_arms}
                if len(ndraw_valid) == 0:
                    return res
                nmin = min(ndraw_valid)
                nmax = max(ndraw_valid)
                if nmin == nmax:
                    self.pending = [a for a in range(self.K) if a in self.valid_arms and a not in skipped_arms]
                else:
                    n = nmin
                    while len(self.pending) == 0:  # n < nmax: # len(self.pending) == 0: # n <= nmax:
                        for a in ndraw_valid:
                            if ndraw_valid[a] <= n:
                                self.pending.append(a)
                        n += 1
                #print(self.pending)
            a = self.pending.pop(0)
            if a in ignore or a in skipped_arms:
                skipped_arms.add(a)
                continue
            res.append(a)
        return res

        """
        ndraw1 = self.ndraw.copy().astype(np.float)
        for ai in chosen:
            ndraw1[ai] += 1
        for ai in ignore:
            ndraw1[ai] += float("inf")

        skipped_arms = set()
        while True:
            if len(skipped_arms) == self.K - len(ignore):
                break
            ai = np.argmin(ndraw1)
            ndraw1[ai] += 1
            if ai not in self.valid_arms:
                skipped_arms.add(ai)
                continue
            res.append(ai)
            if len(res) >= self.batch:
                return res
        return res
        """


class RandomSelection(Policy):
    def __init__(self, **kwargs):
        super(RandomSelection, self).__init__(**kwargs)
        self.K = kwargs.get("K")

    def reset(self, **kwargs):
        super(RandomSelection, self).reset(**kwargs)

    def select(self, ignore=None, chosen=None):
        # ランダムに腕を選択
        res = []
        skipped_arms = set()
        while len(res) < self.batch:
            if len(skipped_arms) == self.K - len(ignore):
                break
            # valueには意味がなくて skipped_armsに入ってないアームを列挙したいだけ
            ndraw_valid = {a: self.ndraw[a] + chosen.count(a) for a in range(self.K) if a not in skipped_arms}
            if len(ndraw_valid) == 0:
                return res
            a = np.random.choice(list(ndraw_valid.keys()))
            if a in ignore or a in skipped_arms:
                skipped_arms.add(a)
                continue
            res.append(a)
        return res


if __name__ == "__main__":
    from setuptools import setup, Extension
    from Cython.Distutils import build_ext
    import sys
    import shutil

    sys.argv += ["build_ext", "--inplace"]
    
    shutil.copy("policies.py", "policies.pyx")
    
    setup(
        cmdclass = {'build_ext': build_ext},
        ext_modules = [Extension("policies", ["policies.pyx"])]
    )

    #import os
    #os.system("python setup.py build_ext --inplace")
