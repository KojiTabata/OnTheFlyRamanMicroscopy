
import numpy as np
from math import log, sqrt, exp
import random
from .policies import Policy


class Arm:
    def __init__(self, f, random_state=None):
        """
        f: function RandomState -> [0, 1]
        """
        if random_state is None:
            random_state = random.randint(0, 2**31)
        self.random_state = random_state
        self.f = f

        self.rnd = np.random.RandomState(seed=self.random_state)

    def randomize(self):
        self.random_state = random.randint(0, 2**31)
        self.reset()
        
    def reset(self):
        self.rnd = np.random.RandomState(seed=self.random_state)

    def draw(self):
        return self.f(self.rnd)


# %% Stop Condition
class StopCondition:
    def reset(self, policy: Policy):
        pass

    def check(self, ais, xs):
        pass


class GoodArmIdentification(StopCondition):
    def __init__(self, **kwargs):
        self.threshold = kwargs["threshold"]
        self.delta = kwargs["delta"]
        self.max_samples = kwargs.get("max_samples", None)
        self.find = kwargs["find"] if "find" in kwargs else 1

    def reset(self, policy: Policy):
        self.policy = policy
        K = self.policy.K
        self.positive_arms = set()
        self.negative_arms = set()
        self.judged = ["U" for _ in range(K)]
        self.stop = False
        self.result = ""

    def lucb(self, i):
        m = self.policy.mean(i)
        if m is None:
            return None
        d = self.delta
        K = self.policy.K
        n = self.policy.ndraw[i]
        if n > 0:
            if self.max_samples:
                N = self.max_samples[i]
                if 2 * n > N:
                    c = (1 - n/N) * (1 + 1/n)
                else:
                    c = 1 - (n-1)/N
                if N == n:
                    w_u, w_l = 0, 0
                else:
                    w_u = sqrt(0.5 * c/ n * (log(2 * K) + 2 * log(n) - log(d)))
                    w_l = sqrt(0.5 * c/ n * (log(2) + 2 * log(n) - log(d)))
            else:
                w_u = sqrt(0.5 / n * (log(2 * K) + 2 * log(n) - log(d)))
                w_l = sqrt(0.5 / n * (log(2) + 2 * log(n) - log(d)))
            return m - w_l, m + w_u
        else:
            return None

    def check(self, ais, xs):
        th = self.threshold
        find = self.find
        for ai in set(ais):
            n = self.policy.ndraw[ai]
            if n == 0:
                continue
            lucb = self.lucb(ai)
            if lucb is None:
                continue
            lcb, ucb = lucb
            if lcb > th:
                self.policy.valid_arms.discard(ai)
                self.positive_arms.add(ai)
                self.judged[ai] = "P"
            if ucb < th:
                self.policy.valid_arms.discard(ai)
                self.negative_arms.add(ai)
                self.judged[ai] = "N"

        if len(self.positive_arms) >= find:
            self.stopped = True
            self.result = "positive"
            self.result_detail = "positive (%s found)" % (self.positive_arms)
            return True
        elif len(self.positive_arms) + len(self.policy.valid_arms) < find:
            self.stopped = True
            self.result = "negative"
            self.result_detail = "negative (%s found)" % (self.positive_arms)
            return True

        return False


#%%
class MultiArmedBandit:
    def __init__(self, arms, **kwargs):
        self.arms = arms
        self.result = None
        self.t = 0
        self.K = 0
    
    def run(self, policy: Policy, stopcondition: StopCondition=None):
        self.t = 0
        arms = []
        for a in self.arms:
            a.reset()
            arms.append(a)
        self.K = len(arms)
        policy.reset()
        if stopcondition is not None:
            stopcondition.reset(policy=policy)

        stop = False
        while not stop:
            self.t += 1
            ais = policy.select()
            xs = []
            for ai in ais:
                x = arms[ai].draw()
                xs.append(x)
            policy.update(ais, xs)
            yield ais, xs
            if stopcondition is not None:
                stop = stopcondition.check(ais, xs)


class MultiArmedBanditWithSelectionControl:
    def __init__(self, arms, **kwargs):
        self.arms = arms
        self.result = None
        self.t = 0
        self.K = 0

    def run(self, policy: Policy, stopcondition: StopCondition = None):
        self.t = 0
        arms = []
        for a in self.arms:
            a.reset()
            arms.append(a)
        self.K = len(arms)
        policy.reset()
        if stopcondition is not None:
            stopcondition.reset(policy=policy)

        stop = False
        while not stop:
            self.t += 1
            policy.satisfied = False
            ais = []
            while not policy.satisfied:
                ais += policy.select()
            xs = []
            for ai in ais:
                x = arms[ai].draw()
                xs.append(x)
            policy.update(ais, xs)
            yield ais, xs
            if stopcondition is not None:
                stop = stopcondition.check(ais, xs)

