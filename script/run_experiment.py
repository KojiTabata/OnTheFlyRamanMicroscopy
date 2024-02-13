import datetime
import matplotlib.pyplot as plt
import numpy as np

from bandit import TopTwoThompsonSampling
from bandit import MultiArmedBandit, Arm, GoodArmIdentification, MultiArmedBanditWithSelectionControl
from bandit import ThompsonSamplingBernoulli, SuccessiveElimination, ThompsonSamplingNonParametric, Policy, RandomSelection
from multiprocessing import Pool

import random
import re, os, pickle
from bisect import bisect
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cbook as cbook

PLOT = False
DEBUG = False
n_jobs = 10


if os.name != "nt":
    import matplotlib
    matplotlib.use('Agg')


def load_anom_index():
    dirname = "../data/anom_index"
    re_fn = re.compile(r"^(FTC133|Nthyori31)_(\d+).pickle$")

    result = {}
    for x in os.listdir(dirname):
        mo = re_fn.search(x)
        if mo is None:
            continue
        with open("%s/%s" % (dirname, x), "rb") as f:
            im = pickle.load(f)
            result[mo.group(1), mo.group(2)] = im

    return result


class Grids:
    def __init__(self, im, gs, type_of_cell, no):
        self.w = w = (im.shape[0] // gs) * gs
        self.h = h = (im.shape[1] // gs) * gs
        self.gs = gs
        self.K = K = (w // gs) * (h // gs)
        self.im = im[:w, :h, :]
        self.remain_points_in_grid = {}
        for i in range(K):
            l, t = self.arm_left_top(i)
            self.remain_points_in_grid[i] = {(l + x, t + y) for x in range(gs) for y in range(gs)}
        self.measured = set()
        self.type_of_cell = type_of_cell
        self.no = no

    def reset(self):
        self.remain_points_in_grid = {}
        for i in range(self.K):
            l, t = self.arm_left_top(i)
            self.remain_points_in_grid[i] = {(l + x, t + y) for x in range(self.gs) for y in range(self.gs)}
        self.measured = set()

    def max_samples(self):
        return [self.gs*self.gs] * self.K

    def mus(self):
        res = []
        for i in range(self.K):
            l, t = self.arm_left_top(i)
            v = np.mean(self.im[l:l+self.gs, t:t+self.gs, 2])
            res.append(v)
        return res

    def imshow(self, pname, title=None, outdir="."):
        if outdir != "." and not os.path.exists(outdir):
            os.mkdir(outdir)
        plt.title("Cancer Index: %s_%s" % (self.type_of_cell, self.no))
        if self.rounding:
            plt.imshow(self.im[:, :, 2] > 0.5)
        else:
            plt.imshow(self.im[:, :, 2])
        plt.savefig("%s/%s_%s_ci.png" % (outdir, self.type_of_cell, self.no))
        plt.show()

        im = np.zeros(self.im.shape[:-1])
        for p in self.measured:
            im[p[0], p[1]] = 1
        plt.imshow(im)
        if title:
            plt.title(title)
        plt.savefig("%s/%s_%s_%s_measured.png" % (outdir, self.type_of_cell, self.no, pname))
        plt.close()

    def arm_left_top(self, i):
        gw = (self.w // self.gs)
        gh = (self.h // self.gs)
        gx, gy = self.w - (i // gh + 1) * self.gs, (i % gh) * self.gs

        return gx, gy

    def select_point(self, i, ocupied_x=None, ocupied_y=None):
        if len(self.remain_points_in_grid[i]) == 0:
            return None
        possible_points = self.remain_points_in_grid[i]
        if ocupied_x is not None:
            possible_points = {(x, y) for (x, y) in possible_points if x not in ocupied_x}
        if ocupied_y is not None:
            possible_points = {(x, y) for (x, y) in possible_points if y not in ocupied_y}
        if len(possible_points) == 0:
            return None

        return random.choice(list(possible_points))

    def evaluate(self, i, pt):
        self.remain_points_in_grid[i].remove(pt)
        self.measured.add(pt)
        return self.im[pt[0], pt[1], 2]


def run(delta, threshold, gs):
    random.seed(42)
    np.random.seed(42)

    anom_data = load_anom_index()

    repeat = 100

    bandit_result = {}

    for (type_of_cell, no), im in anom_data.items():
        grids = Grids(im, gs, type_of_cell, no)
        image_map = get_map(type_of_cell, no, gs)

        if (type_of_cell, no) not in bandit_result:
            bandit_result[type_of_cell, no] = {}

        mus = grids.mus()

        K = grids.K

        policies = {
            "SE": SuccessiveElimination(K=K),
            "TS-RR": ThompsonSamplingBernoulli(K=K, ab=[1, 1], prior_update="random_rounding"),
            "TS-NP": ThompsonSamplingNonParametric(K=K),
            "RAND": RandomSelection(K=K),
            "TTTS": TopTwoThompsonSampling(K=K),
        }

        gai = GoodArmIdentification(threshold=threshold, delta=delta, max_samples=grids.max_samples())

        for policy_name, policy in policies.items():
            if policy_name not in bandit_result[type_of_cell, no]:
                bandit_result[type_of_cell, no][policy_name] = []

            args_list = []
            for rep in range(repeat):
                args = (rep, policy_name, type_of_cell, no, gs, grids, policy, gai, image_map, saveto, threshold)
                args_list.append(args)

            if DEBUG:
                res = []
                for arg in args_list:
                    res.append(exp_fun(arg))
            else:
                with Pool(n_jobs) as pool:
                    res = pool.map(exp_fun, args_list)

            for rep, (t, ndraw, measured_points, measured_points_t, plot_params, gai_result) in enumerate(res):
                bandit_result[type_of_cell, no][policy_name].append((t, ndraw, gai_result, measured_points_t, plot_params))

    return bandit_result


def main():
    fp_reuslt = "result_bandit.pickle"

    if os.path.isfile(fp_reuslt):
        with open(fp_reuslt, 'rb') as f:
            result = pickle.load(f)
    else:
        result = {}
        for delta in [0.01]:
            result[delta] = run(delta, threshold=0.4698551527853685, gs=40)
        with open(fp_reuslt, 'wb') as f:
            pickle.dump(result, f)

    print("ok")


if __name__ == "__main__":
    main()


