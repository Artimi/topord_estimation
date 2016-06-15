# -*- coding: utf-8 -*-
import math
import random
import numpy as np
import scipy.stats
import logging
import igraph

logging.basicConfig(level=logging.DEBUG)


def distance(vector: np.array, linext: np.array) -> int:
    return int(np.max(np.ceil(vector - linext)))


def get_linext(vector):
    return np.ceil(vector)


def precedes(graph, v1, v2, n):
    if v1 == n or v2 == n:
        return False
    return v1 in graph.subcomponent(v2, mode=igraph.IN)


def swap(array, i, j):
    result = np.copy(array)
    result[i], result[j] = result[j], result[i]
    return result


def find_first(arr, i):
    return np.nonzero(arr == i)[0][0]


def f(x, sigma_home, i, beta):
    try:
        diff = find_first(x, i) - find_first(sigma_home, i)
    except IndexError:
        return 1
    if diff <= beta:
        return 1
    elif diff <= math.ceil(beta):
        return 1 + beta - math.ceil(beta)
    else:
        return 0


def w(x, sigma_home, beta):
    result = 1
    for i in range(len(x)):
        result *= f(x, sigma_home, i, beta)
    return result


def g_func(x_star, n, sigma_home):
    if x_star[-1] == n:
        x_star[-1] = sigma_home[np.sum(x_star < n)]
    return x_star


class ExampleTPA:
    def estimate(self, r, beta_shell, beta_center):
        k = 0
        for i in range(1, r+1):
            beta = beta_shell
            k -= 1
            # logging.debug("i: %s", i)
            while beta > beta_center:
                k += 1
                beta = random.uniform(0, beta)
                # logging.debug("k: %s, x: %s", k, beta)
        return math.exp(k / r)


class TopordTPA:
    def __init__(self, graph):
        self.graph = graph
        self.n = len(graph.vs)
        self.sigma_home = np.arange(self.n)

    def estimate(self, r, beta_shell, beta_center):
        k = 0
        for i in range(1, r+1):
            logging.debug("RUN #%s, estimate: %s", i, math.exp(k / i))
            beta = beta_shell
            k -= 1
            while beta > beta_center:
                logging.debug("beta: %s", beta)
                k += 1
                x = self.get_uniform(beta)
                # logging.debug("X: %s", x)
                beta = self.infimum(beta_center, beta_shell, x)
        return math.exp(k / r)

    def get_uniform(self, beta):
        S = self.generate(10**4, beta)
        # assert distance(S, self.sigma_home) <= math.ceil(beta)
        if distance(S, self.sigma_home) > math.ceil(beta):
            logging.warning("Generated linear extension %s have larger distance %s than beta %s", S, distance(S, self.sigma_home), beta)
        X = self.sample_continuous(S, beta)
        return X

    def infimum(self, beta_center, beta_shell, X):
        return np.max(X - self.sigma_home)

    def bounding_chain_step(self, x, x_star, i, H, C, beta):
        # logging.debug("x: %s, x*: %s", x, x_star)
        HH = H if x[i] != x_star[i + 1] else 1 - H
        x_new = self.step(x, i, C, HH, beta)
        x_star_new = g_func(self.step(x_star, i, C, H, beta), self.n, self.sigma_home)
        return x_new, x_star_new

    def step(self, x, i, C, H, beta):
        t = swap(x, i, i + 1)
        if H == 1 \
                and not precedes(self.graph, x[i], x[i+1], self.n)\
                and f(t, self.sigma_home, i+1, beta) > 0\
                and (C == 1 or f(t, self.sigma_home, i + 1, beta) == 1):
            return t
        else:
            return x

    def test_generate(self, t, beta):
        result = []
        x = np.copy(self.sigma_home)
        x_star = np.zeros_like(x) + self.n
        for j in range(t):
            i = np.random.randint(0, self.n - 1)
            H = random.choice((0, 1))
            C = scipy.stats.bernoulli.rvs(1 + beta - math.ceil(beta))
            x, x_star = self.bounding_chain_step(x, x_star, i, H, C, beta)
            result.append(x_star)
        return result

    def generate(self, t, beta):
        logging.debug("Generate t: %s, beta: %s", t, beta)
        x = np.copy(self.sigma_home)
        x_star = np.zeros_like(x) + self.n
        i = np.zeros(t, dtype=int)
        H = np.zeros(t, dtype=int)
        C = np.zeros(t, dtype=int)
        for j in range(t):
            i[j] = np.random.randint(0, self.n - 1)
            H[j] = random.choice((0, 1))
            C[j] = scipy.stats.bernoulli.rvs(1 + beta - math.ceil(beta))
            x, x_star = self.bounding_chain_step(x, x_star, i[j], H[j], C[j], beta)
        if np.all(x_star != self.n):
            x = x_star
        else:
            logging.debug("x*: %s", x_star)
            x = self.generate(t, beta)
            x_star = np.zeros_like(x) + self.n
            for j in range(t):
                x, x_star = self.bounding_chain_step(x, x_star, i[j], H[j], C[j], beta)
        return x

    def sample_continuous(self, S, beta):
        ceil_beta = math.ceil(beta)
        X = np.zeros(self.n)
        for i in range(self.n):
            if S[i] - self.sigma_home[i] < ceil_beta:
                X[i] = np.random.uniform(S[i] - 1, S[i])
            else:
                X[i] = np.random.uniform(S[i] - 1, S[i] + beta - ceil_beta)
        return X


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--example", action='store_true', default=False)
    parser.add_argument("-t", "--topord", action="store_true", default=False)
    parser.add_argument("-d", "--delta", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=0.1)
    args = parser.parse_args()

    if args.example:
        tpa = ExampleTPA()

        r1 = int(math.log(2/args.delta))
        L1 = tpa.estimate(r1, 10, 1)
        A1 = math.log(L1)
        # r2 = int(2 * (A1 + math.sqrt(A1) + 2) * (math.log(1 + args.eps)**2 - math.log(1 + args.eps)**3)**-1 * math.log(4 / args.delta))
        r2 = int(2 * (A1 + math.sqrt(A1) + 2 + math.log(1 + args.eps)) * math.log(1 + args.eps)**-2  * math.log(4 / args.delta))
        print(r2)
        L2 = tpa.estimate(r2, 10, 1)
        print(L2)
    elif args.topord:
        g = igraph.Graph(n=12, directed=True, edges=[(0, 1), (0, 2), (0, 3), (0, 4), (1, 10), (2, 5),
                                                     (2, 6), (3, 7), (4, 8), (5, 9), (6, 10), (9, 10),
                                                     (7, 11), (8, 11), (10, 11)])
        tpa = TopordTPA(g)
        print(tpa.estimate(10, tpa.n - 1, 0))
        # x_stars = tpa.test_generate(10000, 6)
        # for i in x_stars[-10:]:
        #     print(i)


