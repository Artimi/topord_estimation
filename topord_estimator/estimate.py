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


def precedes(graph, v1, v2):
    if v1 == -1 or v2 == -1:
        return False
    return v1 in graph.subcomponent(v2, mode=igraph.IN)


def swap(array, i, j):
    array[i], array[j] = array[j], array[i]


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
        S = self.generate(self.n, beta)
        # assert distance(S, self.sigma_home) <= math.ceil(beta)
        X = self.sample_continuous(S, beta)
        return X

    def infimum(self, beta_center, beta_shell, X):
        return np.max(X - self.sigma_home)

    def bounding_chain_step(self, sigma, B, i, C1, C2, beta):
        C3 = 1 - C1 if sigma[i] == B[i + 1] else C1
        if C1 == 1 and not precedes(self.graph, sigma[i], sigma[i+1]):
            if C2 == 1 or sigma[i] - self.sigma_home[i] != math.ceil(beta) - 1:
                swap(sigma, i, i+1)
        if C3 == 1 and not precedes(self.graph, B[i], B[i+1]):
            if C2 == 1 or B[i] - self.sigma_home[i] != math.ceil(beta) - 1:
                swap(B, i, i+1)
        if B[-1] == -1:
            unique, count = np.unique(B, return_counts=True)
            p = sum(count[1:])
            B[-1] = self.sigma_home[p]
        return sigma, B

    def generate(self, t, beta):
        # logging.debug("Generate t: %s, beta: %s", t, beta)
        sigma = np.copy(self.sigma_home)
        B = np.zeros_like(sigma) - 1
        B[-1] = self.sigma_home[0]
        i = np.zeros(t, dtype=int)
        C1 = np.zeros(t, dtype=int)
        C2 = np.zeros(t, dtype=int)
        for j in range(t):
            i[j] = np.random.randint(0, self.n - 1)
            C1[j] = random.choice((0, 1))
            C2[j] = scipy.stats.bernoulli.rvs(1 + beta - math.ceil(beta))
            sigma, B = self.bounding_chain_step(sigma, B, i[j], C1[j], C2[j], beta)
        if np.all(B != -1):
            sigma = B
        else:
            sigma = self.generate(2*t, beta)
            for j in range(t):
                sigma, B = self.bounding_chain_step(sigma, B, i[j], C1[j], C2[j], beta)
        return sigma

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
        r2 = int(2 * (A1 + math.sqrt(A1) + 2) * (math.log(1 + args.eps)**2 - math.log(1 + args.eps)**3)**-1 * math.log(4 / args.delta))
        print(r2)
        L2 = tpa.estimate(r2, 10, 1)
        print(L2)
    elif args.topord:
        g = igraph.Graph(n=12, directed=True, edges=[(0, 1), (0, 2), (0, 3), (0, 4), (1, 10), (2, 5),
                                                     (2, 6), (3, 7), (4, 8), (5, 9), (6, 10), (9, 10),
                                                     (7, 11), (8, 11), (10, 11)])
        tpa = TopordTPA(g)
        print(tpa.estimate(10, tpa.n - 1, 0))


