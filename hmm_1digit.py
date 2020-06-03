#!/usr/bin/env python3

# Copyright 2018 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import math

import numpy as np


def good_log(x):
    x_log = np.log(x, where=(x != 0))
    x_log[np.where(x == 0)] = -10000000
    return x_log


def log_gaussian(o, mu, r):
    compute = (- 0.5 * np.log(r) - np.divide(
        np.square(o - mu), 2 * r) - 0.5 * np.log(2 * math.pi)).sum()
    return compute


def normalize(x):
    n = np.exp(x - x.max())
    n /= n.sum()
    return n


def lse(x):
    # log sum exp
    m = np.max(x)
    x -= m
    return m + np.log(np.sum(np.exp(x)))


def forward(pi, a, o, mu, r):
    # pi is initial probability over states, a is transition matrix
    T = o.shape[0]
    J = mu.shape[0]
    log_alpha = np.zeros((T, J))

    for j in range(J):
        log_alpha[0][j] = np.log(pi)[j] + log_gaussian(o[0], mu[j], r[j])

    for t in range(1, T):
        for j in range(J):
            log_alpha[t, j] = log_gaussian(o[t], mu[j], r[j]) + lse(good_log(a[:, j].T) + log_alpha[t - 1])

    return log_alpha


def backward(a, o, mu, r):
    # a is transition matrix
    log_a = good_log(a)
    T = o.shape[0]
    J = mu.shape[0]
    # {beta_T(j) = 1}^J_{j=1}
    log_beta = np.zeros((T, J))

    for t in range(T - 1, 0, -1):
        for i in range(J):
            temp = []
            for j in range(J):
                temp.append(log_gaussian(o[t + 1], mu[j], r[j]) + log_beta[t + 1, j] + log_a[i, j])

            log_beta[t, i] = lse(np.array(temp))

    return log_beta


def viterbi(pi, a, data, mu, r):
    states = []
    for s in range(len(data)):
        T = data[s].shape[0]
        J = mu.shape[0]
        s_hat = np.zeros(T, dtype=int)
        log_a = good_log(a)
        log_delta = np.zeros((T, J))
        log_delta[0] = np.log(pi)
        psi = np.zeros((T, J))

        # initialize
        for j in range(J):
            log_delta[0, j] += log_gaussian(data[s][0], mu[j], r[j])

        for t in range(1, T):
            for j in range(J):
                temp = np.zeros(J)
                for i in range(J):
                    temp[i] = log_delta[t - 1, i] + log_a[i, j] + log_gaussian(data[s][t], mu[j], r[j])
                log_delta[t, j] = np.max(temp)
                psi[t, j] = np.argmax(log_delta[t - 1] + log_a[:, j])

        s_hat[T - 1] = np.argmax(log_delta[T - 1])

        for t in reversed(range(T - 1)):
            s_hat[t] = psi[t + 1, s_hat[t+1]]

        states.append(s_hat)

    return states


def get_data_dict(data):
    data_dict = {}
    for line in data:
        if "[" in line:
            key = line.split()[0]
            mat = []
        elif "]" in line:
            line = line.split(']')[0]
            mat.append([float(x) for x in line.split()])
            data_dict[key] = np.array(mat)
        else:
            mat.append([float(x) for x in line.split()])
    return data_dict


# you can add more functions here if needed

class SingleGauss():
    def __init__(self):
        self.dim = None
        self.mu = None
        self.r = None

    def train(self, data):
        data = np.vstack(data)
        self.mu = np.mean(data, axis=0)
        self.r = np.mean(np.square(np.subtract(data, self.mu)), axis=0)

    def loglike(self, data_mat):
        ll = 0
        for each_line in data_mat:
            ll += log_gaussian(each_line, self.mu, self.r)
        return ll


class GMM():
    def __init__(self, sg_model, ncomp):
        # Basic class variable initialized, feel free to add more
        self.ncomp = ncomp
        self.r = np.tile(sg_model.r, (ncomp, 1))
        self.mu = np.tile(sg_model.mu, (ncomp, 1))
        for k in range(ncomp):
            eps_k = np.random.randn()
            self.mu[k] += 0.01 * eps_k * np.sqrt(sg_model.r)

    def e_step(self, data):
        gamma = np.zeros((data.shape[0], self.ncomp))
        for t in range(data.shape[0]):
            gamma[t] = np.log(np.ones(self.ncomp) / self.ncomp)
            for k in range(self.ncomp):
                gamma[t][k] += log_gaussian(data[t], self.mu[k], self.r[k])
            gamma[t] = normalize(gamma[t])
        return gamma

    def m_step(self, data, gamma):
        self.w = np.sum(gamma, axis=0) / np.sum(gamma)
        self.mu = np.zeros(self.mu.shape)
        self.r = np.zeros(self.r.shape)
        denom = np.sum(gamma, axis=0, keepdims=True).T
        for k in range(self.ncomp):
            for t in range(gamma.shape[0]):
                self.mu[k] += gamma[t][k] * data[t]
        self.mu = np.divide(self.mu, denom)
        for k in range(self.ncomp):
            for t in range(gamma.shape[0]):
                self.r[k] += gamma[t][k] * np.square(np.subtract(data[t], self.mu[k]))
        self.r = np.divide(self.r, denom)
        return

    def train(self, data):
        data = np.vstack(data)
        gamma = self.e_step(data)
        self.m_step(data, gamma)

    def loglike(self, data_mat):
        ll = 0
        for t in range(data_mat.shape[0]):
            ll_t = np.array([np.log(self.w[k]) + log_gaussian(data_mat[t], self.mu[k], self.r[k])
                             for k in range(self.ncomp)])
            ll_t = lse(ll_t)
            ll += ll_t
        return ll


class HMM():
    def __init__(self, sg_model, nstate):
        # Basic class variable initialized, feel free to add more
        self.pi = np.full(nstate, 1 / nstate)
        self.mu = np.tile(sg_model.mu, (nstate, 1))
        self.r = np.tile(sg_model.r, (nstate, 1))
        self.nstate = nstate

    def initStates(self, data):
        # states: s elements, each T length
        self.states = []
        for data_s in data:
            T = data_s.shape[0]
            state_seq = np.array([self.nstate * t / T for t in range(T)], dtype=int)
            self.states.append(state_seq)

    def m_step(self, data):
        self.a = np.zeros((self.nstate, self.nstate))
        gamma_0 = np.zeros(self.nstate)
        gamma_1 = np.zeros((self.nstate, data[0].shape[1]))
        gamma_2 = np.zeros((self.nstate, data[0].shape[1]))

        for s in range(len(data)):
            T = data[s].shape[0]

            # state_seq is a list of states with length t
            state_seq = self.states[s]
            # gamma is emission_matrix
            gamma = np.zeros((T, self.nstate))

            # calculate frequency for a and gamma according to current states
            for t, j in enumerate(state_seq[:-1]):
                self.a[j, state_seq[t + 1]] += 1
            for t, j in enumerate(state_seq):
                gamma[t, j] = 1

            # gamma^0_j = \sum^T_{t=1} gamma_t(j)
            gamma_0 += np.sum(gamma, axis=0)
            # gamma^1_j = \sum^T_{t=1}gamma_t(j)o_t
            # gamma^2_j = \sum^\sum^T_{t=1}gamma_t(j)o_t**2
            for t, j in enumerate(state_seq):
                gamma_1[j] += data[s][t]
                gamma_2[j] += np.square(data[s][t])

        for j in range(self.nstate):
            self.a[j] /= np.sum(self.a[j])
            self.mu[j] = gamma_1[j] / gamma_0[j]
            self.r[j] = (gamma_2[j] - np.multiply(gamma_0[j], self.mu[j] ** 2)) / gamma_0[j]

    def train(self, data, iter):
        if iter == 0:
            self.initStates(data)
        self.m_step(data)
        # renew states
        self.states = viterbi(self.pi, self.a, data, self.mu, self.r)

    def loglike(self, data):
        log_alpha_t = forward(self.pi, self.a, data, self.mu, self.r)[-1]
        ll = lse(log_alpha_t)

        return ll


def sg_train(digits, train_data):
    model = {}
    for digit in digits:
        model[digit] = SingleGauss()

    for digit in digits:
        data = [train_data[id] for id in train_data.keys() if digit in id.split('_')[1]]
        logging.info("process %d data for digit %s", len(data), digit)
        model[digit].train(data)

    return model


def gmm_train(digits, train_data, sg_model, ncomp, niter):
    logging.info("Gaussian mixture training, %d components, %d iterations", ncomp, niter)

    gmm_model = {}
    for digit in digits:
        gmm_model[digit] = GMM(sg_model[digit], ncomp=ncomp)

    i = 0
    while i < niter:
        logging.info("iteration: %d", i)
        total_log_like = 0.0
        for digit in digits:
            data = [train_data[id] for id in train_data.keys() if digit in id.split('_')[1]]
            logging.info("process %d data for digit %s", len(data), digit)

            gmm_model[digit].train(data)

            total_log_like += gmm_model[digit].loglike(np.vstack(data))
        logging.info("log likelihood: %f", total_log_like)
        i += 1

    return gmm_model


def hmm_train(digits, train_data, sg_model, nstate, niter):
    logging.info("hidden Markov model training, %d states, %d iterations", nstate, niter)

    hmm_model = {}
    for digit in digits:
        hmm_model[digit] = HMM(sg_model[digit], nstate=nstate)

    i = 0
    while i < niter:
        logging.info("iteration: %d", i)
        total_log_like = 0.0
        for digit in digits:
            data = [train_data[id] for id in train_data.keys() if digit in id.split('_')[1]]
            logging.info("process %d data for digit %s", len(data), digit)

            hmm_model[digit].train(data, i)

            for data_u in data:
                total_log_like += hmm_model[digit].loglike(data_u)

        logging.info("log likelihood: %f", total_log_like)
        i += 1

    return hmm_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train', type=str, help='training data')
    parser.add_argument('test', type=str, help='test data')
    parser.add_argument('--niter', type=int, default=10)
    parser.add_argument('--nstate', type=int, default=5)
    parser.add_argument('--ncomp', type=int, default=8)
    parser.add_argument('--mode', type=str, default='sg',
                        choices=['sg', 'gmm', 'hmm'],
                        help='Type of models')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # set seed
    np.random.seed(777)

    # logging info
    log_format = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s:%(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)

    digits = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "z", "o"]

    # read training data
    with open(args.train) as f:
        train_data = get_data_dict(f.readlines())
    # for debug - use only 100 files
    if args.debug:
        train_data = {key: train_data[key] for key in list(train_data.keys())[:100]}

    # read test data
    with open(args.test) as f:
        test_data = get_data_dict(f.readlines())
    # for debug
    if args.debug:
        test_data = {key: test_data[key] for key in list(test_data.keys())[:100]}

    # Single Gaussian
    sg_model = sg_train(digits, train_data)

    if args.mode == 'sg':
        model = sg_model
    elif args.mode == 'hmm':
        model = hmm_train(digits, train_data, sg_model, args.nstate, args.niter)
    elif args.mode == 'gmm':
        model = gmm_train(digits, train_data, sg_model, args.ncomp, args.niter)

    # test data performance
    total_count = 0
    correct = 0
    for key in test_data.keys():
        lls = []
        for digit in digits:
            ll = model[digit].loglike(test_data[key])
            lls.append(ll)
        predict = digits[np.argmax(np.array(lls))]
        log_like = np.max(np.array(lls))

        logging.info("predict %s for utt %s (log like = %f)", predict, key, log_like)
        if predict in key.split('_')[1]:
            correct += 1
        total_count += 1

    logging.info("accuracy: %f", float(correct / total_count * 100))
