"""
Bayesian Personalized Ranking

Matrix Factorization model and a variety of classes
implementing different sampling strategies.

Original data is a matrix of <user, item>
After training the data we get user_factors(W) and item_factors(H)
"""

import numpy as np
from math import log,exp
import random


class BPRArgs(object):

    def __init__(self, learning_rate=0.05,
                 bias_regularization=1.0,
                 user_regularization=0.0025,
                 positive_item_regularization=0.0025,
                 negative_item_regularization=0.00025,
                 update_negative_item_factors=True):
        self.learning_rate = learning_rate
        self.bias_regularization = bias_regularization
        self.user_regularization = user_regularization
        self.positive_item_regularization = positive_item_regularization
        self.negative_item_regularization = negative_item_regularization
        self.update_negative_item_factors = update_negative_item_factors

class BPR(object):

    def __init__(self, D, args):
        """initialise BPR matrix factorization model
        D: number of factors(10)
        args: parameter table
        """
        self.D = D
        self.learning_rate = args.learning_rate
        self.bias_regularization = args.bias_regularization
        self.user_regularization = args.user_regularization
        self.positive_item_regularization = args.positive_item_regularization
        self.negative_item_regularization = args.negative_item_regularization
        self.update_negative_item_factors = args.update_negative_item_factors

    def train(self, data, sampler, num_iters):
        """train model
        data: user-item matrix as a scipy sparse matrix
              users and items are zero-indexed
              here convergence is replaced by iteration number limit(10)
        """
        self.init(data)

        print 'initial loss = {0}'.format(self.loss())
        """num_iters is set as 10"""

        # TODO: Update factors(W and H) and show negative BPR-OPT during each iteration
        # Show negative BPR-OPT to make sure that we are minimizing it
        for it in xrange(num_iters):
            print 'starting iteration {0}'.format(it)
            for u, i, j in sampler.generate_samples(self.data):
                self.update_factors(u, i, j)
            print 'iteration {0}: loss = {1}'.format(it, self.loss())

    def init(self, data):
        self.data = data
        # data is csr matrix
        self.num_users, self.num_items = self.data.shape
        # item_bias is an one-dimension array, its length = num_items, all elements are 0.
        self.item_bias = np.zeros(self.num_items)
        # user_factors is self.num_users-by-self.D array of random numbers from [0.0, 1.0)
        # it equals the W(factor matrix of users) in the paper
        # TODO: initialise the user factor matrix W with random values between 0 and 1
        self.user_factors = np.random.random_sample((self.num_users, self.D))
        # item_factors is self.num_items-by-self.D array of random numbers from [0.0, 1.0)
        # it equals the H(factor matrix of items) in the paper
        # TODO: initialise the item factor matrix H with random values between 0 and 1
        self.item_factors = np.random.random_sample((self.num_items,self.D))
        self.create_loss_samples()

    def create_loss_samples(self):
        # apply rule of thumb to decide num samples over which to compute loss
        num_loss_samples = int(100*self.num_users**0.5)
        print 'sampling {0} <user,item i,item j> triples...'.format(num_loss_samples)
        # construct a sampler object(sampler)
        sampler = UniformUserUniformItem(True)
        """TODO:compare num_loss_samples with data.nnz in num_samples
        loss_samples is DS
        generate a list of triples(u,i,j)
        """
        self.loss_samples = [t for t in sampler.generate_samples(data, num_loss_samples)]

    def update_factors(self, u, i, j, update_u=True, update_i=True):
        """apply SGD update"""

        update_j = self.update_negative_item_factors
        # item_bias is initialised zeros
        x = self.item_bias[i] - self.item_bias[j] \
            + np.dot(self.user_factors[u, :], self.item_factors[i, :]-self.item_factors[j, :])

        z = 1.0/(1.0+exp(x))
        if update_i:
            d = z - self.bias_regularization * self.item_bias[i]

            self.item_bias[i] += self.learning_rate * d
        if update_j:
            d = -z - self.bias_regularization * self.item_bias[j]
            self.item_bias[j] += self.learning_rate * d
        if update_u:
            d = (self.item_factors[i, :]-self.item_factors[j, :])*z - self.user_regularization*self.user_factors[u, :]
            self.user_factors[u, :] += self.learning_rate*d
        if update_i:
            d = self.user_factors[u, :]*z - self.positive_item_regularization*self.item_factors[i, :]
            self.item_factors[i, :] += self.learning_rate*d
        if update_j:
            d = -self.user_factors[u, :]*z - self.negative_item_regularization*self.item_factors[j, :]
            self.item_factors[j, :] += self.learning_rate*d

    def loss(self):
        ranking_loss = 0
        for u, i, j in self.loss_samples:
            x = self.predict(u, i) - self.predict(u, j)
            ranking_loss += 1.0 / (1.0 + exp(x))
        complexity = 0
        for u, i, j in self.loss_samples:
            complexity += self.user_regularization * np.dot(self.user_factors[u], self.user_factors[u])
            complexity += self.positive_item_regularization * np.dot(self.item_factors[i], self.item_factors[i])
            complexity += self.negative_item_regularization * np.dot(self.item_factors[j], self.item_factors[j])
            complexity += self.bias_regularization * self.item_bias[i]**2
            complexity += self.bias_regularization * self.item_bias[j]**2
        return -log(ranking_loss) + 0.5*complexity
        # return negative BPR-OPT( 0.5 is not necessary)

    def predict(self, u, i):
        return self.item_bias[i] + np.dot(self.user_factors[u], self.item_factors[i])


# sampling strategies
class Sampler(object):

    def __init__(self, sample_negative_items_empirically):
        self.sample_negative_items_empirically = sample_negative_items_empirically

    def init(self, data, max_samples=None):
        self.data = data
        self.num_users, self.num_items = data.shape
        self.max_samples = max_samples

    def sample_user(self):
        u = self.uniform_user()
        num_items = self.data[u].getnnz()
        assert(num_items > 0 and num_items != self.num_items)
        return u

    def sample_negative_item(self, user_items):
        j = self.random_item()
        while j in user_items:
            j = self.random_item()
        return j

    def uniform_user(self):
        return random.randint(0,self.num_users-1)

    def random_item(self):
        """sample an item uniformly or from the empirical distribution
           observed in the training data
        """
        if self.sample_negative_items_empirically:
            # just pick something someone rated!
            #  TODO: choose a user randomly
            u = self.uniform_user()
            i = random.choice(self.data[u].indices)
        else:
            i = random.randint(0, self.num_items - 1)
        return i

    def num_samples(self, n):
        if self.max_samples is None:
            return n
        return min(n, self.max_samples)


# TODO:sample users and items separately
class UniformUserUniformItem(Sampler):

    def generate_samples(self, data, max_samples=None):
        self.init(data,max_samples)
        """nnz is the number of nonzero values in sparse matrix
        and in num_samples() we compare data.nnz with num_loss_samples to identify num_samples
        """
        for _ in xrange(self.num_samples(self.data.nnz)):
            # TODO: choose a user randomly
            u = self.uniform_user()
            # sample positive item
            i = random.choice(self.data[u].indices)
            # TODO: sample negative item
            j = self.sample_negative_item(self.data[u].indices)
            yield u, i, j


# TODO: sample user and item separately without repeating
class UniformUserUniformItemWithoutReplacement(Sampler):

    def generate_samples(self, data, max_samples=None):
        self.init(self, data, max_samples)

        # make a local copy of data as we're going to "forget" some entries
        self.local_data = self.data.copy()

        for _ in xrange(self.num_samples(self.data.nnz)):
            u = self.uniform_user()
            # sample positive item without replacement if we can
            user_items = self.local_data[u].nonzero()[1]
            if len(user_items) == 0:
                # reset user data if it's all been sampled
                for ix in self.local_data[u].indices:
                    self.local_data[u, ix] = self.data[u, ix]
                user_items = self.local_data[u].nonzero()[1]
            i = random.choice(user_items)
            # forget this item so we don't sample it again for the same user
            self.local_data[u, i] = 0
            j = self.sample_negative_item(user_items)
            yield u, i, j


# TODO: sample user and item by pair (they are corresponding)
class UniformPair(Sampler):
    def generate_samples(self, data, max_samples=None):
        self.init(data, max_samples)
        for _ in xrange(self.num_samples(self.data.nnz)):
            idx = random.randint(0, self.data.nnz-1)
            u = self.users[self.idx]
            i = self.items[self.idx]
            j = self.sample_negative_item(self.data[u].indices)
            yield u, i, j


# idxs make it sure that there is not any repeating
class UniformPairWithoutReplacement(Sampler):
    def generate_samples(self, data, max_samples=None):
        self.init(data, max_samples)
        self.users, self.items = self.data.nonzero()
        idxs = range(self.users.size)
        # idxs is a list from 0 to data.nnz-1
        # TODO: make idxs(a list) out of order
        random.shuffle(idxs)

        self.users = self.users[idxs]
        self.items = self.items[idxs]
        self.idx = 0
        for _ in xrange(self.num_samples(self.users.size)):
            u = self.users[self.idx]
            i = self.items[self.idx]
            j = self.sample_negative_item(self.data[u].indices)
            self.idx += 1
            yield u, i, j


class ExternalSchedule(Sampler):
    def __init__(self, filepath, index_offset=0):
        self.filepath = filepath
        self.index_offset = index_offset

    def generate_samples(self, data, max_samples=None):
        self.init(data, max_samples)
        f = open(self.filepath)
        samples = [map(int, line.strip().split()) for line in f]
        random.shuffle(samples)  # important!
        num_samples = self.num_samples(len(samples))
        for u, i, j in samples[:num_samples]:
            yield u-self.index_offset, i-self.index_offset, j-self.index_offset

if __name__ == '__main__':
    # import sys
    # from scipy.io import mmread
    # data = mmread(sys.argv[1]).tocsr()

    # TODO: generate data
    from scipy.sparse import coo_matrix
    from numpy import array
    l = []
    for i in xrange(1000):
        l.append(random.randint(0, 99))
    row = array(l)
    l = []
    for i in xrange(1000):
        l.append(random.randint(0, 99))
    column = array(l)
    l = []
    for _ in xrange(1000):
        l.append(random.randint(0, 1))
    d = array(l)
    da = coo_matrix((d, (row, column)), shape=(100, 100))
    data = da.tocsr()
    args = BPRArgs()
    args.learning_rate = 0.2

    num_factors = 10
    model = BPR(num_factors, args)

    sample_negative_items_empirically = True
    # Change to another sample strategy
    sampler = UniformPairWithoutReplacement(sample_negative_items_empirically)
    # sampler = UniformUserUniformItemWithoutReplacement(sample_negative_items_empirically)

    num_iters = 7
    model.train(data, sampler, num_iters)
