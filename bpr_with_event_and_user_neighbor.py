#!/usr/bin/python
# -*- coding:utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

"""
===============================================================================
@author:    赵明星
@date:      2016-07-08
@desc:      实现BPR算法，融入了user和event的近邻信息。

            Bayesian Personalized Ranking with Neighbor
            
            Matrix Factorization model and a variety of classes
                implementing different sampling strategies.
            
            Original data is a csr_matrix of <user, item>
            After training the data we get user_factors(W) and item_factors(H)

===============================================================================
"""


import numpy as np
from math import log, exp
from event_neighbor import ItemNeighbor
from user_neighbor import UserNeighbor
from getdata import DataMode
import random


class BPRArgs(object):

    def __init__(self, learning_rate=0.01,
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

    def __init__(self, dimension, args, i_neighbors, u_neighbors, t_data):
        """initialise BPR matrix factorization model
        D: number of factors
        args: parameter table
        neighbors are dict generated by find_neighbors
        """
        self.dimension = dimension
        self.learning_rate = args.learning_rate
        self.bias_regularization = args.bias_regularization
        self.user_regularization = args.user_regularization
        self.positive_item_regularization = args.positive_item_regularization
        self.negative_item_regularization = args.negative_item_regularization
        self.update_negative_item_factors = args.update_negative_item_factors
        self.events_neighbors = i_neighbors
        self.users_neighbors = u_neighbors
        self.i_neighbor_counter = 0
        self.j_neighbor_counter = 0
        self.u_neighbor_counter = 0
        self.iter_num_x = []
        self.ranking_loss_y = []
        self.data = None
        self.item_factors = None
        self.item_bias = None
        self.user_factors = None
        self.loss_samples = None
        self.num_users = None
        self.num_items = None
        self.test_data = t_data
        # item_neighbors is a item_number-by-neighbor_number array
        # the element of item_neighbor stands for an item

    def train(self, data, sampler, i_num):
        """train model
        data: user-item matrix as a scipy sparse matrix
              users and items are zero-indexed
              here convergence is replaced by iteration number limit(10)
        """
        self.init(data)
        # initial loss is random
        print 'initial loss = {0}'.format(self.loss())

        # TODO: Update factors(W and H) and show negative BPR-OPT during each iteration
        # TODO: Show negative BPR-OPT to make sure that we are minimizing it
        for it in xrange(i_num):
            self.iter_num_x.append(it)
            print 'starting iteration {0}'.format(it)
            for u, i, j in sampler.generate_samples(self.data):
                self.update_factors(u, i, j)
            print 'iteration {0}: loss = {1}'.format(it, self.loss())
            self.ranking_loss_y.append(self.loss())

    def init(self, data):
        self.data = data
        # data is csr matrix
        self.num_users, self.num_items = self.data.shape
        # item_bias is an one-dimension array, its length = num_items, all elements are initialized to 0
        # it shows users' preference for each item
        # the array of each user differs and it's temporary
        self.item_bias = np.zeros(self.num_items)
        # user_factors is self.num_users-by-self.D array of random numbers from [0.0, 1.0)
        # it equals the W(factor matrix of users) in the paper
        # TODO: initialise the user factor matrix W with random values between 0 and 1
        self.user_factors = np.random.random_sample((self.num_users, self.dimension))
        # item_factors is self.num_items-by-self.D(number of factors, 10) array of random numbers from [0.0, 1.0)
        # it equals the H(factor matrix of items) in the paper
        # TODO: initialise the item factor matrix H with random values between 0 and 1
        self.item_factors = np.random.random_sample((self.num_items, self.dimension))
        self.create_loss_samples()

    def create_loss_samples(self):
        # apply rule of thumb to decide num samples over which to compute loss
        num_loss_samples = int(100*self.num_users**0.5)
        print 'sampling {0} <user,item i,item j> triples...'.format(num_loss_samples)
        # construct a sampler object named sampler
        sampler = UniformUserUniformItem(True)
        """TODO:compare num generated by rule of thumb with num of items of train_data
        the smaller one is the length of the list of triples(u,i,j) generated by sampler
        DS is a triple array and it shows all the information of the data
        data generated by sampler is just a part of DS and may has repetition in it
        """
        self.loss_samples = [t for t in sampler.generate_samples(self.data, num_loss_samples)]

    def update_factors(self, u, i, j, update_u=True, update_i=True):
        """apply SGD update"""

        update_j = self.update_negative_item_factors

        u_factor_with_neighbor = self.user_factors[u]
        for u_neighbor in self.events_neighbors[u]:
            u_factor_with_neighbor += self.user_factors[u_neighbor]
        u_factor_with_neighbor /= (1 + len(self.events_neighbors[u]))

        i_factor_with_neighbor = self.item_factors[i]
        for i_neighbor in self.users_neighbors[i]:
            i_factor_with_neighbor += self.item_factors[i_neighbor]
        i_factor_with_neighbor /= (1 + len(self.users_neighbors[i]))

        j_factor_with_neighbor = self.item_factors[j]
        for j_neighbor in self.users_neighbors[j]:
            j_factor_with_neighbor += self.item_factors[j_neighbor]
        j_factor_with_neighbor /= (1 + len(self.users_neighbors[j]))

        x = self.item_bias[i] - self.item_bias[j] \
            + np.dot(u_factor_with_neighbor, i_factor_with_neighbor - j_factor_with_neighbor)

        z = 1.0 / (1.0 + exp(x))

        if update_i:
            d = -self.bias_regularization * self.item_bias[i] + z
            self.item_bias[i] += self.learning_rate * d
        if update_j:
            d = -z - self.bias_regularization * self.item_bias[j]
            self.item_bias[j] += self.learning_rate * d
        if update_u:
            d = -self.user_regularization * self.user_factors[u, :] + \
                (i_factor_with_neighbor - j_factor_with_neighbor) * z / (1 + self.u_neighbor_counter)
            self.user_factors[u, :] += self.learning_rate * d
        if update_i:
            d = -self.positive_item_regularization * \
                self.item_factors[i, :] + u_factor_with_neighbor * z / (1 + self.i_neighbor_counter)
            self.item_factors[i, :] += self.learning_rate * d
        if update_j:
            d = -self.negative_item_regularization * \
                self.item_factors[j, :] - u_factor_with_neighbor * z / (1 + self.j_neighbor_counter)
            self.item_factors[j, :] += self.learning_rate * d

    def loss(self):
        ranking_loss = 0
        for u, i, j in self.loss_samples:
            x = self.predict(u, i) - self.predict(u, j)
            ranking_loss += 1.0 / (1.0 + exp(-x))
        complexity = 0
        for u, i, j in self.loss_samples:
            complexity += self.user_regularization * np.dot(self.user_factors[u], self.user_factors[u])
            complexity += self.positive_item_regularization * np.dot(self.item_factors[i], self.item_factors[i])
            complexity += self.negative_item_regularization * np.dot(self.item_factors[j], self.item_factors[j])
            complexity += self.bias_regularization * self.item_bias[i]**2
            complexity += self.bias_regularization * self.item_bias[j]**2
        return -log(ranking_loss) + complexity
        # return negative BPR-OPT

    def predict(self, u, i):
        # get user u's preference for item i
        u_factor_with_neighbor = self.user_factors[u]
        for u_neighbor in self.events_neighbors[u]:
            u_factor_with_neighbor += self.user_factors[u_neighbor]
        u_factor_with_neighbor /= (1 + len(self.events_neighbors[u]))

        return self.item_bias[i] + np.dot(u_factor_with_neighbor, self.item_factors[i])

    # TODO: calculate AUC by factors and test_data
    def calculate_auc(self):
        event_sum = 0
        for piece in self.test_data:
            event_id = piece[0]
            user_id = piece[1]
            index = self.data[event_id].indices
            positive_user_list = list(index)
            positive_user_list.append(user_id)
            negative_user_list = list(set(range(self.num_items)) - set(positive_user_list))
            u_sum = 0
            for j in negative_user_list:
                x_u_i = np.dot(self.user_factors[event_id], self.item_factors[user_id]) + self.item_bias[user_id]
                x_u_j = np.dot(self.user_factors[event_id], self.item_factors[j]) + self.item_bias[j]
                if x_u_i > x_u_j:
                    u_sum += 1.0
            event_sum += u_sum/(len(negative_user_list))
            print 'event sum after each step:', event_sum
        auc = event_sum/(len(self.test_data))
        return auc


class Sampler(object):
    def __init__(self, sample_negative_items_empirically):
        self.sample_negative_items_empirically = sample_negative_items_empirically
        self.data = None
        self.num_users = None
        self.num_items = None
        self.max_samples = None

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
        return random.randint(0, self.num_users-1)

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
# this sample strategy includes negative feedback
class UniformUserUniformItem(Sampler):

    def generate_samples(self, data, max_samples=None):
        self.init(data, max_samples)
        """nnz is the number of nonzero values in sparse matrix
        the matrix should have been all positive elements
        and in num_samples() we compare data.nnz with num_loss_samples to identify num_samples
        """
        for _ in xrange(self.num_samples(self.data.nnz)):
            # TODO: choose a user randomly
            u = self.uniform_user()
            i = random.choice(self.data[u].indices)
            # TODO: sample negative item
            j = self.sample_negative_item(self.data[u].indices)
            yield u, i, j


# idxs make it sure that there is not repetition
class UniformPairWithoutReplacement(Sampler):
    # train called generate_samples with only one parameter(data:a csr matrix)
    def generate_samples(self, data, max_samples=None):
        self.init(data, max_samples)
        self.users, self.items = self.data.nonzero()
        idxs = range(self.users.size)
        # idxs is a list from 0 to data.nonzero().size-1
        """TODO: make idxs(a list) out of order
        thus make positive users and items out of order
        """
        random.shuffle(idxs)
        self.users = self.users[idxs]
        self.items = self.items[idxs]
        self.idx = 0
        # get users.size samples at last
        for _ in xrange(self.num_samples(self.users.size)):
            u = self.users[self.idx]
            i = self.items[self.idx]
            j = self.sample_negative_item(self.data[u].indices)
            self.idx += 1
            yield u, i, j

if __name__ == '__main__':
    # TODO: generate data
    d = DataMode()
    d.find_data()
    train_data = d.find_train_data()
    # train_data in csr format is mainly ordered by row_number
    test_data = d.find_test_data()
    # test_data is a list and its element is a tuple of event_id and positive user_id
    args = BPRArgs()
    args.learning_rate = 0.01

    num_factors_list = [10, 20, 30, 40, 50]
    my_file = file('factor_num_and_auc_user_and_event.txt', 'w')
    for num_factors in num_factors_list:
        neighbor_limit = 10

        loc_neighbor_num = 50
        neighbor_model = ItemNeighbor(loc_neighbor_num, neighbor_limit)
        items_neighbors = neighbor_model.get_neighbors()

        cluster_limit_num = 20
        neighbor_top_n = 100
        users_neighbor_model = UserNeighbor(cluster_limit_num, neighbor_top_n, neighbor_limit)
        users_neighbors = users_neighbor_model.generate_users_neighbors()

        model = BPR(num_factors, args, items_neighbors, users_neighbors, test_data)

        sample_negative_items_empirically = True
        # use a sample strategy without repetition
        sampler = UniformPairWithoutReplacement(sample_negative_items_empirically)

        iter_num = 5
        model.train(train_data, sampler, iter_num)
        my_file.write(str(num_factors) + '\t' + str(model.calculate_auc()))
        my_file.write('\n')
