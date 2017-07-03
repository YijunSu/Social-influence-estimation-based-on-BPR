# coding=utf-8
"""
Bayesian Personalized Ranking

Matrix Factorization model and a variety of classes
implementing different sampling strategies.

Original data is a matrix of <event, user>
After training the data we get event_factors(W) and user_factors(H)
"""

import numpy as np
from math import log, exp
import random
from getdata import DataMode
from real_social_influence_order_finder import get_real_social_influence_order


class BPRArgs(object):

    def __init__(self, learning_rate=0.5,    #0.05
                 bias_regularization=1.0,
                 event_regularization=0.0025,
                 positive_user_regularization=0.0025,
                 negative_user_regularization=0.00025,
                 update_negative_user_factors=True):
        self.learning_rate = learning_rate
        self.bias_regularization = bias_regularization
        self.event_regularization = event_regularization
        self.positive_user_regularization = positive_user_regularization
        self.negative_user_regularization = negative_user_regularization
        self.update_negative_user_factors = update_negative_user_factors


class BPR(object):

    def __init__(self, D, args, test_data):
        """initialise BPR matrix factorization model
        D: number of factors(10)
        args: parameter table
        """
        self.D = D
        self.learning_rate = args.learning_rate
        self.bias_regularization = args.bias_regularization
        self.event_regularization = args.event_regularization
        self.positive_user_regularization = args.positive_user_regularization
        self.negative_user_regularization = args.negative_user_regularization
        self.update_negative_user_factors = args.update_negative_user_factors
        self.iter_num_x = []
        self.test_data = test_data
        self.data = None
        self.num_users = None
        self.num_events = None
        self.event_factors = None
        self.user_bias = None
        self.user_factors = None
        self.loss_samples = None

    def train(self, data, sampler, num_iters):
        """train model
        data: event-user matrix as a scipy sparse matrix
              events and users are zero-indexed
        """
        self.init(data)
        print('initial loss = {0}'.format(self.loss()))

        # TODO: Update factors(W and H) and show negative BPR-OPT during each iteration
        # Show negative BPR-OPT to make sure that we are minimizing it
        for it in range(num_iters):                      #num_iters is 500,in main
            self.iter_num_x.append(it)
            print('starting iteration {0}'.format(it))
            for e, i, j in sampler.generate_samples(self.data):
                self.update_factors(e, i, j)
            print('iteration {0}: loss = {1}'.format(it, self.loss()))

        with open('event_factor_file_20_factors.txt', 'w+') as event_factor_file:
            for e_factor in self.event_factors:
                for i in range(self.D):
                    event_factor_file.write(str(e_factor[i]) + ' ')
                event_factor_file.write('\n')
        event_factor_file.close()

        with open('user_factor_file_20_factors.txt', 'w+') as user_factor_file:
            for user_factor in self.user_factors:
                for i in range(self.D):
                    user_factor_file.write(str(user_factor[i]) + ' ')
                user_factor_file.write('\n')
        user_factor_file.close()

        with open('bias_file_20_factors.txt', 'w+') as bias_file:
            for bias in self.user_bias:
                bias_file.write(str(bias) + '\n')
        bias_file.close()

    def init(self, data):
        self.data = data
        # data is csr matrix

        self.num_events, self.num_users = self.data.shape
        self.user_bias = np.zeros(self.num_users)
        # item_bias is an one-dimension array, its length = num_items, all elements are initialized as 0.

        # TODO: initialise the event factor matrix W with random values between 0 and 1
        self.event_factors = np.random.randn(self.num_events, self.D) * 0.5
        # user_factors is self.num_users-by-self.D array of random numbers from [0.0, 1.0)
        # it equals the W(factor matrix of events) in the paper

        # TODO: initialise the user factor matrix H with random values between 0 and 1
        self.user_factors = np.random.randn(self.num_users, self.D) * 0.5
        # item_factors is self.num_items-by-self.D array of random numbers from
        # it equals the H(factor matrix of users) in the paper
        self.create_loss_samples()

    def create_loss_samples(self):
        # apply rule of thumb to decide num samples over which to compute loss
        num_loss_samples = int(100*self.num_events**0.5)

        # TODO: sample num_loss_samples(the number of triples) triples with the first sampler
        print('sampling {0} <event, user i, user j> triples...'.format(num_loss_samples))
        # construct a sampler object(sampler)
        sampler = UniformEventUniformUser(True)

        """TODO:compare num_loss_samples with data.nnz in num_samples
        The number of samples is the smaller one of num_loss_samples and data.nnz
        The sampler sample some triples from DS
        generate a list of triples(u,i,j) called loss_samples
        loss_samples is some part of DS
        """
        self.loss_samples = [t for t in sampler.generate_samples(self.data, num_loss_samples)]

    def update_factors(self, e, i, j, update_e=True, update_i=True):
        """apply SGD update"""

        update_j = self.update_negative_user_factors
        # item_bias is initialised zeros
        x = self.user_bias[i] - self.user_bias[j] \
            + np.dot(self.event_factors[e, :], self.user_factors[i, :]-self.user_factors[j, :])
        temp_e_factors = self.event_factors[e]

        z = 1.0 / (1.0 + exp(x))
        if update_i:
            d = -self.bias_regularization * self.user_bias[i] + z
            self.user_bias[i] += self.learning_rate * d
        if update_j:
            d = -self.bias_regularization * self.user_bias[j] - z
            self.user_bias[j] += self.learning_rate * d
        if update_e:
            d = -self.event_regularization * self.event_factors[e, :] + \
                (self.user_factors[i, :]-self.user_factors[j, :])*z
            self.event_factors[e, :] += self.learning_rate*d
        if update_i:
            d = -self.positive_user_regularization * self.user_factors[i, :] + temp_e_factors * z
            self.user_factors[i, :] += self.learning_rate*d
        if update_j:
            d = -self.negative_user_regularization*self.user_factors[j, :] - temp_e_factors * z
            self.user_factors[j, :] += self.learning_rate*d

    def loss(self):
        ranking_loss = 0
        for e, i, j in self.loss_samples:
            x = self.predict(e, i) - self.predict(e, j)
            ranking_loss += -log(1.0 / (1.0 + exp(-x)))
        complexity = 0
        for e, i, j in self.loss_samples:
            complexity += self.event_regularization * np.dot(self.event_factors[e], self.event_factors[e])
            complexity += self.positive_user_regularization * np.dot(self.user_factors[i], self.user_factors[i])
            complexity += self.negative_user_regularization * np.dot(self.user_factors[j], self.user_factors[j])
            complexity += self.bias_regularization * self.user_bias[i]**2
            complexity += self.bias_regularization * self.user_bias[j]**2
        return ranking_loss + complexity
        # return negative BPR-OPT

    def predict(self, e, i):
        return self.user_bias[i] + np.dot(self.event_factors[e], self.user_factors[i])

    def calculate_auc(self):
        sum2 = 0
        u_sum = 0
        for piece in self.test_data:
            event_id = piece[0]
            user_id = piece[1]
            index = self.data[event_id].indices
            positive_user_list = list(index)
            positive_user_list.append(user_id)
            negative_user_list = list(set(range(self.num_users)) - set(positive_user_list))
            x_u_i = np.dot(self.event_factors[event_id], self.user_factors[user_id]) + self.user_bias[user_id]    #test_data only one
            for j in negative_user_list:  
                x_u_j = np.dot(self.event_factors[event_id], self.user_factors[j]) + self.user_bias[j]
                if x_u_i > x_u_j:
                    u_sum += 1.0
            sum2+=len(negative_user_list)
        auc = u_sum/sum2

        return auc


    def calculate_map(self):
        event_sum = 0
        real_order = get_real_social_influence_order()
        for piece in self.test_data:
            event_id = piece[0]
            user_list = range(self.num_users)

            user_influence_dict = {}
            for user in user_list:
                user_influence_dict[user] = np.dot(self.event_factors[event_id], self.user_factors[user]) \
                                            + self.user_bias[user]
            sorted_users_by_influence = sorted(user_influence_dict.items(), key=lambda d: d[1], reverse=True)
            sorted_user_list = []
            for i in sorted_users_by_influence:
                sorted_user_list.append(i[0])

            # calculate p@n
            limit = 10
            if len(real_order[event_id]) < 10:
                limit = len(real_order[event_id])

            predict_n = 0
            valid_n = 0
            for i in range(limit):
                counter = -1
                for j in range(limit):
                    if real_order[event_id][i] == sorted_user_list[j]:
                        counter = j
                        break
                if counter != -1:
                    valid_n += 1.0
                    predict_n += 1.0 / (1.0 + counter)
            if valid_n != 0:
                predict_n /= valid_n
            event_sum += predict_n
        map_mark = event_sum / len(self.test_data)
        return map_mark


# sampling strategies
class Sampler(object):

    def __init__(self, sample_negative_users_empirically):
        self.sample_negative_users_empirically = sample_negative_users_empirically

    def init(self, data, max_samples=None):
        self.data = data
        self.num_events, self.num_users = data.shape
        self.max_samples = max_samples

    def sample_event(self):
        e = self.uniform_event()
        num_users = self.data[e].getnnz()
        assert(num_users > 0 and num_users != self.num_users)
        return e

    def sample_negative_user(self, event_users):
        j = self.random_user()
        while j in event_users:
            j = self.random_user()
        return j

    def uniform_event(self):
        return random.randint(0, self.num_events-1)

    def random_user(self):
        """sample an item uniformly or from the empirical distribution
           observed in the training data
        """
        if self.sample_negative_users_empirically:         
            # just pick something someone rated!
            e = self.uniform_event()
            i = random.choice(self.data[e].indices)
        else:                                            
            i = random.randint(0, self.num_users - 1)
        return i

    def num_samples(self, n):
        if self.max_samples is None:
            return n
        return min(n, self.max_samples)


# TODO:sample events and users separately
class UniformEventUniformUser(Sampler):
    def generate_samples(self, data, max_samples=None):
        self.init(data, max_samples)
        """nnz is the number of nonzero values(all of the values are 1 in our data) in sparse matrix
        and in num_samples() we compare data.nnz with num_loss_samples to identify num_samples
        """
        for _ in range(self.num_samples(self.data.nnz)):
            # TODO: choose a user randomly
            e = self.uniform_event()
            # sample positive user
            i = random.choice(self.data[e].indices)
            # TODO: sample negative item
            j = self.sample_negative_user(self.data[e].indices)
            yield e, i, j


# idxs make it sure that there is not any repeating
class UniformPairWithoutReplacement(Sampler):
    def generate_samples(self, data, max_samples=None):
        self.init(data, max_samples)
        self.events, self.users = self.data.nonzero()
        idxs = list(range(self.events.size))
        # idxs is a list from 0 to data.nnz-1
        # TODO: make idxs(a list) out of order
        random.shuffle(idxs)

        self.events = self.events[idxs]
        self.users = self.users[idxs]
        self.idx = 0
        for _ in range(self.num_samples(self.events.size)):
            e = self.events[self.idx]
            i = self.users[self.idx]
            j = self.sample_negative_user(self.data[e].indices)
            self.idx += 1
            yield e, i, j

if __name__ == '__main__':
    d = DataMode()
    d.find_data()
    csr_data = d.find_train_data()
    test_data = d.find_test_data()

    args = BPRArgs()
    args.learning_rate = 0.01

    num_factor = 10
    model = BPR(num_factor, args, test_data)
    sample_negative_users_empirically = False
    sampler = UniformPairWithoutReplacement(sample_negative_users_empirically)

    iter_num = 5000
    model.train(csr_data, sampler, iter_num)
    with open('auc.txt', 'w+') as f:
        f.write(str(model.calculate_auc()) + '\n')

    with open('map.txt', 'w') as f:
        f.write(str(model.calculate_map()) + '\n')
