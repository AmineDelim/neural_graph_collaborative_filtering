'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang
'''

import numpy as np
import random as rd
import scipy.sparse as sp
from time import time


class Data(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'

        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}
        self.exist_users = []

        # ----------- READ TRAIN FILE SAFELY -----------
        with open(train_file) as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) < 2:
                    continue

                try:
                    uid = int(parts[0])
                    items = [int(i) for i in parts[1:]]
                except:
                    continue

                if not items:
                    continue

                self.exist_users.append(uid)
                self.n_items = max(self.n_items, max(items))
                self.n_users = max(self.n_users, uid)
                self.n_train += len(items)

        # ----------- READ TEST FILE SAFELY -----------
        with open(test_file) as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) < 2:
                    continue

                try:
                    items = [int(i) for i in parts[1:]]
                except:
                    continue

                if not items:
                    continue

                self.n_items = max(self.n_items, max(items))
                self.n_test += len(items)

        self.n_items += 1
        self.n_users += 1

        self.print_statistics()

        self.R = sp.dok_matrix(
            (self.n_users, self.n_items),
            dtype=np.float32
        )

        self.train_items = {}
        self.test_set = {}

        # ----------- LOAD TRAIN/TEST MATRICES -----------
        with open(train_file) as f_train:
            for line in f_train:
                parts = line.strip().split()

                if len(parts) < 2:
                    continue

                try:
                    items = [int(i) for i in parts]
                except:
                    continue

                uid = items[0]
                train_items = items[1:]

                if not train_items:
                    continue

                for i in train_items:
                    self.R[uid, i] = 1.

                self.train_items[uid] = train_items

        with open(test_file) as f_test:
            for line in f_test:
                parts = line.strip().split()

                if len(parts) < 2:
                    continue

                try:
                    items = [int(i) for i in parts]
                except:
                    continue

                uid = items[0]
                test_items = items[1:]

                if not test_items:
                    continue

                self.test_set[uid] = test_items

    # ----------- ADJ MATRIX -----------

    def get_adj_mat(self):
        try:
            t1 = time()

            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')

            print('already load adj matrix', adj_mat.shape, time() - t1)

        except:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()

            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)

        return adj_mat, norm_adj_mat, mean_adj_mat

    def create_adj_mat(self):
        t1 = time()

        adj_mat = sp.dok_matrix(
            (self.n_users + self.n_items,
             self.n_users + self.n_items),
            dtype=np.float32
        )

        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        print('already create adjacency matrix',
              adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.

            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(adj)

            print('generate single-normalized adjacency matrix.')

            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(
            adj_mat + sp.eye(adj_mat.shape[0])
        )

        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix',
              time() - t2)

        return (
            adj_mat.tocsr(),
            norm_adj_mat.tocsr(),
            mean_adj_mat.tocsr()
        )

    # ----------- NEGATIVE POOL -----------

    def negative_pool(self):
        t1 = time()

        for u in self.train_items.keys():
            neg_items = list(
                set(range(self.n_items))
                - set(self.train_items[u])
            )

            if not neg_items:
                continue

            pools = [rd.choice(neg_items) for _ in range(100)]
            self.neg_pools[u] = pools

        print('refresh negative pools', time() - t1)

    # ----------- SAMPLE -----------

    def sample(self):

        if self.batch_size <= len(self.exist_users):
            users = rd.sample(
                self.exist_users,
                self.batch_size
            )
        else:
            users = [
                rd.choice(self.exist_users)
                for _ in range(self.batch_size)
            ]

        pos_items = []
        neg_items = []

        for u in users:

            pos_list = self.train_items.get(u, [])

            if not pos_list:
                continue

            pos_i = rd.choice(pos_list)

            while True:
                neg_i = np.random.randint(
                    low=0,
                    high=self.n_items
                )

                if neg_i not in pos_list:
                    break

            pos_items.append(pos_i)
            neg_items.append(neg_i)

        return users, pos_items, neg_items

    # ----------- INFO -----------

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' %
              (self.n_users, self.n_items))

        print('n_interactions=%d' %
              (self.n_train + self.n_test))

        print(
            'n_train=%d, n_test=%d, sparsity=%.5f'
            % (
                self.n_train,
                self.n_test,
                (self.n_train + self.n_test)
                / (self.n_users * self.n_items)
            )
        )

    # ----------- SPARSITY -----------

    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []

            lines = open(
                self.path + '/sparsity.split'
            ).readlines()

            for idx, line in enumerate(lines):

                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append(
                        [int(uid)
                         for uid in line.strip().split()]
                    )

            print('get sparsity split.')

        except:
            split_uids, split_state = \
                self.create_sparsity_split()

            with open(
                self.path + '/sparsity.split', 'w'
            ) as f:

                for idx in range(len(split_state)):
                    f.write(split_state[idx] + '\n')
                    f.write(
                        ' '.join(
                            [str(uid)
                             for uid in split_uids[idx]]
                        ) + '\n'
                    )

            print('create sparsity split.')

        return split_uids, split_state

    def create_sparsity_split(self):

        all_users_to_test = list(
            self.test_set.keys()
        )

        user_n_iid = {}

        for uid in all_users_to_test:

            train_iids = self.train_items.get(uid, [])
            test_iids = self.test_set.get(uid, [])

            n_iids = len(train_iids) + len(test_iids)

            user_n_iid.setdefault(
                n_iids, []
            ).append(uid)

        split_uids = []
        temp = []

        n_rates = 0

        split_state = []

        for n_iids in sorted(user_n_iid):

            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])

            if n_rates >= 0.25 * (
                    self.n_train + self.n_test):

                split_uids.append(temp)

                state = (
                    '#inter per user<=[%d], '
                    '#users=[%d], #all rates=[%d]'
                    % (n_iids, len(temp), n_rates)
                )

                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0

        if temp:
            split_uids.append(temp)

        return split_uids, split_state
