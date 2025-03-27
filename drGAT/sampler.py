import numpy as np
import pandas as pd
import scipy.sparse as sp
from myutils import mask, to_coo_matrix, to_tensor


class RandomSampler(object):
    def __init__(
        self, adj_mat_original, train_index, test_index, null_mask, A_cg, A_dg, PATH
    ):
        self.adj_mat = to_coo_matrix(adj_mat_original)
        self.train_index = train_index
        self.test_index = test_index
        self.null_mask = null_mask
        self.train_pos = self.sample(train_index)
        self.test_pos = self.sample(test_index)
        self.train_neg, self.test_neg = self.sample_negative()
        self.train_mask = mask(self.train_pos, self.train_neg, dtype=int)
        self.test_mask = mask(self.test_pos, self.test_neg, dtype=bool)
        self.train_data = to_tensor(self.train_pos)
        self.test_data = to_tensor(self.test_pos)
        self.PATH = PATH
        self.A_cg = A_cg
        self.A_dg = A_dg
        self.edge_index, self.edge_attr = self.update_unified_matrix()

    def update_unified_matrix(self):
        A_dc = pd.DataFrame(
            self.adj_mat.toarray(), index=self.A_dg.index, columns=self.A_cg.index
        )

        indexes = list(A_dc.index) + list(self.A_cg.index) + list(self.A_dg.columns)
        n_all = len(indexes)
        base = pd.DataFrame(np.zeros([n_all, n_all]), index=indexes, columns=indexes)
        # 各行列を統合
        base.loc[self.A_cg.index, self.A_cg.columns] = self.A_cg
        base.loc[self.A_cg.columns, self.A_cg.index] = self.A_cg.T
        base.loc[A_dc.index, A_dc.columns] = A_dc
        base.loc[A_dc.columns, A_dc.index] = A_dc.T
        base.loc[self.A_dg.index, self.A_dg.columns] = self.A_dg
        base.loc[self.A_dg.columns, self.A_dg.index] = self.A_dg.T

        matrix = base
        edge_index = np.array(matrix.values.nonzero())
        edge_attr = np.array(matrix.values[matrix.values.nonzero()])
        return edge_index, edge_attr

    def sample(self, index):
        row = self.adj_mat.row
        col = self.adj_mat.col
        data = self.adj_mat.data
        sample_row = row[index]
        sample_col = col[index]
        sample_data = data[index]
        sample = sp.coo_matrix(
            (sample_data, (sample_row, sample_col)), shape=self.adj_mat.shape
        )
        return sample

    def sample_negative(self):
        # identity indicates whether the adjacency matrix is a bipartite graph
        # bipartite graph: whether the two nodes of an edge belong to the same node set
        pos_adj_mat = self.null_mask + self.adj_mat.toarray()
        neg_adj_mat = sp.coo_matrix(np.abs(pos_adj_mat - np.array(1)))
        all_row = neg_adj_mat.row
        all_col = neg_adj_mat.col
        all_data = neg_adj_mat.data
        index = np.arange(all_data.shape[0])

        # Sample negative test set
        test_n = self.test_index.shape[0]
        test_neg_index = np.random.choice(index, test_n, replace=False)
        test_row = all_row[test_neg_index]
        test_col = all_col[test_neg_index]
        test_data = all_data[test_neg_index]
        test = sp.coo_matrix(
            (test_data, (test_row, test_col)), shape=self.adj_mat.shape
        )

        # Sample training set
        train_neg_index = np.delete(index, test_neg_index)
        # train_n = self.train_index.shape[0]
        # train_neg_index = np.random.choice(train_neg_index, train_n, replace=False)
        train_row = all_row[train_neg_index]
        train_col = all_col[train_neg_index]
        train_data = all_data[train_neg_index]
        train = sp.coo_matrix(
            (train_data, (train_row, train_col)), shape=self.adj_mat.shape
        )
        return train, test


# class NewSampler(object):
#     def __init__(self, original_adj_mat, null_mask, target_dim, target_index):
#         super(NewSampler, self).__init__()
#         self.adj_mat = original_adj_mat
#         self.null_mask = null_mask
#         self.dim = target_dim
#         self.target_index = target_index
#         self.train_data, self.test_data = self.sample_train_test_data()
#         self.train_mask, self.test_mask = self.sample_train_test_mask()

#     def sample_target_test_index(self):
#         if self.dim:
#             target_pos_index = np.where(self.adj_mat[:, self.target_index] == 1)[0]
#         else:
#             target_pos_index = np.where(self.adj_mat[self.target_index, :] == 1)[0]
#         return target_pos_index

#     def sample_train_test_data(self):
#         test_data = np.zeros(self.adj_mat.shape, dtype=np.float32)
#         test_index = self.sample_target_test_index()
#         if self.dim:
#             test_data[test_index, self.target_index] = 1
#         else:
#             test_data[self.target_index, test_index] = 1
#         train_data = self.adj_mat - test_data
#         train_data = torch.from_numpy(train_data)
#         test_data = torch.from_numpy(test_data)
#         return train_data, test_data

#     def sample_train_test_mask(self):
#         test_index = self.sample_target_test_index()
#         neg_value = np.ones(self.adj_mat.shape, dtype=np.float32)
#         neg_value = neg_value - self.adj_mat - self.null_mask
#         neg_test_mask = np.zeros(self.adj_mat.shape, dtype=np.float32)
#         if self.dim:
#             target_neg_index = np.where(neg_value[:, self.target_index] == 1)[0]
#             if test_index.shape[0] < target_neg_index.shape[0]:
#                 target_neg_test_index = np.random.choice(
#                     target_neg_index, test_index.shape[0], replace=False
#                 )
#             else:
#                 target_neg_test_index = target_neg_index
#             neg_test_mask[target_neg_test_index, self.target_index] = 1
#             neg_value[:, self.target_index] = 0
#         else:
#             target_neg_index = np.where(neg_value[self.target_index, :] == 1)[0]
#             if test_index.shape[0] < target_neg_index.shape[0]:
#                 target_neg_test_index = np.random.choice(
#                     target_neg_index, test_index.shape[0], replace=False
#                 )
#             else:
#                 target_neg_test_index = target_neg_index
#             neg_test_mask[self.target_index, target_neg_test_index] = 1
#             neg_value[self.target_index, :] = 0
#         train_mask = (self.train_data.numpy() + neg_value).astype(np.bool)
#         test_mask = (self.test_data.numpy() + neg_test_mask).astype(np.bool)
#         train_mask = torch.from_numpy(train_mask)
#         test_mask = torch.from_numpy(test_mask)
#         return train_mask, test_mask
