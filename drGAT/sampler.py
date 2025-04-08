# type: ignore
# ruff: noqa

import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from .myutils import mask, to_coo_matrix, to_tensor


class RandomSampler(object):
    def __init__(
        self,
        adj_mat_original,
        train_index,
        test_index,
        null_mask,
        S_d,
        S_c,
        S_g,
        A_cg,
        A_dg,
        PATH,
        seed,
    ):
        # Initialize basic attributes
        self.adj_mat_original = adj_mat_original
        self.adj_mat = to_coo_matrix(adj_mat_original)
        self.train_index = train_index
        self.test_index = test_index
        self.null_mask = null_mask
        self.PATH = PATH
        self.seed = seed

        # Initialize similarity matrices
        self.S_d = S_d
        self.S_c = S_c
        self.S_g = S_g

        # Initialize adjacency matrices
        self.A_cg = A_cg
        self.A_dg = A_dg

        # Sample positive and negative examples
        self.train_pos = self.sample(train_index)
        self.test_pos = self.sample(test_index)
        self.train_neg, self.test_neg = self.sample_negative()

        # Create masks and tensors
        self.train_mask = mask(self.train_pos, self.train_neg, dtype=int)
        self.test_mask = mask(self.test_pos, self.test_neg, dtype=bool)
        self.train_data = to_tensor(self.train_pos)
        self.test_data = to_tensor(self.test_pos)

        # Create unified graph representation
        self.edge_index, self.edge_attr = self.update_unified_matrix()
        self.train_labels = self.get_train_labels(is_train=True)
        self.test_labels = self.get_train_labels(is_train=False)

    def get_train_labels(self, is_train=False):
        """Get labels for training or testing data"""
        # Get indices based on mask
        masked_indices = (
            self.train_mask.numpy().nonzero()
            if is_train
            else self.test_mask.numpy().nonzero()
        )

        # Extract labels from original adjacency matrix
        row_labels = [self.adj_mat_original.index[i] for i in masked_indices[0]]
        column_labels = [self.adj_mat_original.columns[j] for j in masked_indices[1]]
        values = [
            self.adj_mat_original.iloc[i, j]
            for i, j in zip(masked_indices[0], masked_indices[1])
        ]

        # Convert indices using mapping
        conv = dict(
            pd.DataFrame(np.load(self.PATH + "idxs.npy", allow_pickle=True))
            .T[[1, 0]]
            .values
        )
        row_labels = [conv[i] for i in row_labels]
        column_labels = [conv[i] for i in column_labels]

        return pd.DataFrame(
            {"Drug": row_labels, "Cell": column_labels, "Label": values}
        )

    def update_unified_matrix(self):
        """Create unified adjacency matrix combining drug-cell, cell-gene and drug-gene interactions"""
        # Create drug-cell adjacency matrix
        A_dc = pd.DataFrame(
            self.adj_mat.toarray(), index=self.A_dg.index, columns=self.A_cg.index
        )

        # Initialize unified matrix
        indexes = list(A_dc.index) + list(self.A_cg.index) + list(self.A_dg.columns)
        n_all = len(indexes)
        base = pd.DataFrame(np.zeros([n_all, n_all]), index=indexes, columns=indexes)

        # Fill unified matrix with interactions
        base.loc[self.A_cg.index, self.A_cg.columns] = self.A_cg
        base.loc[self.A_cg.columns, self.A_cg.index] = self.A_cg.T
        base.loc[A_dc.index, A_dc.columns] = A_dc
        base.loc[A_dc.columns, A_dc.index] = A_dc.T
        base.loc[self.A_dg.index, self.A_dg.columns] = self.A_dg
        base.loc[self.A_dg.columns, self.A_dg.index] = self.A_dg.T

        # Save index mapping
        idxs_path = os.path.join(self.PATH, "idxs.npy")
        if not os.path.exists(idxs_path):
            idxs = np.array([np.arange(len(base.index)), base.index])
            np.save(idxs_path, idxs)

        # Convert to PyTorch tensors
        edge_index = torch.tensor(np.array(base.values.nonzero())).type(torch.int64)
        edge_attr = torch.tensor(np.array(base.values[base.values.nonzero()]))

        return edge_index, edge_attr

    def sample(self, index):
        """Sample positive examples from adjacency matrix"""
        row = self.adj_mat.row
        col = self.adj_mat.col
        data = self.adj_mat.data

        sample_row = row[index]
        sample_col = col[index]
        sample_data = data[index]

        return sp.coo_matrix(
            (sample_data, (sample_row, sample_col)), shape=self.adj_mat.shape
        )

    def sample_negative(self):
        """Sample negative examples for training and testing"""
        # Create negative adjacency matrix
        pos_adj_mat = self.null_mask + self.adj_mat.toarray()
        neg_adj_mat = sp.coo_matrix(np.abs(pos_adj_mat - np.array(1)))

        all_row = neg_adj_mat.row
        all_col = neg_adj_mat.col
        all_data = neg_adj_mat.data
        index = np.arange(all_data.shape[0])

        # Sample negative test examples
        test_n = self.test_index.shape[0]
        np.random.seed(self.seed)
        test_neg_index = np.random.choice(index, test_n, replace=False)
        test = sp.coo_matrix(
            (
                all_data[test_neg_index],
                (all_row[test_neg_index], all_col[test_neg_index]),
            ),
            shape=self.adj_mat.shape,
        )

        # Sample negative training examples
        train_neg_index = np.delete(index, test_neg_index)
        train = sp.coo_matrix(
            (
                all_data[train_neg_index],
                (all_row[train_neg_index], all_col[train_neg_index]),
            ),
            shape=self.adj_mat.shape,
        )

        return train, test


class NewSampler(object):
    def __init__(
        self,
        original_adj_mat,
        null_mask,
        target_dim,
        target_index,
        S_d,
        S_c,
        S_g,
        A_cg,
        A_dg,
        PATH,
        seed,
    ):
        super().__init__()
        self.seed = seed
        self.set_seed()
        self.adj_mat_original = original_adj_mat
        self.adj_mat = original_adj_mat.values
        self.null_mask = null_mask
        self.dim = target_dim
        self.target_index = target_index
        self.train_data, self.test_data = self.sample_train_test_data()
        self.train_mask, self.test_mask = self.sample_train_test_mask()

        self.PATH = PATH

        # Initialize similarity matrices
        self.S_d = S_d
        self.S_c = S_c
        self.S_g = S_g

        # Initialize adjacency matrices
        self.A_cg = A_cg
        self.A_dg = A_dg

        self.train_labels = self.get_train_labels(is_train=True)
        self.test_labels = self.get_train_labels(is_train=False)

        # # Create unified graph representation
        self.edge_index, self.edge_attr = self.update_unified_matrix()

    def set_seed(self):
        np.random.seed(self.seed)  # NumPyのシードを設定
        torch.manual_seed(self.seed)  # PyTorchのシードを設定

    def update_unified_matrix(self):
        """Create unified adjacency matrix combining drug-cell, cell-gene and drug-gene interactions"""
        # Create drug-cell adjacency matrix
        A_dc = pd.DataFrame(
            self.adj_mat.T, columns=self.A_cg.index, index=self.A_dg.index
        )

        # Initialize unified matrix
        indexes = list(A_dc.columns) + list(self.A_cg.columns) + list(self.A_dg.index)
        n_all = len(indexes)
        base = pd.DataFrame(np.zeros([n_all, n_all]), index=indexes, columns=indexes)

        # Fill unified matrix with interactions
        base.loc[self.A_cg.index, self.A_cg.columns] = self.A_cg
        base.loc[self.A_cg.columns, self.A_cg.index] = self.A_cg.T
        base.loc[A_dc.index, A_dc.columns] = A_dc
        base.loc[A_dc.columns, A_dc.index] = A_dc.T
        base.loc[self.A_dg.index, self.A_dg.columns] = self.A_dg
        base.loc[self.A_dg.columns, self.A_dg.index] = self.A_dg.T

        # Save index mapping
        idxs_path = os.path.join(self.PATH, "idxs.npy")
        if not os.path.exists(idxs_path):
            idxs = np.array([np.arange(len(base.index)), base.index])
            np.save(idxs_path, idxs)

        # Convert to PyTorch tensors
        edge_index = torch.tensor(np.array(base.values.nonzero())).type(torch.int64)
        edge_attr = torch.tensor(np.array(base.values[base.values.nonzero()]))

        return edge_index, edge_attr

    def get_train_labels(self, is_train=False):
        """Get labels for training or testing data"""
        # Get indices based on mask
        masked_indices = (
            self.train_mask.numpy().nonzero()
            if is_train
            else self.test_mask.numpy().nonzero()
        )

        # Extract labels from original adjacency matrix
        row_labels = [self.adj_mat_original.index[i] for i in masked_indices[0]]
        column_labels = [self.adj_mat_original.columns[j] for j in masked_indices[1]]
        values = [
            self.adj_mat_original.iloc[i, j]
            for i, j in zip(masked_indices[0], masked_indices[1])
        ]

        # Convert indices using mapping
        conv = dict(
            pd.DataFrame(np.load(self.PATH + "idxs.npy", allow_pickle=True))
            .T[[1, 0]]
            .values
        )
        row_labels = [conv[i] for i in row_labels]
        column_labels = [conv[i] for i in column_labels]

        return pd.DataFrame(
            {"Drug": row_labels, "Cell": column_labels, "Label": values}
        )

    def sample_target_test_index(self):
        if self.dim:
            target_pos_index = np.where(self.adj_mat[:, self.target_index] == 1)[0]
        else:
            target_pos_index = np.where(self.adj_mat[self.target_index, :] == 1)[0]
        return target_pos_index

    def sample_train_test_data(self):
        test_data = np.zeros(self.adj_mat.shape, dtype=np.float32)
        test_index = self.sample_target_test_index()
        if self.dim:
            test_data[test_index, self.target_index] = 1
        else:
            test_data[self.target_index, test_index] = 1
        train_data = self.adj_mat - test_data
        train_data = torch.from_numpy(train_data)
        test_data = torch.from_numpy(test_data)
        return train_data, test_data

    def sample_train_test_mask(self):
        test_index = self.sample_target_test_index()
        neg_value = np.ones(self.adj_mat.shape, dtype=np.float32)
        neg_value = neg_value - self.adj_mat - self.null_mask
        neg_test_mask = np.zeros(self.adj_mat.shape, dtype=np.float32)
        if self.dim:
            target_neg_index = np.where(neg_value[:, self.target_index] == 1)[0]
            if test_index.shape[0] < target_neg_index.shape[0]:
                target_neg_test_index = np.random.choice(
                    target_neg_index, test_index.shape[0], replace=False
                )
            else:
                target_neg_test_index = target_neg_index
            neg_test_mask[target_neg_test_index, self.target_index] = 1
            neg_value[:, self.target_index] = 0
        else:
            target_neg_index = np.where(neg_value[self.target_index, :] == 1)[0]
            if test_index.shape[0] < target_neg_index.shape[0]:
                target_neg_test_index = np.random.choice(
                    target_neg_index, test_index.shape[0], replace=False
                )
            else:
                target_neg_test_index = target_neg_index
            neg_test_mask[self.target_index, target_neg_test_index] = 1
            neg_value[self.target_index, :] = 0
        train_mask = (self.train_data.numpy() + neg_value).astype(bool)
        test_mask = (self.test_data.numpy() + neg_test_mask).astype(bool)
        train_mask = torch.from_numpy(train_mask)
        test_mask = torch.from_numpy(test_mask)
        return train_mask, test_mask
