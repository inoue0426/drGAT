# type: ignore
# ruff: noqa

import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from .myutils import mask, to_coo_matrix


class BalancedSampler:
    def __init__(
        self,
        adj_mat_original,
        all_edges,
        all_labels,
        train_index,
        test_index,
        null_mask,
        S_d,
        S_c,
        S_g,
        A_cg,
        A_dg,
        PATH=None,  # ← デフォルトは None
    ):
        self.PATH = PATH
        self.adj_mat_original = adj_mat_original
        self.null_mask = null_mask
        self.S_d, self.S_c, self.S_g = S_d, S_c, S_g
        self.A_cg, self.A_dg = A_cg, A_dg

        # インデックスとラベルの対応表
        self.row_index = list(adj_mat_original.index)
        self.col_index = list(adj_mat_original.columns)
        self.row_map = {i: name for i, name in enumerate(self.row_index)}
        self.col_map = {i: name for i, name in enumerate(self.col_index)}

        # 残りの処理はそのまま
        self.all_edges = all_edges
        self.all_labels = all_labels

        self.train_edges = all_edges[train_index]
        self.test_edges = all_edges[test_index]
        self.train_labels = all_labels[train_index]
        self.test_labels = all_labels[test_index]

        self.train_pos, self.train_neg = self._create_coo_matrices(
            self.train_edges, self.train_labels
        )
        self.test_pos, self.test_neg = self._create_coo_matrices(
            self.test_edges, self.test_labels
        )

        self.train_data = self._build_coo(self.train_edges, self.train_labels)
        self.test_data = self._build_coo(self.test_edges, self.test_labels)

        self.train_mask = mask(self.train_pos, self.train_neg, dtype=int)
        self.test_mask = mask(self.test_pos, self.test_neg, dtype=bool)

        self.edge_index, self.edge_attr, self.index = self._update_unified_matrix()

        self.train_labels_df = self._get_labels(self.train_edges, self.train_labels)
        self.test_labels_df = self._get_labels(self.test_edges, self.test_labels)

    def _get_labels(self, edges, labels):
        return pd.DataFrame({"Drug": edges[:, 0], "Cell": edges[:, 1], "Label": labels})

    def _edge_list_to_coo(self, edge_list):
        data = np.ones(edge_list.shape[0])
        return sp.coo_matrix(
            (data, (edge_list[:, 0], edge_list[:, 1])),
            shape=self.adj_mat_original.shape,
        )

    def _create_coo_matrices(self, edges, labels):
        pos_edges = edges[labels == 1]
        neg_edges = edges[labels == 0]
        return self._edge_list_to_coo(pos_edges), self._edge_list_to_coo(neg_edges)

    def _build_coo(self, edges, labels):
        return sp.coo_matrix(
            (labels, (edges[:, 0], edges[:, 1])), shape=self.null_mask.shape
        )

    def _update_unified_matrix(self):
        A_dc = pd.DataFrame(
            self.train_data.toarray().T,
            index=self.A_dg.index,
            columns=self.A_cg.index,
        )
        indexes = list(A_dc.index) + list(self.A_cg.index) + list(self.A_dg.columns)
        base = pd.DataFrame(
            np.zeros([len(indexes), len(indexes)]), index=indexes, columns=indexes
        )

        for df, transpose in [(self.A_cg, True), (A_dc, True), (self.A_dg, True)]:
            base.loc[df.index, df.columns] = df
            if transpose:
                base.loc[df.columns, df.index] = df.T

        edge_index = torch.tensor(np.array(base.values.nonzero())).type(torch.int64)
        edge_attr = torch.tensor(base.values[base.values.nonzero()])

        return edge_index, edge_attr, base.index

    def get_label_df(self, edges, labels):
        drugs = [self.row_map[i] for i in edges[:, 0]]
        cells = [self.col_map[j] for j in edges[:, 1]]
        return pd.DataFrame({"Drug": drugs, "Cell": cells, "Label": labels})

    def index_to_label(self, row_idx, col_idx):
        return self.row_map.get(row_idx, f"row{row_idx}"), self.col_map.get(
            col_idx, f"col{col_idx}"
        )


# class RandomSampler(object):
#     def __init__(
#         self,
#         adj_mat_original,
#         train_index,
#         test_index,
#         null_mask,
#         S_d,
#         S_c,
#         S_g,
#         A_cg,
#         A_dg,
#         PATH,
#         seed,
#     ):
#         # Initialize basic attributes
#         self.seed = seed
#         self.set_seed()

#         self.adj_mat_original = adj_mat_original
#         self.adj_mat = to_coo_matrix(adj_mat_original)
#         self.train_index = train_index
#         self.test_index = test_index
#         self.null_mask = null_mask
#         self.PATH = PATH

#         self.train_pos = self.sample(train_index)
#         self.test_pos = self.sample(test_index)
#         self.train_neg, self.test_neg = self.sample_negative()
#         self.train_mask = mask(self.train_pos, self.train_neg, dtype=int)
#         self.test_mask = mask(self.test_pos, self.test_neg, dtype=bool)
#         self.train_data = to_tensor(self.train_pos)
#         self.test_data = to_tensor(self.test_pos)

#         # Initialize similarity matrices
#         self.S_d = S_d
#         self.S_c = S_c
#         self.S_g = S_g

#         # Initialize adjacency matrices
#         self.A_cg = A_cg
#         self.A_dg = A_dg

#         # Create unified graph representation
#         self.edge_index, self.edge_attr = self.update_unified_matrix()
#         self.train_labels = self.get_labels(is_train=True)
#         self.test_labels = self.get_labels(is_train=False)

#     def set_seed(self):
#         np.random.seed(self.seed)  # NumPyのシードを設定
#         torch.manual_seed(self.seed)  # PyTorchのシードを設定

#     def get_train_labels(self, is_train=False):
#         """Get labels for training or testing data"""
#         # Get indices based on mask
#         masked_indices = (
#             self.train_mask.numpy().nonzero()
#             if is_train
#             else self.test_mask.numpy().nonzero()
#         )

#         # Extract labels from original adjacency matrix
#         row_labels = [self.adj_mat_original.index[i] for i in masked_indices[0]]
#         column_labels = [self.adj_mat_original.columns[j] for j in masked_indices[1]]
#         values = [
#             self.adj_mat_original.iloc[i, j]
#             for i, j in zip(masked_indices[0], masked_indices[1])
#         ]

#         # Convert indices using mapping
#         conv = dict(
#             pd.DataFrame(np.load(self.PATH + "idxs.npy", allow_pickle=True))
#             .T[[1, 0]]
#             .values
#         )
#         row_labels = [conv[i] for i in row_labels]
#         column_labels = [conv[i] for i in column_labels]

#         return pd.DataFrame(
#             {"Drug": row_labels, "Cell": column_labels, "Label": values}
#         )

#     def update_unified_matrix(self):
#         """Create unified adjacency matrix combining drug-cell, cell-gene and drug-gene interactions"""
#         # Create drug-cell adjacency matrix
#         A_dc = pd.DataFrame(
#             self.adj_mat.toarray().T, index=self.A_dg.index, columns=self.A_cg.index
#         )

#         # Initialize unified matrix
#         indexes = list(A_dc.index) + list(self.A_cg.index) + list(self.A_dg.columns)
#         n_all = len(indexes)
#         base = pd.DataFrame(np.zeros([n_all, n_all]), index=indexes, columns=indexes)

#         # Fill unified matrix with interactions
#         base.loc[self.A_cg.index, self.A_cg.columns] = self.A_cg
#         base.loc[self.A_cg.columns, self.A_cg.index] = self.A_cg.T
#         base.loc[A_dc.index, A_dc.columns] = A_dc
#         base.loc[A_dc.columns, A_dc.index] = A_dc.T
#         base.loc[self.A_dg.index, self.A_dg.columns] = self.A_dg
#         base.loc[self.A_dg.columns, self.A_dg.index] = self.A_dg.T

#         # Save index mapping
#         idxs_path = os.path.join(self.PATH, "idxs.npy")
#         if not os.path.exists(idxs_path):
#             idxs = np.array([np.arange(len(base.index)), base.index])
#             np.save(idxs_path, idxs)

#         # Convert to PyTorch tensors
#         edge_index = torch.tensor(np.array(base.values.nonzero())).type(torch.int64)
#         edge_attr = torch.tensor(np.array(base.values[base.values.nonzero()]))

#         return edge_index, edge_attr

#     def sample(self, index):
#         """Sample positive examples from adjacency matrix"""
#         row = self.adj_mat.row
#         col = self.adj_mat.col
#         data = self.adj_mat.data

#         sample_row = row[index]
#         sample_col = col[index]
#         sample_data = data[index]

#         return sp.coo_matrix(
#             (sample_data, (sample_row, sample_col)), shape=self.adj_mat.shape
#         )

#     def sample_negative(self):
#         """Sample negative examples for training and testing"""
#         # Create negative adjacency matrix
#         pos_adj_mat = self.null_mask + self.adj_mat.toarray()
#         neg_adj_mat = sp.coo_matrix(np.abs(pos_adj_mat - np.array(1)))

#         all_row = neg_adj_mat.row
#         all_col = neg_adj_mat.col
#         all_data = neg_adj_mat.data
#         index = np.arange(all_data.shape[0])

#         # Sample negative test examples
#         test_n = self.test_index.shape[0]
#         test_neg_index = np.random.choice(index, test_n, replace=False)
#         test = sp.coo_matrix(
#             (
#                 all_data[test_neg_index],
#                 (all_row[test_neg_index], all_col[test_neg_index]),
#             ),
#             shape=self.adj_mat.shape,
#         )

#         # Sample negative training examples
#         train_neg_index = np.delete(index, test_neg_index)
#         train = sp.coo_matrix(
#             (
#                 all_data[train_neg_index],
#                 (all_row[train_neg_index], all_col[train_neg_index]),
#             ),
#             shape=self.adj_mat.shape,
#         )

#         return train, test


class NewSampler:
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
        PATH=None,
    ):
        self.adj_mat_original = original_adj_mat
        self.adj_mat = original_adj_mat.values  # ここで確実に NumPy に変換
        self.null_mask = (
            null_mask.values if isinstance(null_mask, pd.DataFrame) else null_mask
        )
        self.dim = target_dim  # 0 = row (Cell), 1 = col (Drug)
        self.target_index = target_index
        self.PATH = PATH

        self.S_d, self.S_c, self.S_g = S_d, S_c, S_g
        self.A_cg, self.A_dg = A_cg, A_dg

        self.train_data, self.test_data = self._sample_train_test_data()
        self.train_mask, self.test_mask = self._sample_train_test_mask()

        self._save_index_mapping()

        self.train_labels = self._get_labels(is_train=True)
        self.test_labels = self._get_labels(is_train=False)

        self.train_labels_df = self._get_label_df(self.train_data, self.train_mask)
        self.test_labels_df = self._get_label_df(self.test_data, self.test_mask)

        self.edge_index, self.edge_attr = self._update_unified_matrix()

    def _get_label_df(self, data, mask):
        mask = mask.to(bool).cpu().numpy()
        rows, cols = np.where(mask)
        values = data[mask]

        return (
            pd.DataFrame({"Drug": cols, "Cell": rows, "Label": values.cpu().numpy()})
            .sort_values(["Drug", "Cell"])
            .reset_index(drop=True)
        )

    def _save_index_mapping(self):
        if self.PATH is not None:
            os.makedirs(self.PATH, exist_ok=True)
            idxs_path = os.path.join(self.PATH, "idxs.npy")
            if not os.path.exists(idxs_path):
                idxs = np.array(
                    [
                        np.arange(len(self.adj_mat_original.index)),
                        self.adj_mat_original.index,
                    ]
                )
                np.save(idxs_path, idxs)

    def _get_target_indices(self, matrix, value):
        if self.dim == 0:  # 行（Cell）
            return np.where(matrix[self.target_index, :] == value)[0]
        else:  # 列（Drug）
            return np.where(matrix[:, self.target_index] == value)[0]

    def _sample_target_test_index(self):
        return self._get_target_indices(self.adj_mat, 1)

    def _sample_train_test_data(self):
        test_data = np.zeros(self.adj_mat.shape, dtype=np.float32)
        test_index = self._sample_target_test_index()

        if self.dim == 0:
            test_data[self.target_index, test_index] = 1
        else:
            test_data[test_index, self.target_index] = 1

        train_data = self.adj_mat - test_data
        return torch.from_numpy(train_data), torch.from_numpy(test_data)

    def _sample_train_test_mask(self):
        neg_value = (
            np.ones(self.adj_mat.shape, dtype=np.float32)
            - self.adj_mat
            - self.null_mask
        )
        neg_test_mask = np.zeros(self.adj_mat.shape, dtype=np.float32)

        target_neg_index = self._get_target_indices(neg_value, 1)

        if self.dim == 0:  # Cell（行）をターゲット
            neg_test_mask[self.target_index, target_neg_index] = 1
            neg_value[self.target_index, :] = 0
        else:  # Drug（列）をターゲット
            neg_test_mask[target_neg_index, self.target_index] = 1
            neg_value[:, self.target_index] = 0

        train_mask = (self.train_data.numpy() + neg_value).astype(bool)
        test_mask = (self.test_data.numpy() + neg_test_mask).astype(bool)
        # Null Maskを適用
        train_mask[self.null_mask == 1] = False
        return torch.from_numpy(train_mask), torch.from_numpy(test_mask)

    def _get_labels(self, is_train=False):
        mask = self.train_mask if is_train else self.test_mask
        masked_indices = mask.numpy().nonzero()

        row_labels = [self.adj_mat_original.index[i] for i in masked_indices[0]]
        col_labels = [self.adj_mat_original.columns[j] for j in masked_indices[1]]
        values = [
            self.adj_mat_original.iloc[i, j]
            for i, j in zip(masked_indices[0], masked_indices[1])
        ]

        if self.PATH is not None:
            idxs_path = os.path.join(self.PATH, "idxs.npy")
            conv = dict(
                pd.DataFrame(np.load(idxs_path, allow_pickle=True)).T[[1, 0]].values
            )
            row_labels = [conv[i] for i in row_labels]
            col_labels = [conv[i] for i in col_labels]

        return pd.DataFrame({"Drug": col_labels, "Cell": row_labels, "Label": values})

    def _update_unified_matrix(self):
        A_dc = pd.DataFrame(
            self.train_data.numpy(),
            index=self.adj_mat_original.index,  # 正しいCell順
            columns=self.adj_mat_original.columns,  # 正しいDrug順
        ).fillna(0)

        indexes = list(A_dc.index) + list(self.A_cg.columns) + list(self.A_dg.index)
        base = pd.DataFrame(
            np.zeros((len(indexes), len(indexes))), index=indexes, columns=indexes
        )

        base.loc[self.A_cg.index, self.A_cg.columns] = self.A_cg
        base.loc[self.A_cg.columns, self.A_cg.index] = self.A_cg.T
        base.loc[A_dc.index, A_dc.columns] = A_dc
        base.loc[A_dc.columns, A_dc.index] = A_dc.T
        base.loc[self.A_dg.index, self.A_dg.columns] = self.A_dg
        base.loc[self.A_dg.columns, self.A_dg.index] = self.A_dg.T

        edge_index = torch.tensor(np.array(base.values.nonzero())).type(torch.int64)
        edge_attr = torch.tensor(base.values[base.values.nonzero()])
        return edge_index, edge_attr
