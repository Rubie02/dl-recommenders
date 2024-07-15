from torch import nn
import torch
import numpy as np
import math
from sklearn.metrics import accuracy_score, f1_score

class NeutralColabFilteringNet(nn.Module):
    """
    Create an NCF (Neutral Collaborative Filtering) network.
    Arguments:
    - user_count (int): number of unique users
    - product_count (int): number of unique products
    - embedding_size (int)[Optional]: size of the user and product embeddings, defaults to 32
    - hidden_layers (tuple)[Optional]: tuple of integers defining the number of hidden MLP layers
    - dropout_rate (float)[Optional]: dropout rate for the hidden layers [0 1], defaults to None
    - output_range (tuple)[Optional]: tuple of integers defining the output range, defaults to (1, 5)
    """
    def __init__(self,
                 user_count,
                 product_count,
                 embedding_size=32,
                 hidden_layers=(64, 32, 16, 8),
                 dropout_rate=None,
                 output_range=(1, 5)):
        super().__init__()

        self.user_hash_size = user_count
        self.product_hash_size = product_count

        self.user_embedding = nn.Embedding(self.user_hash_size, embedding_size)
        self.product_embedding = nn.Embedding(self.product_hash_size, embedding_size)

        self.MLP = self._gen_MLP(embedding_size, hidden_layers, dropout_rate)
        if (dropout_rate):
            self.dropout = nn.Dropout(dropout_rate)

        assert output_range and len(output_range) == 2, "output_range has to be a tuple with 2 integers"
        self.norm_min = min(output_range)
        self.norm_range = abs(output_range[0] - output_range[1]) + 1

        self._init_params()

    def _gen_MLP(self, embedding_size, hidden_layers_units, dropout_rate):
        assert (embedding_size * 2) == hidden_layers_units[0], "First input layer number of units has to be equal to ..."
        hidden_layers = []
        input_units = hidden_layers_units[0]

        for num_units in hidden_layers_units[1:]:
            hidden_layers.append(nn.Linear(input_units, num_units))
            hidden_layers.append(nn.ReLU())
            if (dropout_rate):
                hidden_layers.append(nn.Dropout(dropout_rate))
            input_units = num_units

        hidden_layers.append(nn.Linear(hidden_layers_units[-1], 1))
        hidden_layers.append(nn.Sigmoid())
        return nn.Sequential(*hidden_layers)

    def _init_params(self):
        def weights_init(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.user_embedding.weight.data.uniform_(-0.05, 0.05)
        self.product_embedding.weight.data.uniform_(-0.05, 0.05)
        self.MLP.apply(weights_init)

    def forward(self, user_id, product_id):
        user_features = self.user_embedding(user_id % self.user_hash_size)
        product_features = self.product_embedding(product_id % self.product_hash_size)
        x = torch.cat([user_features, product_features], dim=1)
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.MLP(x)
        normalized_output = self.norm_min + self.norm_range * x
        return normalized_output

class DatasetBatchIterator:
    def __init__(self, X, Y, batch_size, shuffle=True):
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)

        if shuffle:
            index = np.random.permutation(X.shape[0])
            X = self.X[index]
            Y = self.Y[index]
        self.batch_size = batch_size
        self.n_batches = int(math.ceil(X.shape[0] / batch_size))
        self._current = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if (self._current >= self.n_batches):
            raise StopIteration()
        k = self._current
        self._current += 1
        bs = self.batch_size
        X_batch = torch.LongTensor(self.X[k*bs:(k+1)*bs])
        Y_batch = torch.FloatTensor(self.Y[k*bs:(k+1)*bs])

        return X_batch, Y_batch.view(-1, 1)

def precision_recall_at_k(y_true, y_pred, k=10, threshold=3.5):
    '''Return precision and recall at k metrics for each user.'''

    user_ratings = list(zip(y_pred, y_true))
    user_ratings.sort(key=lambda x: x[0], reverse=True)

    n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

    n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

    n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                          for (est, true_r) in user_ratings[:k])

    precision = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

    recall = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precision, recall

def accuracy_f1_at_k(y_true, y_pred, threshold=3.5):
    '''Return accuracy and F1 score at k metrics for each user.'''

    y_true_binary = (np.array(y_true) >= threshold).astype(int)
    y_pred_binary = (np.array(y_pred) >= threshold).astype(int)

    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    f1 = f1_score(y_true_binary, y_pred_binary)

    return accuracy, f1
