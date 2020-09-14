# codeing: utf-8

import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.block import HybridBlock
from typing import List


def _pos(x: int) -> int:
    """
    map the relative distance between [0, 123)
    :param x: token index
    :return: mapped relative position
    """
    if x < -60:
        return 0
    if x >= -60 and x <= 60:
        return x + 61
    if x > 60:
        return 122


class RelationClassifier(HybridBlock):
    """
    primary model block for attention-based, convolution-based or other classification model
    emb_input_dim: Size of the vocabulary
    emb_output_dim: embedding length
    """
    def __init__(self, emb_input_dim: int, emb_output_dim: int, max_seq_len=100, filters=[2,3,4,5], num_classes=19,
                 dropout=0.2, is_training=True):
        super(RelationClassifier, self).__init__()
        self.max_len = max_seq_len
        # dw - embeding size
        self.dp = 50
        dc = 1000
        np = 123
        nr = num_classes
        self.is_training = is_training

        self.d = emb_output_dim + 2*self.dp
        # define model layers here
        with self.name_scope():
            self.embedding = nn.Embedding(emb_input_dim, emb_output_dim)
            self.dist_embedding = nn.Embedding(np, self.dp)

            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
            # sliding window + convolution layer
            self.conv1 = nn.Conv2D(dc, (filters[0], self.d), (1, self.d), in_channels=1, activation='relu')
            self.conv2 = nn.Conv2D(dc, (filters[1], self.d), (1, self.d), in_channels=1, activation='relu')
            self.conv3 = nn.Conv2D(dc, (filters[2], self.d), (1, self.d), in_channels=1, activation='relu')
            self.conv4 = nn.Conv2D(dc, (filters[3], self.d), (1, self.d), in_channels=1, activation='relu')

            # self.maxpool = nn.MaxPool1D(max_seq_len, strides=1)
            self.wl = nn.Dense(nr, use_bias=False)

    def input_attention(self, data: List[int], inds: List[int]):
        """
        self-implemented input attention
        it compares each argument vector with the input embedding and returns the similarities.
        :param data: The sentence representation (token indices to feed to embedding layer)
        :param inds: A vector - shape (2,) of two indices referring to positions of the two arguments
        :return: attention context vector
        """
        # d1 - relative distance from each word to entity1
        # d2 - relative distance from each word to entity2
        dist1 = []
        dist2 = []
        for sent, pos in zip(data, inds):
            d1 = [_pos(int(pos[0].asscalar()) - idx) for idx, _ in enumerate(sent)]
            d2 = [_pos(int(pos[1].asscalar()) - idx) for idx, _ in enumerate(sent)]
            dist1.append(d1)
            dist2.append(d2)
        dist1 = mx.nd.array(dist1)
        dist2 = mx.nd.array(dist2)
        dist1_emb = self.dist_embedding(dist1) # (batch_size, n=100)
        dist2_emb = self.dist_embedding(dist2) # (batch_size, n=100)
        x_emb = self.embedding(data) # (batch_size, n=100, hidden_units=300)
        x_concat = mx.nd.concat(x_emb, dist1_emb, dist2_emb, dim=2) # (batch_size, n=100, d=350)

        # self-attention layer
        # inds = mx.nd.one_hot(inds, self.max_len)
        # ind_embeddings = mx.nd.batch_dot(inds, x_emb) #(batch_size, 2, hidden_units=300)
        # attention_scores = mx.nd.batch_dot(x_emb, ind_embeddings.transpose((0, 2, 1))) # (batch_size, n=100, 2)
        # attention_scores = mx.nd.mean(mx.nd.softmax(attention_scores, axis=1), axis=2) # (batch_size, n=100)
        # R = x_concat * mx.nd.expand_dims(attention_scores, axis=2) # R shape (batch_size, n=100, dw=350)
        # return R
        return x_concat

    def scoring(self, R_star):
        """
        scoring function
        :param R_star: input vector for the final dense layer
        :return: a score for each label
        """
        # R_star (batch_size, dc=500, n=100)
        # R_star.transpose x WL
        # (1, dc) x (dc, nr)
        score = self.wl(R_star.transpose((0,2,1))) # (batch_size, n=100, nr=19)
        return score

    def hybrid_forward(self, F, data: List[int], inds: List[int]):
        """
        :param data: The sentence representation (token indices to feed to embedding layer)
        :param inds: A vector - shape (2,) of two indices referring to positions of the two arguments
        :return: a score for each label
        """
        R = self.input_attention(data, inds) # R shape (batch_size, n=100, dw=350)
        R = self.dropout1(R)
        R = mx.nd.expand_dims(R, 1) # (batch_size, in_channel=1, n=100, dw=350)

        conv1 = self.conv1(R)[:, :, :, 0]  # (batch_size, dc=500, n=100, 1)
        maxpool1 = mx.nd.max(conv1, 2)
        conv2 = self.conv2(R)[:, :, :, 0]
        maxpool2 = mx.nd.max(conv2, 2)
        conv3 = self.conv3(R)[:, :, :, 0]
        maxpool3 = mx.nd.max(conv3, 2)
        conv4 = self.conv4(R)[:, :, :, 0]
        maxpool4 = mx.nd.max(conv4, 2)
        maxpool_out = mx.nd.concat(maxpool1, maxpool2, maxpool3, maxpool4, dim=1)  # (batch_size, kernal_size*dc)
        maxpool_out = self.dropout2(maxpool_out)
        score = self.wl(maxpool_out) # (batch_size, nr=19)
        return score
