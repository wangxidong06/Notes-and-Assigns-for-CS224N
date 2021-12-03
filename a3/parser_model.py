#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2018-19: Homework 3
parser_model.py: Feed-Forward Neural Network for Dependency Parsing
Sahil Chopra <schopra8@stanford.edu>
"""
import pickle
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class ParserModel(nn.Module):
    """ Feedforward neural network with an embedding layer and single hidden layer.
    The ParserModel will predict which transition should be applied to a
    given partial parse configuration.

    """

    def __init__(self, embeddings, n_features=36,
                 hidden_size=200, n_classes=3, dropout_prob=0.5):
        """ Initialize the parser model.

        @param embeddings (Tensor): word embeddings (num_words, embedding_size)
        @param n_features (int): number of input features
        @param hidden_size (int): number of hidden units
        @param n_classes (int): number of output classes
        @param dropout_prob (float): dropout probability
        """
        super(ParserModel, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.dropout_prob = dropout_prob
        self.embed_size = embeddings.shape[1]
        self.hidden_size = hidden_size
        self.pretrained_embeddings = nn.Embedding(
            embeddings.shape[0], self.embed_size)
        self.pretrained_embeddings.weight = nn.Parameter(
            torch.tensor(embeddings))

        
        self.embed_to_hidden = nn.Linear(
            self.n_features*self.embed_size, self.hidden_size, bias=True)
        nn.init.xavier_uniform_(self.embed_to_hidden.weight)
        self.hidden_to_logits = nn.Linear(
            self.hidden_size, self.n_classes, bias=True)
        nn.init.xavier_uniform_(self.hidden_to_logits.weight)
        self.dropout = nn.Dropout(p=dropout_prob)

        # END YOUR CODE

    def embedding_lookup(self, t):
        """ Utilize `self.pretrained_embeddings` to map input `t` from input tokens (integers)
            to embedding vectors.

            @param t (Tensor): input tensor of tokens (batch_size, n_features)

            @return x (Tensor): tensor of embeddings for words represented in t
                                (batch_size, n_features * embed_size)
        """

        x = self.pretrained_embeddings(t).view(t.shape[0], -1)
        return x

    def forward(self, t):
        """ Run the model forward.

            Note that we will not apply the softmax function here because it is included in the loss function nn.CrossEntropyLoss

        @param t (Tensor): input tensor of tokens (batch_size, n_features)

        @return logits (Tensor): tensor of predictions (output after applying the layers of the network)
                                 without applying softmax (batch_size, n_classes)
        """
        embeddings = self.embedding_lookup(t)
        h = F.relu(self.embed_to_hidden(embeddings))
        logits = self.hidden_to_logits(self.dropout(h))
        # END YOUR CODE
        return logits
