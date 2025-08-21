import warnings
from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from TuckER.TuckerModified import TuckER


from scripts.utils import DataFromJSON
from scripts.ModifiedTorchObjects import TransformerEncoderRelationBias, TransformerEncoderLayerRelationBias

class RotTransformer(nn.module):
    # def __init__(self, model_conf: DataFromJSON, gpu_device: torch.device = None, dicts_set: dict = None):
    #     super(RotTransformer, self).__init__()

    #     ## Overall Model Settings
    #     self.d_emb = model_conf.n_embed
    #     self.epochs = model_conf.epochs
    #     self.gpu_device = gpu_device
    #     self.dicts_set = dicts_set

    #     ## Triplet Transformer Attributes
    #     self.triplet_trans_heads =  model_conf.triplet_trans_heads
    #     self.triplet_trans_layers = model_conf.triplet_trans_layers
    #     self.E = nn.Embedding(NUM_OF_ENTITIES,self.d_emb) ######## figure out how to include NUM_OF_ENTITIES + mask token
    #     self.R = nn.Embedding(NUM_OF_RELATIONS,self.d_emb) ####### figure out how to include NUM_OF_RELATIONS + mask token
    #     encoder_layer = nn.TransformerEncoderLayer(self.n_embed, self.triplet_trans_heads)
    #     self.triplet_transformer = nn.TransformerEncoder(encoder_layer, self.triplet_trans_layers)

    #     ## Graph Transformer Attributes
    #     self.graph_trans_heads =  model_conf.triplet_trans_heads
    #     self.graph_trans_layers = model_conf.triplet_trans_layers
    #     layer = TransformerEncoderLayer(self.d_emb, self.graph_trans_heads)
    #     self.graph_transformer = TransformerEncoderRelationBias(layer, self.graph_trans_layers)

    #     tucker = TuckER(self. d_emb)

    def __init__(self, d_emb, n_entities, n_relations, triplet_trans_heads, triplet_trans_layers, graph_trans_heads, graph_trans_layers):
        super(RotTransformer, self).__init__()

        ## Overall Model Settings
        self.d_emb = d_emb

        ## Triplet Transformer Attributes
        self.triplet_trans_heads =  triplet_trans_heads
        self.triplet_trans_layers = triplet_trans_layers
        self.E = nn.Embedding(n_entities,self.d_emb) ######## figure out how to include NUM_OF_ENTITIES + mask token
        self.R = nn.Embedding(n_relations,self.d_emb) ####### figure out how to include NUM_OF_RELATIONS + mask token
        encoder_layer = nn.TransformerEncoderLayer(self.n_embed, self.triplet_trans_heads)
        self.triplet_transformer = nn.TransformerEncoder(encoder_layer, self.triplet_trans_layers)

        ## Graph Transformer Attributes
        self.graph_trans_heads =  graph_trans_heads
        self.graph_trans_layers = graph_trans_layers
        layer = TransformerEncoderLayerRelationBias(self.d_emb, self.graph_trans_heads)
        self.graph_transformer = TransformerEncoderRelationBias(layer, self.graph_trans_layers)

        tucker = TuckER(self. d_emb)    


    def forward(self, query, args):
        # query : ([entity index, relation index, entity index], int) where int is the location of the anchor

        contextual_triplets = self.context_subgraph(self, query[0][query[1]])
        cooccuring_entities = []
        contextual_relations = []
        out_triplet = 0

        for triple, coocur_loc in contextual_triplets:
            triple = [self.E(triple[0]), self.R(triple[1]), self.E(triple[2])]
            triple = self.triplet_encoder(triple)
            if len(cooccuring_entities) == 0:
                out_triplet = triple[coocur_loc]
            cooccuring_entities.append(triple[coocur_loc])
            contextual_relations.append(triple[1])

        out_triplet = torch.mm(out_graph, self.E.weight.transpose(1,0))
        out_triplet = torch.softmax(out_triplet)

        out_graph = self.graph_transformer(cooccuring_entities, contextual_relations)
        out_graph = self.tucker(out_graph[0], query[0][1])
        out_graph = torch.mm(out_graph, self.E.weight.transpose(1,0))
        pred = torch.softmax(out_graph)

        return out_triplet, out_graph

    def context_subgraph(self, anchor) : #-> [([], int)]:
        """
        Returns a list of tuples representing the contextual triplets given a anchor entity where the first
        item in the tuple is the triplet and the second item is the position of the anchor entity in the triplet

        The output includes the query triplet as the first value 
        """


        pass