import warnings
from typing import Optional, Tuple
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from TuckER.TuckerModified import TuckER
    
class RotationalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super(RotationalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        pass
    
class RotnumFormer(nn.Module):

    def __init__(self, d_emb, n_entities, n_relations, triplet_trans_heads, triplet_trans_layers, graph_trans_heads, graph_trans_layers,
                 contextual_triplets, max_context_triplets=2, batch_first=True, **kwargs):
        super(RotnumFormer, self).__init__()

        ## Overall Model Settings
        self.d_emb = d_emb
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.contextual_triplets = contextual_triplets
        self.max_context_triplets = max_context_triplets
        self.device = "cpu"

        ## Embedding Layers and related variables
        self.E = nn.Embedding(n_entities+2,self.d_emb)
        self.R = nn.Embedding(n_relations+1,self.d_emb)
        self.entity_mask_idx = n_entities
        self.entity_pass_idx = n_entities + 1
        self.relation_pass_idx = n_relations

        ## TuckER Attributes
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d_emb, d_emb, d_emb)), dtype=torch.float, device="cuda", requires_grad=True))
        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        self.bn0 = torch.nn.BatchNorm1d(d_emb)
        self.bn1 = torch.nn.BatchNorm1d(d_emb)

        ## Triplet Transformer Attributes
        self.triplet_trans_heads =  triplet_trans_heads
        self.triplet_trans_layers = triplet_trans_layers
        encoder_layer = nn.TransformerEncoderLayer(self.d_emb, self.triplet_trans_heads, batch_first=batch_first)
        self.triplet_transformer = nn.TransformerEncoder(encoder_layer, self.triplet_trans_layers)

        ## Graph Transformer Attributes
        self.graph_trans_heads =  graph_trans_heads
        self.graph_trans_layers = graph_trans_layers
        layer = nn.TransformerEncoderLayer(self.d_emb, self.graph_trans_heads, batch_first=batch_first)
        self.graph_transformer = nn.TransformerEncoder(layer, self.graph_trans_layers)

        self.tucker = TuckER(self.d_emb, **kwargs)
        self.loss = torch.nn.BCELoss()

    def rebatch_for_triplet_transformer(self, cooccur_idxs, relation_idxs, context_triplets, context_cooccur_locs, contextual_triplets_per_input:list[int]):
        """
        """
        all_idxs = torch.empty(cooccur_idxs.shape[0]+sum(contextual_triplets_per_input), 3, self.d_emb).to(self.device)
        
        count = 0
        idx = 0
        while idx < all_idxs.shape[0]:
            anchor_triplet = [cooccur_idxs[count], relation_idxs[count], torch.LongTensor([self.entity_mask_idx]).to(self.device)[0]]
            anchor_triplet_embedding = torch.stack((self.E(anchor_triplet[0]), self.R(anchor_triplet[1]), self.E(anchor_triplet[2])), dim=0)
            all_idxs[idx] = anchor_triplet_embedding
            for i in range(contextual_triplets_per_input[count]):
                triplet = [torch.LongTensor([val]).to(self.device) for val in context_triplets[count][i]]
                #triplet[context_cooccur_locs[count][i]] = torch.LongTensor([self.n_entities]).to(self.device)[0]
                triplet_embedding = torch.cat((self.E(triplet[0]), self.R(triplet[1]), self.E(triplet[2])), dim=0)
                all_idxs[idx] = triplet_embedding

            ## !! might have to change this to self.max_context_triplets !! ##    
            idx += contextual_triplets_per_input[count]+1
            count += 1
        
        return all_idxs


    def rebatch_for_graph_transformer(self, transformed_embeddings, context_cooccurring_locs, contextual_triplets_per_input:list[int]):
        """
        """
        ## CHANGE FROM MANUALLY FILLING TENSOR TO FUNCTION? ##
        # cooccurring_entity_embeddings = torch.empty(transformed_embeddings.shape[0]-sum(contextual_triplets_per_input), self.max_context_triplets+1, self.d_emb).to(self.device)
        # for i in range(cooccurring_entity_embeddings.shape[0]):
        #     for j in range(cooccurring_entity_embeddings.shape[1]):
        #         cooccurring_entity_embeddings[i][j] = self.E(torch.LongTensor([self.entity_pass_idx]).to(self.device))
        # anchor_relation_embeddings = torch.empty(cooccurring_entity_embeddings.shape[0], self.d_emb).to(self.device)
        # relation_bias = torch.empty(cooccurring_entity_embeddings.shape[0], self.max_context_triplets, self.max_context_triplets).to(self.device)
        cooccurring_entity_embeddings = []
        anchor_relation_embeddings = []
        relation_bias = []
        
        count = 0
        idx = 0
        while count < transformed_embeddings.shape[0]-sum(contextual_triplets_per_input):
            cooccurring_entity_embedding = []
            contextual_relations = []

            cooccurring_anchor_entity = transformed_embeddings[idx][0]
            anchor_relation = transformed_embeddings[idx][1]

            cooccurring_entity_embedding.append(cooccurring_anchor_entity)
            contextual_relations.append(anchor_relation)
            anchor_relation_embeddings.append(anchor_relation)

            idx += 1 

            for i in range(self.max_context_triplets):
                if i<contextual_triplets_per_input[count]:
                    cooccurring_entity = transformed_embeddings[idx+i][context_cooccurring_locs[count][i]]
                    contextual_relation = transformed_embeddings[idx+i][1]
                else:
                    cooccurring_entity = self.E(torch.LongTensor([self.entity_pass_idx]).to(self.device)[0])
                    contextual_relation = self.R(torch.LongTensor([self.relation_pass_idx]).to(self.device)[0])

                cooccurring_entity_embedding.append(cooccurring_entity)
                contextual_relations.append(contextual_relation)
            
            contextual_relations = torch.stack(contextual_relations, dim=0)
            cooccurring_entity_embedding = torch.stack(cooccurring_entity_embedding, dim=0)

            cooccurring_entity_embeddings.append(cooccurring_entity_embedding)
            relation_bias.append(torch.softmax(torch.matmul(contextual_relations,contextual_relations.T),dim=1))
            
            idx += contextual_triplets_per_input[count]
            count += 1
        
        cooccurring_entity_embeddings = torch.stack(cooccurring_entity_embeddings, dim=0) # shape = (transformed_embeddings.shape[0]-sum(contextual_triplets_per_input), self.max_context_triplets+1, self.d_emb)
        anchor_relation_embeddings = torch.stack(anchor_relation_embeddings, dim=0) # shape = (cooccurring_entity_embeddings.shape[0], self.d_emb)
        relation_bias = torch.stack(relation_bias, dim=0)
        
        if not self.check_pass_tokens(cooccurring_entity_embeddings, relation_bias, contextual_triplets_per_input):
            raise Exception("Pass tokens are not in the correct position") # shape = (cooccurring_entity_embeddings.shape[0], self.max_context_triplets, self.max_context_triplets)
        
        return cooccurring_entity_embeddings, relation_bias, anchor_relation_embeddings
    
    def check_pass_tokens(self, cooccurring_entity_embeddings, relation_bias, contextual_triplets_per_input):
        less_than_max_idx = [(idx, self.max_context_triplets-val) for idx,val in enumerate(contextual_triplets_per_input) if val<self.max_context_triplets]

        subset_entities = []
        subset_bias = []
        for (idx,num) in less_than_max_idx:
            for i in range(num):
                subset_entities.append(cooccurring_entity_embeddings[idx][-1-i])
                subset_bias.append(relation_bias[idx][-1-i][-1-i])

        first_entity = subset_entities[0]
        first_bias = subset_bias[0] 
        for i, entity in enumerate(subset_entities):
            if not torch.equal(first_entity,entity):
                return False
        for j, bias in enumerate(subset_bias):
            if not torch.equal(first_bias, bias):
                return False
        
        return True

    def forward_TuckER(self, e1, r):
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat) 
        x = x.view(-1, e1.size(1))     
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.matmul(x, self.E.weight.transpose(1,0))
        x = torch.softmax(x, dim=1)

        return x

    def forward(self, cooccur_idxs, relation_idxs):
        """

        """
        contextual_triplets, contextual_cooccurring_locs, context_triplets_per_input = self.context_subgraph(cooccur_idxs)

        cooccur_context_idxs = self.rebatch_for_triplet_transformer(cooccur_idxs, relation_idxs,contextual_triplets, contextual_cooccurring_locs, context_triplets_per_input)
        out_triplet = self.triplet_transformer(cooccur_context_idxs)

        cooccur_entity_embeddings, relation_bias, anchor_relation_embedding = self.rebatch_for_graph_transformer(out_triplet, contextual_cooccurring_locs, context_triplets_per_input)
        attn_mask = []
        for i in range(self.graph_trans_heads):
            attn_mask.append(relation_bias)
        attn_mask = torch.cat(attn_mask, dim=1).view(self.graph_trans_heads*relation_bias.shape[0], relation_bias.shape[1], relation_bias.shape[2])
        check = attn_mask.view(relation_bias.shape[0], self.graph_trans_heads, -1, 3)
        for i in range(relation_bias.shape[0]):
            if not torch.equal(check[i][0],check[i][1]):
                raise Exception("attn_mask is not equal across heads")
        out_graph = self.graph_transformer(cooccur_entity_embeddings, mask=attn_mask)
        
        tucker_triplet_input = []
        idx = 0
        count = 0
        while idx < cooccur_context_idxs.shape[0]:
            tucker_triplet_input.append(out_triplet[idx][0])
            idx += 1 + context_triplets_per_input[count]
            count += 1
        tucker_triplet_input = torch.stack(tucker_triplet_input, dim=0).to(self.device)

        tucker_triplet_output = self.forward_TuckER(tucker_triplet_input, anchor_relation_embedding)

        tucker_graph_input = [out_graph[count][0] for count in range(out_graph.shape[0])]
        tucker_graph_input = torch.stack(tucker_graph_input, dim=0).to(self.device)

        tucker_graph_output = self.forward_TuckER(tucker_graph_input, anchor_relation_embedding)

        return tucker_triplet_output, tucker_graph_output

    def context_subgraph(self, cooccur_idxs) : #-> [([], int)]:
        """
        Returns a list of tuples representing the contextual triplets given a anchor entity where the first
        item in the tuple is the triplet and the second item is the position of the anchor entity in the triplet

        The output includes the query triplet as the first value 
        """
        context_subgraph = []
        locations = []
        num_context_triplets = []

        for cooccur_idx in cooccur_idxs:
            location = []
            context = []
            n_context = 0
            for i in range(len(self.contextual_triplets[cooccur_idx.item()])):
                if i == self.max_context_triplets: break
                triple = self.contextual_triplets[cooccur_idx.item()][i].copy()
                loc = triple.index(cooccur_idx.item())
                triple[loc] = self.entity_mask_idx # replace cooccurring entity index with entity mask index
                context.append(triple)
                location.append(loc)
                n_context += 1
            locations.append(location)
            context_subgraph.append(context)
            num_context_triplets.append(n_context)

        return context_subgraph, locations, num_context_triplets