import pickle
import torch
import os
import numpy as np

import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from collections import defaultdict

class Data:

    def __init__(self, data_dir="data/FB15k-237/", reverse=False):
        self.train_data = self.load_data(data_dir, "train", reverse=reverse)
        self.valid_data = self.load_data(data_dir, "valid", reverse=reverse)
        self.test_data = self.load_data(data_dir, "test", reverse=reverse)
        self.data = self.train_data + self.valid_data + self.test_data
        self.entities = self.get_entities(self.data)
        self.entity_idxs = {self.entities[i]:i for i in range(len(self.entities))}
        self.train_relations = self.get_relations(self.train_data)
        self.valid_relations = self.get_relations(self.valid_data)
        self.test_relations = self.get_relations(self.test_data)
        self.relations = self.train_relations + [i for i in self.valid_relations \
                if i not in self.train_relations] + [i for i in self.test_relations \
                if i not in self.train_relations]
        self.relation_idxs = {self.relations[i]:i for i in range(len(self.relations))}
        self.train_contextual_triplets = self.get_contextual_triplets(data_dir, "train", self.train_data)
        self.valid_contextual_triplets = self.get_contextual_triplets(data_dir, "valid", self.valid_data)
        self.test_contextual_triplets = self.get_contextual_triplets(data_dir, "test", self.test_data)

    def load_data(self, data_dir, data_type="train", reverse=False):
        with open("%s%s.txt" % (data_dir, data_type), "r") as f:
            data = f.read().strip().split("\n")
            data = [i.split() for i in data]
            if reverse:
                data += [[i[2], i[1]+"_reverse", i[0]] for i in data]
        return data

    def get_relations(self, data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    def get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
        return entities
    
    def get_contextual_triplets(self, data_dir, data_type, data):
        contextual = {}
        try:
            print(f"{data_dir}{data_type}_contextual_triplets.pk")
            with open(f"{data_dir}{data_type}_contextual_triplets.pk", "rb") as file:
                contextual = pickle.load(file)
                print(len(contextual))
        except FileNotFoundError as error:
            entities = self.get_entities(data)
            for e in entities:
                contextual.update({self.entity_idxs[e]: [[self.entity_idxs[d[0]],self.relation_idxs[d[1]],self.entity_idxs[d[2]]] for d in data if e in d]})
            with open(f"{data_dir}{data_type}_contextual_triplets.pk", "wb") as file:
                pickle.dump(contextual, file, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as error:
            print(type(error),error)
        return contextual
    
    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], \
                      self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs
    
    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab
    
    def get_targets(self, er_vocab, er_vocab_pairs):
        #batch = er_vocab_pairs[idx:idx+self.batch_size]
        ## BIG CHANGE IDK IF I CAN DO THIS ##
        targets = np.zeros((len(er_vocab_pairs), len(self.entities)+2))
        for idx, pair in enumerate(er_vocab_pairs):
            targets[idx, er_vocab[pair]] = 1.
        # targets = torch.FloatTensor(targets)
        # if self.cuda:
        #     targets = targets.cuda()
        return targets
    
class TripletDataSet(Dataset):
    def __init__(self, er_vocab, targets, contextual_triplets, entity_mask_idx, max_contextual_triplets):
        self.er_vocab = er_vocab
        self.targets = targets
        self.contextual_triplets = contextual_triplets

        self.entity_mask_idx = entity_mask_idx
        self.max_contextual_triplets = max_contextual_triplets

    def __len__(self):
        return len(self.er_vocab.keys())
    
    def __getitem__(self, idx):
        context_subgraph, locations, num_context_triplets = self.get_context(self.er_vocab[idx,0])
        
        return torch.LongTensor(self.er_vocab[idx,0]), torch.LongTensor(self.er_vocab[idx,1]), torch.LongTensor(self.targets[idx]), context_subgraph, locations, num_context_triplets

    def get_context(self, idx) :
        location = []
        context = []
        n_context = 0
        for i in range(len(self.contextual_triplets[idx])):
            if i == self.max_context_triplets: break
            triple = self.contextual_triplets[idx][i].copy()
            loc = triple.index(idx)
            triple[loc] = self.entity_mask_idx # replace cooccurring entity index with entity mask index
            context.append(triple)
            location.append(loc)
            n_context += 1
        return context, location, n_context

def main(rank):
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl")
    data = Data(data_dir="TuckER/data/WN18RR/", reverse=True)

    train_data_idxs = data.get_data_idxs(data.train_data)
    # valid_data_idxs = data.get_data_idxs(data.valid_data)
    # test_data_idxs = data.get_data_idxs(data.test_data)

    er_vocab = data.get_er_vocab(train_data_idxs)
    er_vocab_pairs = list(er_vocab.keys())
    targets = data.get_targets(er_vocab, er_vocab_pairs)
    context = data.train_contextual_triplets
    n_entities = len(data.entities)
    max_context = 2

    train_dataset = TripletDataSet(er_vocab, targets, context, n_entities, max_context)
    loader = DataLoader(train_dataset, batch_size=1024,pin_memory=True,shuffle=False, sampler=DistributedSampler(train_dataset))

    destroy_process_group()

if __name__ == "__main__":
    mp.spawn(main, args=(), nprocs=1)
