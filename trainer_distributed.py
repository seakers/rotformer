import numpy as np
import torch
import time
from collections import defaultdict
import argparse
import os

from torch.optim.lr_scheduler import ExponentialLR
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as mp
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from RotTransformer import RotTransformer
from RotnumFormer import RotnumFormerDistributed
from load_data import Data, TripletDataSet

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Trainer:

    def __init__(self, model, loader, optimizer, rank, label_smoothing, decay_rate, save_every):
        self.model = DDP(model, device_ids=[rank])
        self.data_loader = loader
        self.optimizer = optimizer
        self.rank = rank
        self.save_every = save_every
        self.label_smoothing = label_smoothing
        if decay_rate:
            self.scheduler = ExponentialLR(self.optimizer, decay_rate)

    def _run_batch(self, coocur_idxs, relation_idxs, targets, context_triplets, locations, num_context_triplets):
        self.optimizer.zero_grad()
        triplet_predictions, graph_predictions = self.model.forward(coocur_idxs, relation_idxs, context_triplets, locations, num_context_triplets)
        if self.label_smoothing:
            targets = ((1.0-self.label_smoothing)*targets) + (1.0/targets.size(1))           
        triplet_loss = self.model.loss(triplet_predictions, targets)
        graph_loss = self.model.loss(graph_predictions, targets)
        loss = triplet_loss+graph_loss
        loss.backward()
        self.optimizer.step()
    
    def _run_epoch(self, epoch):
        batch = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.rank}] Epoch {epoch} | Batchsize: {batch} | Steps: {len(self.data_loader)}")
        self.data_loader.sampler.set_epoch(epoch)
        for coocur_idxs, relation_idxs, targets, context_triplets, locations, num_context_triplets in self.data_loader:
            coocur_idxs = coocur_idxs.to(self.rank)
            relation_idxs = relation_idxs.to(self.rank)
            targets = targets.to(self.rank)
            self._run_batch(coocur_idxs, relation_idxs, targets, context_triplets, locations, num_context_triplets)

    def _save_checkpoint(self, epoch):
        check = self.model.module.state_dict()
        torch.save(check, f"checkpoint_{epoch}.pt")
        print(f"Epoch {epoch} saved.")

    def train(self, num_iterations):
        for epoch in range(num_iterations):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
            if self.decay_rate:
                self.data_loaderscheduler.step()

def prepare_dataloader(data, batch_size):
    train_data_idxs = data.get_data_idxs(data.train_data)

    er_vocab = data.get_er_vocab(train_data_idxs)
    er_vocab_pairs = list(er_vocab.keys())
    targets = data.get_targets(er_vocab, er_vocab_pairs)
    context = data.train_contextual_triplets
    n_entities = len(data.entities)
    max_context = 2

    train_dataset = TripletDataSet(er_vocab, targets, context, n_entities, max_context)
    return DataLoader(train_dataset, batch_size=batch_size ,pin_memory=True,shuffle=False, sampler=DistributedSampler(train_dataset))


def main(rank: int,
         data_dir,
         num_iterations,
         batch_size,
         learning_rate,
         decay_rate,
         d_emb,
         input_dropout,
         hidden_dropout1, 
         hidden_dropout2,
         label_smoothing,
         triplet_trans_heads,
         triplet_trans_layers,
         graph_trans_heads,
         graph_trans_layers):
    
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

    kwargs = {"input_dropout": input_dropout, "hidden_dropout1": hidden_dropout1,"hidden_dropout2": hidden_dropout2}

    data = Data(data_dir, reverse=True)
    model = RotnumFormerDistributed(d_emb = d_emb, n_entities = len(data.entities), n_relations = len(data.relations), contextual_triplets = data.train_contextual_triplets,
                               triplet_trans_heads = triplet_trans_heads, triplet_trans_layers = triplet_trans_layers,
                               graph_trans_heads = graph_trans_heads, graph_trans_layers = graph_trans_layers, 
                               batch_first=True, **kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loader = prepare_dataloader(data, batch_size)
    
    trainer = Trainer(model, loader, optimizer, rank, label_smoothing, decay_rate, save_every=50)
    trainer.train(num_iterations)

    destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FB15k-237", nargs="?",
                    help="Which dataset to use: FB15k, FB15k-237, WN18 or WN18RR.")
    parser.add_argument("--num_iterations", type=int, default=500, nargs="?",
                    help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=128, nargs="?",
                    help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.0005, nargs="?",
                    help="Learning rate.")
    parser.add_argument("--dr", type=float, default=1.0, nargs="?",
                    help="Decay rate.")
    parser.add_argument("--demb", type=int, default=256, nargs="?",
                    help="Embedding dimensionality.")
    parser.add_argument("--cuda", type=bool, default=True, nargs="?",
                    help="Whether to use cuda (GPU) or not (CPU).")
    parser.add_argument("--input_dropout", type=float, default=0.3, nargs="?",
                    help="Input layer dropout.")
    parser.add_argument("--hidden_dropout1", type=float, default=0.4, nargs="?",
                    help="Dropout after the first hidden layer.")
    parser.add_argument("--hidden_dropout2", type=float, default=0.5, nargs="?",
                    help="Dropout after the second hidden layer.")
    parser.add_argument("--label_smoothing", type=float, default=0.1, nargs="?",
                    help="Amount of label smoothing.")
    parser.add_argument("--triplet_heads", type=int, default=8, nargs="?",
                    help="Number of heads in the triplet transformer.")
    parser.add_argument("--triplet_layers", type=int, default=8, nargs="?",
                    help="Number of encoder layers in the triplet transformer.")
    parser.add_argument("--graph_heads", type=int, default=16, nargs="?",
                    help="Number of heads in the graph transformer.")
    parser.add_argument("--graph_layers", type=int, default=16, nargs="?",
                    help="Number of encoder layers in the graph transformer.")
    

    args = parser.parse_args()
    dataset = args.dataset
    data_dir = "TuckER/data/%s/" % dataset
    torch.backends.cudnn.deterministic = True 
    seed = 20
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(data_dir, args.num_iterations, args.batch_size, args.lr, args.dr, 
                        args.demb, args.input_dropout, args.hidden_dropout1, args.hidden_dropout2,
                        args.label_smoothing,args.triplet_heads, args.triplet_layers,args.graph_heads,
                        args.graph_layers)
             , nprocs=world_size)