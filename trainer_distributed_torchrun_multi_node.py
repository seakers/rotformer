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

class Trainer:

    def __init__(self, model, loader, optimizer, label_smoothing, decay_rate, save_every, snapshot_path):
        self.data_loader = loader
        self.optimizer = optimizer
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.save_every = save_every
        self.label_smoothing = label_smoothing
        self.epochs_run = 0
        if decay_rate:
            self.scheduler = ExponentialLR(self.optimizer, decay_rate)

        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)
        self.model = DDP(model, device_ids=[self.local_rank])

    def _load_snapshot(self, snapshot_path):
        snapshot = torch.load(snapshot_path)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

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
        print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {batch} | Steps: {len(self.data_loader)}")
        self.data_loader.sampler.set_epoch(epoch)
        for coocur_idxs, relation_idxs, targets, context_triplets, locations, num_context_triplets in self.data_loader:
            coocur_idxs = coocur_idxs.to(self.local_rank)
            relation_idxs = relation_idxs.to(self.local_rank)
            targets = targets.to(self.local_rank)
            self._run_batch(coocur_idxs, relation_idxs, targets, context_triplets, locations, num_context_triplets)

    def _save_snapshot(self, epoch):
        snapshot = {}
        snapshot["MODEL_STATE"] = self.model.module.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        torch.save(snapshot, f"snapshot.pt")
        print(f"Epoch {epoch} saved.")

    def train(self, num_epochs):
        for epoch in range(self.epochs_run, num_epochs):
            self._run_epoch(epoch)
            if self.global_rank == 0 and epoch % self.save_every == 0:
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


def main(data_dir,
         num_epochs,
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
         graph_trans_layers,
         save_every : int = 50,
         snapshot_path : str = "snapshot.pt"):
    
    init_process_group(backend="nccl")

    kwargs = {"input_dropout": input_dropout, "hidden_dropout1": hidden_dropout1,"hidden_dropout2": hidden_dropout2}

    data = Data(data_dir, reverse=True)
    model = RotnumFormerDistributed(d_emb = d_emb, n_entities = len(data.entities), n_relations = len(data.relations), contextual_triplets = data.train_contextual_triplets,
                               triplet_trans_heads = triplet_trans_heads, triplet_trans_layers = triplet_trans_layers,
                               graph_trans_heads = graph_trans_heads, graph_trans_layers = graph_trans_layers, 
                               batch_first=True, **kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loader = prepare_dataloader(data, batch_size)
    
    trainer = Trainer(model, loader, optimizer, label_smoothing, decay_rate, save_every, snapshot_path)
    trainer.train(num_epochs)

    destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FB15k-237", nargs="?",
                    help="Which dataset to use: FB15k, FB15k-237, WN18 or WN18RR.")
    parser.add_argument("--num_epochs", type=int, default=500, nargs="?",
                    help="Number of training epochs.")
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
    
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1)) 
    minibatch_size = args.batch_size//local_world_size
    main(data_dir, args.num_epochs, minibatch_size, args.lr, args.dr, 
                        args.demb, args.input_dropout, args.hidden_dropout1, args.hidden_dropout2,
                        args.label_smoothing,args.triplet_heads, args.triplet_layers,args.graph_heads,
                        args.graph_layers)
    
    ## torchrun --nproc_per_node=2 trainer.py --dataset=WN18RR --batch_size=1024 --lr=0.0002 --label_smoothing --demb=256 --graph_heads=2 --graph_layer=3 --input_dropout=0.3 --hidden_dropout1=0.4 --hidden_dropout2=0.5