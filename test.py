import argparse
import torch
# from TuckER.load_data import Data

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset", type=str, default="WN18RR", nargs="?",
#                     help="Which dataset to use: FB15k, FB15k-237, WN18 or WN18RR.")
#     parser.add_argument("--num_iterations", type=int, default=500, nargs="?",
#                     help="Number of iterations.")
#     parser.add_argument("--batch_size", type=int, default=128, nargs="?",
#                     help="Batch size.")
#     parser.add_argument("--lr", type=float, default=0.0005, nargs="?",
#                     help="Learning rate.")
#     parser.add_argument("--dr", type=float, default=1.0, nargs="?",
#                     help="Decay rate.")
#     parser.add_argument("--demb", type=int, default=256, nargs="?",
#                     help="Embedding dimensionality.")
#     parser.add_argument("--cuda", type=bool, default=True, nargs="?",
#                     help="Whether to use cuda (GPU) or not (CPU).")
#     parser.add_argument("--input_dropout", type=float, default=0.3, nargs="?",
#                     help="Input layer dropout.")
#     parser.add_argument("--hidden_dropout1", type=float, default=0.4, nargs="?",
#                     help="Dropout after the first hidden layer.")
#     parser.add_argument("--hidden_dropout2", type=float, default=0.5, nargs="?",
#                     help="Dropout after the second hidden layer.")
#     parser.add_argument("--label_smoothing", type=float, default=0.1, nargs="?",
#                     help="Amount of label smoothing.")
#     parser.add_argument("--triplet_heads", type=int, default=8, nargs="?",
#                     help="Number of heads in the triplet transformer.")
#     parser.add_argument("--triplet_layers", type=int, default=8, nargs="?",
#                     help="Number of encoder layers in the triplet transformer.")
#     parser.add_argument("--graph_heads", type=int, default=16, nargs="?",
#                     help="Number of heads in the graph transformer.")
#     parser.add_argument("--graph_layers", type=int, default=16, nargs="?",
#                     help="Number of encoder layers in the graph transformer.")
    

#     args = parser.parse_args()
#     dataset = args.dataset
#     data_dir = "TuckER/data/%s/" % dataset
#     seed = 20
#     d = Data(data_dir=data_dir, reverse=False)

#     print(len(d.train_contextual_triplets.keys()))
#     print(len(d.test_contextual_triplets.keys()))
#     print(len(d.valid_contextual_triplets.keys()))

bias = []
for i in range(5):
    temp = torch.zeros((3,3))
    for x in range(3):
        for y in range(3):
            temp[x][y] = i
    bias.append(temp) 
 
bias = torch.stack(bias, dim=0)
bias = torch.stack((bias,bias), dim=1).view(10,3,3)
print(bias)

weird = bias.view(5, 2, -1, 3)
print(weird.shape)
print([torch.equal(weird[i][0],weird[i][1]) for i in range(5)])