import argparse

import torch
import numpy as np

# from src.MLP import MLP


parser = argparse.ArgumentParser()
parser.add_argument('--hidden-size', type=int, default=256)
parser.add_argument('--embedding-size', type=int, default=16)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=200)

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.device = device
print(args.device)


# x = torch.randn(32, 1684, 1684).to(args.device)

# changzhan_net = MLP(1684, num_hidden=args.hidden_size, num_channel=1684,
#                     num_output=args.embedding_size)
# changzhan_net = changzhan_net.to(args.device)
# print(changzhan_net(x).size())


celue_0_np = np.zeros(shape=(64, 3619, 1))
celue_1_np = np.ones(shape=(64, 3619, 1))

false_tensor = torch.tensor(np.zeros(shape=(64, 3619, 1)), dtype=torch.bool) 
true_tensor = torch.tensor(np.ones(shape=(64, 3619, 1)), dtype=torch.bool) 

