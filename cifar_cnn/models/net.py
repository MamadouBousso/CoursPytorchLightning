from typing import Any, Dict
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CONV1_DIM = 16
CONV2_DIM = 32
CONV3_DIM = 64

FC4_DIM = 500
FC5_DIM = 250
DO = 0.2
MP = 2
IMAGE_SIZE = 32
# define the CNN architecture
class Net(nn.Module):
    def __init__(self,data_config: Dict[str, Any],args: argparse.Namespace = None,) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}

        input_dim = data_config["input_dims"]
        
        num_classes = len(data_config["mapping"])

        conv1_dim = self.args.get("c1", CONV1_DIM)
        conv2_dim = self.args.get("c2", CONV2_DIM)
        conv3_dim = self.args.get("c3", CONV3_DIM)
        fc4_dim = self.args.get("fc4", FC4_DIM)
        fc5_dim = self.args.get("fc5", FC5_DIM)
        
        do = self.args.get("dropout", DO)
        mp = self.args.get("maxpool", MP)


        # convolutional layer
        self.conv1 = nn.Conv2d(input_dim[0], conv1_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(conv1_dim, conv2_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(conv2_dim, conv3_dim, 3, padding=1)
        # linear layer (64 * 4 * 4 -> 500)
        conv_output_size = IMAGE_SIZE // 8
        fc_input_dim = int(conv_output_size * conv_output_size * conv3_dim)
        
        self.fc1 = nn.Linear(fc_input_dim, fc4_dim)
        # linear layer (500 -> 10)
        self.fc2 = nn.Linear(fc4_dim, fc5_dim)
        self.fc3 = nn.Linear(fc5_dim, num_classes)
        # dropout layer (p=0.25)
        self.drop = nn.Dropout(do)
        # max pooling layer
        self.pool = nn.MaxPool2d(mp, mp)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        B_, C_, H, W = x.shape
        assert H == W == IMAGE_SIZE
        x = self.pool(F.relu(self.conv1(x)))
        x = self.drop(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.drop(x)
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = F.relu(x)
     
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
    
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--conv1", type=int, default=CONV1_DIM)
        parser.add_argument("--conv2", type=int, default=CONV2_DIM)
        parser.add_argument("--conv3", type=int, default=CONV3_DIM)
        
        parser.add_argument("--fc4", type=int, default=FC4_DIM)

        parser.add_argument("--fc5", type=int, default=FC5_DIM)
        parser.add_argument("--dropout", type=float, default=DO)
        parser.add_argument("--maxpool", type=int, default=MP)
        return parser

