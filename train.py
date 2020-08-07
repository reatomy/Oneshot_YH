import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import pdb

from model.models import SiameseConvNet
from data_loader.data_loader import DataManager

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="")
parser.add_argument("--result_dir", type=str, default="results/")

# Training parameters
parser.add_argument("--num_epoch", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--reg_penalty", type=float, default=1e-2)
parser.add_argument("--momentum", type=float, default=0.5)

# Dataset parameters
parser.add_argument("--seed", type=int, default=10)
parser.add_argument("--background_dir", type=str, default="data/python/images_background/")
parser.add_argument("--evaluation_dir", type=str, default="data/python/images_evaluation/")

config = parser.parse_args()

os.makedirs(config.result_dir, exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = config.device

def main(config):
    dm = DataManager(bg_dir = config.background_dir, eval_dir = config.evaluation_dir, seed = config.seed)
    train_dl = DataLoader(dataset = dm.verification_dataset, batch_size = config.batch_size)

    do(train_dl, config)

def do(train_dl, config):
    print("LEARNING MODEL...")
    model = SiameseConvNet()

    # This loss combines a Sigmoid layer and the BCELoss in one single class. 
    # This version is more numerically stable than using a plain Sigmoid followed by a BCELoss
    loss_function = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr = config.learning_rate, weight_decay = config.reg_penalty, momentum = config.momentum)

    for ep in range(config.num_epoch):
        model.train()

        total_loss = 0.0
        total = 0.0

        for it, batch_data in enumerate(train_dl):
            batch_img1, batch_img2, batch_label = batch_data
            
            optimizer.zero_grad()
            model.batch_size = len(batch_img1)
            out = model(batch_img1, batch_img2)

            loss = loss_function(out, batch_label)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            


if __name__ == "__main__":
    main(config)
    