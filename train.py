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
parser.add_argument("--model_path", type=str, default="results/model.pt")
parser.add_argument("--log_step", type=int, default=50)

# Training parameters
parser.add_argument("--num_epoch", type=int, default=200)
parser.add_argument("--train_batch_size", type=int, default=128)
parser.add_argument("--eval_batch_size", type=int, default=10)
parser.add_argument("--learning_rate", type=float, default=1e-2)
parser.add_argument("--reg_penalty", type=float, default=1e-2)
parser.add_argument("--momentum", type=float, default=0.5)

# Dataset parameters
parser.add_argument("--seed", type=int, default=10)
parser.add_argument("--background_dir", type=str, default="data/python/images_background/")
parser.add_argument("--evaluation_dir", type=str, default="data/python/images_evaluation/")
parser.add_argument("--n_way", type=int, default=20)

config = parser.parse_args()

os.makedirs(config.result_dir, exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = config.device

def main(config):
    dm = DataManager(bg_dir = config.background_dir, eval_dir = config.evaluation_dir, seed = config.seed, n_way = config.n_way)
    train_dl = DataLoader(dataset = dm.verification_dataset, batch_size = config.train_batch_size)
    valid_dl = DataLoader(dataset = dm.validation_dataset, batch_size = config.eval_batch_size)

    do(train_dl, valid_dl, config)

def do(train_dl, valid_dl, config):
    print("LEARNING MODEL...")
    model = SiameseConvNet()

    # This loss combines a Sigmoid layer and the BCELoss in one single class. 
    # This version is more numerically stable than using a plain Sigmoid followed by a BCELoss
    loss_function = nn.BCELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr = config.learning_rate, weight_decay = config.reg_penalty, momentum = config.momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

    for ep in range(config.num_epoch):
        model.train()

        ep_loss = 0.0
        log_step_loss = 0.0

        for it, batch_data in enumerate(train_dl):
            
            batch_img1, batch_img2, batch_label = batch_data

            if len(config.device) > 0:
                model = model.cuda()
                batch_img1 = batch_img1.cuda()
                batch_img2 = batch_img2.cuda()
                batch_label = batch_label.cuda()
            
            optimizer.zero_grad()
            model.batch_size = len(batch_img1)
            out = model(batch_img1, batch_img2)

            batch_loss = loss_function(out, batch_label)
            log_step_loss += batch_loss.item()
            ep_loss += batch_loss.item()

            batch_loss.backward()
            optimizer.step()
            scheduler.step()

            if (it % config.log_step) == (config.log_step - 1):
                print(" ")
                print("[EPOCH %d , iteration %5d] BATCH LOSS: %.5f" % (ep + 1, it + 1, log_step_loss / config.log_step))
                log_step_loss = 0.0
        
        ep_loss = ep_loss / len(train_dl)
        torch.save(model.state_dict(), config.model_path)
        valid_score, valid_loss = eval(valid_dl, config, model = model)
        print("[EPOCH %d] - TRAINING LOSS: %.5f" % (ep + 1, ep_loss))
        print("\tVALIDATION ACCURACY: %.5f at LOSS: %.5f" % (valid_score, valid_loss))

def eval(test_dl, config, model = None):
    if model is None:
        model = SiameseConvNet()
        model.load_state_dict(torch.load(config.model_path))
    
    model.eval()
    corr_cnt = torch.tensor(0, dtype=torch.int64)

    eval_loss_function = nn.CrossEntropyLoss()
    eval_loss = 0.0

    for it, batch_data in enumerate(test_dl):
        batch_img1, batch_img2_list, batch_label = batch_data
        scores = torch.tensor([], dtype=torch.float32)

        if len(config.device) > 0:
            model = model.cuda()
            batch_img1 = batch_img1.cuda()
            batch_img2_list = [i.cuda() for i in batch_img2_list]
        
        for way_idx in range(config.n_way):
            out = model(batch_img1, batch_img2_list[way_idx])
            out = out.cpu()
            scores = torch.cat((scores, out), dim=1)
        
        # pdb.set_trace()
        pred = scores.argmax(dim=1).type(batch_label.dtype)
        # batch_loss = eval_loss_function(pred, batch_label.view(batch_label.shape[0]))
        batch_loss = eval_loss_function(scores, batch_label.view(batch_label.shape[0]).type(torch.long))
        eval_loss += batch_loss.item()
        mat = torch.sum(torch.eq(pred, batch_label.view(batch_label.shape[0])))
        corr_cnt += mat
    
    eval_loss = eval_loss / len(test_dl)
    acc = int(mat) / len(test_dl)

    return acc, eval_loss
        

        
        
        

            




if __name__ == "__main__":
    main(config)
    
