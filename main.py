import argparse
import logging
import os
import json

from model.net import Net
from model.dataloader import DataEngine

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
import time

parser = argparse.ArgumentParser('VQA network baseline!')

parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--data_dir", type=str, help="Directory with data")
parser.add_argument("--img_dir", type=str, help="Directory with image")
parser.add_argument("--year", type=str, help="VQA release year (either 2014 or 2017)")
parser.add_argument("--test_set", type=str, default="test-dev", help="VQA release year (either 2014 or 2017)")
parser.add_argument("--exp_dir", type=str, help="Directory in which experiments are stored")
parser.add_argument("--config", type=str, help='Config file')
parser.add_argument("--load_checkpoint", type=str, help="Load model parameters from specified checkpoint")
parser.add_argument("--gpu_ratio", type=float, default=0.95, help="How many GPU ram is required? (ratio)")
parser.add_argument("--use_pretrained", action='store_true', help='use pretrained resnet')

args = parser.parse_args()

torch.cuda.set_device(args.gpu)

def load_config(config_file):
    with open(config_file, 'rb') as f_config:
        config_str = f_config.read()
        config = json.loads(config_str.decode('utf-8'))

    return config

config = load_config(args.config)
logger = logging.getLogger()

# required parameters
finetune = config["model"]["image"].get('finetune', list())
use_glove = config["model"]["glove"]
max_iters = config["optimizer"]["max_iters"]
lstm_size = config["model"]["no_hidden_LSTM"]
emb_size = config["model"]["word_embedding_dim"]
resnet_model = config["model"]["image"]["resnet_version"]
lr = config['optimizer']['learning_rate']
batch_size = config['optimizer']['batch_size']
step_size = config['optimizer']['step_size']

'''
Arguments:
    dataloader : to load images, tokens abs glove embeddings
    model : main VQA network
    optimizer : Adam (preferred)

Returns:
    None
'''

def variablize(image, tokens, glove_emb, answer, volatile):
    image = Variable(image.cuda(), volatile=volatile)
    tokens = Variable(tokens.cuda(), volatile=volatile)
    glove_emb = Variable(glove_emb.cuda(), volatile=volatile)
    answer = Variable(answer.cuda(), volatile=volatile)
    return image, tokens, glove_emb, answer

def test(dataloader, model):
    model.eval()
    iteration = 0
    total_num, correct = 0, 0
    loss = 0
    for image, tokens, glove_emb, answer in dataloader:
        image, tokens, glove_emb, answer = variablize(image, tokens, glove_emb, answer, True)
        output = model(image, tokens, glove_emb)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(answer.data.view_as(pred)).long().cpu().sum()
        total_num += len(answer)
        loss += F.nll_loss(output, answer.view(-1)).data[0]    
        iteration += 1
        if iteration > 20:
            break
    return loss/iteration, correct/total_num

        
def train(train_dataloader, val_dataloader, model, optimizer):

    model.train()
    iteration = 0
    
    train_log = open(args.config + '.log.train', 'w')
    valid_log = open(args.config + '.log.valid', 'w')
    for epoch in range(10):
        for image, tokens, glove_emb, answer in train_dataloader:
            image, tokens, glove_emb, answer = variablize(image, tokens, glove_emb, answer, False)
            st = time.time()

            optimizer.zero_grad()
            output = model(image, tokens, glove_emb)
            loss = F.nll_loss(output, answer.view(-1))
            loss.backward()
            optimizer.step()
        
            train_log.write(str(loss.data[0]) + '\n')
            if iteration % 10 == 0:
                print('iteration: ', iteration, 'loss: ', loss.data[0], 'time taken: ', time.time()-st)

            # save model state
            if iteration % 2000 == 0:
                torch.save(model.state_dict(), os.path.join(args.exp_dir, 'iter_%s.pth'%str(iteration)))

            if iteration % 100 == 0:
                valid_loss, valid_acc = test(val_dataloader, model)
                print('Valid loss:', valid_loss, 'Valid acc:', valid_acc)
                valid_log.write(str(valid_loss) + '\t' + str(valid_acc))
        
            iteration += 1

            # decrease learning rate by 10 after each step size
            #if iteration % step_size == 0:
            #    lr = lr*0.1
            #    for param_group in optimizer.param_groups:
            #        param_group['lr'] = lr


def main():
    train_engine = DataEngine(config, args.data_dir, args.img_dir, args.year, args.test_set, 'train')
    train_dataloader = DataLoader(train_engine,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)
    val_engine = DataEngine(config, args.data_dir, args.img_dir, args.year, args.test_set, 'val')
    val_dataloader = DataLoader(val_engine,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=4,
                                pin_memory=True
                                )

    model = Net(config=config, no_words=train_engine.tokenizer.no_words, no_answers=train_engine.tokenizer.no_answers,
                resnet_model=resnet_model, lstm_size=lstm_size, emb_size=emb_size, use_pretrained=args.use_pretrained).cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train(train_dataloader, val_dataloader, model, optimizer)

if __name__ == '__main__':
    main()