import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.stats.stats import pearsonr

from data_preperation import clean_data_strategy_3, prepare_batch_strategy_3

LEARNING_RATE = 1e-4
NUM_EPOCHS = 7


class FCClassifier(nn.Module):
    """
    Classifier class, taking in concatenated sentences embeddings as an input and returning the quality estimation value
    """
    def __init__(self):
        super(FCClassifier, self).__init__()
        self.dropout = 0.25
        self.seq_in_size = 512*3 # three is the number of features
        self.fc_dim = 600
        self.out_dim = 1
        
        self.mlp = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(self.seq_in_size, self.fc_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.fc_dim),
            nn.Linear(self.fc_dim, self.fc_dim//2),
            nn.ReLU(),
            nn.BatchNorm1d(self.fc_dim//2),
            nn.Linear(self.fc_dim//2, self.out_dim))

    def forward(self, en, de):
        features = torch.cat([en, de, torch.abs(en-de)], 1)
        output = self.mlp(features)
        return output

class Model(nn.Module):
    """
    Main model class for the quality estimation task 
    """
    def __init__(self):
      super(Model, self).__init__()
      self.classifier = FCClassifier()

    def forward(self, en, de):
      result = self.classifier(en, de)
      return result

def check_pearson(model, iter):
    all_preds = torch.Tensor()
    all_scores = torch.Tensor()
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(iter):
            en, de = prepare_batch_strategy_3(batch)
            all_preds = torch.cat([all_preds,model.forward(en.float(), de.float()).flatten()])
            all_scores = torch.cat([all_scores,batch.score])
        r = pearsonr(all_scores, all_preds)[0]
        print("\nAverage Pearson coefficient on dev set is {}".format(r))
    model.train()

    return all_preds, all_scores

def train(model, train_iter, dev_iter):
    
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    loss = []

    for eidx in range(1, NUM_EPOCHS + 1):
        model.train()

        epoch_loss = 0
        epoch_items = 0
        # Start training
        for batch_idx, batch in enumerate(train_iter):

            # Clear the gradients
            optim.zero_grad()
            en, de = prepare_batch_strategy_3(batch)
            # loss will be a vector of size (batch_size, ) with losses per every sample
            y_pred = model.forward(en.float(), de.float()).view(-1)
            loss = F.mse_loss(y_pred.double(), batch.score.double(), reduction='none')
            
            # Backprop the average loss and update parameters
            loss.sum().backward()
            optim.step()

            # sum the loss for reporting, along with the denominator
            epoch_loss += loss.detach().sum()
            epoch_items += loss.numel()

            if batch_idx % 10 == 0:
                # Print progress
                loss_per_token = epoch_loss / epoch_items
                print('[Epoch {:<3}] loss: {:6.2f}'.format(eidx, loss_per_token))


        print('\n[Epoch {:<3}] ended with train_loss: {:6.2f}'.format(eidx, loss_per_token))
        
        # Evaluate on valid set
        model.eval()
        check_pearson(model, dev_iter)
        torch.save(model, 'model_strategy_3_epoch_{}'.format(eidx))

def main_strategy_3():
    model = Model()
    train_iter, dev_iter, _ = clean_data_strategy_3()
    train(model, train_iter, dev_iter)
