import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from scipy.stats.stats import pearsonr

from data_preperation import clean_data_strategy_2, get_GloVe_embedding

LEARNING_RATE = 1e-2
NUM_EPOCHS = 5


class SentenceEmbedding(nn.Module):
    """
    Prepare and encode sentence embeddings
    """
    def __init__(self, embed_size, embed_dim, init):
        super(SentenceEmbedding, self).__init__()
        self.word_embedding = nn.Embedding(embed_size, embed_dim).from_pretrained(torch.FloatTensor(init), freeze='True')
        self.encoder = HBMP()

    def forward(self, input_sentence):
        sentence = self.word_embedding(input_sentence)
        embedding = self.encoder(sentence)
        return embedding

    def encode(self, input_sentence):
        embedding = self.encoder(input_sentence)
        return embedding

class HBMP(nn.Module):
    """
    Hierarchical Bi-LSTM Max Pooling Encoder
    """
    def __init__(self):
        super(HBMP, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.cells = 2
        self.hidden_dim = 300
        self.embed_dim = 300
        self.num_layers = 1
        self.dropout = 0.25

        self.rnn1 = nn.LSTM(input_size=self.embed_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            dropout=self.dropout,
                            bidirectional=True)
        self.rnn2 = nn.LSTM(input_size=self.embed_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            dropout=self.dropout,
                            bidirectional=True)
        self.rnn3 = nn.LSTM(input_size=self.embed_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            dropout=self.dropout,
                            bidirectional=True)


    def forward(self, inputs):
        batch_size = inputs.size()[1]
        h_0 = c_0 = Variable(inputs.data.new(self.cells,
                                             batch_size,
                                             self.embed_dim).zero_())

        out1, (ht1, ct1) = self.rnn1(inputs, (h_0, c_0))
        emb1 = self.max_pool(out1.permute(1,2,0)).permute(2,0,1)


        out2, (ht2, ct2) = self.rnn2(inputs, (ht1, ct1))
        emb2 = self.max_pool(out2.permute(1,2,0)).permute(2,0,1)

        out3, (ht3, ct3) = self.rnn3(inputs, (ht2, ct2))
        emb3 = self.max_pool(out3.permute(1,2,0)).permute(2,0,1)

        emb = torch.cat([emb1, emb2, emb3], 2)
        emb = emb.squeeze(0)

        return emb

class FCClassifier(nn.Module):
    """
    NLI classifier class, taking in concatenated enise and dethesis
    embeddings as an input and returning the result
    """
    def __init__(self):
        super(FCClassifier, self).__init__()
        self.dropout = 0.25
        self.activation = nn.ReLU()
        self.seq_in_size = 512*3
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
    Main model class for the NLI task calling SentenceEmbedding and
    Classifier classes34
    """
    def __init__(self, en_embed_size, de_embed_size, embed_dim,embedding_matrix_en, embedding_matrix_de):
      super(Model, self).__init__()
      self.en_sentence_embedding = SentenceEmbedding(en_embed_size, embed_dim, embedding_matrix_en)
      self.de_sentence_embedding = SentenceEmbedding(de_embed_size, embed_dim, embedding_matrix_de)
      self.classifier = FCClassifier()

    def forward(self, batch):
      en = self.en_sentence_embedding(batch.en)
      de = self.de_sentence_embedding(batch.de)
      result = self.classifier(en, de)
      return result

def check_pearson(model, iter):
  all_preds = torch.Tensor()
  all_scores = torch.Tensor()

  model.eval()
  with torch.no_grad():
    for _, batch in enumerate(iter):
      all_preds = torch.cat([all_preds,model.forward(batch).flatten()])
      all_scores = torch.cat([all_scores,batch.score])
    r = pearsonr(all_scores, all_preds)[0]
    print("\nAverage Pearson coefficient on dev set is {}".format(r))
  model.train()

  return all_preds, all_scores

def train(model, train_iter, dev_iter):
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    loss = []

    for eidx in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0
        epoch_items = 0
        # Start training
        for batch_idx, batch in enumerate(train_iter):
            # Clear the gradients
            optim.zero_grad()

            # loss will be a vector of size (batch_size, ) with losses per every sample
            y_pred = model.forward(batch).view(-1)
            loss = F.l1_loss(y_pred.double(), batch.score.double(), reduction='none')
            
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

        torch.save(model, './models/model_strategy_2_epoch_{}'.format(eidx))

def main_strategy_2():
    en_text, de_text, train_iter, dev_iter, _ = clean_data_strategy_2()
    embedding_en, embedding_de = get_GloVe_embedding(en_text, de_text)
    model = Model(len(en_text), len(de_text), 300, embedding_en, embedding_de)
    train(model, train_iter, dev_iter)
