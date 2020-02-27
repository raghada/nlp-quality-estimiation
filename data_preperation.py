import os
import string

import nltk
import numpy as np
import pandas as pd
import spacy
import torch
import torchtext
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sentence_transformers import SentenceTransformer
from torchtext import data, datasets
from tqdm import tqdm

##############################################
##############################################
#####General Methods (for all strategies)#####
##############################################
##############################################

def get_df(split='train'):
    """
    The aim of this method is to read the provided data files
    and convert them to the format of dataframes

    Keyword Arguments:
        split {str} -- [the split of data to work on, train, dev, or test] (default: {'train'})
    
    Returns:
        [dataframe] -- [the processed data contained in a dataframe]
    """
    
    # if we are to follow the split provided or not
    # if not, we will read everything in a single file
    if (split):
      pre = '.'
    else:
      pre = ''        

    src = pre + 'ende.src'
    scores = pre + 'ende.scores'
    mt = pre + 'ende.mt'
    en = pd.DataFrame(columns=['id','en'])
    de = pd.DataFrame(columns=['id','de'])
    score = pd.DataFrame(columns=['id','score'])
    
    count = 0
    with open('./'+split+ mt,'r') as f:
        for line in f:
            de.loc[len(de)] = [count, line[:-1]]
            count+=1
    count = 0
    with open('./'+split+src,'r') as f:
        for line in f:
            en.loc[len(en)] = [count, line[:-1]]
            count+=1
    if split != 'test':
        count = 0
        with open('./'+split+scores,'r') as f:
            for line in f:
                score.loc[len(score)] = [count, float(line[:-1])]
                count+=1
        return en.merge(de,on='id').merge(score,on='id').drop('id', 1)
    return en.merge(de,on='id').drop('id', 1)

def read_data():
    """
        This function checks whether or not data has been previously read and stored in data frames
        if not, it will read them    
    """
    if not os.path.exists("train.csv"):
        get_df('train').to_csv("train.csv", index=False)
        get_df('dev').to_csv("val.csv", index=False)
        get_df('test').to_csv("test.csv", index=False)

def writeScores(scores):
    """
    The aim of this function is to write the test set prediction results to a .txt file
    and save it in the current directory
    
    
    Arguments:
        scores {[list]} -- [a list of the scores predicted for the test-set]
    """
    fn = "predictions.txt"
    with open(fn, 'w') as output_file:
        for x in scores:
            output_file.write(f"{x}\n")

##############################################
##############################################
################ Strategy 1 ##################
##############################################
##############################################






##############################################
##############################################
################ Strategy 2 ##################
##############################################
##############################################


def clean_data_strategy_2():
    nltk.download('stopwords')

    stop_words_en = set(stopwords.words('english'))
    stop_words_de = set(stopwords.words('german'))
    [stop_words_en.add(punct) for punct in string.punctuation]
    [stop_words_de.add(punct) for punct in string.punctuation]

    en_text = data.Field(init_token='<s>',eos_token='</s>',lower=True, tokenize='spacy', tokenizer_language='en', stop_words=stop_words_en)
    de_text = data.Field(init_token='<s>',eos_token='</s>',lower=True, tokenize='spacy', tokenizer_language='de', stop_words=stop_words_de)
    SCORE = data.LabelField(dtype=torch.float, sequential=False, use_vocab=False)

    data_fields = [('en', en_text), ('de', de_text), ('score',SCORE)]
    test_data_fields = [('en', en_text), ('de', de_text)]

    # train, val = data.TabularDataset.splits(path='./', train='train.csv', validation='val.csv', format='csv', fields=data_fields, skip_header=True)
    train_val = data.TabularDataset(path='./train_val.csv', format='csv', fields=data_fields, skip_header=True)
    train, val = train_val.split(split_ratio=0.7)
    test = data.TabularDataset(path='./test.csv', format='csv', fields=test_data_fields, skip_header=True)

    de_text.build_vocab(train, min_freq=2)
    en_text.build_vocab(train, min_freq=2)

    # train_val_iter = data.BucketIterator(train_val, batch_size=64)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, val, test), batch_size=64, sort=False, shuffle=False)
    dev_iter.train = False
    test_iter.train = False
    
    return en_text, de_text, train_iter, dev_iter, test_iter

def get_en_word_emb(word):
  return nlp_en(word).vector

def get_de_word_emb(word):
  return nlp_de(word).vector

def get_GloVe_embedding(en_text, de_text):
    '''
    !spacy download en_core_web_md
    !spacy link en_core_web_md en300

    !spacy download de_core_news_md
    !spacy link de_core_news_md de300
    '''
    nlp_de =spacy.load('de300')
    nlp_en =spacy.load('en300')

    embedding_matrix_en = np.zeros((len(en_text.vocab), 300))
    embedding_matrix_de = np.zeros((len(de_text.vocab), 300))
    
    for i, word in tqdm(enumerate(en_text.vocab.itos)):
        embedding_vector = get_en_word_emb(word)
        embedding_matrix_en[i] = embedding_vector

    for i, word in tqdm(enumerate(de_text.vocab.itos)):
        embedding_vector = get_en_word_emb(word)
        embedding_matrix_de[i] = embedding_vector

    return embedding_matrix_en, embedding_matrix_de

##############################################
##############################################
################ Strategy 3 ##################
##############################################
##############################################

def clean_data_strategy_3():
    nltk.download('stopwords')
    stop_words_en = set()
    stop_words_de = set()
    [stop_words_en.add(punct) for punct in string.punctuation]
    [stop_words_de.add(punct) for punct in string.punctuation]

    en_text = data.RawField()
    de_text = data.RawField()
    SCORE = data.LabelField(dtype=torch.float, sequential=False, use_vocab=False)

    data_fields = [('en', en_text), ('de', de_text), ('score',SCORE)]
    test_data_fields = [('en', en_text), ('de', de_text)]
    train, val = data.TabularDataset.splits(path='./', train='train.csv', validation='val.csv', format='csv', fields=data_fields, skip_header=True)

    test = data.TabularDataset(path='./test.csv', format='csv', fields=test_data_fields, skip_header=True)

    train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, val, test), batch_size=64, sort=False, shuffle=False)
    dev_iter.train = False
    test_iter.train = False

    return train_iter, dev_iter, test_iter

def prepare_batch_strategy_3(batch):
    sntnc_model = SentenceTransformer('distiluse-base-multilingual-cased')
    embed_en = np.zeros((len(batch.en), 512))
    embed_de = np.zeros((len(batch.de), 512))
  
    en_embeddings = sntnc_model.encode(batch.en)
    de_embeddings = sntnc_model.encode(batch.de)
    
    for i in range(len(en_embeddings)):
        embed_en[i] = en_embeddings[i]
        embed_de[i] = de_embeddings[i]

    return torch.from_numpy(embed_en), torch.from_numpy(embed_de)


##############################################
##############################################
################# END OF FILE ################
##############################################
##############################################
