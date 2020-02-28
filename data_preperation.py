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
    src = '.ende.src'
    scores = '.ende.scores'
    mt = '.ende.mt'

    en = pd.DataFrame(columns=['id','en'])
    de = pd.DataFrame(columns=['id','de'])
    score = pd.DataFrame(columns=['id','score'])
    
    count = 0
    with open('./ende_data/'+split+ mt,'r') as f:
        for line in f:
            de.loc[len(de)] = [count, line[:-1]]
            count+=1
    count = 0
    with open('./ende_data/'+split+src,'r') as f:
        for line in f:
            en.loc[len(en)] = [count, line[:-1]]
            count+=1
    if split != 'test':
        count = 0
        with open('./ende_data/'+split+scores,'r') as f:
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
            output_file.write("{}\n".format(x))

def download_dependencies():
    """
    Downloads the necessary packages for the project to run
    """

    os.system('pip install -e git+git://github.com/UKPLab/sentence-transformers@a96ccd3#egg=sentence-transformers')
    os.system('spacy download en_core_web_md')
    os.system('spacy link en_core_web_md en300')

    os.system('spacy download de_core_news_md')
    os.system('spacy link de_core_news_md de300')

    os.system('python3 -m spacy download en')
    os.system('python3 -m spacy download de')


    import nltk
    import ssl

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download('stopwords')


    if not os.path.exists('ende_data.zip'):
        os.system('wget -O ende_data.zip https://competitions.codalab.org/my/datasets/download/c748d2c0-d6be-4e36-9f12-ca0e88819c4d')
        os.system('unzip ende_data.zip')

##############################################
##############################################
################ Strategy 1 ##################
##############################################
##############################################

def clean_data_strategy_1():
    """
    The aim of this method is to prepare the data to be trained using Strategy 1

    Returns:
        [tuple] -- train and validation data
    """
    nlp_de =spacy.load('de300')
    nlp_en =spacy.load('en300')

    #EN-DE files
    de_train_src = get_embeddings("./train.ende.src",nlp_en,'en')
    de_train_mt = get_embeddings("./train.ende.mt",nlp_de,'de')

    f_train_scores = open("./train.ende.scores",'r')
    de_train_scores = f_train_scores.readlines()

    de_val_src = get_embeddings("./dev.ende.src",nlp_en,'en')
    de_val_mt = get_embeddings("./dev.ende.mt",nlp_de,'de')

    f_val_scores = open("./dev.ende.scores",'r')
    de_val_scores = f_val_scores.readlines()

    #Put the features into a list
    X_train= [np.array(de_train_src),np.array(de_train_mt)]
    X_train_de = np.array(X_train).transpose()

    X_val = [np.array(de_val_src),np.array(de_val_mt)]
    X_val_de = np.array(X_val).transpose()

    #Scores
    train_scores = np.array(de_train_scores).astype(float)
    y_train_de =train_scores

    val_scores = np.array(de_val_scores).astype(float)
    y_val_de =val_scores
    
    return X_train_de, X_val_de, y_train_de, y_val_de

def get_sentence_emb(line,nlp,lang, stop_words_en, stop_words_de):
    """
    returns the sentence level embedding
    """
    if lang == 'en':
        text = line.lower()
        l = [token.lemma_ for token in nlp.tokenizer(text)]
        l = ' '.join([word for word in l if word not in stop_words_en])

    elif lang == 'de':
        text = line.lower()
        l = [token.lemma_ for token in nlp.tokenizer(text)]
        l= ' '.join([word for word in l if word not in stop_words_de])

    sen = nlp(l)
    return sen.vector

def get_embeddings(f,nlp,lang):
    """
    Find the embedding for the whole dataset
    """
    file = open(f) 
    lines = file.readlines() 
    sentences_vectors =[]

    stop_words_en = set(stopwords.words('english'))
    stop_words_de = set(stopwords.words('german'))

    for l in lines:
        vec = get_sentence_emb(l,nlp,lang, stop_words_en, stop_words_de)
        if vec is not None:
            vec = np.mean(vec)
            sentences_vectors.append(vec)
        else:
            print("didn't work :", l)
            sentences_vectors.append(0)

    return sentences_vectors

##############################################
##############################################
################ Strategy 2 ##################
##############################################
##############################################

def clean_data_strategy_2():
    """
    The aim of this method is to prepare the data to be trained using Strategy 2
    in this function, we use torchtext to preprocess the data, by removing stopwords and punctuations, building the vocabulary lists
    for both languages, and spliting the training, validation(dev) and testing data

    Returns:
        [tuple] -- [tuple of values, includes the vocabulary lists, as well as train, dev, and test bucket iterators]
    """

    stop_words_en = set(stopwords.words('english'))
    stop_words_de = set(stopwords.words('german'))

    [stop_words_en.add(punct) for punct in string.punctuation]
    [stop_words_de.add(punct) for punct in string.punctuation]

    en_text = data.Field(init_token='<s>',eos_token='</s>',lower=True, tokenize='spacy', tokenizer_language='en', stop_words=stop_words_en)
    de_text = data.Field(init_token='<s>',eos_token='</s>',lower=True, tokenize='spacy', tokenizer_language='de', stop_words=stop_words_de)
    SCORE = data.LabelField(dtype=torch.float, sequential=False, use_vocab=False)

    data_fields = [('en', en_text), ('de', de_text), ('score',SCORE)]
    test_data_fields = [('en', en_text), ('de', de_text)]

    train, val = data.TabularDataset.splits(path='./', train='train.csv', validation='val.csv', format='csv', fields=data_fields, skip_header=True)
    test = data.TabularDataset(path='./test.csv', format='csv', fields=test_data_fields, skip_header=True)

    de_text.build_vocab(train, min_freq=2)
    en_text.build_vocab(train, min_freq=2)

    train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, val, test), batch_size=64, sort=False, shuffle=False)
    dev_iter.train = False
    test_iter.train = False
    
    return en_text, de_text, train_iter, dev_iter, test_iter

def get_en_word_emb(nlp_en, word):
    """return the embedding of the received English word
    
    Arguments:
        word {str} -- [the word that we want to retrieve the embedding of]
    """
    return nlp_en(word).vector

def get_de_word_emb(nlp_de, word):
    """return the embedding of the received German word
    
    Arguments:
        word {str} -- [the word that we want to retrieve the embedding of]
    """
    return nlp_de(word).vector

def get_GloVe_embedding(en_text, de_text):
    """
    Get GloVe embedding for all the words in the dataset
    
    Arguments:
        en_text {list} -- [the English vocabular list]
        de_text {list} -- [the German vocabulary list]
    
    Returns:
        [numpy array] -- [two numpy arrays for the embedding of the two languages]
    """

    nlp_de = spacy.load('de300')
    nlp_en = spacy.load('en300')

    embedding_matrix_en = np.zeros((len(en_text.vocab), 300))
    embedding_matrix_de = np.zeros((len(de_text.vocab), 300))
    
    for i, word in tqdm(enumerate(en_text.vocab.itos)):
        embedding_vector = get_en_word_emb(nlp_en, word)
        embedding_matrix_en[i] = embedding_vector

    for i, word in tqdm(enumerate(de_text.vocab.itos)):
        embedding_vector = get_en_word_emb(nlp_de, word)
        embedding_matrix_de[i] = embedding_vector

    return embedding_matrix_en, embedding_matrix_de

##############################################
##############################################
################ Strategy 3 ##################
##############################################
##############################################

def clean_data_strategy_3():
    """
    The aim of this method is to prepare the data to be trained using Strategy 3
    in this function, we use torchtext to preprocess the data, by removing stopwords and punctuations, building the vocabulary lists
    for both languages, and spliting the training, validation(dev) and testing data
    
    Returns:
        [tuple] -- [tuple of values which includes train, dev, and test bucket iterators]
    """
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
    """
    Find the sentence embedding for all the sentence in the current batch
    
    Arguments:
        batch {batch object} -- [the batch object returned from the bucket iterator to be processed]
    
    Returns:
        [torch tensor] -- [two tensors, one for each language, for both English and German sentences embeddings]
    """
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
