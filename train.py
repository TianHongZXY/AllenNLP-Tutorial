import re
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.nlimodel import nlimodel
from data_preprocess.dataloader import classifier_dataset, collate_fn
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer


def clean_en_corpus(corpus):
    # wnl = WordNetLemmatizer()
    stemmer = SnowballStemmer("english")
    pat_letter = re.compile(r'[^a-zA-Z \']+')
    new_text = pat_letter.sub(' ', corpus).strip().lower()
    new_words = new_text.split()
    new_num = len(new_words)
    # for i in range(new_num):
        # new_words[i] = wnl.lemmatize(word=new_words[i])
        # new_words[i] = stemmer.stem(word=new_words[i])

    return new_text, new_words

def read_data():
    df = pd.read_csv('/Users/tianhongzxy/Downloads/contradictory-my-dear-watson/train.csv')
    text = []
    label = []
    # print(df.head())
    # print(df[['premise', 'hypothesis', 'label', 'lang_abv']][df['lang_abv']=='en'])
    for ins in df[['premise', 'hypothesis', 'label', 'lang_abv']][df['lang_abv']=='en'].values:
        p_text, p_words = clean_en_corpus(ins[0])
        h_text, h_words = clean_en_corpus(ins[1])
        text.append(p_words + ['[SEP]'] + h_words)
        label.append(ins[2])
    return text, label

def build_vocab(text):
    word2id = {'[PAD]': 0, '[SEP]':1}
    id2word = {0: '[PAD]', 1: '[SEP]'}
    idx = 2
    for ins in text:
        for word in ins:
            if word not in word2id:
                word2id[word] = idx
                id2word[idx] = word
                idx += 1
    return word2id, id2word

def tokenizer(vocab, text):
    result = []
    for ins in text:
        result.append([vocab[w] for w in ins])
    return result

if __name__ == '__main__':
    text, label = read_data()
    word2id, id2word = build_vocab(text)
    text = tokenizer(word2id, text)
    print(len(word2id))
    train_data = classifier_dataset(text=text, label=label)
    # print(len(train_data))
    data_loader = DataLoader(dataset=train_data,
                             batch_size=8,
                             collate_fn=collate_fn)
    model = nlimodel(vocab_size=len(word2id),
                     embedding_dim=10,
                     hidden_size=10,
                     output_size=3)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    max_epoch = 50
    for i in range(max_epoch):
        correct = 0
        total = 0
        batch_idx = 0
        for batch in data_loader:
            batch_idx += 1
            optimizer.zero_grad()
            batch_x, batch_x_len, batch_y = batch
            logits = model(batch_x, batch_x_len)
            total += len(batch_y)
            correct += torch.sum( (torch.argmax(logits, dim=-1) == batch_y).int() ).float()
            loss = loss_fn(logits, batch_y)

            loss.backward()
            optimizer.step()
            if batch_idx == 858:
                print('batch_id: {} current acc {}'.format(batch_idx, correct / total))