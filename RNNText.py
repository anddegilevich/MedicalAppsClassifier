import numpy as np
import nltk
import gensim.downloader as api
import torch
import torch.nn as nn
import datasets
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm, trange
import pandas as pd
import sklearn
from google_trans_new import google_translator
import torch.nn.functional as F

data = pd.read_excel('dataNN.xlsx', index_col=0)

N_apps = 600
N_test = 50
N_train = int(N_apps - N_test)

data = data.head(N_apps)
'''translator = google_translator()

for i in range(N_apps):
    translation = translator.translate(data.iloc[i]['description'], lang_tgt='en')
    data.loc[data.index[i], 'description'] = translation'''

data['Type'] = data['Type'] - 1

'''data['Type'] = np.where((data['Type'] >= 0) & (data['Type'] <= 4), 0, data['Type'])
data['Type'] = np.where((data['Type'] >= 5) & (data['Type'] <= 9), 1, data['Type'])
data['Type'] = np.where((data['Type'] >= 10) & (data['Type'] <= 10), 2, data['Type'])
data['Type'] = np.where((data['Type'] >= 11) & (data['Type'] <= 13), 3, data['Type'])
data['Type'] = np.where((data['Type'] >= 14) & (data['Type'] <= 16), 4, data['Type'])
data['Type'] = np.where((data['Type'] >= 17) & (data['Type'] <= 21), 5, data['Type'])'''

data['Type'] = data['Type'].astype('int')

data['Disease']= data['Disease'] - 1
data['Disease'] = data['Disease'].astype('int')

#data = pd.read_pickle('dataRNN.pkl')
data = data.sample(frac=1, replace=True, random_state=9)
#feature = 'Disease'
feature = 'Type'

ntypes = data[feature].nunique()

train_df = pd.DataFrame({'text': data['description'].head(N_train), 'label': data[feature].head(N_train)})
test_df = pd.DataFrame({'text': data['description'].tail(N_test), 'label': data[feature].tail(N_test)})

train= datasets.Dataset.from_dict(train_df)
test= datasets.Dataset.from_dict(test_df)
dataset = datasets.DatasetDict({"train": train, "test": test})

word2vec = api.load('glove-wiki-gigaword-300')

MAX_LENGTH = 256
tokenizer = nltk.WordPunctTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

dataset = dataset.map(
    lambda item: {
        "tokenized": tokenizer.tokenize(item['text'])[:MAX_LENGTH]
    }
)

dataset = dataset.map(
    lambda item: {
          "lemmatized": [lemmatizer.lemmatize(word) for word in item['tokenized']]
    }
)

word2idx = {word: idx for idx, word in enumerate(word2vec.index2word)}

def encode(word):
    if word in word2idx.keys():
        return word2idx[word]
    return word2idx['unk']

dataset = dataset.map(
    lambda item: {
        'features': [encode(word) for word in item['lemmatized']]
    }
)

dataset.remove_columns_(['text','tokenized','lemmatized'])

dataset.set_format(type='torch')

def collate_fn(batch):
    max_len = max(len(row['features']) for row in batch)
    input_embeds = torch.empty((len(batch), max_len), dtype=torch.long)
    labels = torch.empty(len(batch), dtype=torch.long)
    for idx, row in enumerate(batch):
        to_pad = max_len - len(row['features'])
        input_embeds[idx] = torch.cat((row['features'], torch.zeros(to_pad)))
        labels[idx] = row['label']
    return {'features': input_embeds, 'labels':labels}

def collate_fn_pred(batch):
    max_len = max(len(row['features']) for row in batch)
    input_embeds = torch.empty((len(batch), max_len), dtype=torch.long)
    labels = torch.empty(len(batch), dtype=torch.long)
    for idx, row in enumerate(batch):
        to_pad = max_len - len(row['features'])
        input_embeds[idx] = torch.cat((row['features'], torch.zeros(to_pad)))
    return {'features': input_embeds}

loaders = {
    k: DataLoader(
        ds, shuffle=(k == 'train'), batch_size=50, collate_fn=collate_fn
    ) for k, ds in dataset.items()
}

class CNNModel(nn.Module):
    def __init__(self, embed_size, hidden_size, num_classes=ntypes):
        super().__init__()
        self.embed = nn.Embedding(len(word2idx), embedding_dim=embed_size)
        self.cnn = nn.Sequential(
            nn.Conv1d(embed_size, hidden_size, kernel_size=3, padding=2, stride=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=2, stride=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=2, stride=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
        )

        self.cl = nn.Sequential(
            nn.Linear(hidden_size, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.embed(x)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        prediction = self.cl(x)
        return prediction

def training(model, criterion, optimizer, num_epochs, loaders, max_grad_norm=2, num_freeze_iter=25):
    maxAc = 0
    Val_Loss=np.zeros(num_epochs)
    Acc=np.zeros(num_epochs)
    F_value=np.zeros(num_epochs)

    freeze_embeddings(model)
    for e in trange(num_epochs, leave=False):
        model.train()
        num_iter = 0
        pbar = tqdm(loaders['train'], leave=False)
        cur_iter = 0
        for batch in pbar:
            if cur_iter > num_freeze_iter:
                freeze_embeddings(model, True)
            optimizer.zero_grad()
            input_embeds = batch['features'].to(device)
            labels = batch['labels'].to(device)
            prediction = model(input_embeds)
            loss = criterion(prediction, labels)
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            num_iter += 1
        valid_loss = 0
        num_iter = 0
        model.eval()
        with torch.no_grad():
            correct = 0
            num_objs = 0
            F1 = 0
            for batch in loaders['test']:
                input_embeds = batch['features'].to(device)
                labels = batch['labels'].to(device)
                prediction = model(input_embeds)
                valid_loss += criterion(prediction, labels)
                F1 += sklearn.metrics.f1_score(labels.float().numpy(), prediction.argmax(-1).float().numpy(),
                                               average="weighted")
                correct += (labels == prediction.argmax(-1)).float().sum()
                num_objs += len(labels)
                num_iter += 1

        Val_Loss[e] = valid_loss / num_iter
        Acc[e] = correct / num_objs
        F_value[e] = F1 / num_iter

        print(f'Valid Loss: {Val_Loss[e]}, Accuracy: {Acc[e]}, F1: {F_value[e]}')
        if Acc[e] > maxAc:
            maxAc = Acc[e]
            maxF1 = F_value[e]
            emax = e
    input_embeds = batch['features'].to(device)
    labels = batch['labels'].to(device)
    prediction = model(input_embeds)
    sklearn.metrics.confusion_matrix(labels.float().numpy(), prediction.argmax(-1).float().numpy())
    print(f'Max Accuracy: {maxAc}, Max F1: {maxF1}, num_epoh_max: {emax}')
    #np.savetxt('LSTM_Train_Disease_LR-3.txt', (Val_Loss, Acc, F_value))

def freeze_embeddings(model, req_grad=False):
    embeddings = model.embed
    for c_p in embeddings.parameters():
        c_p.requires_grad = req_grad

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):
        patterns = target.shape[0]
        tot = 0
        for b in range(patterns):
            ce_loss = F.cross_entropy(input[b:b + 1, ], target[b:b + 1], reduction=self.reduction, weight=self.weight)
            pt = torch.exp(-ce_loss)
            focal_loss = (1 - pt) ** self.gamma * ce_loss
            tot = tot + focal_loss
        return tot / patterns

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CNNModel(word2vec.vector_size, 50).to(device)
#criterion = FocalLoss(gamma=2, weight=None)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
num_epochs = 200
max_grad_norm = 2

#training(model, criterion, optimizer, num_epochs, loaders, max_grad_norm)

class RNN(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers=2, num_classes=ntypes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(len(word2idx), embedding_dim=embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embed(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = RNN(word2vec.vector_size, 50).to(device)
criterion = FocalLoss(gamma=2, weight=None)
#criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#num_epochs = 180
num_epochs = 180
max_grad_norm = 2

with torch.no_grad():
    for word, idx in word2idx.items():
        if word in word2vec:
            model.embed.weight[idx] = torch.from_numpy(word2vec.get_vector(word))

training(model, criterion, optimizer, num_epochs, loaders, max_grad_norm)

for i in range(1):
    df = pd.read_pickle('dadata/df_%s.pkl' % i)
    Ndf = df.shape[0]

    description_df = pd.DataFrame({'text': df['description']})
    description = datasets.Dataset.from_dict(description_df)

    dataset_pred = datasets.DatasetDict({"test": description})

    dataset_pred = dataset_pred.map(
        lambda item: {
            "tokenized": tokenizer.tokenize(item['text'])[:MAX_LENGTH]
        }
    )

    dataset_pred = dataset_pred.map(
        lambda item: {
            "lemmatized": [lemmatizer.lemmatize(word) for word in item['tokenized']]
        }
    )

    dataset_pred = dataset_pred.map(
        lambda item: {
            'features': [encode(word) for word in item['lemmatized']]
        }
    )

    dataset_pred.remove_columns_(['text', 'tokenized', 'lemmatized'])

    dataset_pred.set_format(type='torch')

    loaders_pred = {
        k: DataLoader(
            ds, shuffle=(k == 'test'), batch_size=50, collate_fn=collate_fn_pred
        ) for k, ds in dataset_pred.items()
    }

    prediction_sum = np.array([])
    for batch in loaders_pred['test']:
        input_embeds = batch['features'].to(device)
        prediction = model(input_embeds)
        prediction = prediction.argmax(-1)
        prediction_sum = np.hstack([prediction_sum, prediction.detach().numpy()])
    np.savetxt('dadata/df_%s_Type.txt' % i, prediction_sum)

for i in range(21,22):
    df = pd.read_pickle('dadata/df_%s.pkl' % i)
    Ndf = df.shape[0]

    description_df = pd.DataFrame({'text': df['description']})
    description = datasets.Dataset.from_dict(description_df)

    dataset_pred = datasets.DatasetDict({"test": description})

    dataset_pred = dataset_pred.map(
        lambda item: {
            "tokenized": tokenizer.tokenize(item['text'])[:MAX_LENGTH]
        }
    )

    dataset_pred = dataset_pred.map(
        lambda item: {
            "lemmatized": [lemmatizer.lemmatize(word) for word in item['tokenized']]
        }
    )

    dataset_pred = dataset_pred.map(
        lambda item: {
            'features': [encode(word) for word in item['lemmatized']]
        }
    )

    dataset_pred.remove_columns_(['text', 'tokenized', 'lemmatized'])

    dataset_pred.set_format(type='torch')

    loaders_pred = {
        k: DataLoader(
            ds, shuffle=(k == 'test'), batch_size=50, collate_fn=collate_fn_pred
        ) for k, ds in dataset_pred.items()
    }

    prediction_sum = np.array([])
    for batch in loaders_pred['test']:
        input_embeds = batch['features'].to(device)
        prediction = model(input_embeds)
        prediction = prediction.argmax(-1)
        prediction_sum = np.hstack([prediction_sum, prediction.detach().numpy()])
    np.savetxt('dadata/df_%s_Type.txt' % i, prediction_sum)