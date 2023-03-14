#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import warnings

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

warnings.simplefilter(action='ignore', category=FutureWarning)  # thx NumPy


# In[2]:


epochs = 10
fixed_length = 19
val_proportion = 0.05
print_every = 50
learning_rate = 1e-4
batch_size = 1024


# In[3]:


Z_train = pd.read_csv('../input/train.csv', index_col='Id', dtype=np.float32)
Z_test = pd.read_csv('../input/test.csv', index_col='Id', dtype=np.float32)


# In[4]:


# See https://www.kaggle.com/c/how-much-did-it-rain-ii/discussion/16622
ref_ = Z_train['Ref'].groupby(level='Id').mean()
Z_train.drop(ref_[ref_.isna()].index, axis=0, inplace=True)

# Just replace all NaN with zero ¯\_(ツ)_/¯
Z_train.fillna(0, inplace=True)
Z_test.fillna(0, inplace=True)


# In[5]:


train_unique_ids = Z_train.index.unique()
train_ids = Z_train.index.unique()[:int((1 - val_proportion) * len(train_unique_ids))]
val_ids = Z_train.index.unique()[int((1 - val_proportion) * len(train_unique_ids)):]

Z_val = Z_train.loc[val_ids]
Z_train = Z_train.loc[train_ids]


# In[6]:


def align_ids(df):
    new_ids = []
    prev_id = df.index[0]

    q = 0
    for _id in df.index:
        q += bool(prev_id - _id)
        prev_id = _id

        new_ids.append(q)
    return new_ids

new_train_ids = align_ids(Z_train)
new_val_ids = align_ids(Z_val)
new_test_ids = align_ids(Z_test)

Z_train.set_index(pd.Index(new_train_ids), inplace=True)
Z_val.set_index(pd.Index(new_val_ids), inplace=True)
Z_test.set_index(pd.Index(new_test_ids), inplace=True)


# In[7]:


def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()

    avg_loss = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if batch_idx % print_every == 0:
            print('Train Epoch: {} [{:.0f}%\tLoss: {:.6f})]'.format(epoch, 100. * batch_idx / len(train_loader), loss.item()))
        avg_loss += loss.item()
    avg_loss /= len(train_loader.dataset)

    print('\nTrain set: Avg. loss: {:.4f}\n'.format(avg_loss))
    
def val(model, device, val_loader, criterion):
    model.eval()

    avg_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            avg_loss += criterion(outputs, labels).item()
        avg_loss /= len(val_loader.dataset)

    print('\nTest set: Avg. loss: {:.4f}\n'.format(avg_loss))

def test(model, device, test_loader):
    model.eval()

    sol = pd.DataFrame(columns=['Id', 'Expected'])
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            partial_sol = pd.DataFrame({'Id': labels.int().numpy(), 'Expected': outputs.cpu().numpy().flatten()})
            sol = sol.append(partial_sol, ignore_index = True)
    return sol.sort_values(by='Id')


# In[8]:


class BidirectionalRNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, 
                 n_layers=5, activation=F.leaky_relu):
        super(BidirectionalRNN, self).__init__()

        kwargs = {'nonlinearity': 'relu', 'batch_first': True, 'bidirectional': True}

        self.n_layers = n_layers
        self.activation = activation

        self.rnn = nn.ModuleList()
        self.forward_linear = nn.ModuleList()
        self.backward_linear = nn.ModuleList()
        self.hidden_linear = nn.ModuleList()

        prev_hi_size = None
        for in_size, hi_size in zip([input_dim] + hidden_dim[:-1], hidden_dim):
            self.rnn.append(nn.RNN(in_size, hi_size, **kwargs))
            self.forward_linear.append(nn.Linear(hi_size, hi_size))
            self.backward_linear.append(nn.Linear(hi_size, hi_size))

            if prev_hi_size is not None:
                self.hidden_linear.append(nn.Linear(prev_hi_size, hi_size))
            prev_hi_size = hi_size

        self.output_linear = nn.Linear(hidden_dim[-1], output_dim)

    def forward(self, x):
        outputs, hidden = x, None

        for idx in range(self.n_layers):
            outputs, hidden = self.rnn[idx](outputs, hidden)

            outputs, output_lengths = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

            outputs_copy = outputs.clone()
            outputs_copy[:, :, :self.rnn[idx].hidden_size] = self.forward_linear[idx](outputs[:, :, self.rnn[idx].hidden_size:])
            outputs_copy[:, :, :self.rnn[idx].hidden_size] = self.activation(outputs[:, :, :self.rnn[idx].hidden_size], negative_slope=0.15)

            outputs_copy[:, :, self.rnn[idx].hidden_size:] = self.backward_linear[idx](outputs[:, :, self.rnn[idx].hidden_size:])
            outputs_copy[:, :, self.rnn[idx].hidden_size:] = self.activation(outputs[:, :, self.rnn[idx].hidden_size:], negative_slope=0.15)

            outputs = outputs_copy[:, :, :self.rnn[idx].hidden_size] + outputs_copy[:, :, self.rnn[idx].hidden_size:]

            if idx < len(self.hidden_linear):
                outputs = nn.utils.rnn.pack_padded_sequence(outputs, output_lengths, batch_first=True)
                hidden = self.hidden_linear[idx](hidden)

        outputs = self.activation(self.output_linear(outputs), negative_slope=0.15).mean(dim=1)
        return outputs


# In[9]:


class RainDataset(data.Dataset):
    def __init__(self, X, y):   
        self.radar_measurements = X
        self.expected = y

    def __len__(self):
        return len(self.radar_measurements.index.unique())

    def __getitem__(self, idx):
        if (type(self.expected.loc[idx]) == np.float32):
            return self.radar_measurements.loc[idx].values.reshape(1, -1), self.expected.loc[idx]
        else:
            return self.radar_measurements.loc[idx].values, self.expected.loc[idx].iloc[0]


# In[10]:


def collate_fn(batch):
    _sorted = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)

    observations, expectations, lengths =     zip(*[(torch.FloatTensor(a), b, a.shape[0]) for (a,b) in _sorted])

    input_tensor = torch.zeros(len(observations), fixed_length, observations[0].size(1)).float()
    for batch_idx in range(input_tensor.size(0)):
        for obs_idx, obs in enumerate(observations[batch_idx]):
            input_tensor[batch_idx, obs_idx, :] = obs

    pack = nn.utils.rnn.pack_padded_sequence(input_tensor, lengths, batch_first=True)
    return pack, torch.FloatTensor(expectations)


# In[11]:


Z_test['Expected'] = Z_test.index + 1  # Hacked

X, y = Z_train.drop('Expected', axis=1), Z_train['Expected']
train_dataset = RainDataset(X, y)
X, y = Z_val.drop('Expected', axis=1), Z_val['Expected']
val_dataset = RainDataset(X, y)
X, y = Z_test.drop('Expected', axis=1), Z_test['Expected'].astype(np.float32)
test_dataset = RainDataset(X, y)


# In[12]:


train_loader = data.DataLoader(train_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
val_loader = data.DataLoader(val_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
test_loader = data.DataLoader(test_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)


# In[13]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BidirectionalRNN(input_dim=22, output_dim=1, hidden_dim=[64, 128, 256, 128, 64]).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.L1Loss()

for epoch in range(epochs):
    train(model, device, train_loader, criterion, optimizer, epoch)
    val(model, device, val_loader, criterion)


# In[14]:


submission = test(model, device, test_loader)
submission.to_csv('submission.csv', index=False)

