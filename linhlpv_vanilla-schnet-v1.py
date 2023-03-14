#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install ase==3.17 schnetpack==0.2.1')


# In[2]:


get_ipython().system('ls ../input')


# In[3]:


import numpy as np
import pandas as pd
molecules = pd.read_csv('../input/champs-scalar-coupling/structures.csv')
molecules = molecules.groupby('molecule_name')
train = pd.read_csv('../input/champs-scalar-coupling/train.csv')
test = pd.read_csv('../input/champs-scalar-coupling/test.csv')
test['scalar_coupling_constant'] = -1

# coupling_type = '1JHC'

# train = train[train.type == coupling_type]
# test = test[test.type == coupling_type]


# In[4]:


J_type = sorted(train.type.unique())
J_type


# In[5]:


type_map = {}
for i, t in enumerate(J_type): 
    type_map[t] = i

inverse_type_map = {}
for i, t in enumerate(J_type): 
    inverse_type_map[i] = t


# In[6]:


len(train)


# In[7]:


train.head()


# In[8]:


len(test)


# In[9]:


test.head()


# In[10]:


train_scalar_couplings = train.groupby('molecule_name')
test_scalar_couplings = test.groupby('molecule_name')


# In[11]:


from ase import Atoms
from ase.db import connect
from tqdm import *

def create_db(db_path, scalar_couplings, molecule_names):
    with connect(db_path) as db:
        with tqdm(total=len(molecule_names)) as pbar:
            for name in molecule_names:
                mol = molecules.get_group(name)
                atoms = Atoms(symbols=mol.atom.values,
                              positions=[(row.x,row.y,row.z) for row in mol.itertuples()])
                numbers = atoms.get_atomic_numbers()
                group = scalar_couplings.get_group(name)
                ai0 = group.atom_index_0.values
                ai1 = group.atom_index_1.values
                scc = group.scalar_coupling_constant.values
                ids = group.id.values
                types = group.type.values
                for i, j, v, w, t in zip(ai0, ai1, scc, ids, types):
                    new_numbers = numbers.copy()
                    new_numbers[i] = 115 - new_numbers[i]
                    new_numbers[j] = 115 - new_numbers[j]
                    atoms.set_atomic_numbers(new_numbers)
                    data = dict(scc=v)
                    coupling_type = t
                    data['type_id'] = w
                    j_type = type_map[coupling_type]
#                     j_type = np.zeros((8))
#                     j_type[type_map[coupling_type]] = 1
                    data['J_type'] = j_type
                    db.write(atoms, name=name+'_H{}_C{}'.format(i,j), data=data)
                    
                pbar.update()
                


# In[12]:


properties=['J_type', 'type_id']


# In[13]:


import schnetpack

import sys
INT_MAX = sys.maxsize

dataset_size = INT_MAX

dataset_molecule_names = train.molecule_name.unique()
print(len(dataset_molecule_names))


# In[14]:


champs_path = 'CHAMPS_train_J_type.db' 
molecule_names = dataset_molecule_names[:dataset_size]
create_db(db_path=champs_path,
          scalar_couplings=train_scalar_couplings,
          molecule_names=molecule_names[:40000])
dataset = schnetpack.data.AtomsData(champs_path, properties=properties)


# In[15]:


#dataset[30]


# In[16]:


len(dataset)


# In[17]:


dataset_molecule_names = test.molecule_name.unique()
test_champs_path = 'CHAMPS_test_J_type.db' 
test_molecule_names = dataset_molecule_names[:9000]
create_db(db_path=test_champs_path,
          scalar_couplings=test_scalar_couplings,
          molecule_names=test_molecule_names)
test_dataset = schnetpack.data.AtomsData(test_champs_path, properties=properties)


# In[18]:


len(test_dataset)


# In[19]:


test_dataset[0]


# In[20]:


import pandas as pd

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

import schnetpack as spk
import schnetpack.atomistic as atm
import schnetpack.representation as rep
from schnetpack.datasets import *

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device('cpu')
torch.manual_seed(21)
np.random.seed(21)


# In[21]:


# The original function comes from the following script:
# https://github.com/atomistic-machine-learning/schnetpack/blob/v0.2.1/src/scripts/schnetpack_qm9.py
def evaluate_dataset(metrics, model, loader, device):
#     for metric in metrics:
#         metric.reset()
    res = []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device)
                for k, v in batch.items()
            }
            result = model(batch)

            for metric in metrics:
                l = metric(batch, result)
                res.append(np.array(l.detach().cpu().data.numpy()))
    results = np.array(res).mean()
    return results


# In[22]:


import torch.nn as nn
from schnetpack.data import Structure

class MolecularOutput(atm.OutputModule):
    def __init__(self, property_name, n_in=128, n_out=64, aggregation_mode='avg',
                 n_layers=2, n_neurons=None,
                 activation=schnetpack.nn.activations.shifted_softplus,
                 outnet=None):
        super(MolecularOutput, self).__init__(n_in, n_out)
        self.property_name = property_name
        self.n_layers = n_layers
        self.create_graph = False
        
        if outnet is None:
            self.out_net = nn.Sequential(
                schnetpack.nn.base.GetItem('representation'),
                schnetpack.nn.blocks.MLP(n_in, n_out, n_neurons, n_layers, activation)
            )
        else:
            self.out_net = outnet
        
        self.FC1 = schnetpack.nn.blocks.MLP(n_out, 32, n_neurons, 1, activation)
        self.out = schnetpack.nn.blocks.MLP(32, 8, n_neurons, 1, None)
        
        if aggregation_mode == 'sum':
            self.atom_pool = schnetpack.nn.base.Aggregate(axis=1, mean=False)
        elif aggregation_mode == 'avg':
            self.atom_pool = schnetpack.nn.base.Aggregate(axis=1, mean=True)
            
    def forward(self, inputs):
        r"""
        predicts molecular property
        """
        atom_mask = inputs[Structure.atom_mask]

        yi = self.out_net(inputs)
        y = self.atom_pool(yi, atom_mask)
        y = self.FC1(y)
        y = self.out(y)
        result = {self.property_name: y}
        return result


# In[23]:


def schnet_model():
    reps = rep.SchNet(n_atom_basis=128, n_filters=128, n_interactions=6, max_z=115)
    output = MolecularOutput('J_type')
    model = atm.AtomisticModel(reps, output)
    model = model.to(device)
    return model


# In[24]:


def train_model(max_epochs=50):
    # print configuration
    print('max_epochs:', max_epochs)
    
    # split in train and val
    n_dataset = len(dataset)
    n_val = n_dataset // 10
    train_data, val_data, test_data = dataset.create_splits(n_dataset-n_val*2, n_val, 'split')
    train_loader = spk.data.AtomsLoader(train_data, batch_size=128, num_workers=4, shuffle=True)
    val_loader = spk.data.AtomsLoader(val_data, batch_size=128, num_workers=4)

    # create model
    model = schnet_model()

    # create trainer
    output_key = "J_type"
    target_key = "J_type"
    opt = Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler = MultiStepLR(opt, milestones=[15, 320], gamma=0.2)
    def loss(b, p): 
        l = nn.CrossEntropyLoss()
        out = l(p[output_key], b[target_key].view(-1).long())
        return out
    metrics = [loss]

    hooks = [
        spk.train.MaxEpochHook(max_epochs),
#         spk.train.CSVHook('log', loss, every_n_epochs=1),
        spk.train.LRScheduleHook(scheduler),
    ]
    trainer = spk.train.Trainer('output', model, loss,
                                opt, train_loader, val_loader, hooks=hooks)

    # start training
    trainer.train(device)
    
    # evaluation
    model.load_state_dict(torch.load('output/best_model'))
    test_loader = spk.data.AtomsLoader(test_data, batch_size=128, num_workers=4)
    model.eval()

    df = pd.DataFrame()

    df['metric'] = [
        'cross_entropy'
    ]
    df['training'] = evaluate_dataset(metrics, model, train_loader, device)
    df['validation'] = evaluate_dataset(metrics, model, val_loader, device)
    df['test'] = evaluate_dataset(metrics, model, test_loader, device)
    df.to_csv('output/evaluation.csv', index=False)
    display(df)
    
    return test_data


# In[25]:


def show_history():
    df = pd.read_csv('log/log.csv')
    display(df.tail())
    
    _ = display(df[['MAE_scc', 'RMSE_scc']].plot())


# In[26]:


def test_prediction(dataset):
    # create model
    model = schnet_model()
    
    # load best parameters
    model.load_state_dict(torch.load('output/best_model'))
    loader = spk.data.AtomsLoader(dataset, batch_size=128, num_workers=4)
    model.eval()
    
    # predict scalar coupling constants
    entry_id = []
    predictions = []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device)
                for k, v in batch.items()
            }
            result = model(batch)
            _, predicted = torch.max(result['J_type'], 1)
#             print(predicted.shape)
            entry_id += batch['type_id'].long().view(-1).tolist()
            predictions += predicted.view(-1).tolist()
    return entry_id, predictions


# In[27]:


get_ipython().run_cell_magic('time', '', 'used_test_data = train_model(max_epochs=1)')


# In[28]:


split = np.load('split.npz')


# In[29]:


list(split.keys())


# In[30]:


used_test_data =  dataset.create_subset(split['test_idx'])


# In[31]:


def make_submission():
    type_id, J_type = test_prediction(test_dataset)
    submission = pd.DataFrame()
    submission['id'] = type_id
    submission['J_type'] = J_type
    
    return submission


# In[32]:


submission = make_submission()


# In[33]:


submission['J_type'] = submission['J_type'].map(lambda x: inverse_type_map[x])


# In[34]:


display(submission.head())


# In[35]:


submission.to_csv('submission_J_type.csv', index=False)


# In[36]:


display(test[:6956].head())


# In[37]:


from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(test['type'])
p = le.transform(submission['J_type'])
t = le.transform(test[:6956]['type'])


# In[38]:


p


# In[39]:


t


# In[40]:


from sklearn.metrics import accuracy_score
accuracy_score(t, p)

