#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os

import gc


# In[2]:


DATA_ROOT = '../input/'
ORIGINAL_DATA_FOLDER = os.path.join(DATA_ROOT, 'movie-review-sentiment-analysis-kernels-only')
TMP_DATA_FOLDER = os.path.join(DATA_ROOT, 'kaggle_review_sentiment_tmp_data')


# In[3]:


train_data_path = os.path.join(ORIGINAL_DATA_FOLDER, 'train.tsv')
test_data_path = os.path.join(ORIGINAL_DATA_FOLDER, 'test.tsv')
sub_data_path = os.path.join(ORIGINAL_DATA_FOLDER, 'sampleSubmission.csv')

train_df = pd.read_csv(train_data_path, sep="\t")
test_df = pd.read_csv(test_data_path, sep="\t")
sub_df = pd.read_csv(sub_data_path, sep=",")


# In[4]:


import seaborn as sns
from sklearn.feature_extraction import text as sktext


# In[5]:


train_df.head()


# In[6]:


test_df.head()


# In[7]:


sub_df.head()


# In[8]:


overlapped = pd.merge(train_df[["Phrase", "Sentiment"]], test_df, on="Phrase", how="inner")
overlap_boolean_mask_test = test_df['Phrase'].isin(overlapped['Phrase'])


# In[9]:


print("training and testing data sentences hist:")
sns.distplot(train_df['SentenceId'], kde_kws={"label": "train"})
sns.distplot(test_df['SentenceId'], kde_kws={"label": "test"})


# In[10]:


print("The number of overlapped SentenceId between training and testing data:")
train_overlapped_sentence_id_df = train_df[train_df['SentenceId'].isin(test_df['SentenceId'])]
print(train_overlapped_sentence_id_df.shape[0])

del train_overlapped_sentence_id_df
gc.collect()


# In[11]:


pd.options.display.max_colwidth = 250
print("Example of sentence and phrases: ")

sample_sentence_id = train_df.sample(1)['SentenceId'].values[0]
sample_sentence_group_df = train_df[train_df['SentenceId'] == sample_sentence_id]
sample_sentence_group_df


# In[12]:


from keras.preprocessing import sequence
import gensim
from sklearn import preprocessing as skp


# In[13]:


max_len = 50
embed_size = 300
max_features = 30000


# In[14]:


class TreeNode:
    
    def __init__(self, left=None, right=None, phrase_id=None, phrase=None, sentiment=None):
        self.left = left
        self.right = right
        self.phrase_id = phrase_id
        self.sentiment = sentiment
        self.phrase = phrase
        
    @classmethod
    def build_preorder_tree(cls, df):
        phrase_ids = df['PhraseId'].values.tolist()
        phrases = df['Phrase'].values.tolist()
        
        if 'Sentiment' in df.columns:
            sentiments = df['Sentiment'].values.tolist()
        else:
            sentiments = None
        
        return TreeNode.__build_preorder_tree(phrases, phrase_ids, sentiments, 0, len(phrases)-1)
        
    @classmethod
    def __build_preorder_tree(cls, phrases, phrase_ids, sentiments, lo, hi):
        if lo > hi:
            return None
        root = TreeNode(
            phrase_id=phrase_ids[lo], phrase=phrases[lo].lower(), 
            sentiment=(2 if sentiments is None else sentiments[lo])
        )
        if lo == hi:
            root = TreeNode.__split_double_node(root)
            return root
        
        left_lo = lo + 1
        
        right_lo = lo + 2
        while(right_lo < len(phrases) and phrases[right_lo].lower() in phrases[left_lo].lower()):
            right_lo += 1
        
        root.left = TreeNode.__build_preorder_tree(phrases, phrase_ids, sentiments, left_lo, right_lo - 1)
        root.right = TreeNode.__build_preorder_tree(phrases, phrase_ids, sentiments, right_lo, hi)
        
        if root.left is not None and root.right is None:
            if root.phrase.startswith(root.left.phrase):
                end_index = root.phrase.rindex(root.left.phrase) + len(root.left.phrase)
                root.right = TreeNode(
                    phrase_id=None, phrase=root.phrase[end_index:].strip(), 
                    sentiment=2
                )
                TreeNode.__split_double_node(root.right)
            else:
                start_index = root.phrase.find(root.left.phrase)
                root.right = root.left
                root.left = TreeNode(
                    phrase_id=None, phrase=root.phrase[:start_index].strip(), 
                    sentiment=2
                )
                TreeNode.__split_double_node(root.left)
                
        return root
    
    @classmethod
    def __split_double_node(cls, root):
        splits = root.phrase.strip().split()
        if len(splits) == 0:
            return None
        if len(splits) == 1:
            return root
        elif len(splits) >= 2 and len(splits) < 11:
            root.left = TreeNode(
                phrase_id=None, phrase=splits[0].lower(), 
                sentiment=2
            )
            root.right = TreeNode(
                phrase_id=None, phrase=splits[1].lower(), 
                sentiment=2
            )
            return root
        else:
            raise ValueError(root.phrase)


# In[15]:


def build_sent_id_tree_map(raw_df):
    sent_id_tree_map = dict()
    for sent_id in raw_df['SentenceId'].unique():
        df = raw_df[raw_df['SentenceId'] == sent_id]
        sent_id_tree_map[sent_id] = TreeNode.build_preorder_tree(df)
    
    return sent_id_tree_map

train_trees = build_sent_id_tree_map(train_df)
test_trees = build_sent_id_tree_map(test_df)


# In[16]:


def ptb_flatten_tree(root):
    if root.left is None and root.right is None:
        return '(' + str(root.sentiment) + ' ' + root.phrase + ')'
    
    return '(' + str(root.sentiment) + ' ' + ptb_flatten_tree(root.left) + ' ' + ptb_flatten_tree(root.right) + ')'

train_trees = dict([(sent_id, ptb_flatten_tree(tree)) for sent_id, tree in train_trees.items()])
test_trees = dict([(sent_id, ptb_flatten_tree(tree)) for sent_id, tree in test_trees.items()])


# In[17]:


train_trees_df = pd.Series(train_trees)
print(train_df[train_df['SentenceId']==4054].iloc[0]['Phrase'])
train_trees_df[4054]


# In[18]:


print(train_df[train_df['Phrase']=='they are few and far between'])
print(train_df[train_df['SentenceId']==899])


# In[19]:


class Node:  # a node in the tree
    def __init__(self, label, word=None):
        self.label = label
        self.word = word
        self.parent = None  # reference to parent
        self.left = None  # reference to left child
        self.right = None  # reference to right child
        # true if I am a leaf (could have probably derived this from if I have
        # a word)
        self.isLeaf = False
        # true if we have finished performing fowardprop on this node (note,
        # there are many ways to implement the recursion.. some might not
        # require this flag)

    def __str__(self):
        if self.isLeaf:
            return '[{0}:{1}]'.format(self.word, self.label)
        return '({0} <- [{1}:{2}] -> {3})'.format(self.left, self.word, self.label, self.right)


class Tree:

    def __init__(self, treeString, openChar='(', closeChar=')'):
        tokens = []
        self.open = '('
        self.close = ')'
        for toks in treeString.strip().split():
            tokens += list(toks)
        self.root = self.parse(tokens)
        # get list of labels as obtained through a post-order traversal
        self.labels = get_labels(self.root)
        self.num_words = len(self.labels)

    def parse(self, tokens, parent=None):
        assert tokens[0] == self.open, "Malformed tree"
        assert tokens[-1] == self.close, "Malformed tree"

        split = 2  # position after open and label
        countOpen = countClose = 0

        if tokens[split] == self.open:
            countOpen += 1
            split += 1
        # Find where left child and right child split
        while countOpen != countClose:
            if tokens[split] == self.open:
                countOpen += 1
            if tokens[split] == self.close:
                countClose += 1
            split += 1

        # New node
        node = Node(int(tokens[1]))  # zero index labels

        node.parent = parent

        # leaf Node
        if countOpen == 0:
            node.word = ''.join(tokens[2: -1]).lower()  # lower case?
            node.isLeaf = True
            return node

        node.left = self.parse(tokens[2: split], parent=node)
        node.right = self.parse(tokens[split: -1], parent=node)

        return node

    def get_words(self):
        leaves = getLeaves(self.root)
        words = [node.word for node in leaves]
        return words

def get_labels(node):
    if node is None:
        return []
    return get_labels(node.left) + get_labels(node.right) + [node.label]

def getLeaves(node):
    if node is None:
        return []
    if node.isLeaf:
        return [node]
    else:
        return getLeaves(node.left) + getLeaves(node.right)

    
def loadTrees(trees_df):
    """
    Loads training trees. Maps leaf node words to word ids.
    """
    trees = [Tree(l) for l in trees_df.values.tolist()]

    return trees


# In[20]:


train_data = loadTrees(train_trees_df)


# In[21]:


flatten = lambda l: [item for sublist in l for item in sublist]
vocab = list(set(flatten([t.get_words() for t in train_data])))

word2index = {'<UNK>': 0}
for vo in vocab:
    if word2index.get(vo) is None:
        word2index[vo] = len(word2index)
        
index2word = {v:k for k, v in word2index.items()}


# In[22]:


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F


# In[23]:


class RNTN(nn.Module):
    
    def __init__(self, word2index, hidden_size, output_size):
        super(RNTN,self).__init__()
        
        self.word2index = word2index
        self.embed = nn.Embedding(len(word2index), hidden_size)
#         self.V = nn.ModuleList([nn.Linear(hidden_size*2,hidden_size*2) for _ in range(hidden_size)])
#         self.W = nn.Linear(hidden_size*2,hidden_size)
        self.V = nn.ParameterList([nn.Parameter(torch.randn(hidden_size * 2, hidden_size * 2)) for _ in range(hidden_size)]) # Tensor
        self.W = nn.Parameter(torch.randn(hidden_size * 2, hidden_size))
        self.b = nn.Parameter(torch.randn(1, hidden_size))
#         self.W_out = nn.Parameter(torch.randn(hidden_size,output_size))
        self.W_out = nn.Linear(hidden_size, output_size)
        
    def init_weight(self):
        nn.init.xavier_uniform(self.embed.state_dict()['weight'])
        nn.init.xavier_uniform(self.W_out.state_dict()['weight'])
        for param in self.V.parameters():
            nn.init.xavier_uniform(param)
        nn.init.xavier_uniform(self.W)
        self.b.data.fill_(0)
#         nn.init.xavier_uniform(self.W_out)
        
    def tree_propagation(self, node):
        
        recursive_tensor = OrderedDict()
        current = None
        if node.isLeaf:
            tensor = Variable(LongTensor([self.word2index[node.word]])) if node.word in self.word2index.keys()                           else Variable(LongTensor([self.word2index['<UNK>']]))
            current = self.embed(tensor) # 1xD
        else:
            recursive_tensor.update(self.tree_propagation(node.left))
            recursive_tensor.update(self.tree_propagation(node.right))
            
            concated = torch.cat([recursive_tensor[node.left], recursive_tensor[node.right]], 1) # 1x2D
            xVx = [] 
            for i, v in enumerate(self.V):
#                 xVx.append(torch.matmul(v(concated),concated.transpose(0,1)))
                xVx.append(torch.matmul(torch.matmul(concated, v), concated.transpose(0, 1)))
            
            xVx = torch.cat(xVx, 1) # 1xD
#             Wx = self.W(concated)
            Wx = torch.matmul(concated, self.W) # 1xD

            current = torch.tanh(xVx + Wx + self.b) # 1xD
        recursive_tensor[node] = current
        return recursive_tensor
        
    def forward(self, Trees, root_only=False):
        
        propagated = []
        if not isinstance(Trees, list):
            Trees = [Trees]
            
        for Tree in Trees:
            recursive_tensor = self.tree_propagation(Tree.root)
            if root_only:
                recursive_tensor = recursive_tensor[Tree.root]
                propagated.append(recursive_tensor)
            else:
                recursive_tensor = [tensor for node,tensor in recursive_tensor.items()]
                propagated.extend(recursive_tensor)
        
        propagated = torch.cat(propagated) # (num_of_node in batch, D)
        
#         return F.log_softmax(propagated.matmul(self.W_out))
        return F.log_softmax(self.W_out(propagated),1)


# In[24]:


import matplotlib.pyplot as plt
from IPython.display import SVG
import random
from collections import Counter, OrderedDict


# In[25]:


HIDDEN_SIZE = 30
ROOT_ONLY = False
BATCH_SIZE = 20
EPOCH = 20
LR = 0.01
LAMBDA = 1e-5
RESCHEDULED = False
USE_CUDA = True


# In[26]:


model = RNTN(word2index, HIDDEN_SIZE,5)
model.init_weight()
if USE_CUDA:
    model = model.cuda()

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# In[27]:


FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor

def getBatch(batch_size, train_data):
    random.shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        batch = train_data[sindex: eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch
    
    if eindex >= len(train_data):
        batch = train_data[sindex:]
        yield batch

for epoch in range(EPOCH):
    losses = []
    
    # learning rate annealing
    if RESCHEDULED == False and epoch == EPOCH//2:
        LR *= 0.1
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=LAMBDA) # L2 norm
        RESCHEDULED = True
    
    for i, batch in enumerate(getBatch(BATCH_SIZE, train_data)):
        
        if ROOT_ONLY:
            labels = [tree.labels[-1] for tree in batch]
            labels = Variable(LongTensor(labels))
        else:
            labels = [tree.labels for tree in batch]
            labels = Variable(LongTensor(flatten(labels)))
        
        model.zero_grad()
        preds = model(batch, ROOT_ONLY)
        
        loss = loss_function(preds, labels)
        losses.append(loss.data.tolist())
        
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print('[%d/%d] mean_loss : %.2f' % (epoch, EPOCH, np.mean(losses)))
            losses = []


# In[28]:


test_data = loadTrees(test_trees_df)


# In[29]:


accuracy = 0
num_node = 0


# In[30]:


for test in test_data:
    model.zero_grad()
    preds = model(test, ROOT_ONLY)
    labels = test.labels[-1:] if ROOT_ONLY else test.labels
    for pred, label in zip(preds.max(1)[1].data.tolist(), labels):
        num_node += 1
        if pred == label:
            accuracy += 1

print(accuracy/num_node * 100)

