#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd # data handling
import numpy as np # base of all
import matplotlib.pyplot as plt # plotting
import seaborn as sns  # advance plotting
from wordcloud import WordCloud # to see the words as image
import torch # PyTorch for building Networks
from torchtext.data import Field,LabelField,BucketIterator,TabularDataset # TorchText has Text processing Function
from torchtext import vocab
from sklearn.model_selection import train_test_split # split the data into training and testing
from sklearn.metrics import accuracy_score # accuracy metric
from nltk import word_tokenize # very popular Text processing Library
import random # to perform randomisation of tasks
from tqdm.notebook import tqdm # for a continuous progress bar style
import time # time module 
import os # import operating system


# In[2]:


SEED = 13 # reproducible results: Same results in every run
IN_PATH = '/kaggle/input/'
DATA_PATH = IN_PATH+'jigsaw-multilingual-toxic-comment-classification/' # input directorypath
OUT_PATH = '/kaggle/working/' # path for output directory 
GLOVE_TEXT_PATH = 'glove6b100dtxt/glove.6B.100d.txt' # glove directory
EPOCH = 3 # number of epochs to run for model

np.random.seed(SEED) 
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True  # cuda algorithms
os.environ['PYTHONHASHSEED'] = str(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # use 'cuda' if available else 'cpu'

plt.style.use('seaborn') # use seaborn style plotting

# nltk.download('punkt') # if using first time. Kaggle has all the things already downloaded


# In[3]:


df = pd.read_csv(DATA_PATH+'jigsaw-toxic-comment-train.csv')
df.head()


# In[4]:


df['toxic'].value_counts().plot(kind='pie',autopct='%.2f%%',labels=['Not Toxic','Toxic'],cmap='Set2') 
# distribution of Toxic or Non Toxic. 1 detemines Toxic 
plt.show()


# In[5]:


text = " ".join(comment for comment in df.comment_text) # join all the comments to make a new one big comment
print(f'There are {len(text)} unique words in the whole dataset')

# show the words as an image. It can take lots of time due to large number of words so I am commenting it out
wordcloud = '''WordCloud(max_words=150,background_color="white").generate(text)

fig = plt.figure(figsize=(15,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()'''


# In[6]:


df = df.loc[:,['comment_text','toxic']] # we need just the two columns
df.drop_duplicates(subset=['comment_text'],inplace=True) # drop duplicates and it'll include empty comments too
df.dropna(subset=['comment_text','toxic'],inplace=True) # we do not need any of the columns with empty values
df.head()


# In[7]:


df,test_df = train_test_split(df,test_size=0.25,random_state=SEED,stratify=df['toxic'])
# stratify tries to split in a manner that distribution of 'toxic' is same in both train and test

train_df,val_df = train_test_split(df,test_size=0.20,random_state=SEED,stratify=df['toxic'])

train_df.reset_index(drop=True),val_df.reset_index(drop=True), test_df.reset_index(drop=True)
# split the data while preserving the type of the data. It preserves the original Index so you need to reset
print(f'train_df is of type {type(train_df,).__name__} and is having a shape {train_df.shape}')

# save the dataframes so that we can directly use those from disk by using PyTorch's modules
train_df.to_csv(OUT_PATH+'train.csv',index=False)
val_df.to_csv(OUT_PATH+'val.csv',index=False)
test_df.to_csv(OUT_PATH+'test.csv',index=False)


# In[8]:


text_field = Field(tokenize=word_tokenize)
# tokenize text using word_tokenize and convert to numerical form using default parameters

label_field = LabelField(dtype=torch.float) 
# useful for label string to LabelEncoding. Not useful here but doesn't hurt either

fields = [('comment_text',text_field),('toxic',label_field)] 
# (column name,field object to use on that column) pair for the dictonary

train, val, test = TabularDataset.splits(path=OUT_PATH, train='train.csv',validation='val.csv',test='test.csv', 
                                         format='csv',skip_header=True,fields=fields)


# In[9]:


print(f'Type of "train:" {type(train)}\n Length of "train": {len(train)}\n' )
i = random.randint(0,len(train)) # generate a random index  within the lenth of train
print(f'Keys at index {i} of "train": {train[i].__dict__.keys()}\n')
print("Contents at random index:\n",vars(train.examples[i])) 
# vars is used to see the whole dictonary when the classes or modules have __dict__() used


# In[10]:


text_field.build_vocab(train,max_size=100000) 
label_field.build_vocab(train) 

# words are stored as integers withn the vocab for internal data structure handling. 
# let us look at the {'word':respective_integer} of first 15 

print({k: text_field.vocab.stoi[k] for k in list(text_field.vocab.stoi)[:15]}) 
# this is just pure python code to get first N elements from a a dictonary, as a dictonary


# In[11]:


print(f"Most common 15 words in the vocab are: {text_field.vocab.freqs.most_common(15)}")
# integers i int the second part of each tuples are the frequencies of words in the vocab. They show that how many
# number of times this specific word has apprered in the whole training data set


# In[12]:


train_iter, val_iter, test_iter = BucketIterator.splits((train,val,test), batch_sizes=(32,128,128),
                                              sort_key=lambda x: len(x.comment_text),
                                              sort_within_batch=False,
                                              device=device) # use the cuda device if available


# In[13]:


class Network(torch.nn.Module):
    '''
    It inherits the functionality of Module class from torch.nn whic includes al the layers, weights, grads setup
    and methods to calculate the same. We just need to put in the required layers and describe the flows as
    which layers comes after which one
    '''
    
    def __init__(self,in_neuron,embedding_dim=128,hidden_size=256,out_neuron=1,m_type='rnn',drop=0.53,**kwargs):
        '''
        Constructor of the class which will instantiate the layers while initialisation.
        
        NOTE: Order of the layer defined here has nothing to do wit hthe working. Just like we can define Drouout()
        layer anywhere in ithe init() but actual working depends on the forward() method  as well as the input
        and output shapes. You should be aware of the in,out shapes as the mismatch can produce error.
        
        args:
            in_neuron: input dimensions of the first layer {int}
            embedding_dim: number of latent features you want to calculate from the input data {int} default=128
            hidden_size: neurons you want to have in your hidden RNN layer {int} default=256
            out_neuron: number of outputs you want to have at the end.{int} default=1
            model: whether to use 'rnn' or 'lstm' {string} 
            drop: proportion of values to dropout from the previous values randomly {float 0-1} default=0.53
            **kwargs: any torch.nn.RNN or torch.nn.LSTM args given m_type='rnn' or'lstm' {dict}
        out: 
            return a tensor of shape {batch,out_neuron} as output 
        '''
        super(Network,self).__init__() # call the constructor of Base Class. To know more, please visit the link
        # https://www.programiz.com/python-programming/methods/built-in/super
        self.m_type = m_type
        
        self.embedding = torch.nn.Embedding(in_neuron,embedding_dim) # embedding layer is always the first layer
        
        if self.m_type == 'lstm':
        # whether to use the LSTM type model or the RNN type model. It'll use only 1 in forward()
            self.lstm = torch.nn.LSTM(embedding_dim,hidden_size,**kwargs)
        else:
            self.rnn = torch.nn.RNN(embedding_dim,hidden_size,**kwargs) 
        
        self.dropout = torch.nn.Dropout(drop) # drop the values by random which comes from previous layer
        
        self.dense = torch.nn.Linear(hidden_size,out_neuron) # last fully connected layer
        
    
    def forward(self,t):
        '''
        Activate the forward propagation of a batch at a time to transform the input bath of tensors through
        the different layers to get an out which then will be compared to original label for computing loss.
        args:
            t: tensors in the form of a batch {torch.tensor}
        '''
        # Step:1 pass the incoming tensor to the first layer to get embeddings
        embedding_t = self.embedding(t) # usually we replace the same tensor as t = self.layer(t)
        # input is a "list" sentences where each sentence is a vector of "encoded words" 
        # out.shape = [sentence_length,batch_size,embedding_dimension], in.shape = [sentence_length,batch_size]
        
        # Step 2: Apply dropout
        drop_emb = self.dropout(embedding_t)
        
        # Step 3: Get hidden state and output. It'll use either LSTM or RNN
        if self.m_type == 'lstm':
            out, (hidden_state,_) = self.lstm(drop_emb)
        else:
            out, hidden_state = self.rnn(drop_emb)
            #  shape of rnn_out = (seq_len, batch, num_directions * hidden_size)
       
        # Step 4: Remove the extra axis from Hidden State 
        hidden_squeezed = hidden_state.squeeze(0) 
        # shape of hidden_state = (num_layers * num_directions, batch, hidden_size) = (1*1,b,h) so extra 1 layer
        
        # Step 5: Assert to check. if failed, AssertionError error will be thrown
        assert torch.equal(out[-1,:,:],hidden_squeezed)
        # out_rnn is concatenation of hidden states so squeezed hidden and last value of out_rnn should be equal
        
        # Step 6: Pass the "last" hidden state only because we only want 1 output based on the last hidden state
        return self.dense(hidden_squeezed) # these are not the probabilities. We still need to use an activation  


# In[14]:


def train_network(network,train_iter,optimizer,loss_fn,epoch_num):
    '''
    train the network using given parameters
    args:
        network: any Neural Network object 
        train_batch: iterator of training data
        optimizer: optimizer for gradients calculation and updation
        loss_fn: appropriate loss function
        epoch_num = Epoch number so that it can show which epoch number in tqdm Bar
    out:
        a tuple of (average_loss,average_accuracy) of floating values for a single epoch
    '''
    epoch_loss = 0 # loss per epoch
    epoch_acc = 0 # accuracy per epoch
    
    network.train() # set the model in training mode as it requires gradients calculation and updtion
    # turn off while testing using  model.eval() and torch.no_grad() block
    
    for batch in tqdm(train_iter,f"Epoch: {epoch_num}"): 
        # data will be shown to model in batches per epoch to calculate gradients per batch
        
        optimizer.zero_grad() # clear all the calculated grdients from previous step
        
        predictions = network(batch.comment_text).squeeze(1) # squeeze out the extra dimension [batch_size,1]
        
        loss = loss_fn(predictions,batch.toxic) # calculate loss on the whole batch
        
        pred_classes = torch.round(torch.sigmoid(predictions))
        # sigmoid will convert each output value (which is a single float value for each sentence in batch) to
        # to probability between {0,1}. round is nothing but setting the threshold at 0.5 that if probability 
        # is greater than 0.5, it belongs to one class and if it is less than 0.5, it belongs to other
        
        correct_preds = (pred_classes == batch.toxic).float()
        # get a floating tensors of predicted classes  which match original true class 
        
        accuracy = correct_preds.sum()/len(correct_preds)# it'll be a tensor of shape [1,]
        
        # below two are must and should be used only after calculation of Loss by optimizer
        loss.backward() # Start Back Propagation so that model can calculate gradients based on loss
        optimizer.step() # update the weights based on gradient corresponding to each neuron
        
        epoch_loss += loss.item()  # add the loss for this batch to calculate the loss for whole epoch
        epoch_acc += accuracy.item() # .item() tend to give the exact number from the tensor of shape [1,]
        
        
        time.sleep(0.001) # for tqdm progess bar
        
    return epoch_loss/len(train_iter), epoch_acc/len(train_iter)


# In[15]:


def evaluate_network(network,val_test_iter,optimizer,loss_fn):
    '''
    evaluate the network using given parameters
    args:
        network: any Neural Network object 
        val_test_iter: iterator of validation/test data
        optimizer: optimizer for gradients calculation and updation
        loss_fn: appropriate loss function
    out:
        a tuple of (average_loss,average_accuracy) of floating values for the incoming dataset
    '''
    total_loss = 0  # total loss for the whole incoming data
    total_acc = 0 # total accuracy for the whole data
    
    network.eval() # set the model in evaluation mode to not compute gradients and reduce overhead
    
    with torch.no_grad(): # turn of gradients calculation 
        
        for batch in val_test_iter:

            predictions = network(batch.comment_text).squeeze(1)

            loss = loss_fn(predictions,batch.toxic)

            pred_classes = torch.round(torch.sigmoid(predictions))

            correct_preds = (pred_classes == batch.toxic).float()

            accuracy = correct_preds.sum()/len(correct_preds)

            total_loss += loss.item() 
            total_acc += accuracy.item()

        return total_loss/len(val_test_iter), total_acc/len(val_test_iter)


# In[16]:


in_neuron = len(text_field.vocab)
lr = 3e-4 # learning rate = 0.0003

network = Network(in_neuron) # instantiate the RNN object. other parameters remain default
if torch.cuda.is_available():
    network.cuda() # activate GPU spport

optimizer = torch.optim.Adam(network.parameters(),lr=lr) # use Adam Optimizer
loss_fn = torch.nn.BCEWithLogitsLoss() # Sigmoid activation with Binary Cross Entropy loss. This is more 
# numerically stable than using a plain Sigmoid followed by a BCELoss as, by combining the operations into one 
# layer,we take advantage of the log-sum-exp trick for numerical stability

for epoch in range(EPOCH):
    train_loss, train_acc = train_network(network,train_iter,optimizer,loss_fn,epoch+1)
    val_loss,val_acc = evaluate_network(network,val_iter,optimizer,loss_fn)
    tqdm.write(f'''End of Epoch: {epoch+1}  |  Train Loss: {train_loss:.3f}  |  Val Loss: {val_loss:.3f}  |  Train Acc: {train_acc*100:.2f}%  |  Val Acc: {val_acc*100:.2f}%''')


# In[17]:


network = Network(in_neuron,m_type='lstm') 

if torch.cuda.is_available():
    network.cuda() # activate GPU spport
    
# optimizer and losses remains the same

for epoch in range(EPOCH):
    train_loss, train_acc = train_network(network,train_iter,optimizer,loss_fn,epoch+1)
    val_loss,val_acc = evaluate_network(network,val_iter,optimizer,loss_fn)
    tqdm.write(f'''End of Epoch: {epoch+1}  |  Train Loss: {train_loss:.3f}  |  Val Loss: {val_loss:.3f}  |  Train Acc: {train_acc*100:.2f}%  |  Val Acc: {val_acc*100:.2f}%''')


# In[18]:


class DeepNetwork(torch.nn.Module):
    '''
    Deep RNN Network which can have either one both of stacked and bi-directional properties
    '''
    
    def __init__(self,in_neuron,embedding_dim=100,hidden_size=256,out_neuron=1,m_type='rnn',drop=0.53,**kwargs):
        '''
        Constructor of the class which will instantiate the layers while initialisation.
        
        args:
            in_neuron: input dimensions of the first layer {int}
            embedding_dim: number of latent features you want to calculate from the input data {int} default=100
            hidden_size: neurons you want to have in your hidden RNN layer {int} default=256
            out_neuron: number of outputs you want to have at the end.{int} default=1
            model: whether to use 'rnn','lstm' or 'gru' {string} 
            drop: proportion of values to dropout from the previous values randomly {float 0-1} default=0.53
            **kwargs: any valid torch.nn.RNN, torch.nn.LSTM or torch.nn.GRU args with either 'bidirectional'=True 
                      or 'num_layers'>1
        out: 
            return a tensor of shape {batch,out_neuron} as output 
        '''
        super(DeepNetwork,self).__init__()
        
        self.m_type = m_type
        
        self.embedding = torch.nn.Embedding(in_neuron,embedding_dim)
        
        if self.m_type == 'lstm':
            self.lstm = torch.nn.LSTM(embedding_dim,hidden_size,**kwargs)
        elif self.m_type == 'gru':
            self.gru = torch.nn.GRU(embedding_dim,hidden_size,**kwargs)
        else:
            self.rnn = torch.nn.RNN(embedding_dim,hidden_size,**kwargs) 
        
        self.dropout = torch.nn.Dropout(drop) 
        
        self.dense = torch.nn.Linear(hidden_size*2,out_neuron)
        # Last output Linear Layer will have the two Hidden States from both the directions to have the result
        
    
    def forward(self,t):
        '''
        Activate the forward propagation
        args:
            t: tensors in the form of a batch {torch.tensor}
        '''
        t = self.dropout(self.embedding(t)) # get embeddings and dropout
    
        if self.m_type == 'lstm':
            out, (hidden,_) = self.lstm(t)
        elif self.m_type == 'gru':
            out, hidden = self.gru(t)
        else:
            out, hidden = self.rnn(t)
        # shape of rnn = (seq_len, batch, num_directions * hidden_size)
        
        # Concatenate the last and second last hidden. One is from backward and one is from forward
        t = self.dropout(torch.cat((hidden[-2,:,:],hidden[-1,:,:]),dim=1))
       
        return self.dense(t)


# In[19]:


rnn_kwargs = {'num_layers':2,'bidirectional':True}
in_neuron = len(text_field.vocab)

network = DeepNetwork(in_neuron,m_type='rnn',**rnn_kwargs) 

if torch.cuda.is_available():
    network.cuda() # activate GPU spport
    
# optimizer and losses remains the same

for epoch in range(EPOCH):
    train_loss, train_acc = train_network(network,train_iter,optimizer,loss_fn,epoch+1)
    val_loss,val_acc = evaluate_network(network,val_iter,optimizer,loss_fn)
    tqdm.write(f'''End of Epoch: {epoch+1}  |  Train Loss: {train_loss:.3f}  |  Val Loss: {val_loss:.3f}  |  Train Acc: {train_acc*100:.2f}%  |  Val Acc: {val_acc*100:.2f}%''')


# In[20]:


glove = vocab.Vectors(IN_PATH+GLOVE_TEXT_PATH, OUT_PATH)

print(f'Shape of GloVe vectors is {glove.vectors.shape}')

# glove = vocab.GloVe('6B',100) # You can also use this one to get vocab from internet 
# by default it'll load (300*840Billion) dimensional matrix but we'll use 100D version


# In[21]:


print(f"eminem is represented by the index location at: {glove.stoi['eminem']} and has the following vector values: \n {glove['eminem']}")


# In[22]:


def get_vector(glove,word):
    '''
    Get the vector corresponding to a word from Glove
    args:
        glove: glove embeddings
        word:  any word
    out: a vector of dimensions according to the embedding size. If a word is not present, it returns zero vector
    '''
    return glove[word.lower()]


def find_closest(glove,input_value,n=6,vector=False):
    '''
    Find the closest words to a given word from the embedding
    args:
        glove: glove embeddings
        input_value: {string,vector} any english word or vector representation from embedding
        n: number of closest words to return
        vector: whether input type is a word or a vector
    out:
        tensor of tuple of words and distances
    '''
    if not vector:
        vector = get_vector(glove,input_value) # get vector of the current word
    else:
        vector  = input_value
        
    distances = []
    for neighbour in glove.itos: # get all the words one by one
        dist = torch.dist(vector,get_vector(glove,neighbour)) # calculate distance of all the words to given
        distances.append((neighbour,dist.item()))
        
    sorted_distances = sorted(distances,key=lambda x: x[1]) # sort the value based on tuple's index=1 value
    return sorted_distances[:n] # return top n


def print_neatly(list_of_tuples):
    '''
    Print a tuple cleanly
    args:
        list_of_tuples: List of tuple of 2 values
    '''
    print('Distances \t Words\n')
    for tup in list_of_tuples:
        print('%.3f \t\t %s'%(tup[1],tup[0]))
    return None
        
      
def find_analogy(glove,w1,w11,w2,n=7):
    '''
    Find analogy of the third word given by analogy of two words
    args:
        w1: first word
        w11: analogy of the first word
        w2: second word
        n: number of analogies to find
    out:
        words that can relate to w2 in the same way w11 is related to w1
    '''
    print(f"{w1} : {w11} :: {w2} : ?")
    v1 = get_vector(glove,w1)
    v11 = get_vector(glove,w11)
    v2 = get_vector(glove,w2)
    v21 = (v11-v1)+v2
    
    closest_n = find_closest(glove,v21,n=n+3,vector=True) # find extra 3 so that we can remove the given 3
    
    closest_n = [i for i in closest_n if i[0] not in [w1,w11,w2]][:n]
    return closest_n


# In[23]:


print_neatly(find_closest(glove,'eminem'))


# In[24]:


print_neatly(find_analogy(glove,'eminem','rapper','messi'))


# In[25]:


text_field = Field(tokenize=word_tokenize)
# tokenize text using word_tokenize and convert to numerical form using default parameters

label_field = LabelField(dtype=torch.float) 
# useful for label string to LabelEncoding. Not useful here but doesn't hurt either

fields = [('comment_text',text_field),('toxic',label_field)] 
# (column name,field object to use on that column) pair for the dictonary


train, val, test = TabularDataset.splits(path=OUT_PATH, train='train.csv',validation='val.csv',test='test.csv', 
                                         format='csv',skip_header=True,fields=fields)


text_field.build_vocab(train,max_size=100000,vectors=glove,unk_init=torch.Tensor.zero_) 

# unk_init = torch.tensor.normal_ set the initial vectors of vocab as the glove vectors and  
# initialize unknown words as normal distribution instead of zeros

label_field.build_vocab(train) 


train_iter, val_iter, test_iter = BucketIterator.splits((train,val,test), batch_sizes=(32,128,128),
                                              sort_key=lambda x: len(x.comment_text),
                                              sort_within_batch=False,
                                              device=device) 


# In[26]:


in_neuron = len(text_field.vocab)
embedding_dim = 100 # dimensions of GloVe which we'll use as the dimension for our embedding layer too
drop = 0.0 # how much to drop

loss_fn = torch.nn.BCEWithLogitsLoss() 
lr = 0.0003 #learning rate for optimizer
optimizer = torch.optim.Adam(network.parameters(),lr=lr) 

network = Network(in_neuron,embedding_dim,drop=) 

pretrained_embeddings = text_field.vocab.vectors  # get all the 100000+2 vectors
network.embedding.weight.data.copy_(pretrained_embeddings) #copy embeddings as the weights to the layer


# now we have 2 extra embeddings so we'll have to get their index and change the values at index to zeros

unknown_index = text_field.vocab.stoi[text_field.unk_token] # get index of unknown token
padding_index = text_field.vocab.stoi[text_field.pad_token] # get index of padding token

network.embedding.weight.data[unknown_index] = torch.zeros(embedding_dim) #change values to zeros
network.embedding.weight.data[padding_index] = torch.zeros(embedding_dim)

if torch.cuda.is_available():
    network.cuda()
    # network = network.to(device)


# if you do not want to train your Embedding weights, you'll have to make 1 extra change
# model.embedding.weight.requires_grad = False


# In[27]:


for epoch in range(EPOCH):
    train_loss, train_acc = train_network(network,train_iter,optimizer,loss_fn,epoch+1)
    val_loss,val_acc = evaluate_network(network,val_iter,optimizer,loss_fn)
    tqdm.write(f'''End of Epoch: {epoch+1}  |  Train Loss: {train_loss:.3f}  |  Val Loss: {val_loss:.3f}  |  Train Acc: {train_acc*100:.2f}%  |  Val Acc: {val_acc*100:.2f}%''')


# In[ ]:




