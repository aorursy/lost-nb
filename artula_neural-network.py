#!/usr/bin/env python
# coding: utf-8

# In[1]:


# The code of the neural network


# In[2]:


from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from scipy import misc
import torch.nn as nn
import skimage as sk
import pylab as plt
import numpy as np
import argparse
import random
import torch
import time
import math
import glob


# In[3]:


# The final neural architecture
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		# Convolution layers
		self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
		self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
		self.conv4 = nn.Conv2d(64, 256, kernel_size=3)
		self.bn0   = nn.BatchNorm2d(3)
		self.bn1   = nn.BatchNorm2d(16)
		self.bn2   = nn.BatchNorm2d(32)
		self.bn3   = nn.BatchNorm2d(64)
		self.bn4   = nn.BatchNorm2d(256)
		# Fully connected layers
		self.fc1 = nn.Conv2d(256, 128, kernel_size=1)
		self.fc2 = nn.Conv2d(128, 64, kernel_size=1)
		self.bnfc1   = nn.BatchNorm2d(128)
		self.bnfc2   = nn.BatchNorm2d(64)
		# Separate classifiers
		# age
		self.age1 = nn.Conv2d(64, 32, kernel_size=1)
		self.age2 = nn.Conv2d(32, 6, kernel_size=1)
		self.bnage = nn.BatchNorm2d(32)
		# gender
		self.gender1 = nn.Conv2d(64, 32, kernel_size=1)
		self.gender2 = nn.Conv2d(32, 3, kernel_size=1)
		self.bngender = nn.BatchNorm2d(32)
		# type
		self.type1 = nn.Conv2d(64, 32, kernel_size=1)
		self.type2 = nn.Conv2d(32, 2, kernel_size=1)
		self.bntype = nn.BatchNorm2d(32)
		# breed
		self.breed1 = nn.Conv2d(64, 32, kernel_size=1)
		self.breed2 = nn.Conv2d(32, 2, kernel_size=1)
		self.bnbreed = nn.BatchNorm2d(32)
		# color1
		self.color1 = nn.Conv2d(64, 32, kernel_size=1)
		self.color2 = nn.Conv2d(32, 7, kernel_size=1)
		self.bncolor = nn.BatchNorm2d(32)
		# color2
		self.color12 = nn.Conv2d(64, 32, kernel_size=1)
		self.color22 = nn.Conv2d(32, 8, kernel_size=1)
		self.bncolor2 = nn.BatchNorm2d(32)
		# color3
		self.color13 = nn.Conv2d(64, 32, kernel_size=1)
		self.color23 = nn.Conv2d(32, 8, kernel_size=1)
		self.bncolor3 = nn.BatchNorm2d(32)
		# maturity size
		self.mat1 = nn.Conv2d(64, 32, kernel_size=1)
		self.mat2 = nn.Conv2d(32, 4, kernel_size=1)
		self.bnmat = nn.BatchNorm2d(32)
		# fur length
		self.fur1 = nn.Conv2d(64, 32, kernel_size=1)
		self.fur2 = nn.Conv2d(32, 3, kernel_size=1)
		self.bnfur = nn.BatchNorm2d(32)
		# adaption speed
		self.speed1 = nn.Conv2d(64, 32, kernel_size=1)
		self.speed2 = nn.Conv2d(32, 5, kernel_size=1)
		self.bnspeed = nn.BatchNorm2d(32)
	def forward(self, x):
		x = F.relu(self.getHidden(x))
		# To separate classifiers
		# age
		a = F.relu(self.bnage(self.age1(x)))
		a = F.softmax(self.age2(a).view(-1, 6), dim=1)
		# gender
		b = F.relu(self.bngender(self.gender1(x)))
		b = F.softmax(self.gender2(b).view(-1, 3), dim=1)
		# type
		c = F.relu(self.bntype(self.type1(x)))
		c = F.softmax(self.type2(c).view(-1, 2), dim=1)
		# breed
		d = F.relu(self.bnbreed(self.breed1(x)))
		d = F.softmax(self.breed2(d).view(-1, 2), dim=1)
		# color1
		e = F.relu(self.bncolor(self.color1(x)))
		e = F.softmax(self.color2(e).view(-1, 7), dim=1)
		# adaption speed
		f = F.relu(self.bnspeed(self.speed1(x)))
		f = F.softmax(self.speed2(f).view(-1, 5), dim=1)
		# color2
		g = F.relu(self.bncolor2(self.color12(x)))
		g = F.softmax(self.color22(g).view(-1, 8), dim=1)
		# color3
		h = F.relu(self.bncolor3(self.color13(x)))
		h = F.softmax(self.color23(h).view(-1, 8), dim=1)
		# maturity size
		i = F.relu(self.bnmat(self.mat1(x)))
		i = F.softmax(self.mat2(i).view(-1, 4), dim=1)
		# fur length
		j = F.relu(self.bnfur(self.fur1(x)))
		j = F.softmax(self.fur2(j).view(-1, 3), dim=1)
		return x.view(-1,64),a,b,c,d,e,f,g,h,i,j
	def getHidden(self, x):
		x = self.bn0(x)
		x = F.relu(F.max_pool2d(self.bn1(self.conv1(x)),2))
		x = F.relu(F.max_pool2d(self.bn2(self.conv2(x)),2))
		x = F.relu(F.max_pool2d(self.bn3(self.conv3(x)),2))
		x = self.bn4(F.relu(self.conv4(x)))
		# colapse
		x = F.max_pool2d(x, 20,20)
		# fully connected
		x = F.relu(self.bnfc1(self.fc1(x)))
		x = self.bnfc2(self.fc2(x))
		return x


# In[4]:


# Set seeds 
seed = 1
random.seed(seed)
torch.manual_seed(seed)
	
# init model and optimizer
model = Net()
optimizer = optim.Adam(model.parameters())


# In[5]:


# Load datalocations and create train and validation set
train = glob.glob('trans/*')
random.shuffle(train)
print(len(train))
trainsize = int((0.8)*len(train))
valset = train[trainsize:]
trainset = train[:trainsize]


# In[6]:


# Function that runs the model on a batch and return the loss
# can be used for training and evaluating
def run(model, batch, labels, weights):
	output = model(batch)
	loss = 0
	for i in xrange(len(labels)):
		loss += F.binary_cross_entropy(output[i+1], labels[i], weight=weights[i])
	return loss

# Function that shows the progress of training
def show(avg=25, fname='log.txt'):
	b = open(fname).read().split('\n')[:-1]
	b = [float(i)for i in b]
	b = [sum(b[i:i+avg])/float(avg) for i in xrange(len(b)-avg)]
	plt.plot(b)
	plt.show()


# In[7]:


# Generates a batch and label set for training or evaluating
def genBatch(bs=1, tset=trainset):
	which = [random.randint(0, len(tset)-1)for i in xrange(bs)]
	ids = [tset[i].split('/')[1] for i in which]
	#misc.imsave('cur.png', misc.imread(tset[which[0]]))
	ids = [i[:i.index('-')]for i in ids]
	dset = [misc.imread(tset[i]) for i in which] # Red in image, after that prepare for neural input
	dset = [augment(i)for i in dset]
	dset = [i.transpose((2,0,1))for i in dset]
	dset = [Variable(torch.Tensor(i).float().contiguous().view(1,3,200,200))for i in dset]
	dset = torch.cat(dset) # == input
	labels = [torch.zeros(bs, 6), # Make one hot vector for all outputs
			  torch.zeros(bs, 3),
			  torch.zeros(bs, 2),
			  torch.zeros(bs, 2),
			  torch.zeros(bs, 7),
			  torch.zeros(bs, 5),
			  torch.zeros(bs, 8),
			  torch.zeros(bs, 8),
			  torch.zeros(bs, 4),
			  torch.zeros(bs, 3),]
    # make one hot vector
	for i in xrange(bs):
		val = values[ids[i]]
		for k in xrange(len(val)):
			labels[k][i][val[k]] = 1
		#print val
	#print labels
	labels = [Variable(i)for i in labels]
	return dset, labels


# In[8]:


# Transforms the photos to neural embeddings
def transformToHidden(model, tset):
  model.eval()
# The file with ids 
  names = open('emb_ids.txt', 'w')
  embeddings = np.zeros((len(tset), 64))
# the file for th eembedding to save in
  csv = open('trainset.txt', 'w')
  print (len(tset))
  for img in xrange(0, len(tset)): # Go over all images
	ids = tset[img].split('/')[1][:-4]
	ids2 = ids[:ids.index('-')]
	val = values[ids2][5]
	if img%100==0:print(img)
	dset = misc.imread(tset[img]) # read in image after that prepare it for neural input
	dset = dset.transpose((2,0,1))
	dset = Variable(torch.Tensor(dset).float().contiguous().view(1,3,200,200))
	hidden = model.getHidden(dset).data.view(-1,64).numpy()
	names.write(ids+'\n')
	embeddings[img] = hidden[0]
	[csv.write(str(i)+',') for i in hidden[0]]
	csv.write(str(val)+'\n')
  csv.close()
  np.save('embeddings', embeddings)
  model.train()


# In[9]:


# Generate the weights for class inbalance compensation
# This by randomly sample from training set (not validation set)
def genWeights(dset, samplesize):
	b,ll = genBatch(100, dset)
	ll = [i.data for i in ll]
	for i in xrange(samplesize-1):
		b,l = genBatch(100, dset)
		for i in xrange(len(l)):
			ll[i] += l[i].data
	for i in xrange(len(ll)):
		for j in xrange(len(ll[i])):
			ll[i][j] = 1./(ll[i][j]+1)
		ll[i] = Variable(ll[i].sum(0))
	return ll


# In[10]:


# Augment the data with random image flip and some rotation
def augment(img):
		img = misc.imrotate(img, random.uniform(-30,30)) 
		if random.random()<.5: img = img[:, ::-1]
		# random crop?
		return img.copy()


# In[11]:


# Function to test the model on the validation set
def testscore(model, bs=64):
  model.eval()
  totscore = 0
  tel = 0.0
  for batch in xrange(0, len(valset)-bs, bs):
	which = range(batch,batch+bs)
	ids = [valset[i].split('/')[1] for i in which]
	ids = [i[:i.index('-')]for i in ids]
	dset = [misc.imread(valset[i]) for i in which]
	#dset = [augment(i)for i in dset]
	dset = [i.transpose((2,0,1))for i in dset]
	dset = [Variable(torch.Tensor(i).float().contiguous().view(1,3,200,200))for i in dset]
	dset = torch.cat(dset)
	labels = [torch.zeros(bs, 6),
			  torch.zeros(bs, 3),
			  torch.zeros(bs, 2),
			  torch.zeros(bs, 2),
			  torch.zeros(bs, 7),
			  torch.zeros(bs, 5),
			  torch.zeros(bs, 8),
			  torch.zeros(bs, 8),
			  torch.zeros(bs, 4),
			  torch.zeros(bs, 3),]
	for i in xrange(bs):
		val = values[ids[i]]
		for k in xrange(len(val)):
			labels[k][i][val[k]] = 1
		#print val
	#print (labels)
	labels = [Variable(i)for i in labels]
	empty = [Variable(torch.ones(i.shape))for i in weights]
	loss = run(model, dset, labels, empty).data.numpy()
	totscore += loss
	tel += 1
  model.train()
  return int(totscore)/tel


# In[12]:


# Function that returns the confusion matrix for all the 
# outputs
def confusionMatrix():
 model = torch.load('eval_model')
 bs=64
 model.eval()
 l1 = [np.zeros((i,i))for i in [6,3,2,2,7,5,8,8,4,3]]
# Go over the complete validation set
 for batch in xrange(0, len(valset)-bs, bs):
 	print batch
	which = range(batch,batch+bs)
	ids = [valset[i].split('/')[1] for i in which]
	ids = [i[:i.index('-')]for i in ids]
	dset = [misc.imread(valset[i]) for i in which]
	#dset = [augment(i)for i in dset]
	dset = [i.transpose((2,0,1))for i in dset]
	dset = [Variable(torch.Tensor(i).float().contiguous().view(1,3,200,200))for i in dset]
	dset = torch.cat(dset) # == batch of input
	labels = [torch.zeros(bs, 6), # create one hot vector
			  torch.zeros(bs, 3),
			  torch.zeros(bs, 2),
			  torch.zeros(bs, 2),
			  torch.zeros(bs, 7),
			  torch.zeros(bs, 5),
			  torch.zeros(bs, 8),
			  torch.zeros(bs, 8),
			  torch.zeros(bs, 4),
			  torch.zeros(bs, 3),]
	for i in xrange(bs): # fill one hot vector
		val = values[ids[i]]
		for k in xrange(len(val)):
			labels[k][i][val[k]] = 1
		#print(val)
	#print() labels)
	labels = [i.numpy() for i in labels]
	o = model(dset) # output of the model
	o=[i.data.numpy()for i in o]
    # Fill the confusion matrixes
	for lab in xrange(len(labels)):
		for sub in xrange(o[lab+1].shape[0]):
			l1[lab][labels[lab][sub].argmax(), o[lab+1][sub].argmax()] += 1
 return l1


# In[13]:


# The final training function
def startTraining():
	# import pretrained model if possible
	if 0:
		try:
			model = torch.load('eval_model')
			log = open('log.txt', 'a')
		except: pass
	else:
		try:
			log = open('log_oud.txt', 'w')
			log = log.write(open('log.txt').read())
			log.close()
			oud = torch.load('model')
			torch.save(oud, 'model_oud')
		except: pass
		log = open('log.txt', 'w')
		
	model.train()
	# The model was sometimed stopped 
    # to prevent to get the exact same batches twice
    # a new seed is introduced
	random.seed(int(time.time()))
	torch.manual_seed(int(time.time()))
	
    
    total_loss = 50 # start score
	evaluate = 5 # after how many batches to print score
	batchsize = 64 # nmb of images per time trained
	
	startValScore = 10e9 # Startscore that the network needs to improve
	for time in xrange(1, 10**12):
		optimizer.zero_grad()
		batch, labels = genBatch(batchsize)
		loss = run(model, batch, labels, weights)
		loss.backward()
		optimizer.step()
		log.write(str(loss.data.numpy()[0])+'\n')
		if time % evaluate == 0:
			total_loss = total_loss/evaluate
			print( time, total_loss )
			total_loss = loss.data.numpy()[0]
		else: 
			total_loss += loss.data.numpy()[0]
		if time % 100 == 0:
			#test()
			torch.save(model,'model')
			log.close()
			log = open('log.txt', 'a')
		if time % 100 == 0:
			evalscore = testscore(model)
			if evalscore < startValScore:
				print()
				print( 'New break!', evalscore)
				startValScore = evalscore
				torch.save(model, 'eval_model')
			else:
				print()
				print( 'Eval score:', evalscore, startValScore)
			model.train()


# In[14]:


# Function that convert photos to fixed size
def make_smaller(location):
	try:
		os.mkdir('trans2') # Location to save in, so action have to be performed only once
	except: pass
	
	photos = glob.glob('%s/test_images/*' % (location))
	for i in photos:
		id1 = i.split('/')[1]
		id2 = id1[:id1.index('-')] # Finding petid 
		img = misc.imread(i) # read in image
		shape = img.shape
		m = float(max(shape))
		shape = [int(ii/m *200) for ii in shape]
	
		img = misc.imresize(img, shape[:2]) # Change shape image
		newimg = np.zeros((200,200,3))
		try:	
			newimg[:shape[0], :shape[1]] = img
		except:
			print i
			for k in xrange(3):
				newimg[:shape[0], :shape[1], k] = img
		#misc.imshow(newimg)
		misc.imsave('trans2/%s' % (id1), newimg.astype('uint8')) # save image

