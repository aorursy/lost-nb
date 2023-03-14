#!/usr/bin/env python
# coding: utf-8

# In[1]:


# apply bayesian model for calculating the probability of sample
def BayesClassify0(testData,dicSubsetMean,dicSubsetVar,dicSubsetPbi):
	dicTestDataBayes = {}
	n = np.shape(testData)[1]
	#
	for key in dicSubsetMean.keys():
		dataMean = dicSubsetMean[key][0]
		dataVar = dicSubsetVar[key][0]
		PBi = dicSubsetPbi[key]
		testData = testData[0]
		PABi = [0]
		for j in range(n):
			if dataVar[0,j]==0:
				break
			PAjBi = np.exp(-pow((testData[0,j]-dataMean[0,j]),2)/(2*dataVar[0,j]))/np.sqrt(2*3.1415*dataVar[0,j])
			if PAjBi>0:
				PABi.append(math.log(PAjBi,2))
		PBiA = sum(PABi) + PBi
		dicTestDataBayes[key] = PBiA
	#
	sortPBiA = sorted(dicTestDataBayes.iteritems(),key=operator.itemgetter(1),reverse=True)
	return sortPBiA[0][0]


# In[2]:


# PCA
def pca(dataSet,topNfeat=9999):
	meanVals = np.mean(dataSet,axis=0)
	dataRemoved = dataSet-meanVals
	covMat = np.cov(dataRemoved,rowvar=0)
	#
	eigVals,eigVects = np.linalg.eig(np.mat(covMat))
	eigValInd = np.argsort(eigVals)
	eigValInd = eigValInd[:-(topNfeat+1):-1]
	redEigVecs = eigVects[:,eigValInd]
	#
	reduceSet = dataRemoved*redEigVecs
	return reduceSet,eigVals,redEigVecs

def contribution(eigVals,tops):
	total = sum(eigVals)
	for i in range(tops):
		conYi = eigVals[i]/total
		print "No.",i+1," contribution rate:",conYi
	sconYi = sum(eigVals[:i+1])/total
	print "the top ",i+1," contribution rate:",sconYi,"\n"

