#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math;
import pandas;
import numpy;
import seaborn;
import matplotlib.pyplot as pyplot;
get_ipython().run_line_magic('matplotlib', 'inline')

# Load training data
data = pandas.read_csv("../input/train.csv");
# Load testing data
testData = pandas.read_csv("../input/test.csv");

def scaleFeatures(data, fNames):
    nFeatures = len(fNames);
    i = 0;
    while (i != nFeatures):
        col = data[fNames[i]];
        a = col.min();
        b = col.max();
        data[fNames[i]] = (col - a) / (b - a);
        i += 1;

def createQuadTerms(data, fNames):
    nFeatures = len(fNames);
    i = 0;
    while (i != nFeatures):
        col = data[fNames[i]];
        j = i;
        while (j != nFeatures):
            newName = (fNames[i] + "^2") if (i == j) else (fNames[i] + "*" + fNames[j]);
            data[newName] = data[fNames[i]] * data[fNames[j]];
            j += 1;
        i += 1;
        
def createCubicTerms(data, fNames):
    nFeatures = len(fNames);
    i = 0;
    while (i != nFeatures):
        col = data[fNames[i]];
        data[fNames[i] + "^3"] = col ** 3;
        i += 1;

def transformData(data):
    data = data.rename(
        index = str,
        columns = {
            "Total Volume" : "totalVol",
            "4046" : "N4046",
            "4225" : "N4225",
            "4770" : "N4770",
            "Total Bags": "bags",
            "Small Bags" : "sBags",
            "Large Bags" : "lBags",
            "XLarge Bags" : "xlBags",
        }
    );
    
    def getMainNutrient(x):
        a = x["N4046perVol"];
        b = x["N4225perVol"];
        c = x["N4770perVol"];
        if (a - b > 0.01 and a - c > 0.01):
            return 4046;
        elif (b - a > 0.01 and b - c > 0.01):
            return 4225;
        else:
            return 4770;
    
    data["N4046perVol"] = data[["N4046", "totalVol"]].apply(
        lambda x: (x["N4046"] / x["totalVol"]), axis = 1);
    data["N4225perVol"] = data[["N4225", "totalVol"]].apply(
        lambda x: (x["N4225"] / x["totalVol"]), axis = 1);
    data["N4770perVol"] = data[["N4770", "totalVol"]].apply(
        lambda x: (x["N4770"] / x["totalVol"]), axis = 1);
    data["volPerBag"] = data[["totalVol", "bags"]].apply(
        lambda x: ((x["totalVol"] / x["bags"]) if (x["bags"] != 0) else 0), axis = 1);
    data["sBagRatio"] = data[["sBags", "bags"]].apply(
        lambda x: ((x["sBags"] / x["bags"]) if (x["bags"] != 0) else 0), axis = 1);
    data["lBagRatio"] = data[["lBags", "bags"]].apply(
        lambda x: ((x["lBags"] / x["bags"]) if (x["bags"] != 0) else 0), axis = 1);
    data["xlBagRatio"] = data[["xlBags", "bags"]].apply(
        lambda x: ((x["xlBags"] / x["bags"]) if (x["bags"] != 0) else 0), axis = 1);
    
    scaleFeatures(
        data,
        ["totalVol", "N4046", "N4225", "N4770", "bags", "sBags", "lBags", "xlBags",
         "N4046perVol", "N4225perVol", "N4770perVol", "volPerBag", "sBagRatio",
         "lBagRatio", "xlBagRatio"]
    );
    createQuadTerms(
        data,
        ["N4046perVol", "N4225perVol", "N4770perVol", "volPerBag",
         "sBagRatio", "lBagRatio", "xlBagRatio"],
    );
    createCubicTerms(
        data,
        ["N4046perVol", "N4225perVol", "N4770perVol", "volPerBag",
         "sBagRatio", "lBagRatio", "xlBagRatio"],
    );
    
    data["mainNutrient"] = data[["N4046perVol", "N4225perVol", "N4770perVol"]].apply(
        lambda x: getMainNutrient(x), axis = 1);
    
    return data;

data = transformData(data);
testData = transformData(testData);


# In[2]:





# In[2]:


def createGroups(data, col1, col2, col3):
    vals1 = data[col1].unique();
    vals2 = data[col2].unique();
    vals3 = data[col3].unique();
    
    groups = [];
    nVals1 = len(vals1);
    nVals2 = len(vals2);
    nVals3 = len(vals3);
    
    i = 0;
    while (i != nVals1):
        j = 0;
        while (j != nVals2):
            k = 0;
            while (k != nVals3):
                f = {};
                f[col1] = vals1[i];
                f[col2] = vals2[j];
                f[col3] = vals3[k];
                groups.append({"filter": f});
                k += 1;
            j += 1;
        i += 1;
    
    return groups;


# In[3]:


def rowPassesFilter(data, i, dataFilter):
    for (k, v) in dataFilter.items():
        if (data[k][i] != v):
            return False;
    return True;


# In[4]:


def linearRegression(data, xCols, yCol, dataFilter = None, regParam = 0):
    
    nFeatures = len(xCols);
    nRows = len(data[yCol]);
    
    listA = [];
    listb = [];
    Ai = numpy.zeros(nFeatures + 1, dtype = numpy.float64);
    i = 0;
    
    while (i != nRows):
        if (dataFilter != None):
            if (not rowPassesFilter(data, i, dataFilter)):
                i += 1;
                continue;
        
        # Add the row to the A and b matrices.
        j = 0;
        bi = data[yCol][i];
        Ai[0] = 1;
        while (j != nFeatures):
            Ai[j + 1] = data[xCols[j]][i];
            j += 1;
            
        listA.append(numpy.copy(Ai));
        listb.append(bi);
        i += 1;
        
    if (len(listA) == 0):
        return (None, None, None);
    
    # Create numpy matrices
    nRows = len(listA);
    A = numpy.array(listA);
    b = numpy.array(listb);
    L = regParam * numpy.identity(nFeatures + 1);
    L[0, 0] = 0;

    # Calculate least squares solution
    At = numpy.transpose(A);
    AtAi = numpy.linalg.pinv(numpy.matmul(At, A) + L);
    w = numpy.matmul(AtAi, numpy.matmul(At, b));
    
    # Calculate predicted values
    predictedVals = numpy.zeros(nRows);
    i = 0;
    while (i != nRows):
        predictedVals[i] = numpy.dot(A[i, :], w);
        i += 1;
    
    return (w, predictedVals, b);


# In[5]:


def trainDataset(data, groups, xCols, yCol, regParam):
    nGroups = len(groups);
    i = 0;
    
    while (i != nGroups):
        g = groups[i];
        # Train group
        (w, pred, actual) = linearRegression(data, xCols, yCol, g["filter"], regParam);
        if (w is None):
            g["count"] = 0;
            g["params"] = None;
            i += 1;
            continue;
            
        # Store count and parameters
        g["count"] = len(actual);
        g["params"] = w;
        # Calculate MSE
        e = pred - actual;
        g["mse"] = (1 / len(actual)) * numpy.dot(e, e);
        # Go to next group
        i += 1;


# In[6]:


def predictTestData(testData, groups, xCols):
    nRows = testData.shape[0];
    nGroups = len(groups);
    nFeatures = len(xCols);
    xValues = numpy.zeros(nFeatures + 1, dtype = numpy.float64);
    yValues = numpy.zeros(nRows, dtype = numpy.float64);
    
    i = 0;
    while (i != nGroups):
        groups[i]["testCount"] = 0;
        i += 1;
    
    i = 0;
    while (i != nRows):
        g = None;
        j = 0;
        while (j != nGroups):
            g = groups[j];
            if (rowPassesFilter(testData, i, g["filter"])):
                g["testCount"] += 1;
                break;
            j += 1;
        
        xValues[0] = 1;
        j = 0;
        while (j != nFeatures):
            xValues[j + 1] = testData[xCols[j]][i];
            j += 1;
            
        yValues[i] = numpy.dot(g["params"], xValues);
        i += 1;
    
    return yValues;


# In[7]:


def writeToOutputFile(ids, yValues, outFileName):
    with open(outFileName, "w") as f:
        f.write("id,AveragePrice\n");
        
        nRows = len(yValues);
        i = 0;
        while (i != nRows):
            f.write(str(ids[i]) + "," + str(yValues[i]) + "\n");
            i += 1;


# In[8]:


featureNames = [
    #"N4046",
    #"N4225",
    #"N4770",
    "N4046perVol",
    "N4225perVol",
    "N4770perVol",
    #"sBags",
    #"lBags",
    #"xlBags",
    "totalVol",
    #"type",
    #"year",
    "N4046perVol^2",
    "N4225perVol^2",
    "N4770perVol^2",
    "volPerBag",
    "volPerBag^2",
    "sBagRatio",
    "lBagRatio",
    "xlBagRatio",
    "sBagRatio^2",
    "lBagRatio^2",
    "xlBagRatio^2",
    "N4046perVol*N4225perVol",
    "N4046perVol*N4770perVol",
    "N4225perVol*N4770perVol",
    "sBagRatio*lBagRatio",
    "sBagRatio*xlBagRatio",
    "lBagRatio*xlBagRatio",
    "N4046perVol^3",
    "N4225perVol^3",
    "N4770perVol^3",
    "sBagRatio^3",
    "lBagRatio^3",
    "xlBagRatio^3",
];

groups = createGroups(data, "type", "year", "mainNutrient");
trainDataset(data, groups, featureNames, "AveragePrice", 0.0001);
testY = predictTestData(testData, groups, featureNames);

nGroups = len(groups);
nFeatures = len(featureNames);
i = 0;
while (i != nGroups):
    g = groups[i];
    w = g["params"];
    if (w is None):
        i += 1;
        continue;
    
    print("Group filter = %s, count = %d, testCount = %d"
          % (str(g["filter"]), g["count"], g["testCount"]));
    
    j = 0;
    while (j != nFeatures):
        print("%s = %.6f" % (featureNames[j], w[j + 1]));
        j += 1;
        
    print("MSE = %.6f" % g["mse"]);
    print("");
    i += 1;

outFileName = "Sub1-2016A7PS0282G.csv";
writeToOutputFile(testData["id"], testY, outFileName);

print("Test outputs written to: " + outFileName);

