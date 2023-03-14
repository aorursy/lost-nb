#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import math
import time


# In[3]:


# configure
inputdir = '../input/ciphertext-challenge-iii/'
outputdir = '../input/mycipher3/'


# In[4]:


dftrain = pd.read_csv(inputdir+'train.csv')
dftest = pd.read_csv(inputdir+'test.csv')
print(dftrain.shape)
print(dftest.shape)
# about 100,000 counts (train set, test set) same count.


# In[5]:


dftrain.head()
# plaintext_id, text, index


# In[6]:


dftest.head()
# ciphertext_id, ciphertext, difficulty


# In[7]:


# count by difficulty level  (1~4)
fig, axs = plt.subplots(1,2)
sns.countplot(dftest['difficulty'], ax=axs[0])
dftest['difficulty'].value_counts().plot.pie(ax=axs[1])
plt.show()
# Each level's count is almost same.


# In[8]:


# get text length and space length
dftrain['length'] = dftrain.text.apply(len)
dftest['length'] = dftest.ciphertext.apply(len)
dftrain['space'] = [x.count(' ') for x in dftrain.text]
dftest['space'] = [x.count(' ') for x in dftest.ciphertext]


# In[9]:


dftrain.head()


# In[10]:


dftest.head()


# In[11]:


# split text data by difficulty level.
dftest1 = dftest.loc[dftest['difficulty']==1]
dftest2 = dftest.loc[dftest['difficulty']==2]
dftest3 = dftest.loc[dftest['difficulty']==3]
dftest4 = dftest.loc[dftest['difficulty']==4]
print(len(dftest1))
dftest1.head()


# In[12]:


# count records by length (train data) plaintext
sns.countplot(x="length", data=dftrain)
# sns.countplot(x="length", data=dftrain, order=dftrain.length.value_counts().sort_index().index)
# sns.countplot(x="length", data=dftrain, order=dftrain.length.value_counts().sort_values().index)


# In[13]:


# count records by length (test data) ciphertext
sns.countplot('length', data=dftest)


# In[14]:


# count plot by level 
fig, ax = plt.subplots(1,4, figsize=(4*4, 4))
sns.countplot(x='length', data=dftest1, ax=ax[0])
sns.countplot(x='length', data=dftest2, ax=ax[1])
sns.countplot(x='length', data=dftest3, ax=ax[2])
sns.countplot(x='length', data=dftest4, ax=ax[3])
fig.show()


# In[15]:


print(dftest1.length.value_counts())


# In[16]:


# all letters (train set) plaintext
alltext = ''.join(dftrain.text)
len(alltext)  # 4580000 letters


# In[17]:


# sample plaintext
alltext[:1000]


# In[18]:


# count by letter
atcounter = Counter(alltext)
dftrainletter = pd.DataFrame( [ [x[0], x[1]] for x in atcounter.items() ], columns=['Letter', 'Count'] )
# order by letter
dftrainletter.sort_values('Letter', inplace=True)
sns.barplot(dftrainletter['Letter'], dftrainletter['Count'])
plt.title('Plain Text')
# space is the most frequency.


# In[19]:


# how is the distribution of the alphabet?
# maybe E>T>A order in dictionary.
alphaCount = dftrainletter.loc[ np.bitwise_and(dftrainletter.Letter>='a' , dftrainletter.Letter<='z') ]
alphaCount.sort_values('Letter', inplace=True)
sns.barplot(alphaCount['Letter'], alphaCount['Count'])
plt.title('Plain Text Lowercase Alphabet Count')
# e t o a 


# In[20]:


# case insensitive.
alphaCountU = dftrainletter.loc[ np.bitwise_and(dftrainletter.Letter>='A' , dftrainletter.Letter<='Z') ]
alphaCountU = alphaCountU.sort_values('Letter')
sns.barplot(alphaCountU['Letter'], alphaCountU['Count'])
plt.title('Plain Text Alphabet Uppercae Count')
# plaintext에 대문자 'Z'가 없다. 


# In[21]:


if ( np.sum(alphaCountU.Letter=='Z')==0 ):
    alphaCountU = alphaCountU.append([{'Letter':'Z', 'Count':0}])

dfAlphaNocase = pd.DataFrame({'Letter':alphaCount['Letter'].values, 'Count':alphaCount.Count.values + alphaCountU.Count.values})
sns.barplot(dfAlphaNocase['Letter'], dfAlphaNocase['Count'])
plt.title('Plain Text Alphabet Count (Case insensitive)')
plt.show()


# In[22]:


dfAlphaNocase.sort_values('Count', ascending=False).iloc[:5]
# e >> t o a order!


# In[23]:


# all cipher text
allcipher = ''.join(dftest.ciphertext)

# level 1 cipher text
allcipher1 = ''.join(dftest1.ciphertext)
print( 'length=' , len(allcipher1) )
print( 'ciphers=', allcipher1[:1000] )


# In[24]:


# count letter 
accounter1 = Counter(allcipher1)
dftestletter1 = pd.DataFrame( [ [x[0], x[1]] for x in accounter1.items() ], columns=['Letter', 'Count'] )
# order by letter
dftestletter1 = dftestletter1.sort_values('Letter')
sns.barplot(dftestletter1['Letter'], dftestletter1['Count'])
plt.title('Cipher Level1')
# 특이하게 level1 ciphertext에도 대문자 'Z'가 없음.


# In[25]:


# letter count. train data  and level1 ciphertext data
# order by count. (letter is not shared) 
dftrainletter = dftrainletter.sort_values(by='Count', ascending=False) 
dftestletter1 = dftestletter1.sort_values(by='Count', ascending=False) 

f, ax = plt.subplots(figsize=(15,5))
plt.bar(np.array(range(len(dftrainletter))), dftrainletter['Count'].values , alpha=0.5, color='blue')
plt.bar(np.array(range(len(dftestletter1))), dftestletter1['Count'].values , alpha=0.5, color='red')
plt.show()


# In[26]:


# letter count (all ciphertext)
accounter = Counter(allcipher)
dftestletter = pd.DataFrame( [ [x[0], x[1]] for x in accounter.items() ], columns=['Letter', 'Count'] )
dftestletter = dftestletter.sort_values('Letter')
sns.barplot(dftestletter['Letter'], dftestletter['Count'])
plt.title('Cipher All')


# In[27]:


dftrainletter = dftrainletter.sort_values(by='Count', ascending=False) # 내림차순 정렬
dftestletter = dftestletter.sort_values(by='Count', ascending=False) # 내림차순 정렬

f, ax = plt.subplots(figsize=(15,5))
plt.bar(np.array(range(len(dftrainletter))), dftrainletter['Count'].values , alpha=0.5, color='blue')
plt.bar(np.array(range(len(dftestletter))), dftestletter['Count'].values , alpha=0.5, color='red')
plt.show()
# ciphertext letter is more than plaintext


# In[28]:


tmpa = dftestletter1.Letter.values
print(np.sort(tmpa))
## level 1 cipher text letter에는 대문자 'Z'가 없음. plain text에도 Z가 없음.
# 즉, 'Z'는 사용 안됨? 변환이 어떤 거로든 알 수 없음???.


# In[29]:


tmpa = dftrainletter.Letter.values
print(np.sort(tmpa))
## plaintext의 도메인은 level 1 cipher text의 문자 셋 도메인과 일치함.


# In[30]:


dftest1.length.sort_values(ascending=False).head()


# In[31]:


# the length of index 45272 is 500. (only one.)
print(dftest1.loc[45272])
print(dftest1.loc[45272].ciphertext)
# ciphertext_id=ID_6100247c5, space count is 81.


# In[32]:


# ciphertext length=50 ==> plaintext length : 401~500
dftrain.loc[np.bitwise_and(dftrain.length.values<=500,dftrain.length.values>400)]
# maybe one of these three items is the plaintext of the ciphertext(45272)
# first one is the same space 81.


# In[33]:


''' compare two string. letter freq
order by letter
t1, t2 : compare string
l1, l2 : labels
'''
def compare_letter(t1, t2, l1=None, l2=None):
    print('compare length=', len(t1), len(t2))
    alpha = [ chr(ord('a')+i) for i in range(26) ]
    alpha.extend([ chr(ord('A')+i) for i in range(26) ] )
    alpha.extend([' ', "'", ',', '-' '.', ':', '!', '$', '(', ')', '?', '[', ']'])
    alpha.extend([ str(i) for i in range(10)])
    
    ''' letter set in cipher test level 1 and plain text. (same) : missing 'Z'!!!
    ' ' '!' '$' "'" '(' ')' ',' '-' '.' '0' '1' '2' '3' '4' '5' '6' '7' '8'
 '9' ':' '?' 'A' 'B' 'C' 'D' 'E' 'F' 'G' 'H' 'I' 'J' 'K' 'L' 'M' 'N' 'O'
 'P' 'Q' 'R' 'S' 'T' 'U' 'V' 'W' 'X' 'Y' '[' ']' 'a' 'b' 'c' 'd' 'e' 'f'
 'g' 'h' 'i' 'j' 'k' 'l' 'm' 'n' 'o' 'p' 'q' 'r' 's' 't' 'u' 'v' 'w' 'x'
 'y' 'z'
    '''
    lettercnt = len(alpha)
    counter1 = Counter(t1)
    dfc1 = pd.DataFrame( {'Letter':alpha, 'Count':np.zeros(lettercnt, dtype=int)})
    for x in counter1.items():
        dfc1.loc[dfc1.Letter == x[0], 'Count'] = x[1]
    dfc1 = dfc1.sort_values('Letter')

    counter2 = Counter(t2)
    dfc2 = pd.DataFrame( {'Letter':alpha, 'Count':np.zeros(lettercnt, dtype=int)})
    for x in counter2.items():
        dfc2.loc[dfc2.Letter == x[0], 'Count'] = x[1]
    dfc2 = dfc2.sort_values('Letter')
    
    plt.figure(figsize=(12,8))
    plt.xticks(rotation=70)
    sns.barplot(dfc1['Letter'], dfc1['Count'], color='blue', alpha=0.5, label=l1)
    sns.barplot(dfc2['Letter'], dfc2['Count'], color='red', alpha=0.5, label=l2)
    plt.legend()
    plt.show()


# In[34]:


''' compare two string. letter freq
order by frequency
'''
def compare_freq(t1, t2, l1=None, l2=None):
    print('compare length=', len(t1), len(t2))
    alpha = [ chr(ord('a')+i) for i in range(26) ]
    alpha.extend([ chr(ord('A')+i) for i in range(26) ] )
    alpha.extend([' ', "'", ',', '-' '.', ':', '!', '$', '(', ')', '?', '[', ']'])
    alpha.extend([ str(i) for i in range(10)])
    lettercnt = len(alpha)
    
    counter1 = Counter(t1)
    dfc1 = pd.DataFrame( {'Letter':alpha, 'Count':np.zeros(lettercnt, dtype=int)})
    for x in counter1.items():
        dfc1.loc[dfc1.Letter == x[0], 'Count'] = x[1]
    dfc1 = dfc1.sort_values('Count', ascending=False)

    counter2 = Counter(t2)
    dfc2 = pd.DataFrame( {'Letter':alpha, 'Count':np.zeros(lettercnt, dtype=int)})
    for x in counter2.items():
        dfc2.loc[dfc2.Letter == x[0], 'Count'] = x[1]
    dfc2 = dfc2.sort_values('Count', ascending=False)
    
    plt.figure(figsize=(12, 8))
    xvalue = list(range(lettercnt))
    plt.xticks(rotation=70)
    sns.barplot(xvalue, dfc1['Count'], color='blue', alpha=0.5, label=l1)
    sns.barplot(xvalue, dfc2['Count'], color='red', alpha=0.5, label=l2)
    plt.legend()
    plt.show()


# In[35]:


compare_letter(dftest1.loc[45272].ciphertext, dftrain.loc[13862].text, 'cipher', 'plain')
# some left parts are same. space, .. 


# In[36]:


compare_freq(dftest1.loc[45272].ciphertext, dftrain.loc[13862].text, 'cipher', 'plain')


# In[37]:


# how many these letter 
print(dftest1.loc[45272].ciphertext.count(' '))
print(dftrain.loc[13862].text.count(' '))
print(dftest1.loc[45272].ciphertext.count('\''))
print(dftrain.loc[13862].text.count('\''))
print(dftest1.loc[45272].ciphertext.count(','))
print(dftrain.loc[13862].text.count(','))
print(dftest1.loc[45272].ciphertext.count('-'))
print(dftrain.loc[13862].text.count('-'))
print(dftest1.loc[45272].ciphertext.count('.'))
print(dftrain.loc[13862].text.count('.'))


# In[38]:


compare_letter(dftest1.loc[45272].ciphertext, dftrain.loc[67817].text)


# In[39]:


compare_letter(dftest1.loc[45272].ciphertext, dftrain.loc[104540].text)


# In[40]:


plt.figure(figsize=(6*2, 6))
plt.subplot(1,2,1)
plt.xticks(rotation=90)
sns.countplot('space', data=dftrain)
plt.subplot(1,2,2)
sns.countplot('space', data=dftest1)
plt.show()


# In[41]:


# found one!
print('plain:\n', dftrain.loc[13862].text)
print('cipher level1:\n', dftest1.loc[45272].ciphertext)
# find the pattern!!!! simliarity...
# word length, special characters, space, ... upper/lower case. 


# In[42]:


# Special characters and space is not changed.
# Uppercase letter to uppercase. and lowercase letter to lowercase 
# paddings are two part. front part and end part.
# word count is same.


# In[43]:


import re
def print2(p, c):
    difflen = len(c)-len(p)
    lpadsize = difflen//2
    c=re.sub('[A-Z]', 'A', c)
    p=re.sub('[A-Z]', 'A', p)
    c=re.sub('[a-z]', 'a', c)
    p=re.sub('[a-z]', 'a', p)
    print(c)
    print(' '*lpadsize+p)
    


# In[44]:


print2(dftrain.loc[13862].text, dftest1.loc[45272].ciphertext )


# In[45]:


# Get special characters. uppercase to 'A'. lowercase to 'a'
# try plaintext and cipher level1.


# In[46]:


def get_plaintext(pid):
    return dftrain.loc[ dftrain['plaintext_id']==pid ].text.values[0]
def get_ciphertext(cid):
    return dftest1.loc[ dftest1['ciphertext_id']==cid ].ciphertext.values[0]


def whitelen(s):
    cnt=0
    for n in s:
        if not n.isalpha():
            cnt+=1
        elif n in ['z', 'Z']:
            cnt+=1
    return cnt

def get_keyindex(ptext, ctext):
    tlen = len(ptext)
    encsize = math.ceil(tlen/100)*100
    padsize = encsize - tlen
    padsizel = padsize//2
    padl = ctext[:padsizel]
    wl = whitelen(padl)
#     val = (ord(ctext[padsizel]) - ord(ptext[0])) % 25
    ki = (padsizel-wl)%4
    kimap = [15, 24, 11, 4]
    return kimap[ki]

def print_comparet(ptext, ctext):
#     print('plaintext_id={} ciphertext_id={}'.format(pid, cid))
#     ptext = get_plaintext(pid)
#     ctext = get_ciphertext(cid)
    tlen = len(ptext)
    encsize = math.ceil(tlen/100)*100
    padsize = encsize - tlen
    padsizel = padsize//2
    padsizer = padsize - padsizel
    padl = ctext[:padsizel]
    print(' '*padsizel+ptext)
    print(ctext)
    wl = whitelen(padl)
    val = (ord(ctext[padsizel]) - ord(ptext[0])) % 25
    kimap = {4:3, 15:0, 24:1, 11:2 }
    print(padl, 'lpadsize=', padsizel, 'whitelen=', wl, 'first diff=', val)
    if val not in [4,15,24,11]:
        print('not found key')
        return
    print('lpad-white len mod=',(padsizel-wl)%4 , 'kimap(first diff)=', kimap[val])


# In[47]:


def decrypt_level1(c):
    keys = [15, 24, 11, 4]
    kimap = {4:3, 15:0, 24:1, 11:2}
    ki = 0
#     for ki in range(4):
    plain=""
#     print('start key index=', ki)
    for l in c:
        if l>='a' and l<='y':
            pl = chr((ord(l)-ord('a') - keys[ki%4])%25+ord('a'))
            ki+=1
        elif l>='A' and l<='Y':
            pl = chr((ord(l)-ord('A') - keys[ki%4])%25+ord('A'))
            ki+=1
        else:
            pl = l
        plain+=pl
#     print(ki,'  ',plain)
    return plain  


# In[48]:


print_comparet(dftrain.loc[13862].text, dftest1.loc[45272].ciphertext )
print( chr((ord('P')-ord('A')+11)%25+ord('A')) )


# In[49]:


decrypt_level1(dftest1.loc[45272].ciphertext)


# In[50]:


cnt_dftest1 = dftest1.shape[0]
dectext = []
for i in range(cnt_dftest1):
    ciphertext = dftest1.iloc[i].ciphertext
    dec = decrypt_level1(ciphertext)
    dectext.append(dec)
dftest1["Dec"] = dectext    


# In[51]:


dftest1.head()


# In[52]:


dict_train=dict()
for ind, row in dftrain.iterrows():
    text = row['text']
    dict_train[text]=row['index']


# In[53]:


cnt_dftrain = dftrain.shape[0]
cnt_dftest1 = dftest1.shape[0]
pid_list=[]
cid_notfound=[]
for i in range(cnt_dftest1):
    dec = dftest1.iloc[i].Dec
    bfound = False
    for padsize in range(100):
        lpadsize = padsize//2
        rpadsize = padsize-lpadsize
        decpart = dec[lpadsize:len(dec)-rpadsize]
        if decpart in dict_train:
            idx = dict_train[decpart]
#             print('plain idx=', idx, 'padsize=', padsize, dftrain.loc[idx].text)
            bfound = True
            pid_list.append(idx)
            break
    if bfound==False:
        print('not found dftest1 iloc=', i, 'dec=', dec)
        pid_list.append(0)
        cid_notfound.append(i)    


# In[54]:


print( len(pid_list), len(cid_notfound))


# In[55]:


pid_list = np.asarray(pid_list, dtype=int)
result=pd.read_csv(inputdir+'sample_submission.csv')


# In[56]:


result.set_index('ciphertext_id', inplace=True)


# In[57]:


result.head()


# In[58]:


for i, cid in enumerate(dftest1.ciphertext_id.values):
    result.loc[cid,"index"] = pid_list[i]


# In[59]:


result.head()

result.reset_index()
# In[60]:


result.head()


# In[61]:


result.to_csv('result.csv')


# In[62]:



def is_white(n):
    if n==' ' or n==',' or n==':' or n=='.' or n=='?' or n=='$' or n=='\'' or             n=='(' or n==')' or n=='[' or n==']':
        return True
    return False

def include_white(s):
    for x in s:
        if is_white(x):
            return True
    return False

def get_sp(text):
    sp=""
#     temp = text.lstrip().rstrip()
    for n in text:
#         if not n.isalnum()
#             sp+=n
#         if not n.isalnum() and n!='(' and n!=')' and n!='[' and n!=']' and n!='$' and n!=':' and n!='\'' and n!='-':
# . , 
#     alpha.extend([' ', "'", ',', '-' '.', ':', '!', '$', '(', ')', '?', '[', ']'])
#         if n==' ' or n==',' or n==':':
# not work letter :   ! -  
        if is_white(n) :
            sp+=n
        elif n.islower():
            sp+='a'
        elif n.isupper():
            sp+='A'
        else:
            sp+='X'  # unknown
#             print('unknown : ', n)
    return sp 


# In[63]:


# check to match pattern.
# e ; cipher text
# p ; plain text
def isSamePattern(e, p, bdebug=False):
    # e에는 패딩이 숨이있다. 길이는 자체로는 알 수 없다. 비교시에 알 수 있다.
    encsize = math.ceil(len(p)/100)*100
    padsize = encsize-len(p)
    # check length match
    if encsize!=len(e):
        if bdebug:
            print('length error: len(p)={} encsize={}, len(e)={}'.format(len(p), encsize, len(e)))
        return False
    if bdebug:
        print(e)
        print(p)
        print(encsize, len(e), len(p))
    lpad = padsize//2
    rpad = padsize-lpad
    if bdebug:
        print(padsize, lpad, rpad)
    lpadblock = e[0:lpad]
    rpadblock = e[encsize-rpad:]
    if len(lpadblock)>0:
        if lpadblock[-1]==' ':
            return False
    if len(rpadblock)>0:
        if rpadblock[0]==' ':
            return False
    if bdebug:
        print(lpadblock)
        print(rpadblock)
#     if include_white(lpadblock):
#         return False
#     if include_white(rpadblock):
#         return False
    encblock = e[lpad:encsize-rpad]
    if bdebug:
        print(p)
        print(encblock)
    sp_e = get_sp(encblock)
    sp_p = get_sp(p)
    if bdebug:
        print(sp_e)
        print(sp_p)
    return sp_e==sp_p


# In[64]:


# 5 ciphertext_id= ID_ac57b8817
# found plain_id= ['ID_a5bac1c5c', 'ID_1f08db396', 'ID_991f7a466', 'ID_d1ad40723']
c=dftest1.loc[ dftest1.ciphertext_id=='ID_ac57b8817']['ciphertext'].values[0]
t=dftrain.loc[dftrain.plaintext_id=='ID_1f08db396']['text'].values[0]
ret = isSamePattern(c,t,True)
print('isSamePattern?', ret)


# In[65]:


# x1=get_sp(dftrain.loc[13862].text)
# x2=get_sp(dftest1.loc[45272].ciphertext)
# print(x1)
# print(x2)
# print(x1==x2)
print(isSamePattern(dftest1.loc[45272].ciphertext,dftrain.loc[13862].text, True))


# In[66]:


if False:
    temp1=dftest1.ciphertext.apply(get_sp)
    dftest1["sp"]=temp1.copy()
    print(dftest1.head())

    temp2 = dftrain.text.apply(get_sp)
    dftrain["sp"]=temp2.copy()
    print(dftrain.head())

    print(len(dftest1.sp))
    dftrain.head()


# In[67]:


print(dftrain.loc[13862].text)
print(dftest1.loc[45272].ciphertext)
cipher = dftest1.iloc[2].ciphertext
clen = len(cipher)
print(cipher)
dftrainflag=np.bitwise_and(dftrain['length']>(clen-100), dftrain['length']<=clen)
print('trainlength=', np.sum(dftrainflag))
for pid, plain in zip(dftrain.loc[dftrainflag]['plaintext_id'], 
                      dftrain.loc[dftrainflag]['text']):
#             print(cipher, plain)
    if isSamePattern(cipher, plain):
        print('found:', pid)
        print(plain)
#         break
        


# In[68]:


# dftrain.head()
# dftrain.loc[ np.bitwise_and(dftrain['length']>100, dftrain['length']<=200) ] 


# In[69]:


# search sp pattern
foundpair1 = [] # test1 idx, train idx
notfoundpair1=[]
foundcnt=0
dftest1_count = dftest1.shape[0]
dftrain_count = dftrain.shape[0]
print('test1 count=', dftest1_count, 'train count=', dftrain_count)

if not os.path.exists(outputdir+'foundpair1.csv'):
    foundcnt=0
    i2=0
    time_s = time.time()
    for (ciphertext_id, cipher) in zip(dftest1.ciphertext_id.values, dftest1.ciphertext.values):
        matched=[]
        matchedidx=[]
        i2+=1
        # comment below, if u want to get all answers. Too long time... 
#         if i2==20 :  # debug 
#             break
        time_s1 = time.time()
        print(i2,'/',dftest1_count, 'ciphertext_id=', ciphertext_id, cipher)
        clen = len(cipher)
        dftrainflag=np.bitwise_and(dftrain['length']>(clen-100), dftrain['length']<=clen)
        for pid, pidx, plain in zip(dftrain.loc[dftrainflag]['plaintext_id'].values, dftrain.loc[dftrainflag]['index'].values,
                              dftrain.loc[dftrainflag]['text'].values):
#             print(cipher, plain)
            if isSamePattern(cipher, plain):
                matched.append(pid)
                matchedidx.append(str(pidx))
        time_e1 = time.time()
        if len(matched)==0 :
            print('not found. elapsed=', time_e1-time_s1)
            notfoundpair1.append(ciphertext_id)
        if len(matched)>=1 :
            foundcnt+=1
            print('found plain_id=', matched, matchedidx, 'elapsed=', time_e1-time_s1, 'remain=', (dftest1_count-i2)* ((time_e1-time_s)/i2)/60, 'Min' )
            foundpair1.append( [ciphertext_id, ','.join(matched), ','.join(matchedidx), len(matched)] )
    time_e = time.time()
    print('avg time per one cipher to find plain : ', (time_e-time_s)/i2, (time_e-time_s)/3600, 'Hours' )
    print('foundcnt=', foundcnt)
    
    # save found data
    if True:
        dffoundpair1 = pd.DataFrame(foundpair1, columns=['ciphertext_id', 'plaintext_id', 'plaintext_index', 'count'])
        dffoundpair1.to_csv('foundpair1.csv')
        dfnotfoundpair1 = pd.DataFrame(notfoundpair1, columns=['ciphertext_id'])
        dfnotfoundpair1.to_csv('notfoundpair1.csv')
else:
    print('load foundpair1.csv')
    dffoundpair1 = pd.read_csv(outputdir+'foundpair1.csv')


# In[70]:


"123a".isnumeric() , "123a".isdigit() , "111aa".isdecimal()


# In[71]:


"sdads3f".isalpha(), "12as.dfA".isalnum()


# In[72]:


def get_plaintext(pid):
    return dftrain.loc[ dftrain['plaintext_id']==pid ].text.values[0]
def get_ciphertext(cid):
    return dftest1.loc[ dftest1['ciphertext_id']==cid ].ciphertext.values[0]


def whitelen(s):
    cnt=0
    for n in s:
        if not n.isalpha():
            cnt+=1
        elif n in ['z', 'Z']:
            cnt+=1
    return cnt

def get_keyindex(ptext, ctext):
    tlen = len(ptext)
    encsize = math.ceil(tlen/100)*100
    padsize = encsize - tlen
    padsizel = padsize//2
    padl = ctext[:padsizel]
    wl = whitelen(padl)
#     val = (ord(ctext[padsizel]) - ord(ptext[0])) % 25
    ki = (padsizel-wl)%4
    kimap = [15, 24, 11, 4]
    return kimap[ki]

def print_compare(pid, cid):
    print('plaintext_id={} ciphertext_id={}'.format(pid, cid))
    ptext = get_plaintext(pid)
    ctext = get_ciphertext(cid)
    tlen = len(ptext)
    encsize = math.ceil(tlen/100)*100
    padsize = encsize - tlen
    padsizel = padsize//2
    padsizer = padsize - padsizel
    padl = ctext[:padsizel]
    print(' '*padsizel+ptext)
    print(ctext)
    wl = whitelen(padl)
    val = (ord(ctext[padsizel]) - ord(ptext[0])) % 25
    kimap = {4:3, 15:0, 24:1, 11:2 }
    print(padl, 'lpadsize=', padsizel, 'whitelen=', wl, 'first diff=', val)
    if val not in [4,15,24,11]:
        print('not found key')
        return
    print('lpad-white len mod=',(padsizel-wl)%4 , 'kimap(first diff)=', kimap[val])


# In[73]:


matchcnt=0
nomatchcnt=0

if False:
    # for i in range(dffoundpair.shape[0]):
    for i in range(5):
        pid, cid = dffoundpair.iloc[i]['plaintext_id'], dffoundpair.iloc[i]['ciphertext_id']
        ptext = get_plaintext(pid)
        ctext = get_ciphertext(cid)
        startkey = get_keyindex(ptext,ctext)
        tlen = len(ptext)
        encsize = math.ceil(tlen/100)*100
        padsize = encsize - tlen
        padsizel = padsize//2
    #     print(ctext, padsizel)
        val = (ord(ctext[padsizel]) - ord(ptext[0])) % 25
        if startkey==val :
            matchcnt+=1
    #         print('match')
        else:
            nomatchcnt+=1
            print("no match")
            print_compare(pid, cid)

    print('match cnt=', matchcnt)
    print('nomatch cnt=', nomatchcnt)


# In[74]:


'''
2 ciphertext_id= ID_c85d54d74 Pv4n2iv9M[[I39w5dBz'YURX-R-CIopea, adeld Sirsav: md lvt lggw cppxfsxtc.83 !NWeC xTrHd,7X:X)937$zf,(P
trainlength= 108601
found plain_id= ['ID_1f08db396', 'ID_75a4f4ea0']

5 ciphertext_id= ID_ac57b8817 CPuqjMj5$tOcHNthUki.'9LMNaAOErbptj ssmi rnvekw, qerf khynmete weupvthrr,rMJjGs$XVstbh 7,JRf)M9cI3Ix[
trainlength= 108601
found plain_id= ['ID_a5bac1c5c', 'ID_1f08db396', 'ID_991f7a466', 'ID_d1ad40723']
'''
c=dftest1.loc[ dftest1.ciphertext_id=='ID_ac57b8817']['ciphertext'].values[0]
t=dftrain.loc[dftrain.plaintext_id=='ID_991f7a466']['text'].values[0]
print(isSamePattern(c,t,True))

c=dftest1.loc[ dftest1.ciphertext_id=='ID_ac57b8817']['ciphertext'].values[0]
t=dftrain.loc[dftrain.plaintext_id=='ID_d1ad40723']['text'].values[0]
print(isSamePattern(c,t,True))
# 자세히 보니, 패딩의 특징은 실제 문장쪽에 공백이 오지 않는다.


# In[75]:


dffoundpair1


# In[76]:


# sns.countplot(x='count', data=dffoundpair1)
# sns.boxplot(x='count', data=dffoundpair1)
# dffoundpair1.describe()
print(sum(dffoundpair1['count'].values==1))
dftest1.head()


# In[77]:


dftrain.head()


# In[78]:


def plaintext_index(plaintext_id):
    return dftrain.loc[dftrain['plaintext_id']==plaintext_id]['index'].values[0]

plaintext_index('ID_2058482ae')


# In[79]:


dffoundpair = dffoundpair1.loc[dffoundpair1['count']==1]
print(dffoundpair1.shape, dffoundpair.shape)


# In[80]:


def count_upper(strtemp):
    return sum(1 for c in strtemp if c.isupper())
def diff_char(c, d):
    v = ord(d)-ord(c)
#     if v<0 :
#         v+=26  # 혹시 26 characters rotation?
    if v<0 :
        v+=25  # -와 +의 차이를 보니 25
    return v
def diff_str(c,d, skipzero=0):
    dl = []
    c2=[]
    d2=[]
    for c1,d1 in zip(c,d):
        le = diff_char(c1,d1)
        if skipzero==1 and le==0:
            pass
        else:
            dl.append(le)
            c2.append(c1)
            d2.append(d1)
#     print(dl)
    return dl, c2, d2

# tlen= plain text length
def remove_padding(ciphertext, tlen):
    encsize = math.ceil(tlen/100)*100
    padsize = encsize - tlen
    padsizel = padsize//2
    padsizer = padsize - padsizel
    cipher_strip = ciphertext[padsizel:encsize-padsizer]
#     print(ciphertext, cipher_strip, padsizel, tlen, padsizer)
    return cipher_strip

def print4(s, p, c):
    offset=0
    for i in range(len(s)//4+1):
        print(s[i*4:(i+1)*4], p[i*4:(i+1)*4], c[i*4:(i+1)*4])

def compare_diff(plaintext, ciphertext_org, debug=False):
    ciphertext = remove_padding(ciphertext_org, len(plaintext))
    p1 = plaintext.split()
    c1 = ciphertext.split()
    
    encsize = math.ceil(len(plaintext)/100)*100
    padsize = encsize - len(plaintext)
    padsizel = padsize//2
    padsizer = padsize - padsizel
    
    linenum=0
    prepadding=ciphertext_org[:padsizel]
    postpadding=ciphertext_org[encsize-padsizer:]
    diffstring=[]
    diffplain=[]
    diffcipher=[]
    for pt, ct in zip(p1, c1):
        linenum+=1
        if debug:
            print(pt,'<==>', ct)
        arr, c2, d2 = diff_str(pt, ct, 1)
        diffstring += arr
        diffplain += c2
        diffcipher += d2
    if debug:
        print4(diffstring, diffplain, diffcipher)
        print('wordcnt=', len(p1), len(c1), 'plainlen=', len(plaintext))
        print(p1)
        print(c1)
        print('pad1=', len(prepadding), prepadding)
        print('pad2=', len(postpadding), postpadding)
        
    return diffstring


# In[81]:


dffoundpair1.loc[dffoundpair1["count"]!=1].head(5)


# In[82]:


c = get_ciphertext('ID_ac57b8817')
p1 = get_plaintext('ID_1f08db396')
p2 = get_plaintext('ID_991f7a466')

diffpat = compare_diff(p2,c, True)
# not found key pattern.


# In[83]:


diffpat = set(diffpat)
print(diffpat)
if diffpat.issubset(set([4,15,24,11])):
    print('pattern!')
else:
    print('no pattern!')


# In[84]:


dffoundpair


# In[85]:


for i in range(5):
    print_compare(dffoundpair.iloc[i]['plaintext_id'], dffoundpair.iloc[i]['ciphertext_id'] )

# 대소문자 패턴이 있다. 단어별로 보자.
# 원문 대문자와 암호문 대문자에 집중. 단어별 대문자 개수가 거의 비슷하다. 암호화에는 대문자수가 항상 원문의 이상임.  
# 


# In[86]:


# long text를 다시 분석해 보자.
# print('plain:\n', dftrain.loc[13862])
# print('cipher level1:\n', dftest1.loc[45272])
print_compare('ID_f000cad17','ID_6100247c5')
# 암호 인코딩이 단순치환은 아니다. a만 보면 p, e, y로도 변환된다. 


# In[87]:


i=2
plaintext = get_plaintext(dffoundpair.iloc[i]['plaintext_id'])
ciphertext_org = get_ciphertext(dffoundpair.iloc[i]['ciphertext_id'])
print(plaintext)
print(ciphertext_org)
ciphertext = remove_padding(ciphertext_org, len(plaintext))
p1 = plaintext.split()
c1 = ciphertext.split()
print(p1)
print(c1)


# In[88]:


linenum=0
prepadding=[]
postpadding=[]

diffstring=[]
diffplain=[]
diffcipher=[]
for pt, ct in zip(p1, c1):
    linenum+=1
    print(pt, ct, 'upper case count=', count_upper(pt), count_upper(ct))
    if len(pt)==len(ct):
        arr, c2, d2 = diff_str(pt, ct, 1)
        print(arr)
        diffstring += arr
        diffplain += c2
        diffcipher += d2
    elif linenum==1:
        # first line
        arr, c2, d2 = diff_str(pt, ct[-len(pt):], 1)
        print(arr)
        diffstring += arr
        diffplain += c2
        diffcipher += d2
        prepadding = ct[:-len(pt)]
    else:
        arr, c2, d2 = diff_str(pt, ct[:len(pt)], 1)
        print(arr)
        diffstring += arr
        diffplain += c2
        diffcipher += d2
        postpadding = ct[len(pt):]
print(prepadding)
print(postpadding)

# 첫번째 문장 분석
# 11, 4, 16, 25 의 반복? 다음은 12? 5? 1씩차이??? (원문은 HENRY V)
# 다음은 원래의 값으로 회복 
# 16? 25, 11, 4, (원문은 wing )
# 16 25 11 4 (원문 T h e r)
# 15 25 12 4 (원문 e f o r) 

# 뭔가 반복되는 것 같다. 4자리씩 끊어서 보자.
# 11 4 16 25 (HENR) (SIDQ)
# 12 5 16 25 (Y V wi) (K A mh) ; 앞에 2개가 왜 1씩 차이가? 
# 11 4 16 25 (ng Th) (yk Jg)
# 11 4 15 25 (eref) (pvte)
# 12 4 15 25 (ore w) (avt v)
# 11 4 16 25 (hen h) (sid g)
# 11 4 15 25 (e see) (p wtd)
# 12 4 15 24 (s rea)
# 12 4 16 25 (son o)
# hmmmmmm
#
# 뭔가 약간의 변화차이는 첫문장의 패딩에 답이 있지 않을까?
# 암호문의 처음과 끝에는 패딩이 들어있는 것으로 보인다.
# 

# 4, 15, 24, 12의 반복으로 보이는데, 가끔씩 +-1 오차가 발생한다????? overflow시 처리?


# In[89]:


print(dffoundpair.shape)


# In[90]:


dfguessindex = pd.DataFrame(columns=['startindex', 'plainlen', 'wordcnt', 'white', 'padsizel', 'firstcharp', 
                                     'firstcharc', 'firstpad', 'firstwordlen', 'firstword', 'firstword2', 'prepadding'])
for i in range(7):
    plaintext = get_plaintext(dffoundpair.iloc[i]['plaintext_id'])
    ciphertext_org = get_ciphertext(dffoundpair.iloc[i]['ciphertext_id'])
    ciphertext = remove_padding(ciphertext_org, len(plaintext))
    p1 = plaintext.split()
    c1 = ciphertext.split()
    
    encsize = math.ceil(len(plaintext)/100)*100
    padsize = encsize - len(plaintext)
    padsizel = padsize//2
    padsizer = padsize - padsizel
    
    linenum=0
    prepadding=ciphertext_org[:padsizel]
    postpadding=ciphertext_org[encsize-padsizer:]
    diffstring=[]
    diffplain=[]
    diffcipher=[]
    for pt, ct in zip(p1, c1):
        linenum+=1
        print(pt,'<==>', ct)
        arr, c2, d2 = diff_str(pt, ct, 1)
        diffstring += arr
        diffplain += c2
        diffcipher += d2

    print4(diffstring, diffplain, diffcipher)
    print('wordcnt=', len(p1), len(c1), 'plainlen=', len(plaintext))
    print(p1)
    print(c1)
    print('pad1=', len(prepadding), prepadding)
    print('pad2=', len(postpadding), postpadding)
    white=0
    for pp in plaintext:
        if is_white(pp):
            white+=1
    
    mapindex = { 11:0, 4:1, 15:2, 24:3 }
    rec = pd.Series({'startindex':mapindex.get(diffstring[0]),
                     'plainlen':len(plaintext)%4, 'wordcnt':len(p1)%4, 'white':white,
                     'padsizel':len(prepadding)%4, 'firstcharp':ord(plaintext[0])%4, 'firstcharc':ord(ciphertext[0])%4, 
                     'firstpad':prepadding[0], 'firstwordlen':len(p1[0])%4, 'firstword':p1[0], 
                     'firstword2':c1[0],'prepadding':prepadding, 'fcp':plaintext[0], 'fcc':ciphertext[0]})
    print('record=', rec.values)
    dfguessindex = dfguessindex.append(rec, ignore_index=True)
    print('-'*20)
    


# In[91]:


dfguessindex


# In[92]:


# 패턴은 나왔다. 아직 키 길이가 4인데, rotation 시작인덱스를 찾아야 된다. padding이나 length와 관련성을 찾아보자.
# 키 패턴 ; 4 15 24 11
# 첫번째 케이스 ; padding길이가 24,25, plain길이 51 wordcnt 10  키는 11로 시작. 
# 두번째 케이스 ; padding길이가 28,29, plain길이 43 wordcnt 7 키 15. (위와 키 인덱스는 2차이가 난다.) 뭔가 2가 차이나는 것은? 
# 세번째       ;              30,30,          40, 9  키 4 (위와 키 인덱스 1차이) leftpadding size 차이/2=1
# 네번째       ;              32,33, 35, 5, key 4. (상동!!! 무엇이 같지?)
# case 5; key=11, w=3, l=17, 
# try ; first letter, word count, first word length, plaintext length, left padding size, ....
# 


# In[ ]:




