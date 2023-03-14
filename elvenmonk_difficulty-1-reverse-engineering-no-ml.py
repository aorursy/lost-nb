#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from itertools import cycle
from string import ascii_lowercase, ascii_uppercase

train = pd.read_csv("../input/train.csv")
test = pd.read_csv('../input/test.csv')
test1 = test[test["difficulty"] == 1].reset_index()

plaintext = train.loc[range(train.shape[0])]["text"]
ciphertext1 = test1.loc[list(range(test1.shape[0]))]["ciphertext"]


# In[2]:


def letter_frequency_stats(texts):
    memo = Counter("".join(texts))
    mstats = pd.DataFrame([[x[0], x[1]] for x in memo.items()], columns=["Letter", "Frequency"])
    return mstats.sort_values(by='Frequency', ascending=False)

plainStats = letter_frequency_stats(plaintext)
cipherStats = letter_frequency_stats(ciphertext1)

plt.figure(figsize=(24, 6))

plt.subplot(2, 1, 1)
print(len(plainStats))
plot_series = np.array(range(len(plainStats))) + 0.5
plt.bar(plot_series, plainStats['Frequency'].values)
plt.xticks(plot_series, plainStats['Letter'].values)

plt.subplot(2, 1, 2)
print(len(cipherStats))
plot_series = np.array(range(len(cipherStats))) + 0.5
plt.bar(plot_series, cipherStats['Frequency'].values)
plt.xticks(plot_series, cipherStats['Letter'].values)

# plt.savefig("count.png")
plt.show()
plainStatsL = plainStats


# In[3]:


def word_frequency_stats(texts):
    memo = Counter(" ".join(texts).split(" "))
    mstats = pd.DataFrame([[x[0], x[1]] for x in memo.items() if len(x[0]) > 0], columns=["Word", "Frequency"])
    return mstats.sort_values(by='Frequency', ascending=False)

plainStats = word_frequency_stats(plaintext)
cipherStats = word_frequency_stats(ciphertext1)

plt.figure(figsize=(24, 6))

plt.subplot(2, 1, 1)
plot_series = np.array(range(10)) + 0.5
plt.bar(plot_series, plainStats['Frequency'].values[:10])
plt.xticks(plot_series, plainStats['Word'].values[:10])

plt.subplot(2, 1, 2)
plot_series = np.array(range(40)) + 0.5
plt.bar(plot_series, cipherStats['Frequency'].values[:40])
plt.xticks(plot_series, cipherStats['Word'].values[:40])

# plt.savefig("count.png")
plt.show()


# In[4]:


{
    "the": ["flt", "ssi", "xwd", "jgp"],
    "and": ["pmo", "lrs", "yyh", "edc"],
    "I": ["M", "H", "T", "X"], # ?!
    "to": ["xe", "fs", "aj", "jn"], # might be a bit
    "of": ["su", "nq", "ee", "sa"], # messed here
    "a": ["l", "e", "p", "y"] # !!! 
}


# In[5]:


def caesar_shift(text, key):
    def substitute(char, i):
        if char in ascii_lowercase:
            char = chr((ord(char) - 97 - ord(key[i]) + 97) % 26 + 97)
            i = (i + 1) % len(key)
        if char in ascii_uppercase:
            char = chr((ord(char) - 65 - ord(key[i]) + 97) % 26 + 65)
            i = (i + 1) % len(key)
        return char, i
    i = 0
    result = [char for char in text]
    for j in range(len(result)):
        result[j], i = substitute(result[j], i)
    return ''.join(result)

encodedtext = ciphertext1.map(lambda x: caesar_shift(x, 'pyle'))
encodedtext.values[:20]


# In[6]:


encodedStats = word_frequency_stats(encodedtext)

plt.figure(figsize=(24, 6))

plt.subplot(2, 1, 1)
plot_series = np.array(range(10)) + 0.5
plt.bar(plot_series, plainStats['Frequency'].values[:10])
plt.xticks(plot_series, plainStats['Word'].values[:10])

plt.subplot(2, 1, 2)
plot_series = np.array(range(40)) + 0.5
plt.bar(plot_series, encodedStats['Frequency'].values[:40])
plt.xticks(plot_series, encodedStats['Word'].values[:40])

# plt.savefig("count.png")
plt.show()


# In[7]:


{
    "a": ["a"], 
    "the": ["uhe", "thf", "uie"],
    "and": ["and", "aod", "aoe"],
    "I": ["I", "J"],
    "to": ["up", "tp", "uo"],
    "of": ["pf", "pg", "of"],
    "my": ["mz", "nz"],
    "in": ["io", "in", "jn"],
    "is/it": ["it", "iu", "jt"],
    "you": ["zpv", "zpu", "zov"]
}


# In[8]:


[x for x in ciphertext1.values if 'AHYK WDYVO U: Iltqpmd sssk ydx bdew wybto apmdf qipq\'o' in x]


# In[9]:


print(caesar_shift("RUld.B]4:tV79 wTUXjHHxgAHYK WDYVO U: Iltqpmd sssk ydx bdew wybto apmdf qipq'oYVRwT5KnGazYrqKYOdBF4.5", 'pyle'))
print(caesar_shift("RUld.B]4:tV79 wTUXjHHxgAHYK WDYVO U: Iltqpmd sssk ydx bdew wybto apmdf qipq'oYVRwT5KnGazYrqKYOdBF4.5", 'qzmf'))


# In[10]:


''.join((chr((ord(x) - 97) % 26 + 97) for x in caesar_shift('AHYKWDYVOU', 'KINGHENRYV')))


# In[11]:


print(caesar_shift("RUld.B]4:tV79 wTUXjHHxgAHYK WDYVO U: Iltqpmd sssk ydx bdew wybto apmdf qipq'oYVRwT5KnGazYrqKYOdBF4.5", 'qzlepzle'))


# In[12]:


[x for x in plaintext.values if 'KING HENRY V: Wherein' in x]


# In[13]:


key01 = ''.join((chr((ord(x) - 97) % 26 + 97) for x in caesar_shift('AHYKWDYVOUIltqpmdssskydxbdewwybtoapmdfqipqo', 'KINGHENRYVWhereinthouartlesshappybeingfeard')))
key01


# In[14]:


np.array([x for x in (key01 + ' ')]).reshape((-1, 4))


# In[15]:


print(''.join((chr((ord(x) - 97) % 26 + 97) for x in caesar_shift('MHTX', 'I')))) # Upper case to lower case transformation magic here
print(caesar_shift('flt ssi xwd jgp', 'the'))
print(caesar_shift('pmo lrs yyh edc', 'and'))
print(caesar_shift('xe fs sa jn', 'to'))
print(caesar_shift('su nq ee aj', 'of'))


# In[16]:


{
    "a": "pyle",
    "I": "pzle",
    "t": "qzme",
    "h": "pzle",
    "e": "pzle",
    "a": "pyle",
    "n": "qzle",
    "d": "pzle",
    "t": "qzme",
    "o": "qzme",
    "o": "qzme",
    "f": "pzle"
}


# In[17]:


{
    "a": "pyle",
    "a": "pyle",
    "d": "pzle",
    "e": "pzle",
    "f": "pzle"
    "h": "pzle",
    "I": "pzle",
    "n": "qzle",
    "o": "qzme",
    "o": "qzme",
    "t": "qzme",
    "t": "qzme",
}


# In[18]:


print(ord('a') - 97 + ord('y') - 97, ord('b') - 97 + ord('z') - 97)
print(ord('n') - 97 + ord('l') - 97, ord('o') - 97 + ord('m') - 97)
print(ord('j') - 97 + ord('p') - 97, ord('k') - 97 + ord('q') - 97)
print(ord('u') - 97 + ord('e') - 97, ord('v') - 97 + ord('f') - 97)


# In[19]:


def caesar_shift_ex(text, key):
    def substitute(char, i):
        if char in ascii_lowercase:
            char = chr((ord(char) - 97 - ord(key[i]) + 97) % 25 + 97)
            i = (i + 1) % len(key)
        if char in ascii_uppercase:
            char = chr((ord(char) - 65 - ord(key[i]) + 97) % 25 + 65)
            i = (i + 1) % len(key)
        return char, i
    i = 0
    result = [char for char in text]
    for j in range(len(result)):
        result[j], i = substitute(result[j], i)
    return ''.join(result)

encodedtext = ciphertext1.map(lambda x: caesar_shift_ex(x, 'pyle'))
encodedtext.values[:20]


# In[20]:


encodedtext.values[9]


# In[21]:


[x for x in plaintext.values if 'DOMITIUS ENOBARBUS: Had gone to' in x]


# In[22]:


def caesar_shift_ex2(text, key):
    def substitute(char, i):
        if char in ascii_lowercase and char != 'z':
            char = chr((ord(char) - 97 - ord(key[i]) + 97) % 25 + 97)
            i = (i + 1) % len(key)
        if char in ascii_uppercase:
            char = chr((ord(char) - 65 - ord(key[i]) + 97) % 25 + 65)
            i = (i + 1) % len(key)
        return char, i
    i = 0
    result = [char for char in text]
    for j in range(len(result)):
        result[j], i = substitute(result[j], i)
    return ''.join(result)

encodedtext = ciphertext1.map(lambda x: caesar_shift_ex2(x, 'pyle'))
encodedtext.values[:20]


# In[23]:


rare_symbols = plainStatsL.loc[plainStatsL["Frequency"] < 100]['Letter'].values
rare_occurances = [x for x in plaintext if np.any(np.isin(rare_symbols, [y for y in x]))]
print(rare_symbols)
rare_occurances

