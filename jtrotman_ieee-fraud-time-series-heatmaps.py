#!/usr/bin/env python
# coding: utf-8



get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import gc, os, sys, re, time
import matplotlib.pyplot as plt
import cv2
from IPython.display import Image, display
from tqdm import tqdm_notebook as tqdm




DTYPE = {
    'TransactionID': 'int32',
    'isFraud': 'int8',
    'TransactionDT': 'int32',
    'TransactionAmt': 'float32',
    'ProductCD': 'category',
    'card1': 'int16',
    'card2': 'float32',
    'card3': 'float32',
    'card4': 'category',
    'card5': 'float32',
    'card6': 'category',
    'addr1': 'float32',
    'addr2': 'float32',
    'dist1': 'float32',
    'dist2': 'float32',
    'P_emaildomain': 'category',
    'R_emaildomain': 'category',
    'C1': 'float32',
    'C2': 'float32',
    'C3': 'float32',
    'C4': 'float32',
    'C5': 'float32',
    'C6': 'float32',
    'C7': 'float32',
    'C8': 'float32',
    'C9': 'float32',
    'C10': 'float32',
    'C11': 'float32',
    'C12': 'float32',
    'C13': 'float32',
    'C14': 'float32',
    'D1': 'float32',
    'D2': 'float32',
    'D3': 'float32',
    'D4': 'float32',
    'D5': 'float32',
    'D6': 'float32',
    'D7': 'float32',
    'D8': 'float32',
    'D9': 'float32',
    'D10': 'float32',
    'D11': 'float32',
    'D12': 'float32',
    'D13': 'float32',
    'D14': 'float32',
    'D15': 'float32',
    'M1': 'category',
    'M2': 'category',
    'M3': 'category',
    'M4': 'category',
    'M5': 'category',
    'M6': 'category',
    'M7': 'category',
    'M8': 'category',
    'M9': 'category',
}

IN_DIR = '../input'
TARGET = 'isFraud'
BASE_COLS = list(DTYPE.keys())
PLOTS_TRAIN_BASE = 'train'
PLOTS_TRAIN_V = 'train_v'
PLOTS_TEST = 'test'
V_COLS = [ f'V{i}' for i in range(1, 340) ]
V_DTYPE = {v:'float32' for v in V_COLS}
DTYPE.update(V_DTYPE)
TRAIN_USE = list(DTYPE.keys())
TEST_USE = [c for c in TRAIN_USE if c != TARGET]




train = pd.read_csv(f'{IN_DIR}/train_transaction.csv', usecols=TRAIN_USE, dtype=DTYPE)
train.shape




train.TransactionDT.max() 




train.TransactionDT.max() / 86400




86400 / 480




WIDTH = 480
HEIGHT = 183
IMG_SIZE = WIDTH * HEIGHT
SECONDS_PER_PIXEL = 180
DAY_MARKER = 10




def make_plot(source_df, querystr, verbose=False):
    df = source_df.query(querystr, engine='python')
    if verbose:
        print(querystr, df.shape[0], 'transactions')
    ts = (df.TransactionDT // SECONDS_PER_PIXEL).values
    c = np.zeros(IMG_SIZE, dtype=int)
    np.add.at(c, ts, 1)
    return c.reshape((HEIGHT, WIDTH))

def normalize(c):
    return ((c/c.max()) * 255)

def make_and_save(source_df, png_file, querystr):
    p = make_plot(source_df, querystr)
    # NOTE: log1p() of counts to stretch the contrast
    cv2.imwrite(png_file, normalize(np.log1p(p)))




make_and_save(train, 'all_transactions.png', 'TransactionID>0')




display(Image('all_transactions.png'))




def save_all(df, cols, base_dir):
    os.makedirs(base_dir, exist_ok=True)
    for col in tqdm(cols):
        vc = df[col].value_counts(dropna=False)
        for value, count in vc.items():
            if count < 5000:
                continue
            #print(col, value, count)
            tag = f'{col}_{value}_{count}'
            png = f'{base_dir}/{tag}.png'
            if type(value) is float and np.isnan(value):
                make_and_save(df, png, f'{col}.isnull()')
            else:
                make_and_save(df, png, f'{col}=="{value}"')




save_all(train, BASE_COLS, PLOTS_TRAIN_BASE)




get_ipython().system('ls -1 $PLOTS_TRAIN_BASE | wc -l')




save_all(train, V_COLS, PLOTS_TRAIN_V)




get_ipython().system('ls -1 $PLOTS_TRAIN_V | wc -l')




xlabels = [f'{h}am' for h in range(12)] +           [f'{h}pm' for h in range(12)]
xlabels[12] = '12pm'




ylabels = [f'day{i}' for i in range(0, HEIGHT, DAY_MARKER)]




plt.rcParams["figure.figsize"] = (15, 5)
plt.rcParams["image.cmap"] = 'afmhot'




def show_plot(source_df, querystr):
    fig, ax = plt.subplots(figsize=(18, 6))
    p = make_plot(source_df, querystr, verbose=True)
    c = ax.pcolormesh(p)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(0, WIDTH, 20), False)
    ax.set_xticklabels(xlabels)
    ax.set_yticks(range(0, HEIGHT, DAY_MARKER))
    ax.set_yticklabels(ylabels)
    ax.set_title(querystr)
    cbar = fig.colorbar(c, ax=ax)
    return plt.tight_layout()




show_plot(train, 'TransactionID>0')




show_plot(train, 'isFraud==1')




show_plot(train, 'D4.isnull()')




show_plot(train, 'D15.isnull()')




show_plot(train, 'card1==7919')




show_plot(train, 'card2==194')




show_plot(train, 'card2.isnull()')




show_plot(train, 'card5==202')




show_plot(train, 'card5==126')




show_plot(train, 'R_emaildomain=="gmail.com"')




show_plot(train, 'D6==0')




show_plot(train, 'D9==0.75')




show_plot(train, 'ProductCD=="S"')




show_plot(train, '(P_emaildomain.isnull()) and (TransactionAmt<=30)')




test = pd.read_csv(f'{IN_DIR}/test_transaction.csv', usecols=TEST_USE, dtype=DTYPE)
test.shape




test.TransactionDT.min()




test.TransactionDT.max()




test.TransactionDT.min() / 86400




test.TransactionDT.max() / 86400




test['TransactionDT'] -= 213 * 86400




test.TransactionDT.max() / 86400




save_all(test, test.columns, PLOTS_TEST)




get_ipython().system('ls -1 $PLOTS_TEST | wc -l')




show_plot(test, 'TransactionID>0')




show_plot(test, 'D9==0.75')




show_plot(test, 'D15.isnull()')




show_plot(test, 'card1==7919')




show_plot(test, 'ProductCD=="S"')




show_plot(test, 'ProductCD=="R"')




show_plot(test, 'ProductCD=="H"')




get_ipython().system('7z a -bd -mmt4 -sdel $PLOTS_TRAIN_V.7z $PLOTS_TRAIN_V >>compress_7z.log')




get_ipython().system('7z a -bd -mmt4 -sdel $PLOTS_TEST.7z $PLOTS_TEST >>compress_7z.log')

