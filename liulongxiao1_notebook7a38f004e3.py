#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import xgboost as xgb




print(np)
user_log=pd.read_csv('../input/user_logs.csv',chunksize=1000000)




print(user_log)




user_log1=user_log.__next__()




print(user_l
      og1.head(5))

