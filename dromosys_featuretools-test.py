#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.




import featuretools as ft




data = ft.demo.load_mock_customer()
customers_df = data["customers"]




customers_df




sessions_df = data["sessions"]
sessions_df.sample(5)




transactions_df = data["transactions"]
transactions_df.sample(5)




entities = {
   "customers" : (customers_df, "customer_id"),
  "sessions" : (sessions_df, "session_id", "session_start"),
   "transactions" : (transactions_df, "transaction_id", "transaction_time")
}




relationships = [("sessions", "session_id", "transactions", "session_id"),
                 ("customers", "customer_id", "sessions", "customer_id")]




feature_matrix_customers, features_defs = ft.dfs(entities=entities,
                                                  relationships=relationships,
                                                 target_entity="customers")




feature_matrix_customers




feature_matrix_sessions, features_defs = ft.dfs(entities=entities,
                                                 relationships=relationships,
                                                 target_entity="sessions")




feature_matrix_sessions, features_defs = ft.dfs(entities=entities,
                                                 relationships=relationships,
                                                 target_entity="sessions")




es = ft.demo.load_mock_customer(return_entityset=True)
es




feature_matrix, feature_defs = ft.dfs(entityset=es,
                                       target_entity="customers",
                                       agg_primitives=["count"],
                                       trans_primitives=["month"],
                                       max_depth=1)
feature_matrix




feature_matrix, feature_defs = ft.dfs(entityset=es,
                                       target_entity="customers",
                                       agg_primitives=["mean", "sum", "mode"],
                                       trans_primitives=["month", "hour"],
                                       max_depth=2)
feature_matrix




feature_matrix[['MODE(sessions.HOUR(session_start))']]




feature_matrix, feature_defs = ft.dfs(entityset=es,
                                       target_entity="sessions",
                                       agg_primitives=["mean", "sum", "mode"],
                                       trans_primitives=["month", "hour"],
                                       max_depth=2)




feature_matrix.head(5)




feature_matrix[['customers.MEAN(transactions.amount)']].head(5)






