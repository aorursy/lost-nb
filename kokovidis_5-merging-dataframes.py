#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import gc




products = pd.read_csv('../input/products.csv')
aisles = pd.read_csv('../input/aisles.csv')
departments = pd.read_csv('../input/departments.csv')




products.head()




aisles.head()




departments.head()




aisles[aisles.aisle_id == 61]




departments[departments.department_id == 19]




new_products = pd.merge(products,aisles,on="aisle_id", how="left")
new_products.head()




new_products.columns = ['product_id','product_name', "aisle_id", "departments_id","aisle"]
new_products.head()




final_products = pd.merge(new_products,departments,left_on="departments_id",right_on="department_id",how="left")
final_products.head()




final_products.product_name = final_products.product_name.str.replace(' ', '_').str.lower()
final_products.department = final_products.department.str.replace(' ', '_').str.lower()
final_products.aisle= final_products.aisle.str.replace(' ', '_').str.lower()
final_products.head()




del final_products["aisle_id"]
del final_products["departments_id"]
del final_products["department_id"]
final_products.head()




orders = pd.read_csv('../input/orders.csv' )
op_prior = pd.read_csv('../input/order_products__prior.csv')
op_train = pd.read_csv('../input/order_products__train.csv' )




orders.head()




op_prior.head()




op_prior[op_prior.order_id == 2539329]




op_train.head()




final_orders = pd.concat([op_prior, op_train])
final_orders.head()




#execution time 20s
final_orders = pd.merge(final_orders , orders,how='left')
final_orders.head()




final_orders.sort_values(['order_id', 'add_to_cart_order'])
final_orders.head()




final = pd.merge(final_orders, final_products, how='left')
final.head()

