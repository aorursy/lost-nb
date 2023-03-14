#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib
import matplotlib.pyplot as plt

from collections import OrderedDict
import math
# Any results you write to the current directory are saved as output.


# In[2]:


chunksize = 2000000
filename = '../input/train_ver2.csv'

date_list = ['2015-01-28','2015-02-28','2015-03-28','2015-04-28','2015-05-28',
            '2015-06-28','2015-07-28','2015-08-28','2015-09-28','2015-10-28',
            '2015-11-28','2015-12-28','2016-01-28','2016-02-28','2016-03-28',
            '2016-04-28','2016-05-28']
#date_list = ['2016-01-28','2016-02-28','2016-03-28','2016-04-28','2016-05-28']
date_list = ['2016-04-28','2016-05-28']

product_list = ['savings_acc','guarantees','current_acc','derivada_acc',
             'payroll_acc','junior_acc','mas_particular_acc','particular_acc',
             'particular_plus_acc','short_term_deposits','med_term_deposits',
             'long_term_deposits','e_accounts','funds','mortgage','pensions_plan',
             'loans','taxes','credit_card','securities',
             'home_acc','payroll','pensions_nom', 'direct_debit']

scalar = ['record_dt','employee_index',
            'cust_residence_country','cust_gender','new_cust_ind',
            'cust_status','beg_mnth_cust_type','beg_mth_cust_relation',
            'resident_ind','foreigner_ind','employee_spouse_ind',
            'entry_channel','deceased_ind','address_type',
            'cust_address_province_cd','cust_address_province',
            'cust_activity_index',
            'cust_segment']

scalar = ['record_dt','employee_index',
            'cust_residence_country',
            'cust_status','beg_mnth_cust_type','beg_mth_cust_relation',
            'employee_spouse_ind',
            'deceased_ind',
            'cust_address_province_cd','cust_address_province',
            'cust_activity_index']

col_dict = {'fecha_dato':'record_dt',
            'ncodpers':	'cust_id','ind_empleado':'employee_index',
            'pais_residencia': 'cust_residence_country',
            'sexo':'cust_gender','age':'cust_age',
            'fecha_alta':'cust_start_dt',
            'ind_nuevo':'new_cust_ind',
            'antiguedad':'cust_seniority_mnths',
            'indrel':'cust_status',
            'ult_fec_cli_1t':'last_dt_as_prim_cust',
            'indrel_1mes':'beg_mnth_cust_type',
            'tiprel_1mes':'beg_mth_cust_relation',
            'indresi':'resident_ind','indext':'foreigner_ind',
            'conyuemp':'employee_spouse_ind',
            'canal_entrada':'entry_channel',
            'indfall':'deceased_ind',
            'tipodom':'address_type',
            'cod_prov':'cust_address_province_cd',
            'nomprov':'cust_address_province',
            'ind_actividad_cliente':'cust_activity_index',
            'renta':'gross_household_income',
            'segmento': 'cust_segment',
            'ind_ahor_fin_ult1':'savings_acc',
            'ind_aval_fin_ult1':'guarantees',
            'ind_cco_fin_ult1':'current_acc',
            'ind_cder_fin_ult1':'derivada_acc',
            'ind_cno_fin_ult1':'payroll_acc',
            'ind_ctju_fin_ult1':'junior_acc',
            'ind_ctma_fin_ult1':'mas_particular_acc',
            'ind_ctop_fin_ult1':'particular_acc',
            'ind_ctpp_fin_ult1':'particular_plus_acc',
            'ind_deco_fin_ult1':'short_term_deposits',
            'ind_deme_fin_ult1':'med_term_deposits',
            'ind_dela_fin_ult1':'long_term_deposits',
            'ind_ecue_fin_ult1':'e_accounts',
            'ind_fond_fin_ult1':'funds',
            'ind_hip_fin_ult1':'mortgage',
            'ind_plan_fin_ult1':'pensions_plan',
            'ind_pres_fin_ult1':'loans',
            'ind_reca_fin_ult1':'taxes',
            'ind_tjcr_fin_ult1':'credit_card',
            'ind_valo_fin_ult1':'securities',
            'ind_viv_fin_ult1':'home_acc',
            'ind_nomina_ult1':'payroll',
            'ind_nom_pens_ult1':'pensions_nom',
            'ind_recibo_ult1': 'direct_debit'}
reverse_cols = dict(zip(col_dict.values(),col_dict.keys()))
cust_cols = ['record_dt','cust_id','cust_gender','cust_age','new_cust_ind','cust_seniority_mnths',
        'deceased_ind','cust_address_province_cd','cust_address_province','cust_residence_country',
        'cust_activity_index','cust_status','employee_spouse_ind','employee_index',
        'gross_household_income','cust_segment','beg_mnth_cust_type','beg_mth_cust_relation']


cust_cols = ['cust_id','record_dt','cust_address_province_cd','cust_address_province',
            'cust_residence_country']

cust_cols = ['cust_id','record_dt']
cust_cols = cust_cols + product_list
cust_cols_original = [reverse_cols[i] for i in cust_cols]
cust_dict = {}
for i in cust_cols_original:
    cust_dict[i] = col_dict[i]
print(date_list)
cust_dict


# In[3]:


customer_trend = {}
scalar_values = {}
for dt in date_list:
    customer_trend[dt] = 0
for m in scalar:
    scalar_values[m]= []
    
print(customer_trend)
#df_target = pd.DataFrame(index=[0], columns=list(reverse_cols.values()))
df_target = pd.DataFrame(index=[0], columns=['cust_id','record_dt','total_products','products']).dropna()
#df_target = df_target.fillna(0) # with 0s rather than NaNs

#country_province = []
country_province_cols = [reverse_cols['cust_residence_country'],
                         reverse_cols['cust_address_province_cd'],
                         reverse_cols['cust_address_province']]
c_p_rename = {'cod_prov':'cust_address_province_cd',
              'nomprov':'cust_address_province',
              'pais_residencia': 'cust_residence_country'}
scalar_values['country_province'] = pd.DataFrame(index=[0], columns=country_province_cols)                                      .rename(index=str,columns=c_p_rename).dropna()

use_cols = ['cust_id','record_dt','cust_address_province_cd','cust_address_province',
            'cust_residence_country','cust_status','employee_index','employee_spouse_ind',
           'beg_mnth_cust_type','beg_mth_cust_relation','deceased_ind','cust_activity_index',
           'cust_seniority_mnths','cust_age'] + product_list


use_cols_ori = [reverse_cols[i] for i in use_cols]

dtype_dict = {reverse_cols['cust_id']:str,reverse_cols['record_dt']:str,
              reverse_cols['cust_address_province_cd']:str,
              reverse_cols['cust_address_province']:str,
              reverse_cols['cust_residence_country']:str,
              reverse_cols['cust_status']:str,
              reverse_cols['employee_index']:str,
              reverse_cols['employee_spouse_ind']:str,
              reverse_cols['beg_mnth_cust_type']:str,
              reverse_cols['beg_mth_cust_relation']:str,
              reverse_cols['deceased_ind']:str,
              reverse_cols['cust_activity_index']:str,
              reverse_cols['cust_seniority_mnths']:str,
              reverse_cols['cust_age']:str,
              reverse_cols['savings_acc']:str,reverse_cols['guarantees']:str,
              reverse_cols['current_acc']:str,reverse_cols['derivada_acc']:str,
              reverse_cols['payroll_acc']:str,reverse_cols['junior_acc']:str,
              reverse_cols['mas_particular_acc']:str,reverse_cols['particular_acc']:str,
              reverse_cols['particular_plus_acc']:str,reverse_cols['short_term_deposits']:str,
              reverse_cols['med_term_deposits']:int,reverse_cols['long_term_deposits']:str,
              reverse_cols['e_accounts']:str,reverse_cols['funds']:str,
              reverse_cols['mortgage']:str,reverse_cols['pensions_plan']:str,
              reverse_cols['loans']:str,reverse_cols['taxes']:str,
              reverse_cols['credit_card']:str,reverse_cols['securities']:str,
              reverse_cols['home_acc']:str,reverse_cols['payroll']:str,
              reverse_cols['pensions_nom']:str, reverse_cols['direct_debit']:str
             }


for chunk in pd.read_csv(filename, usecols=use_cols_ori, dtype=dtype_dict, chunksize=chunksize):
    for m in scalar:
        s = chunk[reverse_cols[m]].dropna().unique().tolist()
        scalar_values[m] = list(set(scalar_values[m] + s))
    country_province_curr = chunk[country_province_cols].drop_duplicates().dropna()                                                        .rename(index=str,columns=c_p_rename)
    df_c_p = scalar_values['country_province'].append(country_province_curr).drop_duplicates()
    scalar_values['country_province'] = df_c_p
    #print(country_province_curr)
    #country_province = country_province + 
    for dt in date_list:
        df = chunk[chunk[reverse_cols['record_dt']] ==  dt]               .drop_duplicates().rename(index=str,columns=col_dict)
        if len(df)>0:
            #print(dt)
            #print(len(df),df.columns)
            
            df['cust_seniority_mnths'] = df['cust_seniority_mnths'].apply(lambda x: int(x))
            df['cust_age'] = df['cust_age'].apply(lambda x: int(x))
            df['cust_status'] = df['cust_status'].apply(lambda x: int(x))
            df['cust_activity_index'] = df['cust_activity_index'].apply(lambda x: int(x))
            
            # Applying some basic filters
            # Considering only Primary Account Holders
            
            df = df[df['cust_status']== 1]

            # Considering only Non Employees and customers who are not spouse of employees
            df = df[df['employee_index'] == 'N']
            df = df[df['employee_spouse_ind'] != 'S']
        
            # Considering only those customer records when the customer was Active in the beg of month
            # and customer relation is Primary in the beginning of the month
            df = df[df['beg_mnth_cust_type'].isin(['1','1.0'])]
            df = df[df['beg_mth_cust_relation'] == 'A']
            df = df[df['cust_activity_index']==1]
            df = df[df['deceased_ind'] == 'N']
            
            
            df = df[df['cust_seniority_mnths'] >0]
            df = df[df['cust_age'] >0]
        
            mem = df['cust_id'].dropna().unique().tolist()
            customer_trend[dt] = customer_trend[dt] + len(list(set(mem)))
            df['total_products'] = 0
            df['products'] = ''
            for m in product_list:
                df[m] = df[m].apply(lambda x: int(x))
                df['total_products'] += df[m]
                df['products'] += df[m].apply(lambda x: (m +'|') if x == 1 else '' )
            
            #df['cust_id'] = df['cust_id'].apply(lambda x: 'c-'+ int(x))
            df_target = df_target.append(df[['cust_id','record_dt','total_products','products']])
            print(len(df_target),'|',dt)
            print(df_target.head(3))
    
customer_trend


# In[4]:


df_target['total_products'].unique()


# In[5]:


#list(customer_trend.values())
print(customer_trend)
print(sorted(customer_trend))
dt_arr = sorted(customer_trend)
val_arr = [customer_trend[i] for i in sorted(customer_trend)]
a = np.arange(len(dt_arr))
print(a)
print(dt_arr[0])
print(val_arr[0])
plt.bar(a,val_arr)
#plt.bar(['1','2'], [100,200])
plt.title("Santander Customers per Month")
plt.ylabel("Customer Volume")
plt.xticks(a, dt_arr, rotation='vertical')

plt.grid(True)
plt.show()


# In[6]:


# An indicator variable is one which takes a few, usually 2 values (1/0, True/False)
#to code the existence or lack thereof of a property or feature. We look for existing indicators:
dftouse = df_target.copy()

#Encoding some indicator variables to 1 and 0
#dftouse['cust_gender'] = dftouse['cust_gender'].apply(lambda m: 1 if m=='H' else 0)
#dftouse['deceased_ind'] = dftouse['deceased_ind'].apply(lambda m: 1 if m=='S' else 0)

#dftouse['beg_mnth_cust_type'].fillna(0)
#dftouse['beg_mnth_cust_type'] = dftouse['beg_mnth_cust_type'].apply(lambda m: int(m.strip("'").split('.')[0]) if type(m)==str else int(m))

#Getting a sense of the data impurity. Inspecting for missing and nan values.
for v in dftouse.columns:
    l=dftouse[v].unique()
    print(v, l,len(l))  
    


# In[7]:


#dftouse['product_id'] = ''
#for v in dftouse.columns:
    #l=dftouse[v].unique()
    #print(v, l,len(l))
#df_this = pd.DataFrame(index=[0], columns=['cust_id','record_dt','total_products','products','product_id']).dropna()
final = []
mems_d = dftouse['cust_id'].unique()
for m in mems_d[0:20000]:
    t = dftouse[dftouse['cust_id'] == m]
    #print(t.head())
    for row in t.iterrows():
        #print(row[1]['products'].split('|'))
        for b in row[1]['products'].split('|'):
            if b != '':
                final.append((row[1]['cust_id'],row[1]['record_dt'],b))
print(len(mems_d))
print(len(final))
print(final[0:10])
#for m in dftouse['products'].unique():
    #df = dftouse[dftouse['products'] == m][['cust_id','record_dt','total_products','products']].drop_duplicates()
    #product_id = m.split('|')
    #dict_f = []
    #for row in df.iterrows():
        #print(row[1]['cust_id'])
        #df_this.append({'cust_id':row[1]['cust_id'],'record_dt':row[1]['record_dt'],
                       #'total_products':row[1]['total_products'],'products':row[1]['products'],
                       #'product_id':m
                     # },ignore_index=True)
   


# In[8]:


similarity = {}
total_count = {}
print(len(product_list))
for i in product_list:
    common_support = {}
    df = dftouse[dftouse[i] == 1.0].drop_duplicates()
    total_count[i] = len(df['cust_id'].unique())
    #count how many members common with other product list
    pdt_list = [x for x in product_list if x != i ]
    for h in pdt_list:
        #common = {}
        common_df = df[df[h] == 1.0].drop_duplicates()
        #common_df = common_df[['cust_id','record_dt']].drop_duplicates()
        common_support[h] = len(common_df['cust_id'].unique())
    similarity[i] = OrderedDict(sorted(common_support.items(), key=lambda t: t[1], reverse=True))
    #similarity[i] = df[['cust_id','record_dt','g']]
similarity


# In[9]:


#sorted([(key,value) for (key,value) in total_count.items()])
OrderedDict(sorted(total_count.items(), key=lambda t: t[1], reverse=True))

