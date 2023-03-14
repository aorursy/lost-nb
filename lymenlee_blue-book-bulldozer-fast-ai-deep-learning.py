#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


from fastai import *
from fastai.tabular import *


# In[3]:


path = Path('data')
dest = path
dest.mkdir(parents=True, exist_ok=True)


# In[4]:


# copy the data over to working directory for easier manipulations
get_ipython().system('cp -r ../input/* {path}/')


# In[5]:


path.ls()


# In[6]:


ls data/bluebook-for-bulldozers


# In[7]:


# read in the dataset. Since the Test.csv and Valid.csv doesn't have label, it will be used to create our own validation set. 
train_df = pd.read_csv('/kaggle/working/data/bluebook-for-bulldozers/train/Train.csv', low_memory=False, parse_dates=["saledate"])
valid_df = pd.read_csv('/kaggle/working/data/bluebook-for-bulldozers/Valid.csv', low_memory=False, parse_dates=["saledate"])
test_df = pd.read_csv('/kaggle/working/data/bluebook-for-bulldozers/Test.csv', low_memory=False, parse_dates=["saledate"])


# In[8]:


len(train_df), len(test_df)


# In[9]:


train_df.head()


# In[10]:


len(train_df),len(valid_df), len(test_df)


# In[11]:


# Sort the dataframe on 'saledate' so we can easily create a validation set that data is in the 'future' of what's in the training set
train_df = train_df.sort_values(by='saledate', ascending=False)
train_df = train_df.reset_index(drop=True)


# In[12]:


# The evaluation method for this Kaggle competition is REMLE, so if we take the log on dependant variable, we can just use RSME as evaluation metrics. 
# Simpler handling this way. 
train_df.SalePrice = np.log(train_df.SalePrice)


# In[13]:


# The only feature engineering we do is add some meta-data from the sale date column, using 'add_datepart' function in fast.ai
add_datepart(train_df, "saledate", drop=False)
add_datepart(test_df, "saledate", drop=False)


# In[14]:


# check and see whether all date related meta data is added.
def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)
        
display_all(train_df.tail(10).T)


# In[15]:


# Defining pre-processing we want for our fast.ai DataBunch
procs=[FillMissing, Categorify, Normalize]


# In[16]:


train_df.dtypes
g = train_df.columns.to_series().groupby(train_df.dtypes).groups
g


# In[17]:


# prepare categorical and continous data columns for building Tabular DataBunch.
cat_vars = ['SalesID', 'YearMade', 'MachineID', 'ModelID', 'datasource', 'auctioneerID', 'UsageBand', 'fiModelDesc', 'fiBaseModel', 'fiSecondaryDesc', 'fiModelSeries', 'fiModelDescriptor', 'ProductSize', 
            'fiProductClassDesc', 'state', 'ProductGroup', 'ProductGroupDesc', 'Drive_System', 'Enclosure', 'Forks', 'Pad_Type', 'Ride_Control', 'Stick', 'Transmission', 'Turbocharged', 'Blade_Extension', 
            'Blade_Width', 'Enclosure_Type', 'Engine_Horsepower', 'Hydraulics', 'Pushblock', 'Ripper', 'Scarifier', 'Tip_Control', 'Tire_Size', 'Coupler', 'Coupler_System', 'Grouser_Tracks', 'Hydraulics_Flow', 
            'Track_Type', 'Undercarriage_Pad_Width', 'Stick_Length', 'Thumb', 'Pattern_Changer', 'Grouser_Type', 'Backhoe_Mounting', 'Blade_Type', 'Travel_Controls', 'Differential_Type', 'Steering_Controls', 
            'saleYear', 'saleMonth', 'saleWeek', 'saleDay', 'saleDayofweek', 'saleDayofyear', 'saleIs_month_end', 'saleIs_month_start', 'saleIs_quarter_end', 'saleIs_quarter_start', 'saleIs_year_end', 
            'saleIs_year_start'
           ]

cont_vars = ['MachineHoursCurrentMeter', 'saleElapsed']


# In[18]:


# rearrange training set before feed into the databunch
dep_var = 'SalePrice'
df = train_df[cat_vars + cont_vars + [dep_var,'saledate']].copy()


# In[19]:


# Look at the time period of test set, make sure it's more recent
test_df['saledate'].min(), test_df['saledate'].max()


# In[20]:


# Calculate where we should cut the validation set. We pick the most recent 'n' records in training set where n is the number of entries in test set. 
cut = train_df['saledate'][(train_df['saledate'] == train_df['saledate'][len(test_df)])].index.max()
cut


# In[21]:


valid_idx = range(cut)


# In[22]:


df[dep_var].head()


# In[23]:


# Use fast.ai datablock api to put our training data into the DataBunch, getting ready for training
data = (TabularList.from_df(df, path=path, cat_names=cat_vars, cont_names=cont_vars, procs=procs)
                   .split_by_idx(valid_idx)
                   .label_from_df(cols=dep_var, label_cls=FloatList)
                   .databunch())


# In[24]:


# We want to limit the price range for our prediction to be within the history sale price range, so we need to calculate the y_range
# Note that we multiplied the maximum of 'SalePrice' by 1.2 so when we apply sigmoid, the upper limit will also be covered. 
max_y = np.max(train_df['SalePrice'])*1.2
y_range = torch.tensor([0, max_y], device=defaults.device)
y_range


# In[25]:


# Create our tabular learner. The dense layer is 1000 and 500 two layer NN. We used dropout, hai 
learn = tabular_learner(data, layers=[1000,500], ps=[0.001,0.01], emb_drop=0.04, 
                        y_range=y_range, metrics=rmse)


# In[26]:


learn.model


# In[27]:


learn.lr_find()


# In[28]:


learn.recorder.plot()


# In[29]:


learn.fit_one_cycle(2, 1e-2, wd=0.2)


# In[30]:


learn.fit_one_cycle(5, 3e-4, wd=0.2)


# In[31]:


# learn.fit_one_cycle(5, 3e-4, wd=0.2)


# In[ ]:




