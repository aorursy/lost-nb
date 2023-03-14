#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import pandas as pd, numpy as np
np.set_printoptions(suppress=True)


# In[2]:


get_ipython().system(' ls ../input')


# In[3]:


DATA_ROOT = Path("../input/m5-forecasting-accuracy")


# In[ ]:





# In[4]:


class M5Config:
    def __init__(self,cat_cols=None,sales_path=None, add_fake_categories=True, start=1,
                     end = 1913, days=None, evaluation=False,
                read_calendar=True, read_sales=True, read_prices=True, read_sample_submission=False):
        self.cat_cols = ["id", "cat_id", "state_id", "dept_id", "store_id", "item_id"] if cat_cols is None else cat_cols
        self.col_groups = [
                ('Total', 'X'),
                ('cat_id', 'X'),
                ('state_id', 'X'),
                ('dept_id', 'X'),
                ('store_id', 'X'),
                ('item_id', 'X'),
                ('state_id', 'cat_id'),
                ('state_id', 'dept_id'),
                ('store_id', 'cat_id'),
                ('store_id', 'dept_id'),
                ('state_id','item_id'),
                ('item_id', 'store_id')]

        self.evaluation = False
        self.suffix = "evaluation" if self.evaluation else "validation"

        self.sales_path = DATA_ROOT/f'sales_train_{self.suffix}.csv' if sales_path is None else sales_path
        self.calendar_path = DATA_ROOT/"calendar.csv"
        self.prices_path = DATA_ROOT/"sell_prices.csv"
        self.sample_submission_path = DATA_ROOT/"sample_submission.csv"

        self.add_fake_categories = add_fake_categories

        self.start = start
        self.end = end

        
        if days is None:
            self.set_days()
        else:
            self.days = days

        assert end > 28
        self.set_weight_days()
        
        self.read_calendar = read_calendar
        self.read_sales = read_sales
        self.read_prices = read_prices
        self.read_sample_submission = False
        
    def set_days(self):
        self.days = [f"d_{i}" for i in range(self.start,self.end+1)]
    
    def set_weight_days(self):
        self.weight_days = [f"d_{i}" for i in range(self.end-27, self.end+1)]


# In[ ]:





# In[5]:


class M5Data:
    def __init__(self, config=None):
        self.config = config if config is not None else M5Config()
        
        self.cal = self.read(self.config.calendar_path) if self.config.read_calendar else None
        self.sales = self.read(self.config.sales_path, usecols = self.config.cat_cols
                               + self.config.days) if self.config.read_sales else None
        self.prices = self.read(self.config.prices_path) if self.config.read_prices else None
        self.sample_submission = self.read(self.config.sample_submission_path)                                            if self.config.read_sample_submission else None
        
        if self.config.add_fake_categories:
            self.add_fake_categories()
            
        
    def read(self, path, usecols=None):
        return pd.read_csv(path.as_posix(), usecols=usecols)
    
    
    def add_fake_categories(self):
        self.sales["Total"] = "Total"
        self.sales["X"] = "X"
        
        
        
    
    def _to_42840(self, group):
        assert group in self.config.col_groups
#         print(group, self.sales.columns)
        group = list(group)
        df = self.sales[group+self.config.days].groupby(group)[self.config.days].sum()
        df.reset_index(inplace=True)
        df.rename(columns={group[0]:"level1_val", group[1]:"level2_val"}, inplace=True)
        df["level1_name"] = group[0]
        df["level2_name"] = group[1]
        df["group_id"] = self.config.col_groups.index(tuple(group))
        df = df[["group_id", "level1_name", "level2_name", "level1_val", "level2_val"]+self.config.days]
        
        return df
    
    def to_42840(self):
        
        df = pd.concat([self._to_42840(group) for group in self.config.col_groups], axis=0,sort=False)
        df.sort_values([ "level1_val", "level2_val"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        return df


# In[6]:


get_ipython().run_cell_magic('time', '', '\ndata = M5Data()')


# In[7]:


data.sales.head()


# In[8]:


get_ipython().run_cell_magic('time', '', '\ndata._to_42840(("item_id", "store_id")).head()')


# In[ ]:





# In[9]:


class WeightAndScaleComputer:
    def __init__(self, config=None, data=None):
        self.config = config if config is not None else M5Config()
        self.data = data if data is not None else M5Data(config=self.config)
        
        self.df_42840 = self.data.to_42840()
        
    def get_prices(self):
        prices = self.data.prices.copy()
        cal = self.data.cal

        prices["id"] = ["{}_{}_{}".format(item_id, store_id, self.config.suffix) 
                            for item_id, store_id in zip(prices.item_id, prices.store_id)] 
        
        day_count = cal["d"].str.replace("d_", "").astype(int)

        prices = prices[["wm_yr_wk", "id", "sell_price"]].merge(
                cal.loc[(day_count>= self.config.end-27) & (day_count<= self.config.end), ["wm_yr_wk", "d"]],
                                                on = ["wm_yr_wk"])
        prices = prices.set_index(["id", "d"]).sell_price.fillna(0.).unstack().fillna(0.)
        return prices
        
    
    def get_weights(self):
        
        # Backup old data
        sales_backup = self.data.sales
        df_42840_backup  = self.df_42840

        data = self.data
        sales = data.sales
        
        sales.sort_values("id",inplace=True)
        sales.reset_index(inplace=True, drop=True)
        
        prices = self.get_prices()
        prices.sort_index(inplace=True)
        prices.reset_index(inplace=True, drop=True)
        prices = prices[self.config.weight_days]
        
        for i,col in enumerate(self.config.weight_days):
            sales[col] = sales[col]*prices[col].values
            
        data.sales = sales
        df_42840 = data.to_42840()
        
        df_42840["turnover"] = df_42840[self.config.weight_days].sum(axis=1)
        df_42840["level_turnover"] = df_42840.groupby(["level1_name","level2_name"]).turnover.transform("sum")
        df = df_42840[["group_id", "level1_name", "level2_name", "level1_val", "level2_val"]].copy()
        df["weights"] = df_42840["turnover"]/df_42840["level_turnover"].values
        
        df.sort_values(["level1_val", "level2_val"], inplace=True)
        df.reset_index(drop=True,inplace=True)
        
        # Restore old data
        self.data.sales = sales_backup
        self.df_42840 = df_42840_backup
        
        return df
    
    
    def get_scales(self, kind="mae"): # kind in ['mae', 'mse']
        assert kind in ['mae', 'mse']
        
        df = self.df_42840[self.config.days].values
        
        diff = (np.abs if kind == "mae" else np.square )(df[:, 1:] - df[:, :-1]) 
        
        is_start = df[:, :-1].cumsum(1) >= 1
        
        diff *= is_start
        
        starts = is_start.argmax(1)
        size = df.shape[1] - starts - 1
        
        scales = diff.sum(1)/size
        
        df = self.df_42840[["level1_val", "level2_val"]].copy()
        df["scales"] = scales
        
        df.sort_values(["level1_val", "level2_val"], inplace=True)
        df.reset_index(drop=True,inplace=True)
        
        return df


# In[ ]:





# In[10]:


class WeightedRootMeanSquaredScaledError:
        """A fast routine for the  Weighted Root Mean Squared Scaled Error (WRMSSE).

This might be slow (one to two minutes) at initialisation in order to initiate all the routines required 
to accelerate on-the-fly WRMSSE computation.
"""
        
        
        def __init__(self, scales, weights, data=None):
            self.set_weights_and_scales(weights = weights, scales =  scales)
            self._scales = self.weights_and_scales["scales"].values
            self._weights = self.weights_and_scales["weights"].values
            
            
            if data is None:
                config = M5Config(sales_path=DATA_ROOT/"sales_train_evaluation.csv", start = 1914, end=1914+27,
                                        read_calendar=False,read_prices=False,read_sample_submission=False)
                data = M5Data(config)
                data.sales["id"] = data.sales["id"].str.replace("evaluation", "validation")
                
            self.data = data
            self.df_42840 = self.data.to_42840()[self.data.config.days]
                
            self.cat_data = M5Data(M5Config(read_calendar=False,read_prices=False,read_sample_submission=False))
            self.cat_data  = self.cat_data.sales[self.cat_data.config.cat_cols]
            
            self.submission_config = M5Config(cat_cols = ["id"], days = [f"F{i}" for  i in range(1,29)],
                                read_calendar=False,read_prices=False,read_sample_submission=False)
            
        def set_weights_and_scales(self, weights, scales):
            weights = weights.merge(scales,on=["level1_val", "level2_val"])
            weights = weights.sort_values(["level1_val", "level2_val"]).reset_index(drop=True)
            
            self.weights_and_scales = weights
            
        def score(self, y_pred ):
            """Compute the WRMSSE.
            
            Parameters:
            -----------
            y_true: pd.DataFrame, Path,str-path, M5Data 
                pd.DataFram or path to a pd.DataFrame that consists of daily 30490-42840x28 evaluation data.
                This dataframe must includes the 'id' column. 
                
            y_pred: pd.DataFrame, Path,str-path, M5Data 
                pd.DataFram or path to a pd.DataFrame that consists of daily 30490-42840x28 prediction data.
                This dataframe must includes the 'id' column.
            """
            y_true = self.df_42840.values
            y_pred = self.get_sub_data(y_pred)
            assert y_true.shape == y_pred.shape
            
            rmsse = np.sqrt(np.square(y_true - y_pred).mean(1)/self._scales)
            wrmsse = np.sum(self._weights*rmsse)/12.
            
            return wrmsse
        
        def get_sub_data(self, data):
            config = self.submission_config
            days = list(config.days)
            
            if isinstance(data, (str, Path)):
                config.sales_path = Path(data)
                data = M5Data(config)
            
            elif isinstance(data, pd.DataFrame):
                sales = data
                config.read_sales = False
                config.add_fake_categories = False
                data = M5Data(config)
                data.sales = sales
                config.read_sales = True
                config.add_fake_categories = True
                data.add_fake_categories()
                
            else:
                assert isinstance(data, M5Data), "The object type `{}` is not valid.".format(type(data))
                
                    
            data.sales = data.sales[["id","X","Total"]+ days].merge(self.cat_data, on="id")
            
            sales = data.to_42840()  if len(data.sales) < 42840 else data.sales
            
            return sales[days].values


# In[11]:


get_ipython().run_cell_magic('time', '', '\nwsc = WeightAndScaleComputer()\nwrmsse_scales = wsc.get_scales(kind="mse")\nweights = wsc.get_weights()\nwrmsse_scales.shape, weights.shape')


# In[12]:


wrmsse_scales[(weights.level1_name=="state_id")&(weights.level2_name=="cat_id")]


# In[13]:


WRMSSE =  WeightedRootMeanSquaredScaledError(scales = wrmsse_scales, weights=weights )


# In[14]:


get_ipython().system(' ls ../input/accuracy-best-public-lbs')


# In[15]:


kkiller_048874 = pd.read_csv("../input/accuracy-best-public-lbs/Kkiller_FirstPublicNotebookUnder050_048874.csv")


# In[16]:


get_ipython().run_cell_magic('time', '', '\nWRMSSE.score(kkiller_048874)')


# In[ ]:





# In[17]:


ragnar_064127 = pd.read_csv("../input/accuracy-best-public-lbs/Ragnar_VeryFirstModel_064127.csv")


# In[18]:


get_ipython().run_cell_magic('time', '', '\nWRMSSE.score(ragnar_064127)')


# In[19]:


konstantin_064127 = pd.read_csv("../input/accuracy-best-public-lbs/Konstantin_ThreeShadesofDark_047506.csv")


# In[20]:


get_ipython().run_cell_magic('time', '', '\nWRMSSE.score(konstantin_064127)')


# In[ ]:





# In[21]:


# public LB rank
def get_lb_rank(score):
    """
    Get rank on public LB as of 2020-05-31 23:59:59
    """
    df_lb = pd.read_csv("../input/m5-accuracy-final-public-lb/m5-forecasting-accuracy-publicleaderboard-rank.csv")

    return (df_lb.Score <= score).sum() + 1


# In[ ]:





# In[22]:


get_ipython().run_cell_magic('time', '', '\nprint("Kkiller\'s LB:", get_lb_rank(WRMSSE.score(kkiller_048874)))')


# In[23]:


get_ipython().run_cell_magic('time', '', '\nprint("Ragnar\'s LB:", get_lb_rank(WRMSSE.score(ragnar_064127)))')


# In[24]:


get_ipython().run_cell_magic('time', '', '\nprint("Konstantin\'s LB:", get_lb_rank(WRMSSE.score(konstantin_064127)))')


# In[ ]:




