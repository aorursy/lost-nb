#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import csv
from collections import defaultdict


# In[2]:


SALES = "../input/m5-forecasting-accuracy/sales_train_validation.csv"
PRICES = "../input/m5-forecasting-accuracy/sell_prices.csv"
CALENDAR = "../input/m5-forecasting-accuracy/calendar.csv"

# SALES = "../data/raw/sales_train_validation.csv"
# PRICES = "../data/raw/sell_prices.csv"
# CALENDAR = "../data/raw/calendar.csv"

NUM_SERIES = 30490
NUM_TRAINING = 1913
NUM_TEST = NUM_TRAINING + 2 * 28


# In[3]:


series_ids = np.empty(NUM_SERIES, dtype=object)
item_ids = np.empty(NUM_SERIES, dtype=object)
dept_ids = np.empty(NUM_SERIES, dtype=object)
cat_ids = np.empty(NUM_SERIES, dtype=object)
store_ids = np.empty(NUM_SERIES, dtype=object)
state_ids = np.empty(NUM_SERIES, dtype=object)


# In[4]:


qties = np.zeros((NUM_TRAINING, NUM_SERIES), dtype=float)
sell_prices = np.zeros((NUM_TEST, NUM_SERIES), dtype=float)


# In[5]:


get_ipython().run_cell_magic('time', '', 'id_idx = {}\nwith open(SALES, "r", newline=\'\') as f:\n    is_header = True\n    i = 0\n    for row in csv.reader(f):\n        if is_header:\n            is_header = False\n            continue\n        series_id, item_id, dept_id, cat_id, store_id, state_id = row[0:6]\n        # Remove \'_validation/_evaluation\' at end by regenerating series_id\n        series_id = f"{item_id}_{store_id}"\n\n        qty = np.array(row[6:], dtype=float)\n\n        series_ids[i] = series_id\n\n        item_ids[i] = item_id\n        dept_ids[i] = dept_id\n        cat_ids[i] = cat_id\n        store_ids[i] = store_id\n        state_ids[i] = state_id\n\n        qties[:, i] = qty\n\n        id_idx[series_id] = i\n\n        i += 1')


# In[6]:


get_ipython().run_cell_magic('time', '', 'wm_yr_wk_idx = defaultdict(list)  # map wmyrwk to d:s\nwith open(CALENDAR, "r", newline=\'\') as f:\n    for row in csv.DictReader(f):\n        d = int(row[\'d\'][2:])\n        wm_yr_wk_idx[row[\'wm_yr_wk\']].append(d)\n        # TODO: Import the rest of the data')


# In[7]:


get_ipython().run_cell_magic('time', '', 'with open(PRICES, "r", newline=\'\') as f:\n    is_header = True\n    for row in csv.reader(f):\n        if is_header:\n            is_header = False\n            continue\n        store_id, item_id, wm_yr_wk, sell_price = row\n        series_id = f"{item_id}_{store_id}"\n        series_idx = id_idx[series_id]\n        for d in wm_yr_wk_idx[wm_yr_wk]:\n            sell_prices[d - 1, series_idx] = float(sell_price)')


# In[8]:


qty_ts = pd.DataFrame(qties,
                      index=range(1, NUM_TRAINING + 1),
                      columns=[state_ids, store_ids,
                               cat_ids, dept_ids, item_ids])

qty_ts.index.names = ['d']
qty_ts.columns.names = ['state_id', 'store_id',
                        'cat_id', 'dept_id', 'item_id']

price_ts = pd.DataFrame(sell_prices,
                        index=range(1, NUM_TEST + 1),
                        columns=[state_ids, store_ids,
                                 cat_ids, dept_ids, item_ids])
price_ts.index.names = ['d']
price_ts.columns.names = ['state_id', 'store_id',
                          'cat_id', 'dept_id', 'item_id']


# In[9]:


qty_ts


# In[10]:


price_ts


# In[11]:


LEVELS = {
    1: [],
    2: ['state_id'],
    3: ['store_id'],
    4: ['cat_id'],
    5: ['dept_id'],
    6: ['state_id', 'cat_id'],
    7: ['state_id', 'dept_id'],
    8: ['store_id', 'cat_id'],
    9: ['store_id', 'dept_id'],
    10: ['item_id'],
    11: ['state_id', 'item_id'],
    12: ['item_id', 'store_id']
}


# In[12]:


COARSER = {
    'state_id': [],
    'store_id': ['state_id'],
    'cat_id': [],
    'dept_id': ['cat_id'],
    'item_id': ['cat_id', 'dept_id']
}


# In[13]:


def aggregate_all_levels(df):
    levels = []
    for i in range(1, max(LEVELS.keys()) + 1):
        level = aggregate_groupings(df, i, *LEVELS[i])
        levels.append(level)
    return pd.concat(levels, axis=1)

def aggregate_groupings(df, level_id, grouping_a=None, grouping_b=None):
    """Aggregate time series by summing over optional levels

    New columns are named according to the m5 competition.

    :param df: Time series as columns
    :param level_id: Numeric ID of level
    :param grouping_a: Grouping to aggregate over, if any
    :param grouping_b: Additional grouping to aggregate over, if any
    :return: Aggregated DataFrame with columns as series id:s
    """
    if grouping_a is None and grouping_b is None:
        new_df = df.sum(axis=1).to_frame()
    elif grouping_b is None:
        new_df = df.groupby(COARSER[grouping_a] + [grouping_a], axis=1).sum()
    else:
        assert grouping_a is not None
        new_df = df.groupby(COARSER[grouping_a] + COARSER[grouping_b] +
                            [grouping_a, grouping_b], axis=1).sum()

    new_df.columns = _restore_columns(df.columns, new_df.columns, level_id,
                                      grouping_a, grouping_b)
    return new_df


# In[14]:


def _restore_columns(original_index, new_index, level_id, grouping_a, grouping_b):
    original_df = original_index.to_frame()
    new_df = new_index.to_frame()
    for column in original_df.columns:
        if column not in new_df.columns:
            new_df[column] = None

    # Set up `level` column
    new_df['level'] = level_id

    # Set up `id` column
    if grouping_a is None and grouping_b is None:
        new_df['id'] = 'Total_X'
    elif grouping_b is None:
        new_df['id'] = new_df[grouping_a] + '_X'
    else:
        assert grouping_a is not None
        new_df['id'] = new_df[grouping_a] + '_' + new_df[grouping_b]

    new_index = pd.MultiIndex.from_frame(new_df)
    # Remove "unnamed" level if no grouping
    if grouping_a is None and grouping_b is None:
        new_index = new_index.droplevel(0)
    new_levels = ['level'] + original_index.names + ['id']
    return new_index.reorder_levels(new_levels)


# In[15]:


aggregate_all_levels(qty_ts)


# In[16]:


def calculate_weights(totals):
    """Calculate weights from total sales.

    Uses all data in the dataframe so remember to calculate total sales
    (quantity times sell price) and .

    :param totals: Total sales
    :return: Series of weights with (level, *_id, id:) as multi-index
    """
    summed = aggregate_all_levels(totals).sum()
    
    return summed / summed.groupby(level='level').sum()


# In[17]:


final_month_totals = (qty_ts.loc[NUM_TRAINING - 28 + 1:NUM_TRAINING + 1] *
                      price_ts.loc[NUM_TRAINING - 28 + 1:NUM_TRAINING + 1])

weights = calculate_weights(final_month_totals)


# In[18]:


def cumulative_scales(history, f):
    """Calculate column-wise cumulative scales.
    
    :param history: Values (in day-order)
    :param f: Function to apply to differeces, eg., square for RMSSE, abs for SPL"""
    # Number of values after the first non-zero
    ns = (history.cumsum() > 0).cumsum().shift(1, fill_value=0)
    scales = f(history - history.shift(1)).cumsum() / ns
    
    # Fill parts where no sales with ∞ (effectively ignore series there)
    return scales.fillna(np.inf)


def cumulative_squared_scales(history):
    """Calculate column-wise cumulative scales for RMSSE (squared)."""
    return cumulative_scales(history, np.square)


# In[19]:


def calculate_scales(history):
    """Calculate scales using all of history."""
    return cumulative_squared_scales(history).iloc[-1]


# In[20]:


def evaluate_rmsse(actual_full, forecast_full, history_full):
    scale = calculate_scales(history_full)

    rmsse = ((actual_full - forecast_full).pow(2).mean() / scale)         .pow(1 / 2)
    return rmsse

def evaluate_all_rmsse(actual, forecast, history):
    """Evaluate per-series RMSSE after aggregation"""
    actual_full = aggregate_all_levels(actual)
    forecast_full = aggregate_all_levels(forecast)
    history_full = aggregate_all_levels(history)

    return evaluate_rmsse(actual_full, forecast_full, history_full)

def evaluate_rmsse_wrmsse_per_level(actual, forecast, history, weights):
    """Aggregate series and return per-level RMSSE"""
    rmsse = evaluate_all_rmsse(actual, forecast, history)
    # Average per-series RMSSE over levels
    return rmsse.mean(level='level'), (weights * rmsse).sum(level='level')


# In[21]:


final_month = qty_ts.loc[NUM_TRAINING - 28 + 1:NUM_TRAINING + 1]
final_month_noise = np.clip(final_month + np.random.normal(loc=0.0, scale=0.5, size=(28, 30490)), 0, None)


# In[22]:


noise_rmsse, noise_wrmsse = evaluate_rmsse_wrmsse_per_level(final_month, final_month_noise, 
                                 qty_ts.loc[:NUM_TRAINING - 28 + 1], weights)


# In[23]:


noise_rmsse


# In[24]:


noise_rmsse.mean()


# In[25]:


noise_wrmsse


# In[26]:


noise_wrmsse.mean()


# In[27]:


qty_train = qty_ts.loc[:NUM_TRAINING - 28 + 1]
qty_test = qty_ts.loc[NUM_TRAINING - 28 + 1:NUM_TRAINING + 1]

def evaluate_model(model):
    model.fit(None, qty_train)
    qty_pred = model.forecast(None, 28)
    _, wrmsses = evaluate_rmsse_wrmsse_per_level(qty_test, qty_pred, qty_train, weights)
    return wrmsses.mean()


# In[28]:


class SeasonalNaive(object):
    def __init__(self, period):
        self.period = period

    def fit(self, features, target):
        self.history = target.iloc[-self.period:]
        self.d = self.history.index[-1]

        return self

    def forecast(self, features, h):
        """Forecast the next h days"""
        fs = []
        for i in range(h):
            self.d += 1
            assert self.history.index[0] + self.period == self.d
            f = self.history.iloc[0:1]
            f.index = [self.d]
            fs.append(f)
            self.history = self.history.iloc[1:].append(f)
        return pd.concat(fs)


# In[29]:


evaluate_model(SeasonalNaive(7))


# In[30]:


evaluate_model(SeasonalNaive(28))


# In[31]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


# In[32]:


ACTIVATION = {
    'relu': F.relu,
    'tanh': F.tanh,
    'sigmoid': F.sigmoid,
    'linear': lambda x: x
}

ACTIVATION_FUNCTIONS = list(ACTIVATION.keys())


# In[33]:


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Network(nn.Module):
    def __init__(self, lookback, layer_1_size, layer_1_activation, layer_2_size,
                 layer_2_activation):
        """Initialize parameters and build model."""
        super().__init__()
        self.fc1 = nn.Linear(lookback, layer_1_size)
        self.d1 = nn.Dropout()
        self.f1 = ACTIVATION[layer_1_activation]

        self.fc2 = nn.Linear(layer_1_size, layer_2_size)
        self.f2 = ACTIVATION[layer_2_activation]
        self.d2 = nn.Dropout()

        self.fc3 = nn.Linear(layer_2_size, FORECAST_DAYS)

        self.initialize_weights()

    def initialize_weights(self):
        """Initializes the weights with random values"""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, qties):
        x = self.fc1(qties)
        x = self.d1(x)
        x = self.f1(x)

        x = self.fc2(x)
        x = self.d2(x)
        x = self.f2(x)

        x = self.fc3(x)

        return x


# In[34]:


def rmsse_loss(input, target, scales):
    return (((input - target)**2 / scales).sum() / input.data.nelement()).sqrt()

FORECAST_DAYS = 28

class NeuralNet(object):
    def __init__(self, lookback,
                 layer_1_size, layer_1_activation,
                 layer_2_size, layer_2_activation,
                 batches, shuffle,
                 epochs,
                 device):
        self.device = device

        self.lookback = lookback
        self.layer_1_size = layer_1_size
        self.layer_1_activation = layer_1_activation
        self.layer_2_size = layer_2_size
        self.layer_2_activation = layer_2_activation
        self.batches = batches
        self.shuffle = shuffle

        self.epochs = epochs

    def fit(self, features, target):
        """Attempts to predict the last 28 days"""
        y = (target.iloc[-(FORECAST_DAYS + self.batches):].values
             .transpose())
        X = (target.iloc[-(FORECAST_DAYS + self.lookback
                           + self.batches):-FORECAST_DAYS]
             .values.transpose())

        y = torch.from_numpy(y).float().to(self.device)
        X = torch.from_numpy(X).float().to(self.device)

        # Calculate scales (remember to avoid leaks from the future!)
        scales = cumulative_squared_scales(target)                      .values[
                 -(FORECAST_DAYS + self.batches):-(FORECAST_DAYS - 1)]
        scales = scales.transpose()
        scales = torch.from_numpy(scales).float().to(self.device)


        net = Network(self.lookback,
                      self.layer_1_size,
                      self.layer_1_activation,
                      self.layer_2_size,
                      self.layer_2_activation).to(self.device)
        self.net = net

        optimizer = optim.Adam(net.parameters())

        for epoch in tqdm(range(self.epochs)):
            running_loss = 0.0

            batch_idxs = np.arange(self.batches + 1)
            if self.shuffle:
                np.random.shuffle(batch_idxs)
            for i in batch_idxs:
                optimizer.zero_grad()

                X_run = X[:, i:(i + self.lookback)]
                y_run = y[:, i:(i + FORECAST_DAYS)]
                scales_run = scales[:, i:(i + 1)]


                forecast = net(X_run)

                loss = rmsse_loss(forecast, y_run, scales_run)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            mean_loss = running_loss / self.batches
            # print(f"Epoch {epoch + 1}: Loss {mean_loss:.2f}")

        # Store history
        self.history = target.iloc[-self.lookback:]
        self.d = self.history.index[-1]

        return self

    def forecast(self, features, h):
        # For now, only handle full period
        assert h == FORECAST_DAYS

        assert h <= FORECAST_DAYS

        with torch.no_grad():
            X = self.history.values.transpose()
            X = torch.from_numpy(X).float().to(self.device)
            forecast = self.net(X).cpu().numpy()

            forecast = forecast.transpose()
            self.d += 1

            # TODO: Update self.d properly

            forecast = pd.DataFrame(forecast,
                                    index=range(self.d, self.d + h),
                                    columns=self.history.columns)
            # Remove any negative values
            forecast = forecast.clip(lower=0)

            # TODO: Truncate to h days only and store into history
            self.d += 1 + h
            self.history = forecast
            return forecast


# In[35]:


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# In[36]:


nnet = NeuralNet(lookback=140, 
                 layer_1_size=512, layer_1_activation='relu',
                 layer_2_size=256, layer_2_activation='relu',
                 batches=7, shuffle=True, epochs=64, 
                 device=device)


# In[37]:


evaluate_model(nnet)


# In[38]:


from functools import reduce
import operator

class Ensemble(object):
    def __init__(self, models):
        self.models = models
    
    def fit(self, features, target):
        for model in self.models:
            model.fit(features, target)
    
    def forecast(self, features, h):
        return reduce(operator.add, 
                      [model.forecast(features, h) for model in self.models]) / len(self.models)


# In[39]:


naive_ensemble = Ensemble([SeasonalNaive(7), SeasonalNaive(28)])


# In[40]:


evaluate_model(naive_ensemble)


# In[41]:


large_ensemble = Ensemble([naive_ensemble, nnet])


# In[42]:


evaluate_model(large_ensemble)


# In[43]:


huge_ensemble =  Ensemble([
    SeasonalNaive(7), 
    SeasonalNaive(14), 
    SeasonalNaive(21), 
    SeasonalNaive(28),
    SeasonalNaive(56),
    NeuralNet(lookback=140,
              layer_1_size=512, layer_1_activation='relu',
              layer_2_size=256, layer_2_activation='relu',
              batches=7, shuffle=True, epochs=64, 
              device=device),
    NeuralNet(lookback=365, 
              layer_1_size=1024, layer_1_activation='relu',
              layer_2_size=512, layer_2_activation='relu',
              batches=140, shuffle=True, epochs=64, 
              device=device)])


# In[44]:


evaluate_model(huge_ensemble)


# In[45]:


get_ipython().run_cell_magic('time', '', 'huge_ensemble.fit(None, qty_ts)\nqty_pred = huge_ensemble.forecast(None, 28)')


# In[46]:


def convert_to_submission(forecast):
    """Convert level 12-predictions to submssion"""
    df = aggregate_all_levels(qty_pred)        .transpose()        .reset_index(level=['level', 'state_id', 'store_id', 'cat_id', 'dept_id', 'item_id'],
                    drop=True)
    df.columns = [f"F{i}" for i in range(1, 29)]
    validation = df
    evaluation = df.copy()
    
    validation.index += "_validation"
    evaluation.index += "_evaluation"
    
    return pd.concat([validation, evaluation])


# In[47]:


submission = convert_to_submission(qty_pred)


# In[48]:


# You can't submit zip-files directly from notebooks, otherwise one could use this instead:
# submission.to_csv("submission.zip")
submission.to_csv("submission.csv")

