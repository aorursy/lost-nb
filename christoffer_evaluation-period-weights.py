#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import csv
from collections import defaultdict




SALES = "../input/m5-forecasting-accuracy/sales_train_evaluation.csv"
PRICES = "../input/m5-forecasting-accuracy/sell_prices.csv"
CALENDAR = "../input/m5-forecasting-accuracy/calendar.csv"

# SALES = "../data/raw/sales_train_validation.csv"
# PRICES = "../data/raw/sell_prices.csv"
# CALENDAR = "../data/raw/calendar.csv"

NUM_SERIES = 30490
NUM_TRAINING = 1941
NUM_TEST = NUM_TRAINING + 1 * 28




series_ids = np.empty(NUM_SERIES, dtype=object)
item_ids = np.empty(NUM_SERIES, dtype=object)
dept_ids = np.empty(NUM_SERIES, dtype=object)
cat_ids = np.empty(NUM_SERIES, dtype=object)
store_ids = np.empty(NUM_SERIES, dtype=object)
state_ids = np.empty(NUM_SERIES, dtype=object)




qties = np.zeros((NUM_TRAINING, NUM_SERIES), dtype=float)
sell_prices = np.zeros((NUM_TEST, NUM_SERIES), dtype=float)




get_ipython().run_cell_magic('time', '', 'id_idx = {}\nwith open(SALES, "r", newline=\'\') as f:\n    is_header = True\n    i = 0\n    for row in csv.reader(f):\n        if is_header:\n            is_header = False\n            continue\n        series_id, item_id, dept_id, cat_id, store_id, state_id = row[0:6]\n        # Remove \'_validation/_evaluation\' at end by regenerating series_id\n        series_id = f"{item_id}_{store_id}"\n\n        qty = np.array(row[6:], dtype=float)\n\n        series_ids[i] = series_id\n\n        item_ids[i] = item_id\n        dept_ids[i] = dept_id\n        cat_ids[i] = cat_id\n        store_ids[i] = store_id\n        state_ids[i] = state_id\n\n        qties[:, i] = qty\n\n        id_idx[series_id] = i\n\n        i += 1')




get_ipython().run_cell_magic('time', '', 'wm_yr_wk_idx = defaultdict(list)  # map wmyrwk to d:s\nwith open(CALENDAR, "r", newline=\'\') as f:\n    for row in csv.DictReader(f):\n        d = int(row[\'d\'][2:])\n        wm_yr_wk_idx[row[\'wm_yr_wk\']].append(d)\n        # TODO: Import the rest of the data')




get_ipython().run_cell_magic('time', '', 'with open(PRICES, "r", newline=\'\') as f:\n    is_header = True\n    for row in csv.reader(f):\n        if is_header:\n            is_header = False\n            continue\n        store_id, item_id, wm_yr_wk, sell_price = row\n        series_id = f"{item_id}_{store_id}"\n        series_idx = id_idx[series_id]\n        for d in wm_yr_wk_idx[wm_yr_wk]:\n            sell_prices[d - 1, series_idx] = float(sell_price)')




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




qty_ts




price_ts




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




COARSER = {
    'state_id': [],
    'store_id': ['state_id'],
    'cat_id': [],
    'dept_id': ['cat_id'],
    'item_id': ['cat_id', 'dept_id']
}




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




aggregate_all_levels(qty_ts)




def calculate_weights(totals):
    """Calculate weights from total sales.

    Uses all data in the dataframe so remember to calculate total sales
    (quantity times sell price) and .

    :param totals: Total sales
    :return: Series of weights with (level, *_id, id:) as multi-index
    """
    summed = aggregate_all_levels(totals).sum()
    
    return summed / summed.groupby(level='level').sum()




final_month_totals = (qty_ts.loc[NUM_TRAINING - 28 + 1:NUM_TRAINING + 1] *
                      price_ts.loc[NUM_TRAINING - 28 + 1:NUM_TRAINING + 1])

weights = calculate_weights(final_month_totals)




weights_export = weights.transpose()        .reset_index(level=['level', 'state_id', 'store_id', 'cat_id', 'dept_id', 'item_id'],
                    drop=True)
weights_export.to_csv("weights.csv")

