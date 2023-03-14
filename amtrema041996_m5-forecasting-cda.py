#!/usr/bin/env python
# coding: utf-8



get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import warnings
import seaborn as sns 
import statsmodels.formula.api as smf
import statsmodels.api         as sm
from sklearn.utils import shuffle
from scipy.stats import yeojohnson, yeojohnson_normplot, probplot, boxcox
from statsmodels.stats.outliers_influence import OLSInfluence
warnings.filterwarnings("ignore")

# Display all columns
pd.set_option('display.max_columns', None)

# Use a ggplot style for graphics
plt.style.use('ggplot')




# Load data
files = ['/kaggle/input/m5-forecasting-accuracy/calendar.csv', 
         '/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv',
         '/kaggle/input/m5-forecasting-accuracy/sell_prices.csv']
data = [pd.read_csv(f) for f in files]
dt_calendar, dt_sales, dt_prices = data

# Merge calendar and prices
dt_prices = shuffle(dt_prices, n_samples = 3000000)
dt_complementary = dt_prices.merge(dt_calendar, how='left', on='wm_yr_wk')
del dt_prices
del dt_calendar

# Shuffle data (it is originally ordered) and take n rows (if you don't have enough RAM)
#dt_complementary = shuffle(dt_complementary, n_samples=10000000, random_state=0)




# Count the number of zeros in data 
dt_sales['num_zeros'] = (dt_sales == 0).sum(axis=1)
_ = plt.hist(dt_sales.num_zeros)
plt.xlabel('Number of zeros')
plt.ylabel('Frequency')
plt.title("Distribution of number of zeros by observation")




# Transform date variable to datetime
dt_complementary.date = pd.to_datetime(dt_complementary.date)

# Append the first day of sales to each item 
first_date = dt_complementary.groupby(['store_id','item_id']).agg({'date':'min'}).reset_index().rename(columns={'date':'date_first_sale'})
dt_sales = dt_sales.merge(first_date, how='left', on=['store_id','item_id'])

# Delete data to save RAM
del first_date

# Difference in days between date_first_sales and d_1
dt_sales['since_d_1'] = dt_sales.date_first_sale - pd.to_datetime('2011-01-29')
dt_sales['since_d_1'] = dt_sales['since_d_1'].apply(lambda x: x.days)




# Percentage of zeros since first day of sale
dt_sales['%_zeros_of_total'] = round(((dt_sales.num_zeros - dt_sales.since_d_1) / (1913 - dt_sales.since_d_1)) * 100, 2)
_ = sns.distplot(dt_sales['%_zeros_of_total'])
#_ = plt.hist(dt_sales['%_zeros_of_total'], bins=10)
plt.xlabel('% of zeros')
plt.ylabel('Frequency')
plt.title("Distribution of % of zeros by item")




def perc_bin(num:int):
    if num <= 20:
        output = 'perc_bin_1'
    elif num <= 40:
        output = 'perc_bin_2'
    elif num <= 60:
        output = 'perc_bin_3'
    elif num <= 80:
        output = 'perc_bin_4'
    else:
        output = 'perc_bin_5'
    return output

dt_sales['perc_zeros_bin'] = dt_sales['%_zeros_of_total'].apply(lambda x: perc_bin(x))




# Items with less percentage of zeros
dt_sales_bin1 = dt_sales[dt_sales.perc_zeros_bin == 'perc_bin_1'].drop(columns=['id','num_zeros','date_first_sale','since_d_1','%_zeros_of_total','perc_zeros_bin'])
del dt_sales

# Melt sales data
indicators = [f'd_{i}' for i in range(1,1914)]

dt_sales_bin1_melt = pd.melt(dt_sales_bin1, 
                             id_vars = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
                             value_vars = indicators, 
                             var_name = 'day_key', 
                             value_name = 'sales_day')
del dt_sales_bin1

# Extract the number of day from the day_key variable
dt_sales_bin1_melt['day'] = dt_sales_bin1_melt['day_key'].apply(lambda x: x[2:]).astype(int)




# Data to work with
columns = ['store_id','item_id','sell_price','date','year','d','event_name_1','event_type_1','event_name_2','event_type_2','snap_CA','snap_TX','snap_WI']
dt_work = dt_sales_bin1_melt.merge(dt_complementary[columns], how = 'inner', left_on=['item_id','store_id','day_key'], right_on=['item_id','store_id','d'])
del dt_complementary
print(dt_work.shape)




# If there are null values, print the unique values of the column 
for k,v in dict(dt_work.isnull().sum()).items():
    if v > 0:
        print(f"The unique values for the column {k} are:", dt_work[k].unique(), "\n")




dt_work['event_name_1'] = dt_work['event_name_1'].fillna('Normal')
dt_work['event_name_2'] = dt_work['event_name_2'].fillna('Normal')
dt_work['event_type_1'] = dt_work['event_type_1'].fillna('Non-Special')
dt_work['event_type_2'] = dt_work['event_type_2'].fillna('Non-Special')




# Taken from https://datascience.stackexchange.com/questions/10459/calculation-and-visualization-of-correlation-matrix-with-pandas
corr = dt_work.corr()

cmap = cmap=sns.diverging_palette(5, 250, as_cmap=True)

def magnify():
    return [dict(selector="th",
                 props=[("font-size", "7pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])
]

corr.style.background_gradient(cmap, axis=1)    .set_properties(**{'max-width': '100px', 'font-size': '10pt'})    .set_caption("Correlation between variables")    .set_precision(2)    .set_table_styles(magnify())




print(dt_work.sales_day.describe(),
      f"The 1th percentile is {dt_work.sales_day.quantile(.01)}", "\n",
      f"The 5th percentile is {dt_work.sales_day.quantile(.05)}", "\n",
      f"The 10th percentile is {dt_work.sales_day.quantile(.1)}", "\n",
      f"The 15th percentile is {dt_work.sales_day.quantile(.15)}", "\n",
      f"The 20th percentile is {dt_work.sales_day.quantile(.15)}", "\n",
      f"The 90th percentile is {dt_work.sales_day.quantile(.90)}", "\n",
      f"The 98th percentile is {dt_work.sales_day.quantile(.98)}", "\n",
      f"The 99th percentile is {dt_work.sales_day.quantile(.99)}", "\n",
      f"The 99th percentile is {dt_work.sales_day.quantile(.995)}", "\n",
      f"The 99th percentile is {dt_work.sales_day.quantile(.999)}", "\n")




fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)

# Yeo-Johson Normality Plot 
lmbd_yj = yeojohnson_normplot(dt_work.sales_day, -10, 10, plot=ax)
sales_transformed, maxlmbd = yeojohnson(dt_work.sales_day)

ax.axvline(maxlmbd, color='r')
plt.show()




# QQ-plot of sales transformation 
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
qq_plot = probplot(sales_transformed, dist="norm", plot=ax)
ax.set_title("QQ-plot for normal distribution")
plt.show()




plt.figure(figsize=(12, 6))

# Left side figure
plt.subplot(1,2,1)
_ = sns.distplot(dt_work.sales_day)
plt.title("Original Distribution")

# Right side figure
plt.subplot(1,2,2)
_ = sns.distplot(sales_transformed, rug=True)
plt.title("With Yeo-Johnson Transformation")
plt.tight_layout(pad=1)
plt.show()




dt_work['sales_day_yj'] = sales_transformed

# Filter by columns of interest
cols_to_drop = ['item_id', 'day_key', 'day', 'date', 'd']
dt_reg = dt_work.drop(columns=cols_to_drop)

# Covert to category type 
for k, v in dict(dt_reg.dtypes).items(): 
    if v == 'object':
        dt_reg[k] = dt_reg[k].astype('category')
 
# Dummy variables    
dt_reg = pd.get_dummies(dt_reg)




# Model with Yeo-Johnson transformation
formula_yj="sales_day_yj ~ "

for col in dt_reg.columns[8:]:
    formula_yj+='Q("'+col+'")+'
    
formula_yj = formula_yj + 'sell_price + year'

model_yj = smf.ols(formula = formula_yj, data = dt_reg).fit()
model_yj.summary()




# Model with Yeo-Johnson transformation
formula="sales_day ~ "

for col in dt_reg.columns[8:]:
    formula+='Q("'+col+'")+'
    
formula = formula + 'sell_price + year'

model_orig = smf.ols(formula = formula, data = dt_reg).fit()
model_orig.summary()




# Residuals vs fitted values 
model_fitted = model_yj.fittedvalues
model_residuals = model_yj.resid
fig = plt.figure(figsize = (8, 6))
sns.scatterplot(model_fitted, 
                model_residuals,
                alpha=0.5
                  )
plt.title('Residuals vs Fitted')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')




# Cook's distance values
outlierInfluence = OLSInfluence(model_yj)
(c, p) = outlierInfluence.cooks_distance

# Leverage and normalized residuals
model_leverage = model_yj.get_influence().hat_matrix_diag
model_norm_residuals = model_yj.get_influence().resid_studentized_internal
model_cooks = model_yj.get_influence().cooks_distance[0]




plt.figure(figsize=(12, 6))

# Cook's distance plot
plt.subplot(1,2,1)
plt.stem(np.arange(20000), c[:20000], markerfmt=",")
plt.title("Cook's distance plot for the residuals",fontsize=16)
plt.grid(False)


# Scatterplot of leverage vs normalized residuals
plt.subplot(1,2,2)
plt.scatter(model_leverage[:200000], 
            model_norm_residuals[:200000], alpha=0.5)
sns.regplot(model_leverage[:200000], 
            model_norm_residuals[:200000],
            scatter=False,
            ci=False,
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8});
plt.xlim(-0.0005, 0.0010)
plt.title('Residuals vs Leverage')
plt.xlabel('Leverage')
plt.ylabel('Standardized Residuals')

plt.tight_layout(1.0)
plt.show()




# Taken from https://stackoverflow.com/questions/42658379/variance-inflation-factor-in-python/54857466

def variance_inflation_factors(exog_df):
    '''
    Parameters
    ----------
    exog_df : dataframe, (nobs, k_vars)
        design matrix with all explanatory variables, as for example used in
        regression. One recommendation is that if VIF is greater than 5, then 
        the explanatory variable given by exog_idx is highly collinear with the 
        other explanatory variables, and the parameter estimates will have large 
        standard errors because of this.

    Returns
    -------
    vif : Series
        variance inflation factors
    '''
    exog_df = sm.add_constant(exog_df)
    vifs = pd.Series([1 / (1. - sm.OLS(exog_df[col].values, 
                                       exog_df.loc[:, exog_df.columns != col].values) \
                                   .fit() \
                                   .rsquared
                           ) 
                           for col in exog_df],
        index=exog_df.columns,
        name='VIF'
    )
    return vifs

cols = ['sell_price', 'year', 'snap_CA', 'snap_TX', 'snap_WI', 'sales_day_yj']
variance_inflation_factors(dt_reg[cols])




iterables = [('state_id_CA','snap_CA'), ('state_id_TX', 'snap_TX'), ('state_id_WI', 'snap_WI')]


for i in iterables:
    state, snap = i
    sns.lmplot(x='year',
               y='sales_day_yj', 
               data=dt_reg[dt_reg[state]==1], 
               hue=snap,
               col=snap,
               height=8, 
               scatter_kws={"s": 10},
               x_jitter=.25,
               y_jitter=.05
              )
    
plt.tight_layout(1)
plt.show()




for i in iterables:
    state, snap = i
    formula = (f'sales_day_yj ~ year*C({snap})')
    model = smf.ols(formula=formula, data=dt_reg[dt_reg[state]==1]).fit()
    print(f"Model for {state}",
          "\n",
          "-----------------------------------------------------------",
          "\n",
          model.summary(),
          "\n",
          "-----------------------------------------------------------")




dt_reg['log_sell_price'] = np.log(dt_reg.sell_price + 1)

for i in iterables:
    state, snap = i
    sns.lmplot(x='year',
               y='log_sell_price', 
               data=dt_reg[dt_reg[state]==1], 
               hue=snap,
               col=snap,
               height=8, 
               scatter_kws={"s": 10},
               x_jitter=.15,
               y_jitter=.05)
    
plt.tight_layout(1)
plt.show()




for i in iterables:
    state, snap = i
    formula = (f'log_sell_price ~ year*C({snap})')
    model = smf.ols(formula=formula, data=dt_reg[dt_reg[state]==1]).fit()
    print(f"Model for {state}",
          "\n",
          "-----------------------------------------------------------",
          "\n",
          model.summary(),
          "\n",
          "-----------------------------------------------------------")




for i in iterables:
    state, snap = i
    formula = (f'log_sell_price ~ dept_id_FOODS_1*C({snap}) + dept_id_FOODS_2*C({snap}) + dept_id_FOODS_3*C({snap}) + dept_id_HOBBIES_1*C({snap}) + dept_id_HOUSEHOLD_1*C({snap}) + dept_id_HOUSEHOLD_2*C({snap})')
    model = smf.ols(formula=formula, data=dt_reg[dt_reg[state]==1]).fit()
    print(f"Model for {state}",
          "\n",
          "-----------------------------------------------------------",
          "\n",
          model.summary(),
          "\n",
          "-----------------------------------------------------------")




iterables = [('CA','snap_CA'), ('TX', 'snap_TX'), ('WI', 'snap_WI')]

for i in iterables:
    state, snap = i
    formula = (f'np.log(sell_price+1) ~ C(dept_id)*C({snap})')
    model = smf.ols(formula=formula, data=dt_work[dt_work.state_id==state]).fit()
    print(f"Model for {state}",
          "\n",
          "-----------------------------------------------------------",
          "\n",
          model.summary(),
          "\n",
          "-----------------------------------------------------------")

