#!/usr/bin/env python
# coding: utf-8

# In[88]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import time
from datetime import datetime
from scipy import integrate, optimize
import warnings
warnings.filterwarnings('ignore')

# ML libraries
import tensorflow as tf
print(tf.__version__)


# In[ ]:


submission_example = pd.read_csv("../input/covid19-global-forecasting-week-2/submission.csv")
test = pd.read_csv("../input/covid19-global-forecasting-week-2/test.csv")
train = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")
world_population = pd.read_csv("/kaggle/input/population-by-country-2020/population_by_country_2020.csv")


# In[ ]:


train.head()


# In[ ]:


submission_example


# In[ ]:


train.Country_Region.value_counts()


# In[ ]:


# Merge train and test, exclude overlap
dates_overlap = ['2020-03-19','2020-03-20','2020-03-21','2020-03-22','2020-03-23', '2020-03-24', '2020-03-25', '2020-03-26', '2020-03-27', '2020-03-28', '2020-03-29', '2020-03-30']
train2 = train.loc[~train['Date'].isin(dates_overlap)]
df = pd.concat([train2, test], axis = 0, sort=False)


# In[ ]:


train.dtypes


# In[ ]:


df['Date'] = pd.to_datetime(df.Date)
df.sort_values(by='Date', inplace = True)
df.reset_index(drop=True)


# In[ ]:


world_population


# In[ ]:


df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Dayofweek'] = df['Date'].dt.dayofweek
df['Is_weekend'] = np.where(df['Dayofweek'].isin([6,7]),1,0)
df['Quarter'] = df['Date'].dt.quarter
df['Year'] = df['Date'].dt.year

# Fill null values 
df['Province_State'].fillna("None", inplace=True)
df['ConfirmedCases'].fillna(0, inplace=True)
df['Fatalities'].fillna(0, inplace=True)
df['Id'].fillna(-1, inplace=True)
df['ForecastId'].fillna(-1, inplace=True)


# In[ ]:


df


# In[ ]:


df.columns


# In[ ]:


# Apply log transformation to all ConfirmedCases and Fatalities columns, except for trends
df[['ConfirmedCases', 'Fatalities']] = df[['ConfirmedCases', 'Fatalities']].astype('float64')
df[['ConfirmedCases', 'Fatalities']] = df[['ConfirmedCases', 'Fatalities']].apply(lambda x: np.log(x))


# In[ ]:


# Replace infinites
df.replace([np.inf, -np.inf], 0, inplace=True)


# In[ ]:


df.dtypes


# In[ ]:


world_population = world_population[['Country (or dependency)', 'Population (2020)', 'Density (P/Km²)', 'Land Area (Km²)', 'Med. Age', 'Urban Pop %']]
world_population.columns = ['Country (or dependency)', 'Population (2020)', 'Density', 'Land Area', 'Med Age', 'Urban Pop']


# In[ ]:


world_population['Urban Pop']


# In[ ]:


# Replace United States by US
world_population.loc[world_population['Country (or dependency)']=='United States', 'Country (or dependency)'] = 'US'

# Remove the % character from Urban Pop values
world_population['Urban Pop'] = world_population['Urban Pop'].str.rstrip('%')


# In[ ]:


# Replace Urban Pop and Med Age "N.A" by their respective modes, then transform to int
world_population.loc[world_population['Urban Pop']=='N.A.', 'Urban Pop'] = int(world_population.loc[world_population['Urban Pop']!='N.A.', 'Urban Pop'].mode()[0])
world_population['Urban Pop'] = world_population['Urban Pop'].astype('int16')
world_population.loc[world_population['Med Age']=='N.A.', 'Med Age'] = int(world_population.loc[world_population['Med Age']!='N.A.', 'Med Age'].mode()[0])
world_population['Med Age'] = world_population['Med Age'].astype('int16')


# In[ ]:


world_population


# In[ ]:


# join the dataset, fill na
df = df.merge(world_population, left_on='Country_Region', right_on='Country (or dependency)', how='left')
df[['Population (2020)', 'Density', 'Land Area', 'Med Age', 'Urban Pop']] = df[['Population (2020)', 'Density', 'Land Area', 'Med Age', 'Urban Pop']].fillna(0)


# In[ ]:


df


# In[ ]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()


# Label encode the day, countries and provinces. 

df['Day_num'] = le.fit_transform(df.Date)

#df.drop('Country (or dependency)', inplace=True, axis=1)
df['Country_Region'] = le.fit_transform(df['Country_Region'])
number_c = df['Country_Region']
countries = le.inverse_transform(df['Country_Region'])
# Save dictionary for exploration purposes
country_dict = dict(zip(countries, number_c)) 


df['Province_State'] = le.fit_transform(df['Province_State'])
number_p = df['Province_State']
province = le.inverse_transform(df['Province_State'])
province_dict = dict(zip(province, number_p)) 


# In[ ]:


df.to_csv('features.csv', index=False)


# In[ ]:


# Split data into train/test
def split_data(data):
    
    # Train set
    x_train = data[data.ForecastId == -1].drop(['ConfirmedCases', 'Fatalities'], axis=1)
    y_train_1 = data[data.ForecastId == -1]['ConfirmedCases']
    y_train_2 = data[data.ForecastId == -1]['Fatalities']

    # Test set
    x_test = data[data.ForecastId != -1].drop(['ConfirmedCases', 'Fatalities'], axis=1)

    # Clean Id columns and keep ForecastId as index
    x_train.drop('Id', inplace=True, errors='ignore', axis=1)
    x_train.drop('ForecastId', inplace=True, errors='ignore', axis=1)
    x_test.drop('Id', inplace=True, errors='ignore', axis=1)
    x_test.drop('ForecastId', inplace=True, errors='ignore', axis=1)
    
    return x_train, y_train_1, y_train_2, x_test


# In[ ]:


Y_train_confirmed


# In[ ]:


X_train


# In[ ]:


# Linear regression model
from sklearn.linear_model import LinearRegression

def lin_reg(X_train, Y_train, X_test):
    lr_model = LinearRegression()

    # Train the model
    lr_model.fit(X_train, Y_train)

    # Make predictions using the testing set
    y_pred = lr_model.predict(X_test)
    
    w = lr_model.coef_ 
    b = lr_model.intercept_ 
    
    print("w: ", w)
    print("b: ", b)
    
    return lr_model, y_pred


# In[ ]:


def lin_regression(X_train, Y_train, X_test):
    lr_model = LinearRegression()

    # Train the model
    lr_model.fit(X_train, Y_train)

    # Make predictions using the testing set
    y_pred = lr_model.predict(X_test)
    
    w = lr_model.coef_ 
    b = lr_model.intercept_ 
    
    print("w: ", w)
    print("b: ", b)
    
    return lr_model, y_pred, w, b


# In[ ]:


# Submission function
def prepare_submissionFormat(df, target1, target2):
    
    prediction_1 = df[target1]
    prediction_2 = df[target2]

    # Submit predictions
    prediction_1 = [int(item) for item in list(map(round, prediction_1))]
    prediction_2 = [int(item) for item in list(map(round, prediction_2))]
    
    submission = pd.DataFrame({
        "ForecastId": df['ForecastId'].astype('int32'), 
        "ConfirmedCases": prediction_1, 
        "Fatalities": prediction_2
    })
    submission.to_csv('test_submission.csv', index=False)


# In[ ]:


ts = time.time()

day_start = 52
df2 = df.loc[df.Day_num >= day_start]

# Set the dfframe where we will update the predictions
df_pred = df[df.ForecastId != -1][['Country_Region', 'Province_State', 'Day_num', 'ForecastId']]
df_pred = df_pred.loc[df_pred['Day_num']>=day_start]
df_pred['Predicted_ConfirmedCases'] = [0]*len(df_pred)
df_pred['Predicted_Fatalities'] = [0]*len(df_pred)


# In[ ]:


df2['Country_Region'].unique()


# In[ ]:


data2 = data.loc[data.Day_num >= day_start]

# Set the dataframe where we will update the predictions
data_pred = data[data.ForecastId != -1][['Country_Region', 'Province_State', 'Day_num', 'ForecastId']]
data_pred = data_pred.loc[data_pred['Day_num']>=day_start]
data_pred['Predicted_ConfirmedCases'] = [0]*len(data_pred)
data_pred['Predicted_Fatalities'] = [0]*len(data_pred)


# In[ ]:


# For every countries, run the linear regression
for c in df2['Country_Region'].unique():
    
    # List of provinces
    provinces_list = df2[df2['Country_Region']==c]['Province_State'].unique()
        
    # If the country has several Province/State informed
    if len(provinces_list)>1:
        for p in provinces_list:
            df_cp = df2[(df2['Country_Region']==c) & (df2['Province_State']==p)]
            X_train1, Y_train_1, Y_train_2, X_test1 = split_data(df_cp)
            model_1, pred_1, w1, b1 = lin_regression(X_train1, Y_train_1, X_test1)
            model_2, pred_2, w2, b2 = lin_regression(X_train1, Y_train_2, X_test1)
            df_pred.loc[((df_pred['Country_Region']==c) & (df2['Province_State']==p)), 'Pred_ConfirmedCases'] = pred_1
            df_pred.loc[((df_pred['Country_Region']==c) & (df2['Province_State']==p)), 'Pred_Fatalities'] = pred_2

    # Predict only the country
    else:
        df_c = df2[(df2['Country_Region']==c)]
        X_train1, Y_train_1, Y_train_2, X_test1 = split_data(df_c)
        model_1, pred_1, w1, b1 = lin_regression(X_train1, Y_train_1, X_test1)
        model_2, pred_2, w2, b2 = lin_regression(X_train1, Y_train_2, X_test1)
        df_pred.loc[(df_pred['Country_Region']==c), 'Pred_ConfirmedCases'] = pred_1
        df_pred.loc[(df_pred['Country_Region']==c), 'Pred_Fatalities'] = pred_2


# In[ ]:


def linreg_basic_all_countries(data, day_start):
    
    data2 = data.loc[data.Day_num >= day_start]

    # Set the dataframe where we will update the predictions
    data_pred = data[data.ForecastId != -1][['Country_Region', 'Province_State', 'Day_num', 'ForecastId']]
    data_pred = data_pred.loc[data_pred['Day_num']>=day_start]
    data_pred['Predicted_ConfirmedCases'] = [0]*len(data_pred)
    data_pred['Predicted_Fatalities'] = [0]*len(data_pred)

    print("Currently running Logistic Regression for all countries")

    # Main loop for countries
    for c in data2['Country_Region'].unique():

        # List of provinces
        provinces_list = data2[data2['Country_Region']==c]['Province_State'].unique()

        # If the country has several Province/State informed
        if len(provinces_list)>1:
            for p in provinces_list:
                data_cp = data2[(data2['Country_Region']==c) & (data2['Province_State']==p)]
                X_train, Y_train_1, Y_train_2, X_test = split_data(data_cp)
                model_1, pred_1 = lin_reg(X_train, Y_train_1, X_test)
                model_2, pred_2 = lin_reg(X_train, Y_train_2, X_test)
                data_pred.loc[((data_pred['Country_Region']==c) & (data2['Province_State']==p)), 'Predicted_ConfirmedCases'] = pred_1
                data_pred.loc[((data_pred['Country_Region']==c) & (data2['Province_State']==p)), 'Predicted_Fatalities'] = pred_2

        # No Province/State informed
        else:
            data_c = data2[(data2['Country_Region']==c)]
            X_train, Y_train_1, Y_train_2, X_test = split_data(data_c)
            model_1, pred_1 = lin_reg(X_train, Y_train_1, X_test)
            model_2, pred_2 = lin_reg(X_train, Y_train_2, X_test)
            data_pred.loc[(data_pred['Country_Region']==c), 'Predicted_ConfirmedCases'] = pred_1
            data_pred.loc[(data_pred['Country_Region']==c), 'Predicted_Fatalities'] = pred_2

    # Apply exponential transf. and clean potential infinites due to final numerical precision
    data_pred[['Predicted_ConfirmedCases', 'Predicted_Fatalities']] = data_pred[['Predicted_ConfirmedCases', 'Predicted_Fatalities']].apply(lambda x: np.expm1(x))
    data_pred.replace([np.inf, -np.inf], 0, inplace=True) 
    
    return data_pred


# In[ ]:


ts = time.time()
day_start = 52
df_pred = linreg_basic_all_countries(df, day_start)
get_submission(df_pred, 'Predicted_ConfirmedCases', 'Predicted_Fatalities')

print("Process finished in ", round(time.time() - ts, 2), " seconds")


# In[ ]:


def get_train_data(scaled_x_data, scaled_y_data, divide_train_valid_index, time_step):
    train_x, train_y = [], []
    normalized_train_feature = scaled_x_data[0: -divide_train_valid_index]
    normalized_train_label = scaled_y_data[0: -divide_train_valid_index]
    for i in range(len(normalized_train_feature) - time_step + 1):
        train_x.append(normalized_train_feature[i:i + time_step].tolist())
        train_y.append(normalized_train_label[i:i + time_step].tolist())
    return train_x, train_y

def get_train_fit_data(scaled_x_data, scaled_y_data, divide_train_valid_index, time_step):
    train_fit_x, train_fit_y = [], []
    normalized_train_feature = scaled_x_data[0: -divide_train_valid_index]
    normalized_train_label = scaled_y_data[0: -divide_train_valid_index]
    train_fit_remain = len(normalized_train_label) % time_step
    train_fit_num = int((len(normalized_train_label) - train_fit_remain) / time_step)
    temp = []
    for i in range(train_fit_num):
        train_fit_x.append(normalized_train_feature[i * time_step:(i + 1) * time_step].tolist())
        temp.extend(normalized_train_label[i * time_step:(i + 1) * time_step].tolist())
    if train_fit_remain > 0:
        train_fit_x.append(normalized_train_feature[-time_step:].tolist())
        temp.extend(normalized_train_label[-train_fit_remain:].tolist())
    for i in temp:
        train_fit_y.append(i[0])
    return train_fit_x, train_fit_y, train_fit_remain

def get_valid_data(scaled_x_data, scaled_y_data, divide_train_valid_index, divide_valid_test_index, time_step):
    valid_x, valid_y = [], []
    normalized_valid_feature = scaled_x_data[-divide_train_valid_index: -divide_valid_test_index]
    normalized_valid_label = scaled_y_data[-divide_train_valid_index: -divide_valid_test_index]
    valid_remain = len(normalized_valid_label) % time_step
    valid_num = int((len(normalized_valid_label) - valid_remain) / time_step)
    temp = []
    for i in range(valid_num):
        valid_x.append(normalized_valid_feature[i * time_step:(i + 1) * time_step].tolist())
        temp.extend(normalized_valid_label[i * time_step:(i + 1) * time_step].tolist())
    if valid_remain > 0:
        valid_x.append(normalized_valid_feature[-time_step:].tolist())
        temp.extend(normalized_valid_label[-valid_remain:].tolist())
    for i in temp:
        valid_y.append(i[0])
    return valid_x, valid_y, valid_remain


def get_test_data(scaled_x_data, scaled_y_data, divide_valid_test_index, time_step):
    test_x, test_y = [], []
    normalized_test_feature = scaled_x_data[-divide_valid_test_index:]
    normalized_test_label = scaled_y_data[-divide_valid_test_index:]
    test_remain = len(normalized_test_label) % time_step
    test_num = int((len(normalized_test_label) - test_remain) / time_step)
    temp = []
    for i in range(test_num):
        test_x.append(normalized_test_feature[i * time_step:(i + 1) * time_step].tolist())
        temp.extend(normalized_test_label[i * time_step:(i + 1) * time_step].tolist())
    if test_remain > 0:
        test_x.append(scaled_x_data[-time_step:].tolist())
        temp.extend(normalized_test_label[-test_remain:].tolist())
    for i in temp:
        test_y.append(i[0])
    return test_x, test_y, test_remain


# In[ ]:


import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
lr = 1e-3  
batch_size = 5  # minibatch 
rnn_unit = 30  # LSTM hiden unit number
input_size = 1  # 
output_size = 1  # 
time_step = 15  # 
epochs = 1000  # 
gradient_threshold = 15  # 
stop_loss = np.float32(0.04)  # 
train_keep_prob = [1.0, 0.5, 1.0] 


# In[ ]:


divide_train_valid_index = 39
divide_valid_test_index = 5


# In[ ]:


def lstm(X, keep_prob):
    batch_size = tf.shape(X)[0]  # minibatch 

    # reshape the input for LSTM, start weights with truncated normal distribution 
    weights = tf.Variable(tf.truncated_normal(shape=[input_size, rnn_unit]))
    biases = tf.Variable(tf.constant(0.1, shape=[rnn_unit, ]))
    input = tf.reshape(X, [-1, input_size])

    input_layer = tf.nn.tanh(tf.matmul(input, weights) + biases)
    input_rnn = tf.nn.dropout(input_layer, keep_prob[0])
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])

    # two layer LSTM，activation function by tanh, suggest to switch to relu if the network's deeper
    initializer = tf.truncated_normal_initializer()
    cell_1 = tf.nn.rnn_cell.LSTMCell(forget_bias=1.0, num_units=rnn_unit, use_peepholes=True, num_proj=None, initializer=initializer, name='lstm_cell_1')
    cell_1_drop = tf.nn.rnn_cell.DropoutWrapper(cell=cell_1, output_keep_prob=keep_prob[1])

    cell_2 = tf.nn.rnn_cell.LSTMCell(forget_bias=1.0, num_units=rnn_unit, use_peepholes=True, num_proj=output_size, initializer=initializer, name='lstm_cell_2')
    cell_2_drop = tf.nn.rnn_cell.DropoutWrapper(cell=cell_2, output_keep_prob=keep_prob[2])

    mutilstm_cell = tf.nn.rnn_cell.MultiRNNCell(cells=[cell_1_drop, cell_2_drop], state_is_tuple=True)
    init_state = mutilstm_cell.zero_state(batch_size, dtype=tf.float32)

    with tf.variable_scope('lstm', reuse=tf.AUTO_REUSE):
        output, state = tf.nn.dynamic_rnn(cell=mutilstm_cell, inputs=input_rnn, initial_state=init_state, dtype=tf.float32)

    return output, state


# In[ ]:


def get_fit_seq(x, remain, sess, output, X, keep_prob, scaler, inverse):
    fit_seq = []
    if inverse:
        # used the scaler for input data before, inverse it back to the original measurement
        temp = []
        for i in range(len(x)):
            next_seq = sess.run(output, feed_dict={X: [x[i]], keep_prob: [1.0, 1.0, 1.0]})
            if i == len(x) - 1:
                temp.extend(scaler.inverse_transform(next_seq[0].reshape(-1, 1))[-remain:])
            else:
                temp.extend(scaler.inverse_transform(next_seq[0].reshape(-1, 1)))
        for i in temp:
            fit_seq.append(i[0])
    else:
        for i in range(len(x)):
            next_seq = sess.run(output,
                                feed_dict={X: [x[i]], keep_prob: [1.0, 1.0, 1.0]})
            if i == len(x) - 1:
                fit_seq.extend(next_seq[0].reshape(1, -1).tolist()[0][-remain:])
            else:
                fit_seq.extend(next_seq[0].reshape(1, -1).tolist()[0])

    return fit_seq


# In[ ]:


def train_lstm():
    X = tf.placeholder(tf.float32, [10, time_step, input_size])
    Y = tf.placeholder(tf.float32, [1, time_step, output_size])

    keep_prob = tf.placeholder(tf.float32, [None])
    output, state = lstm(X, keep_prob)
    loss = tf.losses.mean_squared_error(tf.reshape(output, [-1]), tf.reshape(Y, [-1]))

    # gradients 
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    grads, variables = zip(*optimizer.compute_gradients(loss))
    grads, global_norm = tf.clip_by_global_norm(grads, gradient_threshold)
    train_op = optimizer.apply_gradients(zip(grads, variables))

    X_train_fit = tf.placeholder(tf.float32, [None])
    Y_train_fit = tf.placeholder(tf.float32, [None])
    train_fit_loss = tf.losses.mean_squared_error(tf.reshape(X_train_fit, [-1]), tf.reshape(Y_train_fit, [-1]))

    X_valid = tf.placeholder(tf.float32, [None])
    Y_valid = tf.placeholder(tf.float32, [None])
    valid_fit_loss = tf.losses.mean_squared_error(tf.reshape(X_valid, [-1]), tf.reshape(Y_valid, [-1]))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        fit_loss_seq = []
        valid_loss_seq = []

        for epoch in range(epochs):
            for index in range(len(train_x) - batch_size + 1):
               sess.run(train_op, feed_dict={X: train_x[index: index + batch_size], Y: train_y[index: index + batch_size], keep_prob: train_keep_prob})

            # fit the training sequence and validation sequence
            train_fit_seq = get_fit_seq(train_fit_x, train_fit_remain, sess, output, X, keep_prob, scaler_y, False)
            train_loss = sess.run(train_fit_loss, {X_train_fit: train_fit_seq, Y_train_fit: train_fit_y})
            fit_loss_seq.append(train_loss)

            valid_seq = get_fit_seq(valid_x, valid_remain, sess, output, X, keep_prob, scaler_y, False)
            valid_loss = sess.run(valid_fit_loss, {X_valid: valid_seq, Y_valid: valid_y})
            valid_loss_seq.append(valid_loss)

            print('epoch:', epoch + 1, 'fit loss:', train_loss, 'valid loss:', valid_loss)

            # earily stop 
            # stop_loss needs to be tried multiple times
            if train_loss + valid_loss <= stop_loss:
                train_fit_seq = get_fit_seq(train_fit_x, train_fit_remain, sess, output, X, keep_prob, scaler_y, True)
                valid_fit_seq = get_fit_seq(valid_x, valid_remain, sess, output, X, keep_prob, scaler_y, True)
                test_fit_seq = get_fit_seq(test_x, test_remain, sess, output, X, keep_prob, scaler_y, True)
                print('best epoch: ', epoch + 1)
                break

    return fit_loss_seq, valid_loss_seq, train_fit_seq, valid_fit_seq, test_fit_seq

#fit_loss_seq, valid_loss_seq, train_fit_seq, valid_fit_seq, test_fit_seq = train_lstm()


# In[ ]:




