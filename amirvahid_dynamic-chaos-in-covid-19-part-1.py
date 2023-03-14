#!/usr/bin/env python
# coding: utf-8



# Import libraries
import torch

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters
from torch import nn, optim
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 14, 10
register_matplotlib_converters()

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)




df_train = pd.read_csv("../input/covid19-local-us-ca-forecasting-week-1/ca_train.csv")
df_test = pd.read_csv("../input/covid19-local-us-ca-forecasting-week-1/ca_test.csv")




df_train.head()




df_test.head()




train_drop_cols = df_train.columns[:-3]
test_drop_cols = df_test.columns[1:-1]

train = df_train.copy().drop(train_drop_cols, axis=1)
test = df_test.copy().drop(test_drop_cols, axis=1)




train.head()




test.head()




df = train
df.head()




df.isnull().sum().sum()




train.index = pd.to_datetime(train['Date'])
train.drop(['Date'], axis=1, inplace=True)

test.index = pd.to_datetime(test['Date'])
test.drop(['Date'], axis=1, inplace=True)




train.head()




test.head()




daily_cases = train




plt.plot(daily_cases['ConfirmedCases'])
plt.title("Cumulative confirmed daily cases");




plt.plot(daily_cases['Fatalities'])
plt.title("Cumulative fatalities daily cases");




daily_cases_infected = daily_cases['ConfirmedCases'].diff().fillna(daily_cases['ConfirmedCases'][0]).astype(np.int64)
daily_cases_infected.head()




daily_cases_fatality = daily_cases['Fatalities'].diff().fillna(daily_cases['Fatalities'][0]).astype(np.int64)
daily_cases_fatality.head()




plt.plot(daily_cases_infected)
plt.title("Daily infected cases");




plt.plot(daily_cases_fatality)
plt.title("Daily fatality cases");




daily_cases_infected.shape




daily_cases_fatality.shape




# train = train[train['ConfirmedCases'] > 0]
# train_data, test_data = train_test_split(train, test_size=0.33, random_state=42)
# infection_train = train_data['ConfirmedCases']
# infection_test = test_data['ConfirmedCases']
# fatality_train = train_data['Fatalities']
# fatality_test = test_data['Fatalities']




train_data_infected, test_data_infected = train_test_split(daily_cases_infected, test_size=0.33, random_state=42)
train_data_fatality, test_data_fatality = train_test_split(daily_cases_fatality, test_size=0.33, random_state=42)
infection_train = train_data_infected
fatality_train = train_data_fatality
infection_test = test_data_infected
fatality_test = test_data_fatality




train_data_infected.shape




train_data_fatality.shape




scaler_infection = MinMaxScaler()

scaler_infection = scaler_infection.fit(np.expand_dims(infection_train, axis=1))

infection_train = scaler_infection.transform(np.expand_dims(infection_train, axis=1))

infection_test = scaler_infection.transform(np.expand_dims(infection_test, axis=1))

scaler_fatality = MinMaxScaler()

scaler_fatality = scaler_fatality.fit(np.expand_dims(fatality_train, axis=1))

fatality_train = scaler_fatality.transform(np.expand_dims(fatality_train, axis=1))

fatality_test = scaler_fatality.transform(np.expand_dims(fatality_test, axis=1))





fatality_test




def create_sequences(data, seq_length):
    xs = []
    ys = []

    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)




seq_length = 2

# confirmed cases
X_train_infection, y_train_infection = create_sequences(infection_train, seq_length)
X_test_infection, y_test_infection = create_sequences(infection_test, seq_length)

X_train_infection = torch.from_numpy(X_train_infection).float()
y_train_infection = torch.from_numpy(y_train_infection).float()

X_test_infection = torch.from_numpy(X_test_infection).float()
y_test_infection = torch.from_numpy(y_test_infection).float()

# fatalities
X_train_fatality, y_train_fatality = create_sequences(fatality_train, seq_length)
X_test_fatality, y_test_fatality = create_sequences(fatality_test, seq_length)

X_train_fatality = torch.from_numpy(X_train_fatality).float()
y_train_fatality = torch.from_numpy(y_train_fatality).float()

X_test_fatality = torch.from_numpy(X_test_fatality).float()
y_test_fatality = torch.from_numpy(y_test_fatality).float()




y_test_infection




X_train_infection.shape




X_train_fatality.shape




X_train_infection[:2]




X_train_fatality[:2]




y_train_infection.shape




y_train_fatality.shape




y_train_infection[:2]




y_train_fatality[:2]




X_test_infection.shape




infection_train[:10]




fatality_train[:10]




class CoronaVirusForecast(nn.Module):

  def __init__(self, n_features, n_hidden, seq_len, n_layers=2):
    super(CoronaVirusForecast, self).__init__()

    self.n_hidden = n_hidden
    self.seq_len = seq_len
    self.n_layers = n_layers

    self.lstm = nn.LSTM(
      input_size=n_features,
      hidden_size=n_hidden,
      num_layers=n_layers,
      dropout=0.5
    )

    self.linear = nn.Linear(in_features=n_hidden, out_features=1)

  def reset_hidden_state(self):
    self.hidden = (
        torch.zeros(self.n_layers, self.seq_len, self.n_hidden),
        torch.zeros(self.n_layers, self.seq_len, self.n_hidden)
    )

  def forward(self, sequences):
    lstm_out, self.hidden = self.lstm(
      sequences.view(len(sequences), self.seq_len, -1),
      self.hidden
    )
    last_time_step =       lstm_out.view(self.seq_len, len(sequences), self.n_hidden)[-1]
    y_pred = self.linear(last_time_step)
    return y_pred




def train_model_infection(
  model, 
  infection_train, 
  train_labels, 
  infection_test=None, 
  test_labels=None
):
  loss_fn = torch.nn.MSELoss(reduction='sum')

  optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
  num_epochs = 400

  infection_train_hist = np.zeros(num_epochs)
  infection_test_hist = np.zeros(num_epochs)

  for t in range(num_epochs):
    model.reset_hidden_state()

    y_pred_infection = model(X_train_infection)

    loss = loss_fn(y_pred_infection.float(), y_train_infection)

    if infection_test is not None:
      with torch.no_grad():
        y_test_pred_infection = model(X_test_infection)
        test_loss = loss_fn(y_test_pred_infection.float(), y_test_infection)
      infection_test_hist[t] = test_loss.item()

      if t % 10 == 0:  
        print(f'Epoch {t} train loss: {loss.item()} test loss: {test_loss.item()}')
    elif t % 10 == 0:
      print(f'Epoch {t} train loss: {loss.item()}')

    infection_train_hist[t] = loss.item()
    
    optimiser.zero_grad()

    loss.backward()

    optimiser.step()
  
  return model.eval(), infection_train_hist, infection_test_hist




def train_model_fatality(
  model, 
  fatality_train, 
  train_labels, 
  fatality_test=None, 
  test_labels=None
):
  loss_fn = torch.nn.MSELoss(reduction='sum')

  optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
  num_epochs = 400

  fatality_train_hist = np.zeros(num_epochs)
  fatality_test_hist = np.zeros(num_epochs)

  for t in range(num_epochs):
    model.reset_hidden_state()

    y_pred_fatality = model(X_train_fatality)

    loss = loss_fn(y_pred_fatality.float(), y_train_fatality)

    if fatality_test is not None:
      with torch.no_grad():
        y_test_pred_fatality = model(X_test_fatality)
        test_loss = loss_fn(y_test_pred_fatality.float(), y_test_fatality)
      fatality_test_hist[t] = test_loss.item()

      if t % 10 == 0:  
        print(f'Epoch {t} train loss: {loss.item()} test loss: {test_loss.item()}')
    elif t % 10 == 0:
      print(f'Epoch {t} train loss: {loss.item()}')

    fatality_train_hist[t] = loss.item()
    
    optimiser.zero_grad()

    loss.backward()

    optimiser.step()
  
  return model.eval(), fatality_train_hist, fatality_test_hist




model = CoronaVirusForecast(
  n_features=1, 
  n_hidden=512, 
  seq_len=seq_length, 
  n_layers=2
)
model, infection_train_hist, infection_test_hist = train_model_infection(
  model, 
  X_train_infection, 
  y_train_infection, 
  X_test_infection, 
  y_test_infection
)




plt.plot(infection_train_hist, label="Training loss")
plt.plot(infection_test_hist, label="Test loss")
plt.ylim((0, 5))
plt.legend();




model = CoronaVirusForecast(
  n_features=1, 
  n_hidden=512, 
  seq_len=seq_length, 
  n_layers=2
)
model, fatality_train_hist, fatality_test_hist = train_model_fatality(
  model, 
  X_train_fatality, 
  y_train_fatality, 
  X_test_fatality, 
  y_test_fatality
)




plt.plot(fatality_train_hist, label="Training loss")
plt.plot(fatality_test_hist, label="Test loss")
plt.ylim((0, 5))
plt.legend();




with torch.no_grad():
  test_seq_infection = X_test_infection[:1]
  preds_infection = []
  for _ in range(len(X_test_infection)):
    y_test_pred_infection = model(test_seq_infection)
    pred_infection = torch.flatten(y_test_pred_infection).item()
    preds_infection.append(pred_infection)
    new_seq_infection = test_seq_infection.numpy().flatten()
    new_seq_infection = np.append(new_seq_infection, [pred_infection])
    new_seq_infection = new_seq_infection[1:]
    test_seq_infection = torch.as_tensor(new_seq_infection).view(1, seq_length, 1).float()




with torch.no_grad():
  test_seq_fatality = X_test_fatality[:1]
  preds_fatality = []
  for _ in range(len(X_test_fatality)):
    y_test_pred_fatality = model(test_seq_fatality)
    pred_fatality = torch.flatten(y_test_pred_fatality).item()
    preds_fatality.append(pred_fatality)
    new_seq_fatality = test_seq_fatality.numpy().flatten()
    new_seq_fatality = np.append(new_seq_fatality, [pred_fatality])
    new_seq_fatality = new_seq_fatality[1:]
    test_seq_fatality = torch.as_tensor(new_seq_fatality).view(1, seq_length, 1).float()




true_cases_infection = scaler_infection.inverse_transform(
    np.expand_dims(y_test_infection.flatten().numpy(), axis=0)
).flatten()

predicted_cases_infection = scaler_infection.inverse_transform(
  np.expand_dims(preds_infection, axis=0)
).flatten()




true_cases_fatality = scaler_fatality.inverse_transform(
    np.expand_dims(y_test_fatality.flatten().numpy(), axis=0)
).flatten()

predicted_cases_fatality = scaler_fatality.inverse_transform(
  np.expand_dims(preds_fatality, axis=0)
).flatten()




plt.plot(
  daily_cases_infected.index[:len(infection_train)], 
  scaler_infection.inverse_transform(infection_train).flatten(),
  label='Historical Infected Daily Cases'
)

plt.plot(
  daily_cases_infected.index[len(infection_train):len(infection_train) + len(true_cases_infection)], 
  true_cases_infection,
  label='Real Infected Daily Cases'
)

plt.plot(
  daily_cases_infected.index[len(infection_train):len(infection_train) + len(true_cases_infection)], 
  predicted_cases_infection, 
  label='Predicted Infected Daily Cases'
)

plt.legend();




scaler_infection = MinMaxScaler()

scaler_infection = scaler_infection.fit(np.expand_dims(daily_cases_infected, axis=1))

all_data_infection = scaler_infection.transform(np.expand_dims(daily_cases_infected, axis=1))

all_data_infection.shape




scaler_fatality = MinMaxScaler()

scaler_fatality = scaler_fatality.fit(np.expand_dims(daily_cases_fatality, axis=1))

all_data_fatality = scaler_fatality.transform(np.expand_dims(daily_cases_fatality, axis=1))

all_data_fatality.shape




X_all_infection, y_all_infection = create_sequences(all_data_infection, seq_length)

X_all_infection = torch.from_numpy(X_all_infection).float()
y_all_infection = torch.from_numpy(y_all_infection).float()

model = CoronaVirusForecast(
  n_features=1, 
  n_hidden=512, 
  seq_len=seq_length, 
  n_layers=2
)
model, train_hist_infection, _ = train_model_infection(model, X_all_infection, y_all_infection)




X_all_fatality, y_all_fatality = create_sequences(all_data_fatality, seq_length)

X_all_fatality = torch.from_numpy(X_all_fatality).float()
y_all_fatality = torch.from_numpy(y_all_fatality).float()

model = CoronaVirusForecast(
  n_features=1, 
  n_hidden=512, 
  seq_len=seq_length, 
  n_layers=2
)
model, train_hist_fatality, _ = train_model_fatality(model, X_all_fatality, y_all_fatality)




DAYS_TO_PREDICT_INFECTION = 43

with torch.no_grad():
  test_seq = X_all_infection[:1]
  preds_infection = []
  for _ in range(DAYS_TO_PREDICT_INFECTION):
    y_test_pred_infection = model(test_seq_infection)
    pred_infection = torch.flatten(y_test_pred_infection).item()
    preds_infection.append(pred_infection)
    new_seq_infection = test_seq_infection.numpy().flatten()
    new_seq_infection = np.append(new_seq_infection, [pred_infection])
    new_seq_infection = new_seq_infection[1:]
    test_seq_infection = torch.as_tensor(new_seq_infection).view(1, seq_length, 1).float()




DAYS_TO_PREDICT_FATALITY = 43

with torch.no_grad():
  test_seq = X_all_fatality[:1]
  preds_fatality = []
  for _ in range(DAYS_TO_PREDICT_FATALITY):
    y_test_pred_fatality = model(test_seq_fatality)
    pred_fatality = torch.flatten(y_test_pred_fatality).item()
    preds_fatality.append(pred_fatality)
    new_seq_fatality = test_seq_fatality.numpy().flatten()
    new_seq_fatality = np.append(new_seq_fatality, [pred_fatality])
    new_seq_fatality = new_seq_fatality[1:]
    test_seq_fatality = torch.as_tensor(new_seq_fatality).view(1, seq_length, 1).float()




predicted_cases_infection = scaler_infection.inverse_transform(
  np.expand_dims(preds_infection, axis=0)
).flatten()




predicted_cases_fatality = scaler_fatality.inverse_transform(
  np.expand_dims(preds_fatality, axis=0)
).flatten()




daily_cases_infected.index[-1]




daily_cases_fatality.index[-1]




predicted_index_infection = pd.date_range(
  start=daily_cases_infected.index[-14],
  periods=DAYS_TO_PREDICT_INFECTION + 1,
  closed='right'
)

predicted_cases_infection = pd.Series(
  data=predicted_cases_infection,
  index=predicted_index_infection
)

plt.plot(predicted_cases_infection, label='Predicted Infected Daily Cases')
plt.legend();




predicted_index_fatality = pd.date_range(
  start=daily_cases_fatality.index[-14],
  periods=DAYS_TO_PREDICT_FATALITY + 1,
  closed='right'
)

predicted_cases_fatality = pd.Series(
  data=predicted_cases_fatality,
  index=predicted_index_fatality
)

plt.plot(predicted_cases_fatality, label='Predicted Fatality Daily Cases')
plt.legend();




predicted_index_infection




plt.plot(daily_cases_infected, label='Historical Infected Daily Cases')
plt.plot(predicted_cases_infection, label='Predicted Infected Daily Cases')
plt.legend();




plt.plot(daily_cases_fatality, label='Historical Fatality Daily Cases')
plt.plot(predicted_cases_fatality, label='Predicted Fatality Daily Cases')
plt.legend();









sample_submission = pd.read_csv("../input/covid19-local-us-ca-forecasting-week-1/ca_submission.csv")
sample_submission
submission = pd.DataFrame({
                           'ConfirmedCases': predicted_cases_infection,
                           'Fatalities': predicted_cases_fatality})
submission.index = sample_submission.index
submission['ForecastId'] = sample_submission['ForecastId']
submission = submission[['ForecastId','ConfirmedCases','Fatalities']]
submission.tail()




submission.to_csv("submission.csv", index=False)

