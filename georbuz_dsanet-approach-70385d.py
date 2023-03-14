#!/usr/bin/env python
# coding: utf-8



week = 4




import numpy as np 
import pandas as pd
import datetime
import os
import matplotlib.pyplot as plt




train = pd.read_csv(f"/kaggle/input/covid19-global-forecasting-week-{week}/train.csv")

train.loc[:, "geo"] = np.where(train.loc[:, "Province_State"].isna(), train.loc[:, "Country_Region"], train.loc[:, "Country_Region"] + "_" + train.loc[:, "Province_State"])

train.loc[:, 'Date'] = pd.to_datetime(train.loc[:, 'Date'])
train = train.loc[train.loc[:, 'Date'] > '2020-02-20', :]
#train.drop(train[train.ConfirmedCases == 0].index, inplace = True)
train_last_date = train.Date.unique()[-1]
print(f"Dataset has training data untill : {str(train_last_date)[:10]}")
print(f"Training dates: {len(train.Date.unique())}")




all_countries = train.groupby(['Date']).agg({'ConfirmedCases': np.sum}).reset_index()

all_countries['sh'] = train.groupby(['Date']).agg({'ConfirmedCases': np.sum}).reset_index().shift(1)['ConfirmedCases']

all_countries['sh'] = (all_countries['ConfirmedCases'] / all_countries['sh']) - 1

all_countries = all_countries.set_index("Date")

all_countries['sh'].plot()




all_countries = train.groupby(['Date']).agg({'Fatalities': np.sum}).reset_index()

all_countries['sh'] = train.groupby(['Date']).agg({'Fatalities': np.sum}).reset_index().shift(1)['Fatalities']

all_countries['sh'] = all_countries['Fatalities'] - all_countries['sh']

all_countries = all_countries.set_index("Date")

all_countries['sh'].plot()




train_old = train.loc[train.loc[:, 'Date'] > '2020-04-01', :]




test = pd.read_csv(f"/kaggle/input/covid19-global-forecasting-week-{week}/test.csv")
test.loc[:, 'Date'] = pd.to_datetime(test.loc[:, 'Date'])
test_first_date = test.loc[:, 'Date'].values[0]
test_last_date = test.loc[:, 'Date'].values[-1]
print(f'Test period from {str(test_first_date)[:10]} to {str(test_last_date)[:10]}')




test




period = (np.array(test_last_date, dtype='datetime64[D]').astype(np.int64) - np.array(train_last_date, dtype='datetime64[D]').astype(np.int64))




print(f"Prediction days: {(np.array(test_last_date, dtype='datetime64[D]').astype(np.int64) - np.array(train_last_date, dtype='datetime64[D]').astype(np.int64))+1}")
print(f"Public set: {(np.array(train_last_date, dtype='datetime64[D]').astype(np.int64) - np.array(test_first_date, dtype='datetime64[D]').astype(np.int64))+1}")
print(f"Full prediction set: {(np.array(test_last_date, dtype='datetime64[D]').astype(np.int64) - np.array(test_first_date, dtype='datetime64[D]').astype(np.int64))+1}")




win = 32
hor = 2




base_1 = train.pivot(index='Date', columns="geo", values='ConfirmedCases').iloc[-(win+1),:].values
base_2 = train.pivot(index='Date', columns="geo", values='Fatalities').iloc[-(win+1),:].values




train.loc[:, 'ConfirmedCases'] = train.loc[:, 'ConfirmedCases'] - train.groupby('geo')['ConfirmedCases'].shift(periods=1)
train.loc[:, 'Fatalities'] = train.loc[:, 'Fatalities'] - train.groupby('geo')['Fatalities'].shift(periods=1)

train = train.groupby('geo').tail(train.groupby('geo').size().values[0]-1)

train.loc[train.loc[:, 'ConfirmedCases'] < 0, 'ConfirmedCases'] =  0.0
train.loc[train.loc[:, 'Fatalities'] < 0, 'Fatalities'] = 0.0
train




train.Country_Region.value_counts()




train_cases = train.pivot(index='Date', columns="geo", values='ConfirmedCases').iloc[:-hor,:].values
valid_cases = train.pivot(index='Date', columns="geo", values='ConfirmedCases').iloc[-(win+hor):,:].values

train_fatal = train.pivot(index='Date', columns="geo", values='Fatalities').iloc[:-hor,:].values
valid_fatal = train.pivot(index='Date', columns="geo", values='Fatalities').iloc[-(win+hor):,:].values




get_ipython().run_cell_magic('bash', '', '\npip install pytorch_lightning')




import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as ptl

from torch import optim
from torch.utils.data import DataLoader
from collections import OrderedDict




def rmsle(predict, target): 
    return torch.sqrt(((torch.log(predict + 1) - torch.log(target + 1))**2).mean())




class MTSFDataset(torch.utils.data.Dataset):

    def __init__(self, window, horizon, set_type, tra, validation):
        
        assert type(set_type) == type('str')
        
        self.window = window
        self.horizon = horizon
        self.tra = tra
        self.validation = validation
        self.set_type = set_type
        
        if set_type == 'train':
            rawdata = tra
        elif set_type == 'validation':
            rawdata = validation

        _, self.len, self.var_num = rawdata.shape
        self.sample_num = max(self.len - self.window - self.horizon + 1, 0)
        self.samples, self.labels = self.__getsamples(rawdata)
    
    def __getsamples(self, data):
        
        x, y = [], []

        for j in range(len(data)):
            X = torch.zeros((self.sample_num, self.window, self.var_num))
            Y = torch.zeros((self.sample_num, 1, self.var_num))#1))#

            for i in range(self.sample_num):
                start = i
                end = i + self.window
                X[i, :, :] = torch.from_numpy(data[j, start:end, :])
                Y[i, :, :] = torch.from_numpy(data[j, end+self.horizon-1, :])#torch.from_numpy(np.array())#-1])#.reshape(1,1,1)
            
            x.append(X)
            y.append(Y)

        return (torch.cat(x), torch.cat(y))

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        sample = [self.samples[idx, :, :], self.labels[idx, :, :]]

        return sample




class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)

        output, attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input)

        enc_output = self.pos_ffn(enc_output)

        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)

        dec_output = self.pos_ffn(dec_output)

        return dec_output, dec_slf_attn, dec_enc_attn


class Single_Global_SelfAttn_Module(nn.Module):

    def __init__(
            self,
            window, n_multiv, n_kernels, w_kernel,
            d_k, d_v, d_model, d_inner,
            n_layers, n_head, drop_prob=0.1):
        '''
        Args:
        window (int): the length of the input window size
        n_multiv (int): num of univariate time series
        n_kernels (int): the num of channels
        w_kernel (int): the default is 1
        d_k (int): d_model / n_head
        d_v (int): d_model / n_head
        d_model (int): outputs of dimension
        d_inner (int): the inner-layer dimension of Position-wise Feed-Forward Networks
        n_layers (int): num of layers in Encoder
        n_head (int): num of Multi-head
        drop_prob (float): the probability of dropout
        '''

        super(Single_Global_SelfAttn_Module, self).__init__()

        self.window = window
        self.w_kernel = w_kernel
        self.n_multiv = n_multiv
        self.d_model = d_model
        self.drop_prob = drop_prob
        self.conv2 = nn.Conv2d(1, n_kernels, (window, w_kernel))
        self.in_linear = nn.Linear(n_kernels, d_model)
        self.out_linear = nn.Linear(d_model, n_kernels)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=drop_prob)
            for _ in range(n_layers)])

    def forward(self, x, return_attns=False):

        x = x.view(-1, self.w_kernel, self.window, self.n_multiv)
        x2 = F.relu(self.conv2(x))
        x2 = nn.Dropout(p=self.drop_prob)(x2)
        x = torch.squeeze(x2, 2)
        x = torch.transpose(x, 1, 2)
        src_seq = self.in_linear(x)

        enc_slf_attn_list = []

        enc_output = src_seq

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        enc_output = self.out_linear(enc_output)
        return enc_output,


class Single_Local_SelfAttn_Module(nn.Module):

    def __init__(
            self,
            window, local, n_multiv, n_kernels, w_kernel,
            d_k, d_v, d_model, d_inner,
            n_layers, n_head, drop_prob=0.1):
        '''
        Args:
        window (int): the length of the input window size
        n_multiv (int): num of univariate time series
        n_kernels (int): the num of channels
        w_kernel (int): the default is 1
        d_k (int): d_model / n_head
        d_v (int): d_model / n_head
        d_model (int): outputs of dimension
        d_inner (int): the inner-layer dimension of Position-wise Feed-Forward Networks
        n_layers (int): num of layers in Encoder
        n_head (int): num of Multi-head
        drop_prob (float): the probability of dropout
        '''

        super(Single_Local_SelfAttn_Module, self).__init__()

        self.window = window
        self.w_kernel = w_kernel
        self.n_multiv = n_multiv
        self.d_model = d_model
        self.drop_prob = drop_prob
        self.conv1 = nn.Conv2d(1, n_kernels, (local, w_kernel))
        self.pooling1 = nn.AdaptiveMaxPool2d((1, n_multiv))
        self.in_linear = nn.Linear(n_kernels, d_model)
        self.out_linear = nn.Linear(d_model, n_kernels)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=drop_prob)
            for _ in range(n_layers)])

    def forward(self, x, return_attns=False):

        x = x.view(-1, self.w_kernel, self.window, self.n_multiv)
        x1 = F.relu(self.conv1(x))
        x1 = self.pooling1(x1)
        x1 = nn.Dropout(p=self.drop_prob)(x1)
        x = torch.squeeze(x1, 2)
        x = torch.transpose(x, 1, 2)
        src_seq = self.in_linear(x)

        enc_slf_attn_list = []

        enc_output = src_seq

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        enc_output = self.out_linear(enc_output)
        return enc_output,

class AR(nn.Module):

    def __init__(self, window):
        super(AR, self).__init__()
        self.linear = nn.Linear(window, 1)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.linear(x)
        x = torch.transpose(x, 1, 2)
        return x

class DSANet(ptl.LightningModule):

    def __init__(self, tra, validation, n_multiv, batch_size=16, window=64, local=3, n_kernels=32, 
                 drop_prob=0.1, criterion='rmsle_loss', learning_rate=0.005, horizon=14):
        
        super(DSANet, self).__init__()

        self.batch_size = batch_size

        self.window = window
        self.local = local
        self.n_multiv = n_multiv
        self.n_kernels = n_kernels
        self.w_kernel = 1

        self.d_model = 512
        self.d_inner = 2048
        self.n_layers = 6
        self.n_head = 8
        self.d_k = 64
        self.d_v = 64
        self.drop_prob = drop_prob

        self.criterion = criterion
        self.learning_rate = learning_rate
        self.horizon = horizon
        self.tra = tra
        self.validation = validation

        self.__build_model()

    def __build_model(self):

        self.sgsf = Single_Global_SelfAttn_Module(
            window=self.window, n_multiv=self.n_multiv, n_kernels=self.n_kernels,
            w_kernel=self.w_kernel, d_k=self.d_k, d_v=self.d_v, d_model=self.d_model,
            d_inner=self.d_inner, n_layers=self.n_layers, n_head=self.n_head, drop_prob=self.drop_prob)

        self.slsf = Single_Local_SelfAttn_Module(
            window=self.window, local=self.local, n_multiv=self.n_multiv, n_kernels=self.n_kernels,
            w_kernel=self.w_kernel, d_k=self.d_k, d_v=self.d_v, d_model=self.d_model,
            d_inner=self.d_inner, n_layers=self.n_layers, n_head=self.n_head, drop_prob=self.drop_prob)

        self.ar = AR(window=self.window)
        self.W_output1 = nn.Linear(2 * self.n_kernels, 1)
        self.dropout = nn.Dropout(p=self.drop_prob)
        self.active_func = nn.Tanh()

    def forward(self, x):
 
        sgsf_output, *_ = self.sgsf(x)
        slsf_output, *_ = self.slsf(x)
        sf_output = torch.cat((sgsf_output, slsf_output), 2)
        sf_output = self.dropout(sf_output)
        sf_output = self.W_output1(sf_output)

        sf_output = torch.transpose(sf_output, 1, 2)

        ar_output = self.ar(x)

        output = sf_output + ar_output
        output[output < 0] = 0.0

        return output

    def loss(self, labels, predictions):
        if self.criterion == 'l1_loss':
            loss = F.l1_loss(predictions, labels)
        elif self.criterion == 'mse_loss':
            loss = F.mse_loss(predictions, labels)
        elif self.criterion == 'rmsle_loss':
            loss = rmsle(predictions, labels)
        return loss

    def training_step(self, data_batch, batch_i):

        x, y = data_batch

        y_hat = self.forward(x)

        loss_val = self.loss(y, y_hat)

        if self.trainer.use_dp:
            loss_val = loss_val.unsqueeze(0)

        output = OrderedDict({
            'loss': loss_val
        })

        return output

    def validation_step(self, data_batch, batch_i):

        x, y = data_batch

        y_hat = self.forward(x)

        loss_val = self.loss(y, y_hat)

        if self.trainer.use_dp:
            loss_val = loss_val.unsqueeze(0)

        output = OrderedDict({
            'val_loss': loss_val,
            'y': y,
            'y_hat': y_hat,
        })

        return output

    def validation_epoch_end(self, outputs):

        loss_sum = 0
        for x in outputs:
            loss_sum += x['val_loss'].item()
        val_loss_mean = loss_sum / len(outputs)

        y = torch.cat(([x['y'] for x in outputs]), 0)
        y_hat = torch.cat(([x['y_hat'] for x in outputs]), 0)

        num_var = y.size(-1)
        y = y.view(-1, num_var)
        y_hat = y_hat.view(-1, num_var)
        sample_num = y.size(0)

        y_diff = y_hat - y
        y_mean = torch.mean(y)
        y_translation = y - y_mean

        val_rrse = torch.sqrt(torch.sum(torch.pow(y_diff, 2))) / torch.sqrt(torch.sum(torch.pow(y_translation, 2)))

        y_m = torch.mean(y, 0, True)
        y_hat_m = torch.mean(y_hat, 0, True)
        y_d = y - y_m
        y_hat_d = y_hat - y_hat_m
        corr_top = torch.sum(y_d * y_hat_d, 0)
        corr_bottom = torch.sqrt((torch.sum(torch.pow(y_d, 2), 0) * torch.sum(torch.pow(y_hat_d, 2), 0)))
        corr_inter = corr_top / corr_bottom
        val_corr = (1. / num_var) * torch.sum(corr_inter)

        val_mae = (1. / (sample_num * num_var)) * torch.sum(torch.abs(y_diff))

        tqdm_dic = {
            'val_loss': val_loss_mean,
            'RRSE': val_rrse.item(),
            'CORR': val_corr.item(),
            'MAE': val_mae.item()
        }
        return tqdm_dic

    def configure_optimizers(self):

        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler] 

    def __dataloader(self, train):

        set_type = train
        dataset = MTSFDataset(window=self.window, horizon=self.horizon,
                              set_type=set_type, 
                              tra=self.tra, validation=self.validation)

        train_sampler = None
        batch_size = self.batch_size

        should_shuffle = train_sampler is None
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=should_shuffle,
            sampler=train_sampler,
            num_workers=4
        )

        return loader

    @ptl.data_loader
    def train_dataloader(self):
        return self.__dataloader(train='train')

    @ptl.data_loader
    def val_dataloader(self):
        return self.__dataloader(train='validation')




model_cases = DSANet(np.array([train_fatal, train_cases]), np.array([valid_fatal, valid_cases]), train_cases.shape[1], window=win, 
                     learning_rate=0.01, horizon=hor, drop_prob=0.5, batch_size=128)

trainer = ptl.Trainer(val_check_interval=1, max_steps=5000, gpus=1, show_progress_bar=False) 
trainer.fit(model_cases) 




input = np.array([
    train.pivot(index='Date', columns="geo", values='Fatalities').iloc[-win:,:].values,
    train.pivot(index='Date', columns="geo", values='ConfirmedCases').iloc[-win:,:].values
])

for i in range(period):
    
    ins = torch.tensor(input[:, -win:, :]).cuda()
    pred = model_cases(ins.float())
    
    input = np.concatenate([input, np.array(pred.detach().cpu().numpy(), dtype=np.int)], axis=1)




input.shape




pred_size = (np.array(test_last_date, dtype='datetime64[D]').astype(np.int64) - np.array(test_first_date, dtype='datetime64[D]').astype(np.int64))+1




pd.DataFrame(np.array(input[1,:,:].cumsum(0) + base_1, dtype=np.int)[-pred_size:,:], columns=train.pivot(index='Date', columns="geo", values='ConfirmedCases').columns).loc[:, ['US_New York', 'Ukraine', 'Italy', 'Spain']]




pd.DataFrame(np.array(input[0, :, :].cumsum(0) + base_2, dtype=np.int)[-pred_size:,:], columns=train.pivot(index='Date', columns="geo", values='ConfirmedCases').columns).loc[:, ['US_New York', 'Ukraine', 'Italy', 'Spain']]




input[1, :, :] = input[1, :, :].cumsum(0) + base_1
input[0, :, :] = input[0, :, :].cumsum(0) + base_2




import datetime 

def prov(i):
    try:
        return i.split("_")[1]
    except:
        return None

res = pd.DataFrame(input[0, -pred_size:,:], columns=train.pivot(index='Date', columns="geo", values='Fatalities').columns).unstack().reset_index(name='Fatalities')     .merge(
    pd.DataFrame(input[1, -pred_size:,:], columns=train.pivot(index='Date', columns="geo", values='ConfirmedCases').columns).unstack().reset_index(name='ConfirmedCases'),
          how='left', on=['geo', 'level_1']
)

res['Date'] = [test.Date[0] + datetime.timedelta(days=i) for i in res['level_1']]
res['Province_State'] = [prov(i) for i in res['geo']]
res['Country_Region'] = [i.split("_")[0] for i in res['geo']]

res




sub = pd.read_csv("/kaggle/input/covid-sberbank/sample_submission.csv")
sub




sub.country.value_counts().sort_values()




sub = pd.read_csv(f"/kaggle/input/covid19-global-forecasting-week-{week}/submission.csv")

sub = test.merge(res, how='left', on=['Date', 'Province_State', 'Country_Region']).loc[:, ["ForecastId", "ConfirmedCases", "Fatalities"]]

sub['Fatalities'] = np.array(sub['Fatalities'], dtype=np.int)
sub["ConfirmedCases"] = np.array(sub["ConfirmedCases"], dtype=np.int)

sub




sub.to_csv("submission.csv", index=False)

