#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import Image


# In[2]:


import pickle

import pandas as pd
from pathlib import Path
from fastai import *
from fastai.text import *
from sklearn import metrics
from torch import nn


# In[3]:


import neptune


# In[4]:


class MSELoss(nn.MSELoss):
  "Mean Absolute Error Loss"
  def forward(self, input:Tensor, target:Tensor) -> Rank0Tensor:
    return super().forward(input.view(-1), target.view(-1))


# In[5]:


k_folds = 8
target_col_name = 'target'
target_class_col_name = 'class_target'
x_col_name = 'comment_text'

epochs = 1
layer1_lr = 2e-2
layer2_lr = slice(1e-2/(2.6**4),1e-2)
momentum = (0.8,0.7)
loss_function = MSELoss
DROP_MULT = 0.5
freeze_layer_idx = -2
architecture = AWD_LSTM

bs = 40
SEED = 777

identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']


# In[6]:


get_ipython().system('cp ../input/toxic-regression-model/exported_reg_mse_toxic_lm_extended_15ep.pkl ../working/')


# In[7]:


# data
path = Path('../input/jigsaw-unintended-bias-in-toxicity-classification')
clas_csv_file = 'train.csv'
test_csv_file = 'test.csv'

path_lm = Path('../input/data')
lm_csv_file = 'combined_toxic_lm.csv'

# pretrained
pretrained_lm_path = Path('../input/basic-toxic-languagemodel-extended-15ep')
encoder_file = 'fine_tuned_encoder_basic_toxic_lm15ep_extended'
data_lm_file = 'data_reg_basic_toxic_lm_extended_15ep.pkl'
pretrained_reg_path = Path('../working')
reg_model_file = 'exported_reg_mse_toxic_lm_extended_15ep.pkl'

# results
exp_nb = 2
model_cv_result_file = 'toxic_test_cv_{}'
model_prod_cv_result_file = 'toxic_production_cv_{}'
k_folds_file = 'holdout_and_k_folds_idxs_exp_nb_{}.pkl'.format(exp_nb)
models_performance_file = 'toxic_CV_models_performance_exp_nb_{}.pkl'.format(exp_nb)
submission_predictions = 'toxic_CV_submission_exp_nb_{}.csv'.format(exp_nb)

final_model_result_file = 'toxic_final_model_{}'.format(exp_nb)
final_model_prod_result_file = 'toxic_production_final_{}'.format(exp_nb)
final_submission_predictions = 'toxic_final_submission_exp_nb_{}.csv'.format(exp_nb)


# In[8]:


neptune.init(api_token='YOUR_API_TOKEN',
             project_qualified_name='lachonman/toxic-bias')

nep_exp = neptune.create_experiment(name='FastAI_regression_exp_nb_{}'.format(exp_nb),
                          description=' Regression model implemented in FastAI that predicts'
                          'probablity that given text belongs to toxic category. It uses pretrained'
                          'Language Model that was trained on train+test and old toxic text corpora.'
                          'Official metric used to evaluate model performance is "toxic_metric".'
                          'Sorted stratificated k fold cross validation with 1'
                          'holdout is used to tune paramters and measure metrics.',
                          params={'loss_func': str(loss_function.__name__),
                                  'lr_layer1': layer1_lr,
                                  'lr_layer2': layer2_lr,
                                  'momentum': momentum,
                                  'epochs_count': epochs,
                                  'dropout': DROP_MULT,
                                  'freeze_layer_idx': freeze_layer_idx,
                                  'architecture': architecture,
                                  'k_folds': k_folds})

nep_exp.append_tag('regression')
nep_exp.append_tag('k folds CV')
nep_exp.append_tag('pretrained extended LangModel')


# In[9]:


data_lm = (TextList.from_csv(path_lm, lm_csv_file, cols='comment_text')
                .split_by_rand_pct(0.1)
                .label_for_lm()
                .databunch(bs=bs))


# In[10]:


data_lm.save(pre_trained_path/data_lm_file)


# In[11]:


data_lm = load_data(pre_trained_path, data_lm_file, bs=bs)


# In[12]:


data_lm.show_batch()


# In[13]:


learn = language_model_learner(data_lm, AWD_LSTM, pretrained=True, drop_mult=0.3)


# In[14]:


learn.lr_find()


# In[15]:


learn.recorder.plot(skip_end=15, suggestion=True)


# In[16]:


Image("../input/fastai-lm-on-extended-datasetcv-regression-source/lm_lr_find1.png")


# In[17]:


learn.fit_one_cycle(1, 1e-2, moms=(0.8,0.7))


# In[18]:


Image("../input/fastai-lm-on-extended-datasetcv-regression-source/lm_fit_ep1_loss_table.png")


# In[19]:


learn.save('fit_head_basic_toxic_lm1ep_extended')


# In[20]:


learn.load('fit_head_basic_toxic_lm1ep_extended');


# In[21]:


learn.unfreeze()


# In[22]:


learn.fit_one_cycle(15, 1e-3, moms=(0.8,0.7))


# In[23]:


Image("../input/fastai-lm-on-extended-datasetcv-regression-source/lm_fit_ep15_loss_table.png")


# In[24]:


learn.save('fine_tuned_basic_toxic_lm15ep_extended')


# In[25]:


learn.load('fine_tuned_basic_toxic_lm15ep_extended');


# In[26]:


learn.save_encoder('fine_tuned_encoder_basic_toxic_lm15ep_extended')


# In[27]:


with open('fine_tuned_basic_toxic_vocab_lm15ep_extended.pcl', 'wb') as f:
    pickle.dump(data_lm.vocab, f)


# In[28]:


def get_k_stratified_folds_indexes_given_continues_variable(df, k_folds, continuous_target_col_name):
    nb_samples = df.shape[0]
    split_remainder = nb_samples%k_folds
    groups_count = nb_samples//k_folds
    
    idxs_train_remainder = df[:split_remainder].index.values
    idxs_train_trimmed = df[split_remainder:]

    idxs_train_sorted = idxs_train_trimmed.sort_values(by=continuous_target_col_name).index.values

    k_groups = np.split(idxs_train_sorted, groups_count)
    list(map(np.random.shuffle, k_groups))
    folds_idxs = [np.array(x) for x in zip(*k_groups)]
    folds_idxs[0] = np.concatenate([folds_idxs[0], idxs_train_remainder])
    
    return np.array(folds_idxs)

def get_train_and_valid_idxs_split_given_fold_nb(folds_idxs, k):
    mask = np.ones(len(folds_idxs), dtype=bool)
    mask[k] = 0
    folds_idxs[mask]
    train_idxs = np.concatenate(folds_idxs[mask])
    
    test_idxs = folds_idxs[1]
    return train_idxs, test_idxs


# In[29]:



# Convert taget and identity columns to booleans
def get_col_converted_to_bool(df, col_name):
    return np.where(df[col_name] >= 0.5, True, False)
    
def get_df_with_converted_categorical_cols_to_bool(df, categorical_cols):
    for col in categorical_cols:
        df[col] = get_col_converted_to_bool(df, col)
    return df

def get_df_with_class_target_col(df, target_col_name, target_class_col_name):
    df[target_class_col_name] = get_col_converted_to_bool(df, target_col_name)
    return df


# In[30]:


df_train_initial = pd.read_csv(path/clas_csv_file)


# In[31]:


df_train_initial = get_df_with_converted_categorical_cols_to_bool(
    df=df_train_initial,
    categorical_cols=identity_columns)
df_train_initial = get_df_with_class_target_col(
    df=df_train_initial,
    target_col_name=target_col_name,
    target_class_col_name=target_class_col_name)


# In[32]:


df_train_initial = df_train_initial.sample(frac=1 ,random_state=SEED)

holdout_and_k_folds_idxs = get_k_stratified_folds_indexes_given_continues_variable(df_train_initial, k_folds, target_col_name)


# In[33]:


models_performance = {}
# if holdout then k-1 cuz one fold went on holdout
for k in range(k_folds-1):

    train_idxs, valid_idxs = get_train_and_valid_idxs_split_given_fold_nb(folds_idxs, k)
    
    data_clas = (TextList.from_df(df_train_initial, path=path, cols=x_col_name, vocab=data_lm.vocab)
             .split_by_idxs(train_idx=train_idxs, valid_idx=valid_idxs)
             .label_from_df(cols=target_col_name, label_cls=FloatList)             
             .databunch(bs=bs))

    learn = text_classifier_learner(data_clas, architecture, drop_mult=DROP_MULT, metrics=[mse, ToxicMetric()])
    learn.load_encoder(encoder_file)
    
    learn.loss = loss_function
    learn.fit_one_cycle(epochs, layer1_lr, moms=momentum)
    learn.freeze_to(freeze_layer_idx)
    learn.fit_one_cycle(epochs, layer2_lr, moms=momentum)
    learn.save(model_cv_result_file.format(k))
    learn.export(model_prod_cv_result_file.format(k))
    
    learn.data.add_test(df_train_initial[x_col_name][holdout_idx])
    prob_preds = learn.get_preds(ds_type=DatasetType.Test, ordered=True)
    batch_preds = [pred[0] for pred in prob_preds[0].numpy()]
    models_performance[k] = {'holdout_preds': batch_preds,
                             'train_loss': learn.recorder.losses,
                             'valid_loss': learn.recorder.val_losses}


# In[34]:


train_folds_losses = []
for model_nb in models_performance:
    train_loss = models_performance[model_nb]['train_loss']
    train_score = [float(step.numpy()) for step in train_loss]
    train_folds_losses.append(train_score)

folds_train_losses = [x[-1] for x in train_folds_losses]
print('folds train losses', folds_train_losses)
avg_train_losses = np.mean(train_folds_losses, axis=0)
avg_train_loss = avg_train_losses[-1]
print('avg train loss', avg_train_loss)


# In[35]:


folds train losses [0.018044915050268173, 0.019446002319455147, 0.018564680591225624, 0.017788488417863846, 0.018882934004068375, 0.01710507646203041, 0.017085056751966476]
avg train loss 0.01813102194241115


# In[36]:


valid_folds_losses = [models_performance[x]['valid_loss'][-1] for x in models_performance]
print('folds valid losses', valid_folds_losses)
avg_valid_loss = np.mean(valid_folds_losses, axis=0)
print('avg valid loss', avg_valid_loss)


# In[37]:


folds valid losses [0.5292527, 54.811653, 0.017135596, 0.29703587, 0.026240299, 26.354286, 0.049951334]
avg valid loss 11.726507


# In[38]:


holdout_group_preds = [models_performance[x]['holdout_preds'] for x in models_performance]


# In[39]:


avg_holdout_preds = np.mean(holdout_group_preds, axis=0)


# In[40]:


correct_preds_count = 0
mse_loss_holdout = []
sample_count = len(avg_holdout_preds)
for i in range(sample_count):
    single_holdout_idx = holdout_idx[i]
    target = df_train_initial['target'].loc[single_holdout_idx]
    ensamble_pred = avg_holdout_preds[i]
    mse_pred = (ensamble_pred - target)**2
    mse_loss_holdout.append(mse_pred)
    if (ensamble_pred>0.5 and target>0.5) or (ensamble_pred<0.5 and target<0.5):
        correct_preds_count += 1
holdout_acc = correct_preds_count/sample_count
print('holdout_acc', holdout_acc)
holdout_loss = sum(mse_loss_holdout)/len(mse_loss_holdout)
print('holdout_loss ', holdout_loss)


# In[41]:


holdout_acc 0.9462703502932038
holdout_loss  5.278712082137559


# In[42]:


SUBGROUP_AUC = 'subgroup_auc'
BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative
BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positive

def compute_auc(y_true, y_pred):
    try:
        return metrics.roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan

def compute_subgroup_auc(df, subgroup, label, model_name):
    subgroup_examples = df[df[subgroup]]
    return compute_auc(subgroup_examples[label], subgroup_examples[model_name])

def compute_bpsn_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
    subgroup_negative_examples = df[df[subgroup] & ~df[label]]
    non_subgroup_positive_examples = df[~df[subgroup] & df[label]]
    examples = subgroup_negative_examples.append(non_subgroup_positive_examples)
    return compute_auc(examples[label], examples[model_name])

def compute_bnsp_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
    subgroup_positive_examples = df[df[subgroup] & df[label]]
    non_subgroup_negative_examples = df[~df[subgroup] & ~df[label]]
    examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
    return compute_auc(examples[label], examples[model_name])

def compute_bias_metrics_for_model(dataset,
                                   subgroups,
                                   model,
                                   label_col,
                                   include_asegs=False):
    """Computes per-subgroup metrics for all subgroups and one model."""
    records = []
    for subgroup in subgroups:
        record = {
            'subgroup': subgroup,
            'subgroup_size': len(dataset[dataset[subgroup]])
        }
        record[SUBGROUP_AUC] = compute_subgroup_auc(dataset, subgroup, label_col, model)
        record[BPSN_AUC] = compute_bpsn_auc(dataset, subgroup, label_col, model)
        record[BNSP_AUC] = compute_bnsp_auc(dataset, subgroup, label_col, model)
        records.append(record)
    return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)


# In[43]:


def calculate_overall_auc(df, model_name):
    true_labels = df[TOXICITY_COLUMN]
    predicted_labels = df[model_name]
    return metrics.roc_auc_score(true_labels, predicted_labels)

def power_mean(series, p):
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)

def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):
    bias_score = np.average([
        power_mean(bias_df[SUBGROUP_AUC], POWER),
        power_mean(bias_df[BPSN_AUC], POWER),
        power_mean(bias_df[BNSP_AUC], POWER)
    ])
    return (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score)


# In[44]:


MODEL_NAME = 'regression'
validate_df = df_train_initial.loc[holdout_idx]
validate_df[MODEL_NAME] = avg_holdout_preds
TOXICITY_COLUMN = target_class_col_name

bias_metrics_df = compute_bias_metrics_for_model(validate_df, identity_columns, MODEL_NAME, TOXICITY_COLUMN)
bias_metrics_df


# In[45]:


Image("../input/fastai-lm-on-extended-datasetcv-regression-source/toxic_subgroups_auc.png")


# In[46]:


toxic_metric_acc = get_final_metric(bias_metrics_df, calculate_overall_auc(validate_df, MODEL_NAME))


# In[47]:


for fold_nb in range(len(valid_folds_losses)):
    neptune.send_metric(channel_name='valid_loss per fold', x=fold_nb, y=valid_folds_losses[fold_nb])


# In[48]:


for step in range(len(avg_train_losses)):
    neptune.send_metric(channel_name='train_avg_folds_losses', x=step, y=avg_train_losses[step])


# In[49]:


neptune.send_metric(channel_name='avg_train_loss', x=exp_nb, y=avg_train_loss)
neptune.send_metric(channel_name='avg_valid_loss', x=exp_nb, y=avg_valid_loss)

neptune.send_metric(channel_name='holdout_loss', x=exp_nb, y=holdout_loss)
neptune.send_metric(channel_name='holdout_acc', x=exp_nb, y=holdout_acc)

neptune.send_metric(channel_name='toxic_metric', x=exp_nb, y=toxic_metric_acc)


# In[50]:


for _, row in bias_metrics_df.iterrows():
    key = 'subgroup_'+row['subgroup']+'_auc'
    neptune.send_metric(channel_name=key, x=exp_nb, y=row['subgroup_auc'])


# In[51]:


neptune.experiments.get_current_experiment().get_hardware_utilization()


# In[52]:


neptune.stop()


# In[53]:


data_clas = (TextList.from_df(df_train_initial, path=path, cols=x_col_name, vocab=data_lm.vocab)
         .split_none()
         .label_from_df(cols=target_col_name, label_cls=FloatList)             
         .databunch(bs=bs))

learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=DROP_MULT)
learn.load_encoder(encoder_file)

learn.loss = loss_function
learn.fit_one_cycle(epochs, layer1_lr, moms=momentum)
learn.freeze_to(freeze_layer_idx)
learn.fit_one_cycle(epochs, layer2_lr, moms=momentum)


# In[54]:


Image("../input/fastai-lm-on-extended-datasetcv-regression-source/final_model_table_loss.png")


# In[55]:


learn.export(final_model_prod_result_file)


# In[56]:


test_df = pd.read_csv(path/test_csv_file)


# In[57]:


#learn = load_learner(pretrained_reg_path, reg_model_file)

learn.data.add_test(test_df[x_col_name])
prob_preds = learn.get_preds(ds_type=DatasetType.Test, ordered=True)
batch_preds = [pred[0] for pred in prob_preds[0].numpy()]
batch_preds[:5]


# In[58]:


test_df['prediction'] = batch_preds
submission = test_df[['id','prediction']]
submission[:5]


# In[59]:


submission.to_csv(path/final_submission_predictions, index=False)


# In[60]:


df_submit = pd.read_csv('../input/preds-toxic-reg/toxic_final_submission_exp_nb_2.csv')


# In[61]:


df_submit.to_csv('submission.csv', index=False)

