#!/usr/bin/env python
# coding: utf-8

# In[1]:


import abc
import codecs
import inspect
import json
import logging
import gc
import pickle
import sys
import time
import warnings

import catboost as cat
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from abc import abstractmethod
from contextlib import contextmanager
from pathlib import Path
from numba import jit
from typing import List, Optional, Union, Tuple, Dict
from collections import Counter
from functools import partial
import scipy as sp

from sklearn.model_selection import train_test_split, GroupKFold
from tqdm import tqdm_notebook
from sklearn.metrics import cohen_kappa_score, mean_squared_error

from sklearn.utils import shuffle
from numpy.random import RandomState
import sys
import pdb


# In[2]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[3]:


conf_string = '''
dataset:
  dir: "../input/data-science-bowl-2019/"
  feature_dir: "features"
  params:

features:
  - Basic

av:
  split_params:
    test_size: 0.33
    random_state: 42

  model_params:
    objective: "binary"
    metric: "auc"
    boosting: "gbdt"
    max_depth: 7
    num_leaves: 75
    learning_rate: 0.01
    colsample_bytree: 0.7
    subsample: 0.1
    subsample_freq: 1
    seed: 111
    feature_fraction_seed: 111
    drop_seed: 111
    verbose: -1
    first_metric_only: True

  train_params:
    num_boost_round: 1000
    early_stopping_rounds: 100
    verbose_eval: 100

cat_model:
  name: "catboost"
  model_params:
    loss_function: "RMSE"
    task_type: "CPU"
    iterations: 6000
    colsample_bylevel: 0.5
    early_stopping_rounds: 400
    l2_leaf_reg: 18
    random_seed: 2019
    use_best_model: True
    
lgbm_model:
  name: "lgbm"
  model_params:
    n_estimators: 5000
    boosting_type: 'gbdt'
    metric: 'rmse'
    subsample: 0.75
    subsample_freq: 1
    learning_rate: 0.01
    feature_fraction: 0.9
    max_depth: 15
    lambda_l1: 1
    lambda_l2: 1
    early_stopping_rounds: 100
    
xgb_model:
   name: "xgb"
   model_params:
    colsample_bytree: 0.8
    learning_rate: 0.01
    max_depth: 10
    subsample: 1
    objective: 'reg:squarederror'
    min_child_weight: 3
    gamma: 0.25
    num_boost_round: 5000
    early_stopping_rounds: 100

nn_model:
    name: "nn"
    
train_params:
    mode: "regression"

val:
  name: "group_kfold"
  params:
    n_splits: 5

output_dir: "output"
'''


# In[4]:


config = dict(yaml.load(conf_string, Loader=yaml.SafeLoader))


# In[5]:


def feature_existence_checker(feature_path: Path,
                              feature_names: List[str]) -> bool:
    features = [f.name for f in feature_path.glob("*.ftr")]
    for f in feature_names:
        if f + "_train.ftr" not in features:
            return False
        if f + "_test.ftr" not in features:
            return False
    return True


# In[6]:


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def save_json(config: dict, save_path: Union[str, Path]):
    f = codecs.open(str(save_path), mode="w", encoding="utf-8")
    json.dump(config, f, indent=4, cls=MyEncoder, ensure_ascii=False)
    f.close()


# In[7]:


def configure_logger(config_name: str, log_dir: Union[Path, str], debug: bool):
    if isinstance(log_dir, str):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    else:
        log_dir.mkdir(parents=True, exist_ok=True)

    log_filename = config_name.split("/")[-1].replace(".yml", ".log")
    log_filepath = log_dir / log_filename         if isinstance(log_dir, Path) else Path(log_dir) / log_filename

    # delete the old log
    if log_filepath.exists():
        with open(log_filepath, mode="w"):
            pass

    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        filename=str(log_filepath),
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p")


# In[8]:


@contextmanager
def timer(name: str, log: bool = False):
    t0 = time.time()
    msg = f"[{name}] start"
    if not log:
        print(msg)
    else:
        logging.info(msg)
    yield

    msg = f"[{name}] done in {time.time() - t0:.2f} s"
    if not log:
        print(msg)
    else:
        logging.info(msg)


# In[9]:


def RandGroupKfold(groups, n_splits, random_state=None, shuffle_groups=False):

    ix = np.array(range(len(groups)))
    unique_groups = np.unique(groups)
    if shuffle_groups:
        prng = RandomState(random_state)
        prng.shuffle(unique_groups)
    splits = np.array_split(unique_groups, n_splits)
    train_test_indices = []

    for split in splits:
        mask = [el in split for el in groups]
        train = ix[np.invert(mask)]
        test = ix[mask]
        train_test_indices.append((train, test))
    return train_test_indices


# In[10]:


def group_kfold(df: pd.DataFrame, groups: pd.Series,random_state,
                config: dict) -> List[Tuple[np.ndarray, np.ndarray]]:
    params = config["val"]["params"]
    #kf = GroupKFold(n_splits=params["n_splits"])
    splits = RandGroupKfold(groups, n_splits=params["n_splits"], random_state=random_state, shuffle_groups=True)
    #split = list(kf.split(df, groups=groups))
    return list(splits)


def get_validation(df: pd.DataFrame,random_state,
                   config: dict) -> List[Tuple[np.ndarray, np.ndarray]]:
    name: str = config["val"]["name"]

    func = globals().get(name)
    if func is None:
        raise NotImplementedError

    if "group" in name:
        cols = df.columns.tolist()
        cols.remove("group")
        groups = df["group"]
        return func(df[cols], groups,random_state, config)
    else:
        return func(df, config)


# In[11]:


class OptimizedRounder(object):
    """
    An optimizer for rounding thresholds
    to maximize Quadratic Weighted Kappa (QWK) score
    # https://www.kaggle.com/naveenasaithambi/optimizedrounder-improved
    """

    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        """
        Get loss according to
        using current coefficients

        :param coef: A list of coefficients that will be used for rounding
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels=[0, 1, 2, 3])

        return -qwk(y, X_p)

    def fit(self, X, y):
        """
        Optimize rounding thresholds

        :param X: The raw predictions
        :param y: The ground truth labels
        """
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [1.10, 1.72, 2.25]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead', options={
            'maxiter': 5000})

    def predict(self, X, coef):
        """
        Make predictions with specified thresholds

        :param X: The raw predictions
        :param coef: A list of coefficients that will be used for rounding
        """
        return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels=[0, 1, 2, 3])

    def coefficients(self):
        """
        Return the optimized coefficients
        """
        return self.coef_['x']


# In[12]:


@jit
def qwk(y_true: Union[np.ndarray, list],
        y_pred: Union[np.ndarray, list],
        max_rat: int = 3) -> float:
    y_true_ = np.asarray(y_true, dtype=int)
    y_pred_ = np.asarray(y_pred, dtype=int)

    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))

    numerator = 0
    for k in range(y_true_.shape[0]):
        i, j = y_true_[k], y_pred_[k]
        hist1[i] += 1
        hist2[j] += 1
        numerator += (i - j) * (i - j)

    denominator = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            denominator += hist1[i] * hist2[j] * (i - j) * (i - j)

    denominator /= y_true_.shape[0]
    return 1 - numerator / denominator


def calc_metric(y_true: Union[np.ndarray, list],
                y_pred: Union[np.ndarray, list]) -> float:
    return qwk(y_true, y_pred)


# In[13]:


def eval_qwk_lgb_regr(y_true, y_pred):
    """
    Fast cappa eval function for lgb.
    """
    dist = Counter(y_train)
    for k in dist:
        dist[k] /= len(y_train)
    
    acum = 0
    bound = {}
    for i in range(3):
        acum += dist[i]
        bound[i] = np.percentile(y_pred, acum * 100)

    def classify(x):
        if x <= bound[0]:
            return 0
        elif x <= bound[1]:
            return 1
        elif x <= bound[2]:
            return 2
        else:
            return 3

    y_pred = np.array(list(map(classify, y_pred))).reshape(y_true.shape)

    return 'cappa', cohen_kappa_score(y_true, y_pred, weights='quadratic'), True

def cohenkappa(ypred, y):
    y = y.get_label().astype("int")
    ypred = ypred.reshape((4, -1)).argmax(axis = 0)
    loss = cohenkappascore(y, y_pred, weights = 'quadratic')
    return "cappa", loss, True


# In[14]:


from catboost import CatBoostClassifier, CatBoostRegressor
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from xgboost import plot_importance
from lightgbm import LGBMClassifier, LGBMRegressor

# type alias
AoD = Union[np.ndarray, pd.DataFrame]
AoS = Union[np.ndarray, pd.Series]
CatModel = Union[cat.CatBoostClassifier, cat.CatBoostRegressor]
LGBModel = Union[lgb.LGBMClassifier, lgb.LGBMRegressor]
XGBModel = Union[xgb.XGBClassifier, xgb.XGBRegressor]
Model = Union[CatModel, LGBModel, XGBModel]


class BaseModel(object):
    @abstractmethod
    def fit(self, x_train: AoD, y_train: AoS, x_valid: AoD, y_valid: AoS,
            config: dict) -> Tuple[Model, dict]:
        raise NotImplementedError

    @abstractmethod
    def get_best_iteration(self, model: Model) -> int:
        raise NotImplementedError

    @abstractmethod
    def predict(self, model: Model, features: AoD) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_feature_importance(self, model: Model) -> np.ndarray:
        raise NotImplementedError

    def cv(self,
           y_train: AoS,
           train_features: AoD,
           test_features: AoD,
           feature_name: List[str],
           folds_ids: List[Tuple[np.ndarray, np.ndarray]],
           config: dict,
           log: bool = True
           ) -> Tuple[List[Model], np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, dict]:
        # initialize
        test_preds = np.zeros(len(test_features))
        oof_preds = np.zeros(len(train_features))
        oof_true = np.zeros(len(train_features))
        importances = pd.DataFrame(index=feature_name)
        best_iteration = 0.0
        cv_score_list: List[dict] = []
        models: List[Model] = []
        X = train_features.values if isinstance(train_features, pd.DataFrame)             else train_features
        y = y_train.values if isinstance(y_train, pd.Series)             else y_train
        
        #test_features = self.convert_x(test_features.values)
        for i_fold, (trn_idx, val_idx) in enumerate(folds_ids):
            # get train data and valid data
            x_trn = X[trn_idx]
            y_trn = y[trn_idx]
            x_val = X[val_idx]
            y_val = y[val_idx]

            # train model
            model, best_score = self.fit(x_trn, y_trn, x_val, y_val, config)
            cv_score_list.append(best_score)
            models.append(model)
            best_iteration += self.get_best_iteration(model) / len(folds_ids)

            # predict oof and test
            
            ## Here something strange
            x_val = self.convert_x(x_val)
            oof_preds[val_idx] = self.predict(model, x_val).reshape(-1)
            oof_true[val_idx] = y_val
            
            _, oof_score, _ = eval_qwk_lgb_regr(y_val, oof_preds[val_idx])
            print(f"fold : {i_fold}| oof score: {oof_score:.5f}")
            
            test_preds += self.predict(
                model, test_features).reshape(-1) / len(folds_ids)
            
            # get feature importances
            importances_tmp = pd.DataFrame(
                self.get_feature_importance(model),
                columns=[f"gain_{i_fold+1}"],
                index=feature_name)
            importances = importances.join(importances_tmp, how="inner")

        # summary of feature importance
        feature_importance = importances.mean(axis=1)

        # print oof score
        _, oof_score, _ = eval_qwk_lgb_regr(oof_true, oof_preds)
        print(f"oof score: {oof_score:.5f}")

        if log:
            logging.info(f"oof score: {oof_score:.5f}")

        evals_results = {
            "evals_result": {
                "oof_score":
                oof_score,
                "cv_score": {
                    f"cv{i + 1}": cv_score
                    for i, cv_score in enumerate(cv_score_list)
                },
                "n_data":
                len(train_features),
                "best_iteration":
                best_iteration,
                "n_features":
                len(train_features.columns),
                "feature_importance":
                feature_importance.sort_values(ascending=False).to_dict()
            }
        }

        return models, oof_preds,oof_true, test_preds, feature_importance, evals_results


# In[15]:


class NN(BaseModel):
    
    def fit(self, x_train: np.ndarray, y_train: np.ndarray,
                x_valid: np.ndarray, y_valid:np.ndarray, config: dict) -> Tuple[Model,dict]:
        train_set = {'X': x_train, 'y': y_train}
        valid_set = {'X': x_valid, 'y': y_valid}
         
        scaler = MinMaxScaler()
        train_set['X'] = scaler.fit_transform(train_set['X'])
        valid_set['X'] = scaler.fit_transform(valid_set['X'])
        
        print(x_train.shape)
        verbosity = 100
        size = train_set['X'].shape[1]
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(size,)),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(25, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(1, activation='relu')
        ])
            
        model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=4e-4), loss='mse')
        save_best = tf.keras.callbacks.ModelCheckpoint('nn_model.w8',
                                                        save_weights_only=True, save_best_only=True,
                                                        verbose=1)
        early_stop = tf.keras.callbacks.EarlyStopping(patience=10)
            
        model.fit(train_set['X'],train_set['y'],
                    validation_data = (valid_set['X'],valid_set['y']),
                    epochs = 100,
                    callbacks=[save_best, early_stop])
        model.load_weights('nn_model.w8')
            
        return model, 0
    def convert_x(self,x):
        scaler = MinMaxScaler()
        return scaler.fit_transform(x)
    
    def predict(self,model
                ,features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        return model.predict(np.array(features,dtype=float))
        
    def get_best_iteration(self, model):
        return 0
        
    def get_feature_importance(self, model):
        return 0


# In[16]:


CatModel = Union[CatBoostClassifier, CatBoostRegressor]

class CatBoost(BaseModel):
    def fit(self, x_train: np.ndarray, y_train: np.ndarray,
            x_valid: np.ndarray, y_valid: np.ndarray,
            config: dict) -> Tuple[CatModel, dict]:
        model_params = config["cat_model"]["model_params"]
        mode = config["train_params"]["mode"]
        if mode == "regression":
            model = CatBoostRegressor(**model_params)
        else:
            model = CatBoostClassifier(**model_params)
        
        train_set = {'X': x_train, 'y': y_train}
        valid_set = {'X': x_valid, 'y': y_valid}
        
        model.fit(
            train_set['X'],
            train_set['y'],
            eval_set=(valid_set['X'], valid_set['y']),
            use_best_model=True,
            verbose=model_params["early_stopping_rounds"])
        
        best_score = model.best_score_
        return model, best_score
    
    def convert_x(self,x):
        return x
    
    def get_best_iteration(self, model: CatModel):
        return model.best_iteration_

    def predict(self, model: CatModel,
                features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        return model.predict(features)

    def get_feature_importance(self, model: CatModel) -> np.ndarray:
        return model.feature_importances_


# In[17]:


LgbmModel = Union[LGBMClassifier, LGBMRegressor]
class Lgbm(BaseModel):
    def fit(self, x_train: np.ndarray, y_train: np.ndarray,
           x_valid: np.ndarray, y_valid: np.ndarray,
           config: dict) -> Tuple[LgbmModel,dict]:
        
        model_params = config["lgbm_model"]["model_params"]
            
        train_set = lgb.Dataset(x_train, y_train)
        valid_set = lgb.Dataset(x_valid, y_valid)
        
        model = lgb.train(
            model_params,
            train_set,
            valid_sets=[train_set, valid_set],
            valid_names=["train", "valid"],
            verbose_eval=model_params["early_stopping_rounds"])
        
        best_score = model.best_score
        
        return model, best_score
    
    def convert_x(self,x):
        return x
    
    def get_best_iteration(self, model: LgbmModel):
        return model.best_iteration
    
    def predict(self, model:LgbmModel,
               features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        return model.predict(features)
    
    def get_feature_importance(self,model: LgbmModel) -> np.ndarray:
        return model.feature_importance(importance_type="gain")


# In[18]:


XgbModel = Union[XGBRegressor,XGBClassifier]
class Xgb(BaseModel):
    def fit(self, x_train: np.ndarray, y_train: np.ndarray,
           x_valid: np.ndarray, y_valid: np.ndarray,
           config: dict) -> Tuple[XgbModel,dict]:
        model_params = config["xgb_model"]["model_params"]
        import xgboost as xgb
        train_set = xgb.DMatrix(x_train, y_train)
        valid_set = xgb.DMatrix(x_valid, y_valid)
        
        model = xgb.train(model_params, train_set,
                  evals=[(train_set,'train'),(valid_set,'valid')],
                  verbose_eval = model_params["early_stopping_rounds"]
                 )
        
        best_score = 0
        
        return model, best_score
    
    def convert_x(self,x):
        import xgboost as xgb
        return xgb.DMatrix(x)
    def get_best_iteration(self, model: LgbmModel):
        return model.best_iteration
    
    def predict(self, model:XgbModel,
               features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        return model.predict(features)
    
    def get_feature_importance(self,model: LgbmModel) -> np.ndarray:
        return model.get_fscore()


# In[19]:


def catboost() -> CatBoost:
    return CatBoost()

def lgbm() -> Lgbm:
    return Lgbm()

def xgb() -> Xgb:
    return Xgb()

def nn() -> NN:
    return NN()

def get_model(config: dict, model:str):
    model_name = config[model]["name"]
    func = globals().get(model_name)
    if func is None:
        raise NotImplementedError
    return func()


# In[20]:


class Feature(metaclass=abc.ABCMeta):
    prefix = ""
    suffix = ""
    save_dir = "features"
    is_feature = True

    def __init__(self):
        self.name = self.__class__.__name__
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        self.train_path = Path(self.save_dir) / f"{self.name}_train.ftr"
        self.test_path = Path(self.save_dir) / f"{self.name}_test.ftr"

    def run(self,
            train_df: pd.DataFrame,
            test_df: Optional[pd.DataFrame] = None,
            log: bool = False):
        with timer(self.name, log=log):
            self.create_features(train_df, test_df)
            prefix = self.prefix + "_" if self.prefix else ""
            suffix = self.suffix + "_" if self.suffix else ""
            self.train.columns = [str(c) for c in self.train.columns]
            self.test.columns = [str(c) for c in self.test.columns]
            self.train.columns = prefix + self.train.columns + suffix
            self.test.columns = prefix + self.test.columns + suffix
        return self

    @abc.abstractmethod
    def create_features(self, train_df: pd.DataFrame,
                        test_df: Optional[pd.DataFrame]):
        raise NotImplementedError

    def save(self):
        self.train.to_feather(str(self.train_path))
        self.test.to_feather(str(self.test_path))


class PartialFeature(metaclass=abc.ABCMeta):
    def __init__(self):
        self.df = pd.DataFrame

    @abc.abstractmethod
    def create_features(self, df: pd.DataFrame, test: bool = False):
        raise NotImplementedError


def is_feature(klass) -> bool:
    return "is_feature" in set(dir(klass))


def get_features(namespace: dict):
    for v in namespace.values():
        if inspect.isclass(v) and is_feature(v) and not inspect.isabstract(v):
            yield v()


def generate_features(train_df: pd.DataFrame,
                      test_df: pd.DataFrame,
                      namespace: dict,
                      overwrite: bool,
                      log: bool = False):
    for f in get_features(namespace):
        if f.train_path.exists() and f.test_path.exists() and not overwrite:
            if not log:
                print(f.name, "was skipped")
            else:
                logging.info(f"{f.name} was skipped")
        else:
            f.run(train_df, test_df, log).save()


def load_features(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feather_path = config["dataset"]["feature_dir"]

    dfs = [
        pd.read_feather(f"{feather_path}/{f}_train.ftr", nthreads=-1)
        for f in config["features"]
        if Path(f"{feather_path}/{f}_train.ftr").exists()
    ]
    x_train = pd.concat(dfs, axis=1)

    dfs = [
        pd.read_feather(f"{feather_path}/{f}_test.ftr", nthreads=-1)
        for f in config["features"]
        if Path(f"{feather_path}/{f}_test.ftr").exists()
    ]
    x_test = pd.concat(dfs, axis=1)
    return x_train, x_test


# In[21]:


IoF = Union[int, float]
IoS = Union[int, str]


class Basic(Feature):
    def create_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        
        #Title Event
        train_df['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), train_df['title'], train_df['event_code']))
        test_df['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test_df['title'], test_df['event_code']))
        all_title_event_codes = set(train_df['title_event_code'].unique()).union(
            set(test_df['title_event_code'].unique()))
        
        title_event_code_map = dict(zip(all_title_event_codes,np.arange(len(all_title_event_codes))))
        
        #Type World
        train_df['type_world'] = list(map(lambda x, y: str(x) + '_' + str(y), train_df['type'], train_df['world']))
        test_df['type_world'] = list(map(lambda x, y: str(x) + '_' + str(y), test_df['type'], test_df['world']))
        all_type_worlds = set(train_df['type_world'].unique()).union(
            set(test_df['type_world'].unique()))
        type_world_map = dict(zip(all_type_worlds,np.arange(len(all_type_worlds))))
        
        #Assess Title
        all_assess_titles = set(train_df[train_df['type']=='Assessment']['title'].unique()).union(
            set(test_df[test_df['type']=='Assessment']['title'].unique()))
        assess_title_map = dict(zip(all_assess_titles,np.arange(len(all_assess_titles))))
        
        #Activities
        all_activities = set(train_df["title"].unique()).union(set(test_df["title"].unique()))
        activities_map = dict(zip(all_activities, np.arange(len(all_activities))))
        inverse_activities_map = dict(zip(np.arange(len(all_activities)), all_activities))
        
        train_df["title"] = train_df["title"].map(activities_map)
        test_df["title"] = test_df["title"].map(activities_map)

        
        #Worlds
        all_worlds = set(train_df['world'].unique()).union(
        set(test_df['world'].unique()))
        worlds_map = dict(zip(np.arange(len(all_worlds)),all_worlds))
        
        #Event Codes
        all_event_codes = set(train_df["event_code"].unique()).union(
            test_df["event_code"].unique())
        
        event_codes_map = dict(zip(all_event_codes,np.arange(len(all_event_codes))))
        
        #Event Ids
        all_event_ids = set(train_df['event_id'].unique()).union(
            set(test_df['event_id'].unique()))
        
        event_ids_map = dict(zip(all_event_ids,np.arange(len(all_event_ids))))
        
        
        compiled_data_train: List[List[IoF]] = []
        compiled_data_test: List[List[IoF]] = []

        installation_ids_train = []
        installation_ids_test = []


        train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
        test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])
        
        feats = KernelFeatures(all_activities, all_event_codes, all_worlds,
                                   all_event_ids,all_title_event_codes,all_type_worlds,
                                   all_assess_titles,
                                   activities_map, inverse_activities_map, worlds_map,
                                   event_codes_map,
                                   event_ids_map,
                                   title_event_code_map,
                                   type_world_map,
                                   assess_title_map)
        
        for ins_id, user_sample in tqdm_notebook(
                train_df.groupby("installation_id", sort=False),
                total=train_df["installation_id"].nunique(),
                desc="train features"):
            '''
            if "Assessment" not in user_sample["type"].unique():
                continue
            '''
            feat_df = feats.create_features(user_sample, test=False)
            installation_ids_train.extend([ins_id] * len(feat_df))
            compiled_data_train.append(feat_df)
        self.train = pd.concat(compiled_data_train, axis=0, sort=False)
        self.train["installation_id"] = installation_ids_train
        self.train.reset_index(drop=True, inplace=True)

        for ins_id, user_sample in tqdm_notebook(
                test_df.groupby("installation_id", sort=False),
                total=test_df["installation_id"].nunique(),
                desc="test features"):
            
            feat_df = feats.create_features(user_sample, test=True)
            installation_ids_test.extend([ins_id] * len(feat_df))
            compiled_data_test.append(feat_df)
        self.test = pd.concat(compiled_data_test, axis=0, sort=False)
        self.test["installation_id"] = installation_ids_test
        self.test.reset_index(drop=True, inplace=True)


class KernelFeatures(PartialFeature):
    def __init__(self, all_activities: set, all_event_codes: set,all_worlds: set,
                all_event_ids: set,all_title_event_codes: set, all_type_worlds:set,
                all_assess_titles: set,
                activities_map: Dict[str, float],
                inverse_activities_map: Dict[float, str],
                event_codes_map: Dict[str, float],
                worlds_map: Dict[str, float],
                event_ids_map: Dict[str, float],
                title_event_code_map: Dict[str,float],
                type_world_map: Dict[str,float],
                assess_title_map: Dict[str,float]):
        
        self.all_activities = all_activities
        self.all_event_codes = all_event_codes
        self.all_worlds = all_worlds
        self.all_event_ids = all_event_ids
        self.all_title_event_codes = all_title_event_codes
        self.all_type_worlds = all_type_worlds
        self.all_assess_titles = all_assess_titles
        
        self.activities_map = activities_map
        self.inverse_activities_map = inverse_activities_map
        self.worlds_map = worlds_map
        self.event_codes_map = event_codes_map
        self.event_ids_map = event_ids_map
        self.title_event_code_map = title_event_code_map
        self.type_world_map = type_world_map
        self.assess_title_map = assess_title_map

        win_code = dict(
            zip(activities_map.values(),
                (4100 * np.ones(len(activities_map))).astype(int)))
        win_code[activities_map["Bird Measurer (Assessment)"]] = 4110
        self.win_code = win_code

        super().__init__()
        
    def cnt_miss(self,df):
        cnt = 0
        for e in range(len(df)):
            x = df['event_data'].iloc[e]
            y = json.loads(x)['misses']
            cnt += y
        return cnt
    
    def get_4020_acc(self,df, counter_dict):
        for e in ['Cauldron Filler (Assessment)', 'Bird Measurer (Assessment)',
                  'Mushroom Sorter (Assessment)', 'Chest Sorter (Assessment)']:
            Assess_4020 = df[(df.event_code == 4020) & (df.title == self.activities_map[e])]
            true_attempts_ = Assess_4020['event_data'].str.contains('true').sum()
            false_attempts_ = Assess_4020['event_data'].str.contains('false').sum()

            measure_assess_accuracy_ = true_attempts_ / (true_attempts_ + false_attempts_) if (
                                                                                                      true_attempts_ + false_attempts_) != 0 else 0
            counter_dict[e + "_4020_accuracy"] += (counter_dict[e + "_4020_accuracy"] + measure_assess_accuracy_) / 2.0

        return counter_dict

    def create_features(self, df: pd.DataFrame, test: bool = False):
        
        time_spent_each_act: Dict[str, int] = {act: 0 for act in self.all_activities}
        event_code_count: Dict[int, int] = {ev: 0 for ev in self.all_event_codes}
        world_count: Dict[str, int] = {wrd: 0 for wrd in self.all_worlds}
        title_count: Dict[str, int] = {tit: 0 for tit in self.activities_map.keys()}
        event_id_count: Dict[str, int] = {ids: 0 for ids in self.all_event_ids}
        title_event_count: Dict[str, int] = {te: 0 for te in self.all_title_event_codes}
        type_world_count: Dict[str, int] = {tw: 0 for tw in self.all_type_worlds}
        assess_title_count: Dict[str, int] = {at: 0 for at in self.all_assess_titles}
        last_acc_title_count: Dict[str,int] = {'acc_' + at: -1 for at in self.all_assess_titles}
            
        user_activities_count: Dict[IoS, IoF] = {
            "Clip": 0,
            "Activity": 0,
            "Assessment": 0,
            "Game": 0
        }
        assess_4020_acc_dict: Dict[IoS, IoF] = {
            'Cauldron Filler (Assessment)_4020_accuracy': 0,
                            'Mushroom Sorter (Assessment)_4020_accuracy': 0,
                            'Bird Measurer (Assessment)_4020_accuracy': 0,
                            'Chest Sorter (Assessment)_4020_accuracy': 0}

        all_assesments = []

        accumulated_acc_groups = 0
        accumulated_acc = 0
        accumulated_correct_attempts = 0
        accumulated_failed_attempts = 0
        accumulated_actions = 0
        
        #New
        accumulated_game_miss = 0
        Cauldron_Filler_4025 = 0
        mean_game_round = 0
        mean_game_duration = 0
        mean_game_level = 0
        Assessment_mean_event_count = 0
        Game_mean_event_count = 0
        Activity_mean_event_count = 0
        chest_assessment_uncorrect_sum = 0

        counter = 0

        accuracy_group: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}

        durations: List[float] = []
        durations_game: List[float] = []
        durations_activity: List[float] = []
        last_activity = ""

        for i, sess in df.groupby("game_session", sort=False):
            sess_type = sess["type"].iloc[0]
            sess_title = sess["title"].iloc[0]
            sess_title_text = self.inverse_activities_map[sess_title]
            
            if sess_type == "Actiity":
                Activity_mean_event_count = (Activity_mean_event_count + sess['event_count'].iloc[-1])/2.0
            if sess_type == "Game":
                Game_mean_event_count = (Game_mean_event_count + sess['event_count'].iloc[-1]) / 2.0
                game_s = sess[sess.event_code == 2030]
                misses_cnt = self.cnt_miss(game_s)
                accumulated_game_miss += misses_cnt
                
                try:
                    game_round = json.loads(sess['event_data'].iloc[-1])['round']
                    mean_game_round = (mean_game_round + game_round)/2.0
                except:
                    pass
                try:
                    game_duration = json.loads(sess['event_data'].iloc[-1])['duration']
                    mean_game_duration = (mean_game_duration + game_duration)/2.0
                except:
                    pass
                try:
                    game_level = json.loads(sess['event_data'].iloc[-1])['level']
                    mean_game_level = (mean_game_level + game_level) /2.0
                except:
                    pass
                
            
            if sess_type == 'Game':
                durations_game.append((sess.iloc[-1,2] - sess.iloc[0,2]).seconds)
            if sess_type == 'Activity':
                durations_activity.append((sess.iloc[-1,2] - sess.iloc[0,2]).seconds)

            if sess_type != "Assessment":
                time_spent = int(sess["game_time"].iloc[-1] / 1000)
                time_spent_each_act[
                    self.inverse_activities_map[sess_title]] += time_spent
                
            if sess_type == "Assessment" and (test or len(sess) > 1):

                all_attempts: pd.DataFrame = sess.query(
                    f"event_code == {self.win_code[sess_title]}")
                true_attempt = all_attempts["event_data"].str.contains(
                    "true").sum()
                false_attempt = all_attempts["event_data"].str.contains(
                    "false").sum()

                features = user_activities_count.copy()
                features.update(time_spent_each_act.copy())
                features.update(event_code_count.copy())
                features.update(world_count.copy())
                features.update(title_count.copy())
                features.update(title_event_count.copy())
                features.update(type_world_count.copy())
                features.update(event_id_count.copy())
                features.update(last_acc_title_count.copy())
                features.update(assess_4020_acc_dict.copy())
                
                features['session_id'] = i
                features['accumulated_game_miss'] = accumulated_game_miss
                features['mean_game_round'] = mean_game_round
                features['mean_game_duration'] = mean_game_duration
                features['mean_game_level'] = mean_game_level
                features['chest_assessment_uncorrect_sum'] = chest_assessment_uncorrect_sum
                features['accumulated_game_miss'] = accumulated_game_miss
                features['Assessment_mean_event_count'] = Assessment_mean_event_count
                features['Activity_mean_event_count'] = Activity_mean_event_count
                features['Game_mean_event_count'] = Game_mean_event_count
                '''
                variety_features = [('var_event_code', event_code_count),
                                ('var_event_id', event_id_count),
                                ('var_title', title_count),
                                ('var_title_event_code', title_event_count),
                                ('var_type_world', type_world_count)]
                
                for name, dict_counts in variety_features:
                    arr = np.array(list(dict_counts.values()))
                    features[name] = np.count_nonzero(arr)
                '''
                
                
                features["session_title"] = sess_title

                features["accumulated_correct_attempts"] =                     accumulated_correct_attempts
                features["accumulated_failed_attempts"] =                     accumulated_failed_attempts

                accumulated_correct_attempts += true_attempt
                accumulated_failed_attempts += false_attempt
                
    
                if durations == []:
                    features["duration_mean"] = 0
                    #features["duration_medain"] = 0
                    features["duration_max"] = 0
                    features["duration_min"] = 0
                    #features["duration_std"] = 0
                    features['last_duration'] = 0
                    #features['first_duration'] = 0
                    
                else:
                    features["duration_mean"] = np.mean(durations)
                    #features["duration_medain"] = np.median(durations)
                    features["duration_max"] = np.max(durations)
                    features["duration_min"] = np.min(durations)
                    #features["duration_std"] = np.std(durations)
                    features['last_duration'] = durations[-1]
                    #features['first_duration'] = durations[0]
                    
                durations.append((sess.iloc[-1, 2] - sess.iloc[0, 2]).seconds)
                
                if durations_game == []:
                    
                    features["duration_game_mean"] = 0
                    #features["duration_game_medain"] = 0
                    features["duration_game_max"] = 0
                    features["duration_game_min"] = 0
                    #features["duration_game_std"] = 0
                    features['last_game_duration'] = 0
                    #features['first_game_duration'] = 0
                else:
                    features["duration_game_mean"] = np.mean(durations_game)
                    #features["duration_game_medain"] = np.median(durations_game)
                    features["duration_game_max"] = np.max(durations_game)
                    features["duration_game_min"] = np.min(durations_game)
                    #features["duration_game_std"] = np.std(durations_game)
                    features['last_game_duration'] = durations_game[-1]
                    #features['first_game_duration'] = durations_game[0]
                    
                if durations_activity == []:
                    features["duration_activity_mean"] = 0
                    #features["duration_activity_medain"] = 0
                    #features["duration_activity_max"] = 0
                    features["duration_activity_min"] = 0
                    features["duration_activity_std"] = 0
                    features['last_activity_duration'] = 0
                    #features['first_activity_duration'] = 0
                    
                else:
                    features["duration_activity_mean"] = np.mean(durations_activity)
                    #features["duration_activity_medain"] = np.median(durations_activity)
                    features["duration_activity_max"] = np.max(durations_activity)
                    features["duration_activity_min"] = np.min(durations_activity)
                    #features["duration_activity_std"] = np.std(durations_activity)
                    features['last_activity_duration'] = durations_activity[-1]
                    #features['first_activity_duration'] = durations_activity[0]

                features["accumulated_acc"] =                     accumulated_acc / counter if counter > 0 else 0
                
                acc = true_attempt / (true_attempt + false_attempt)                     if (true_attempt + false_attempt) != 0 else 0
                accumulated_acc += acc
                
                last_acc_title_count['acc_' + sess_title_text] = acc
                
                #features['Couldron_Filler_4025'] = \
                    #Cauldron_Filler_4025 / counter if counter > 0 else 0
                
                #Assess_4025 = sess[(sess.event_code == 4025) & (sess.title == 'Cauldron Filler (Assessment)')]
                #true_attempts_ = Assess_4025['event_data'].str.contains('true').sum()
                #false_attempts_ = Assess_4025['event_data'].str.contains('false').sum()
                
                #cau_assess_accuracy_ = true_attempts_ / (true_attempts_ + false_attempts_) if (
                                                                                            #true_attempts_ + false_attempts_) != 0 else 0
                #Cauldron_Filler_4025 += cau_assess_accuracy_
                
                #chest_assessment_uncorrect_sum += len(sess[sess.event_id == "df4fe8b6"])
                
                #Assessment_mean_event_count = (Assessment_mean_event_count + sess['event_count'].iloc[-1]) / 2.0
                

                if acc == 0:
                    features["accuracy_group"] = 0
                elif acc == 1:
                    features["accuracy_group"] = 3
                elif acc == 0.5:
                    features["accuracy_group"] = 2
                else:
                    features["accuracy_group"] = 1

                features.update(accuracy_group.copy())
                accuracy_group[features["accuracy_group"]] += 1

                features["accumulated_accuracy_group"] =                     accumulated_acc_groups / counter if counter > 0 else 0
                accumulated_acc_groups += features["accuracy_group"]

                features["accumulated_actions"] = accumulated_actions
                if test:

                    all_assesments.append(features)
                elif true_attempt + false_attempt > 0:
                    all_assesments.append(features)

                counter += 1
            
            def update_counters(counter: dict, col: str):
                num_of_count = Counter(sess[col])
                for k in num_of_count.keys():
                    if col == 'title':
                        counter[self.inverse_activities_map[k]] += num_of_count[k]
                    else:
                        counter[k] += num_of_count[k]
                        
                return counter
                        
            event_code_count = update_counters(event_code_count,'event_code')
            world_count = update_counters(world_count,'world')
            title_count = update_counters(title_count,'title')
            event_ids = update_counters(event_id_count, 'event_id')
            title_event_count = update_counters(title_event_count,'title_event_code')
            type_world_count = update_counters(type_world_count, 'type_world')
            #assess_title_count = update_counters(assess_title_count,'type')
            
            #assess_4020_acc_dict = self.get_4020_acc(sess, assess_4020_acc_dict)
            

            accumulated_actions += len(sess)
            if last_activity != sess_type:
                user_activities_count[sess_type] +=1
                last_activity = sess_type

        if test:
            self.df = pd.DataFrame([all_assesments[-1]])
        else:
            self.df = pd.DataFrame(all_assesments)

        return self.df


# In[22]:


warnings.filterwarnings("ignore")

debug = True
config_path = "../config/cat_0.yml"
log_dir = "../log/"

configure_logger(config_path, log_dir, debug)

logging.info(f"config: {config_path}")
logging.info(f"debug: {debug}")

config["args"] = dict()
config["args"]["config"] = config_path

# make output dir
output_root_dir = Path(config["output_dir"])
feature_dir = Path(config["dataset"]["feature_dir"])

config_name: str = config_path.split("/")[-1].replace(".yml", "")
output_dir = output_root_dir / config_name
output_dir.mkdir(parents=True, exist_ok=True)

logging.info(f"model output dir: {str(output_dir)}")

config["model_output_dir"] = str(output_dir)


# In[23]:


input_dir = Path(config["dataset"]["dir"])
train = reduce_mem_usage(pd.read_csv(input_dir / "train.csv"))
inst_id = train.loc[train.type=='Assessment'].installation_id.unique()
train = train.loc[train.installation_id.isin(inst_id)]
test = reduce_mem_usage(pd.read_csv(input_dir / "test.csv"))
inst_testid = test.loc[test.type=='Assessment'].installation_id.unique()
test = test.loc[test.installation_id.isin(inst_testid)]
specs = pd.read_csv(input_dir / "specs.csv")


# In[24]:


input_dir = Path(config["dataset"]["dir"])
if not feature_existence_checker(feature_dir, config["features"]):
    with timer(name="load data", log=True):
        
        generate_features(
            train, test, namespace=globals(), overwrite=False, log=True)

        del train, test
        gc.collect()


with timer("feature laoding", log=True):
    x_train = pd.concat([
        pd.read_feather(feature_dir / (f + "_train.ftr"), nthreads=-1)
        for f in config["features"]
    ],
                        axis=1,
                        sort=False)
    x_test = pd.concat([
        pd.read_feather(feature_dir / (f + "_test.ftr"), nthreads=-1)
        for f in config["features"]
    ])

cols: List[str] = x_train.columns.tolist()

x_train, x_test = x_train[cols], x_test[cols]

groups = x_train["installation_id"].values
y_train = x_train["accuracy_group"].values.reshape(-1)
cols.remove("installation_id")
cols.remove("accuracy_group")
cols.remove('session_id')

assert len(x_train) == len(y_train)
logging.debug(f"number of features: {len(cols)}")
logging.debug(f"number of train samples: {len(x_train)}")
logging.debug(f"numbber of test samples: {len(x_test)}")


# In[25]:


def stract_hists(feature, train, test, adjust=False, plot=False):
    n_bins = 10
    train_data = train[feature]
    test_data = test[feature]
    if adjust:
        test_data *= train_data.mean() / test_data.mean()
    perc_90 = np.percentile(train_data, 95)
    train_data = np.clip(train_data, 0, perc_90)
    test_data = np.clip(test_data, 0, perc_90)
    train_hist = np.histogram(train_data, bins=n_bins)[0] / len(train_data)
    test_hist = np.histogram(test_data, bins=n_bins)[0] / len(test_data)
    msre = mean_squared_error(train_hist, test_hist)
    if plot:
        print(msre)
        plt.bar(range(n_bins), train_hist, color='blue', alpha=0.5)
        plt.bar(range(n_bins), test_hist, color='red', alpha=0.5)
        plt.show()
    return msre


# In[26]:


def remove_correlated_features(reduce_train, features):
    counter = 0
    to_remove = []
    for feat_a in features:
        for feat_b in features:
            if feat_a != feat_b and feat_a not in to_remove and feat_b not in to_remove:
                c = np.corrcoef(reduce_train[feat_a], reduce_train[feat_b])[0][1]
                if c > 0.99:
                    counter += 1
                    to_remove.append(feat_b)
                    print('{}: FEAT_A: {} FEAT_B: {} - Correlation: {}'.format(counter, feat_a, feat_b, c))
    return to_remove


# In[27]:


to_remove = remove_correlated_features(x_train, cols)
features = [col for col in x_train.columns if col not in to_remove]
features = [col for col in features if col not in ['Heavy, Heavier, Heaviest_2000', 'Heavy, Heavier, Heaviest']]


# In[28]:


to_exclude = [] 
ajusted_x_test = x_test.copy()
for feature in features:
    if feature not in ['accuracy_group', 'installation_id', 'session_title']:
        data = x_train[feature]
        train_mean = data.mean()
        data = ajusted_x_test[feature] 
        test_mean = data.mean()
        try:
            error = stract_hists(feature,x_train,ajusted_x_test, adjust=True)
            ajust_factor = train_mean / test_mean
            if ajust_factor > 10 or ajust_factor < 0.1:# or error > 0.01:
                to_exclude.append(feature)
                print(feature, train_mean, test_mean, error)
            else:
                ajusted_x_test[feature] *= ajust_factor
        except:
            to_exclude.append(feature)
            print(feature, train_mean, test_mean)


# In[29]:


to_exclude = list(set(to_exclude).union(['installation_id','accuracy_group','session_id']))
features = [x for x in features if x not in (to_exclude)]


# In[30]:


def select_uncorrelated_features(reduce_train, features):
    counter = 0
    to_remove1 = []
    to_remove2 = []
    lists= []
    for feat_a in features:
        for feat_b in features:
            if feat_a != feat_b and feat_a not in lists and feat_b not in lists:
                c = np.corrcoef(reduce_train[feat_a], reduce_train[feat_b])[0][1]
                if c <= 0.6:
                    counter += 1
                    to_remove1.append(feat_a)
                    to_remove2.append(feat_b)
                    lists.append(feat_a)
                    lists.append(feat_b)
                    print('{}: FEAT_A: {} FEAT_B: {} - Correlation: {}'.format(counter, feat_a, feat_b, c))
                    break
    return to_remove1,to_remove2


# In[31]:


to_seperate1,to_seperate2 = select_uncorrelated_features(x_train, features)


# In[32]:


def select_uncorrelated_features(reduce_train, features):
    counter = 0
    to_remove1 = []
    to_remove2 = []
    lists= []
    for feat_a in features:
        for feat_b in features:
            if feat_a != feat_b and feat_a not in lists and feat_b not in lists:
                c = np.corrcoef(reduce_train[feat_a], reduce_train[feat_b])[0][1]
                if c <= 0.6:
                    counter += 1
                    to_remove1.append(feat_a)
                    to_remove2.append(feat_b)
                    lists.append(feat_a)
                    lists.append(feat_b)
                    print('{}: FEAT_A: {} FEAT_B: {} - Correlation: {}'.format(counter, feat_a, feat_b, c))
                    break
    return to_remove1,to_remove2


# In[33]:


to_seperate11,to_seperate12 = select_uncorrelated_features(x_train, to_seperate1)


# In[34]:


to_seperate21,to_seperate22 = select_uncorrelated_features(x_train, to_seperate2)


# In[35]:


features1 = [feat for feat in features if feat in to_seperate11]
features2 = [feat for feat in features if feat in to_seperate12]
features3 = [feat for feat in features if feat in to_seperate21]
features4 = [feat for feat in features if feat in to_seperate22]


# In[36]:


features1 = list(set(features1).union(['session_title']))
features2 = list(set(features2).union(['session_title']))
features3 = list(set(features3).union(['session_title']))
features4 = list(set(features4).union(['session_title']))


# In[37]:


logging.info("Adversarial Validation")
train_adv = x_train.copy()
test_adv = x_test.copy()

train_adv["target"] = 0
test_adv["target"] = 1
train_test_adv = pd.concat([train_adv, test_adv], axis=0,
                           sort=False).reset_index(drop=True)

split_params: dict = config["av"]["split_params"]
train_set, val_set = train_test_split(
    train_test_adv,
    random_state=split_params["random_state"],
    test_size=split_params["test_size"])
x_train_adv = train_set[features1]
y_train_adv = train_set["target"]
x_val_adv = val_set[features1]
y_val_adv = val_set["target"]

logging.debug(f"The number of train set: {len(x_train_adv)}")
logging.debug(f"The number of valid set: {len(x_val_adv)}")

train_lgb = lgb.Dataset(x_train_adv, label=y_train_adv)
valid_lgb = lgb.Dataset(x_val_adv, label=y_val_adv)

model_params = config["av"]["model_params"]
train_params = config["av"]["train_params"]
clf = lgb.train(
    model_params,
    train_lgb,
    valid_sets=[train_lgb, valid_lgb],
    valid_names=["train", "valid"],
    **train_params)

# Check the feature importance
feature_imp = pd.DataFrame(
    sorted(zip(clf.feature_importance(importance_type="gain"), features1)),
    columns=["value", "feature"])

plt.figure(figsize=(20, 10))
sns.barplot(
    x="value",
    y="feature",
    data=feature_imp.sort_values(by="value", ascending=False).head(50))
plt.title("LightGBM Features")
plt.tight_layout()
plt.savefig(output_dir / "feature_importance_adv.png")

config["av_result"] = dict()
config["av_result"]["score"] = clf.best_score
config["av_result"]["feature_importances"] =     feature_imp.set_index("feature").sort_values(
        by="value",
        ascending=False
    ).head(100).to_dict()["value"]


# In[38]:


logging.info("Adversarial Validation")
train_adv = x_train.copy()
test_adv = x_test.copy()

train_adv["target"] = 0
test_adv["target"] = 1
train_test_adv = pd.concat([train_adv, test_adv], axis=0,
                           sort=False).reset_index(drop=True)

split_params: dict = config["av"]["split_params"]
train_set, val_set = train_test_split(
    train_test_adv,
    random_state=split_params["random_state"],
    test_size=split_params["test_size"])
x_train_adv = train_set[features2]
y_train_adv = train_set["target"]
x_val_adv = val_set[features2]
y_val_adv = val_set["target"]

logging.debug(f"The number of train set: {len(x_train_adv)}")
logging.debug(f"The number of valid set: {len(x_val_adv)}")

train_lgb = lgb.Dataset(x_train_adv, label=y_train_adv)
valid_lgb = lgb.Dataset(x_val_adv, label=y_val_adv)

model_params = config["av"]["model_params"]
train_params = config["av"]["train_params"]
clf = lgb.train(
    model_params,
    train_lgb,
    valid_sets=[train_lgb, valid_lgb],
    valid_names=["train", "valid"],
    **train_params)

# Check the feature importance
feature_imp = pd.DataFrame(
    sorted(zip(clf.feature_importance(importance_type="gain"), features2)),
    columns=["value", "feature"])

plt.figure(figsize=(20, 10))
sns.barplot(
    x="value",
    y="feature",
    data=feature_imp.sort_values(by="value", ascending=False).head(50))
plt.title("LightGBM Features")
plt.tight_layout()
plt.savefig(output_dir / "feature_importance_adv.png")

config["av_result"] = dict()
config["av_result"]["score"] = clf.best_score
config["av_result"]["feature_importances"] =     feature_imp.set_index("feature").sort_values(
        by="value",
        ascending=False
    ).head(100).to_dict()["value"]


# In[39]:


logging.info("Adversarial Validation")
train_adv = x_train.copy()
test_adv = x_test.copy()

train_adv["target"] = 0
test_adv["target"] = 1
train_test_adv = pd.concat([train_adv, test_adv], axis=0,
                           sort=False).reset_index(drop=True)

split_params: dict = config["av"]["split_params"]
train_set, val_set = train_test_split(
    train_test_adv,
    random_state=split_params["random_state"],
    test_size=split_params["test_size"])
x_train_adv = train_set[features3]
y_train_adv = train_set["target"]
x_val_adv = val_set[features3]
y_val_adv = val_set["target"]

logging.debug(f"The number of train set: {len(x_train_adv)}")
logging.debug(f"The number of valid set: {len(x_val_adv)}")

train_lgb = lgb.Dataset(x_train_adv, label=y_train_adv)
valid_lgb = lgb.Dataset(x_val_adv, label=y_val_adv)

model_params = config["av"]["model_params"]
train_params = config["av"]["train_params"]
clf = lgb.train(
    model_params,
    train_lgb,
    valid_sets=[train_lgb, valid_lgb],
    valid_names=["train", "valid"],
    **train_params)

# Check the feature importance
feature_imp = pd.DataFrame(
    sorted(zip(clf.feature_importance(importance_type="gain"), features3)),
    columns=["value", "feature"])

plt.figure(figsize=(20, 10))
sns.barplot(
    x="value",
    y="feature",
    data=feature_imp.sort_values(by="value", ascending=False).head(50))
plt.title("LightGBM Features")
plt.tight_layout()
plt.savefig(output_dir / "feature_importance_adv.png")

config["av_result"] = dict()
config["av_result"]["score"] = clf.best_score
config["av_result"]["feature_importances"] =     feature_imp.set_index("feature").sort_values(
        by="value",
        ascending=False
    ).head(100).to_dict()["value"]


# In[40]:


logging.info("Adversarial Validation")
train_adv = x_train.copy()
test_adv = x_test.copy()

train_adv["target"] = 0
test_adv["target"] = 1
train_test_adv = pd.concat([train_adv, test_adv], axis=0,
                           sort=False).reset_index(drop=True)

split_params: dict = config["av"]["split_params"]
train_set, val_set = train_test_split(
    train_test_adv,
    random_state=split_params["random_state"],
    test_size=split_params["test_size"])
x_train_adv = train_set[features4]
y_train_adv = train_set["target"]
x_val_adv = val_set[features4]
y_val_adv = val_set["target"]

logging.debug(f"The number of train set: {len(x_train_adv)}")
logging.debug(f"The number of valid set: {len(x_val_adv)}")

train_lgb = lgb.Dataset(x_train_adv, label=y_train_adv)
valid_lgb = lgb.Dataset(x_val_adv, label=y_val_adv)

model_params = config["av"]["model_params"]
train_params = config["av"]["train_params"]
clf = lgb.train(
    model_params,
    train_lgb,
    valid_sets=[train_lgb, valid_lgb],
    valid_names=["train", "valid"],
    **train_params)

# Check the feature importance
feature_imp = pd.DataFrame(
    sorted(zip(clf.feature_importance(importance_type="gain"), features4)),
    columns=["value", "feature"])

plt.figure(figsize=(20, 10))
sns.barplot(
    x="value",
    y="feature",
    data=feature_imp.sort_values(by="value", ascending=False).head(50))
plt.title("LightGBM Features")
plt.tight_layout()
plt.savefig(output_dir / "feature_importance_adv.png")

config["av_result"] = dict()
config["av_result"]["score"] = clf.best_score
config["av_result"]["feature_importances"] =     feature_imp.set_index("feature").sort_values(
        by="value",
        ascending=False
    ).head(100).to_dict()["value"]


# In[41]:


import gc
gc.collect()


# In[42]:


'''
logging.info("Train model")

# get folds
x_train["group"] = x_train["installation_id"].values
splits = get_validation(x_train,7, config)
x_train.drop("group", axis=1, inplace=True)

model = get_model(config,'cat_model')
models, oof_preds1,oof_true1, cat_preds1, feature_importance, eval_results = model.cv(
    y_train, x_train[features1], x_test[features1], features1, splits, config, log=True)

config["eval_results"] = dict()
for k, v in eval_results.items():
    config["eval_results"][k] = v

feature_imp = feature_importance.reset_index().rename(columns={
    "index": "feature",
    0: "value"
})
plt.figure(figsize=(20, 10))
sns.barplot(
    x="value",
    y="feature",
    data=feature_imp.sort_values(by="value", ascending=False).head(50))
plt.title("Model Features")
plt.tight_layout()
plt.savefig(output_dir / "feature_importance_model.png")
'''


# In[43]:


'''
logging.info("Train model")

# get folds
x_train["group"] = x_train["installation_id"].values
splits = get_validation(x_train,27, config)
x_train.drop("group", axis=1, inplace=True)

model = get_model(config,'cat_model')
models, oof_preds2,oof_true2, cat_preds2, feature_importance, eval_results = model.cv(
    y_train, x_train[features2], x_test[features2], features2, splits, config, log=True)

config["eval_results"] = dict()
for k, v in eval_results.items():
    config["eval_results"][k] = v

feature_imp = feature_importance.reset_index().rename(columns={
    "index": "feature",
    0: "value"
})
plt.figure(figsize=(20, 10))
sns.barplot(
    x="value",
    y="feature",
    data=feature_imp.sort_values(by="value", ascending=False).head(50))
plt.title("Model Features")
plt.tight_layout()
plt.savefig(output_dir / "feature_importance_model.png")
'''


# In[44]:


'''
logging.info("Train model")

# get folds
x_train["group"] = x_train["installation_id"].values
splits = get_validation(x_train,1989, config)
x_train.drop("group", axis=1, inplace=True)

model = get_model(config,'cat_model')
models, oof_preds3,oof_true3, cat_preds3, feature_importance, eval_results = model.cv(
    y_train, x_train[features3], x_test[features3], features3, splits, config, log=True)

config["eval_results"] = dict()
for k, v in eval_results.items():
    config["eval_results"][k] = v

feature_imp = feature_importance.reset_index().rename(columns={
    "index": "feature",
    0: "value"
})
plt.figure(figsize=(20, 10))
sns.barplot(
    x="value",
    y="feature",
    data=feature_imp.sort_values(by="value", ascending=False).head(50))
plt.title("Model Features")
plt.tight_layout()
plt.savefig(output_dir / "feature_importance_model.png")
'''


# In[45]:


'''
logging.info("Train model")

# get folds
x_train["group"] = x_train["installation_id"].values
splits = get_validation(x_train,2019, config)
x_train.drop("group", axis=1, inplace=True)

model = get_model(config,'cat_model')
models, oof_preds4,oof_true4, cat_preds4, feature_importance, eval_results = model.cv(
    y_train, x_train[features2], x_test[features2], features2, splits, config, log=True)

config["eval_results"] = dict()
for k, v in eval_results.items():
    config["eval_results"][k] = v

feature_imp = feature_importance.reset_index().rename(columns={
    "index": "feature",
    0: "value"
})
plt.figure(figsize=(20, 10))
sns.barplot(
    x="value",
    y="feature",
    data=feature_imp.sort_values(by="value", ascending=False).head(50))
plt.title("Model Features")
plt.tight_layout()
plt.savefig(output_dir / "feature_importance_model.png")
'''


# In[46]:


#gc.collect()


# In[47]:


x_train['session_title'] = x_train['session_title'].astype('category')
x_test['session_title'] = x_test['session_title'].astype('category')


# In[48]:


logging.info("Train model")

# get folds
x_train["group"] = x_train["session_id"].values
splits = get_validation(x_train,4567, config)
x_train.drop("group", axis=1, inplace=True)

model = get_model(config,'lgbm_model')
models, oof_preds5,oof_true5, lgbm_preds1, feature_importance, eval_results = model.cv(
    y_train, x_train[features1], x_test[features1], features1, splits, config, log=True)

config["eval_results"] = dict()
for k, v in eval_results.items():
    config["eval_results"][k] = v

feature_imp = feature_importance.reset_index().rename(columns={
    "index": "feature",
    0: "value"
})
plt.figure(figsize=(20, 10))
sns.barplot(
    x="value",
    y="feature",
    data=feature_imp.sort_values(by="value", ascending=False).head(50))
plt.title("Model Features")
plt.tight_layout()
plt.savefig(output_dir / "feature_importance_model.png")


# In[49]:


logging.info("Train model")

# get folds
x_train["group"] = x_train["session_id"].values
splits = get_validation(x_train,8262, config)
x_train.drop("group", axis=1, inplace=True)

model = get_model(config,'lgbm_model')
models, oof_preds6,oof_true6, lgbm_preds2, feature_importance, eval_results = model.cv(
    y_train, x_train[features2], x_test[features2], features2, splits, config, log=True)

config["eval_results"] = dict()
for k, v in eval_results.items():
    config["eval_results"][k] = v

feature_imp = feature_importance.reset_index().rename(columns={
    "index": "feature",
    0: "value"
})
plt.figure(figsize=(20, 10))
sns.barplot(
    x="value",
    y="feature",
    data=feature_imp.sort_values(by="value", ascending=False).head(50))
plt.title("Model Features")
plt.tight_layout()
plt.savefig(output_dir / "feature_importance_model.png")


# In[50]:


logging.info("Train model")

# get folds
x_train["group"] = x_train["session_id"].values
splits = get_validation(x_train,287, config)
x_train.drop("group", axis=1, inplace=True)

model = get_model(config,'lgbm_model')
models, oof_preds7,oof_true7, lgbm_preds3, feature_importance, eval_results = model.cv(
    y_train, x_train[features3], x_test[features3], features3, splits, config, log=True)

config["eval_results"] = dict()
for k, v in eval_results.items():
    config["eval_results"][k] = v

feature_imp = feature_importance.reset_index().rename(columns={
    "index": "feature",
    0: "value"
})
plt.figure(figsize=(20, 10))
sns.barplot(
    x="value",
    y="feature",
    data=feature_imp.sort_values(by="value", ascending=False).head(50))
plt.title("Model Features")
plt.tight_layout()
plt.savefig(output_dir / "feature_importance_model.png")


# In[51]:


logging.info("Train model")

# get folds
x_train["group"] = x_train["session_id"].values
splits = get_validation(x_train,2530, config)
x_train.drop("group", axis=1, inplace=True)

model = get_model(config,'lgbm_model')
models, oof_preds8,oof_true8, lgbm_preds4, feature_importance, eval_results = model.cv(
    y_train, x_train[features4], x_test[features4], features4, splits, config, log=True)

config["eval_results"] = dict()
for k, v in eval_results.items():
    config["eval_results"][k] = v

feature_imp = feature_importance.reset_index().rename(columns={
    "index": "feature",
    0: "value"
})
plt.figure(figsize=(20, 10))
sns.barplot(
    x="value",
    y="feature",
    data=feature_imp.sort_values(by="value", ascending=False).head(50))
plt.title("Model Features")
plt.tight_layout()
plt.savefig(output_dir / "feature_importance_model.png")


# In[ ]:





# In[52]:


#np.corrcoef([cat_preds1,cat_preds2,cat_preds3,cat_preds4,lgbm_preds1,lgbm_preds2,lgbm_preds3,lgbm_preds4])
np.corrcoef([lgbm_preds1,lgbm_preds2,lgbm_preds3,lgbm_preds4])


# In[53]:


#np.corrcoef([oof_preds1,oof_preds2,oof_preds3,oof_preds4,oof_preds5,oof_preds6,oof_preds7,oof_preds8])
np.corrcoef([oof_preds5,oof_preds6,oof_preds7,oof_preds8])


# In[54]:


#np.corrcoef([oof_true1,oof_true2,oof_true3,oof_true4,oof_true5,oof_true6,oof_true7,oof_true8])
np.corrcoef([oof_true5,oof_true6,oof_true7,oof_true8])


# In[55]:


'''
logging.info("Train model")

# get folds
x_train["group"] = groups
splits = get_validation(x_train, config)
x_train.drop("group", axis=1, inplace=True)

model = get_model(config,'xgb_model')
models, oof_preds, test_preds, feature_importance, eval_results = model.cv(
    y_train, x_train, x_test, cols, splits, config, log=True)

config["eval_results"] = dict()
for k, v in eval_results.items():
    config["eval_results"][k] = v

feature_imp = feature_importance.reset_index().rename(columns={
    "index": "feature",
    0: "value"
})
plt.figure(figsize=(20, 10))
sns.barplot(
    x="value",
    y="feature",
    data=feature_imp.sort_values(by="value", ascending=False).head(50))
plt.title("Model Features")
plt.tight_layout()
plt.savefig(output_dir / "feature_importance_model.png")
'''


# In[56]:


save_path = output_dir / "output.json"
save_json(config, save_path)
np.save(output_dir / "oof_preds.npy", oof_preds)

with open(output_dir / "model.pkl", "wb") as m:
    pickle.dump(models, m)


# In[57]:


'''
def keras_model(X):
    model = Sequential([
            Dense(units=1, input_shape=(X.shape[1],))
        ])
    return model
def train_keras(X, y, run_lr_finder=False, epochs=5):
    print("train_keras")
    # Parameters
    VAL_SPLIT = 0.3

    def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred-y_true)))

    model = keras_model(X)

    model.compile(optimizer=Adam(lr=0.001), # Default of adam is 0.001. Check large and small values, use a value slighly lower than a diverging lr
                 loss=root_mean_squared_error)
    clr = CyclicLR(base_lr=1e-3, 
                   max_lr=1e-1,
                   step_size=2*int(len(y)/BS), # 2 times the number of iterations
                   mode='exp_range',
                   gamma=0.99994
                  )
    checkpointer=ModelCheckpoint('best_val.hdf5', monitor='val_loss', verbose=1, save_best_only=True,mode='min', period=1)
    
    if run_lr_finder:
        model.fit(X, y, epochs=1, validation_split=VAL_SPLIT, callbacks=[LRFinder(min_lr=1e-4, max_lr=10)])
    else:
        print("Fitting Keras Model")
        model.fit(X, y, epochs=epochs, validation_split=0.2, callbacks=[checkpointer], verbose=0)
        model.load_weights('best_val.hdf5')
        return model
'''


# In[58]:


#model_lists = ['cat1','cat2','cat3','cat4','lgbm1','lgbm2','lgbm3','lgbm4']
model_lists = ['lgbm1','lgbm2','lgbm3','lgbm4']


# In[59]:


import seaborn as sns
def ridgecv_predict():
    PRINT_CORR_HEATMAP=True
    PRINT_RIDGE_WEIGHTS = True
    RIDGE_ALPHAS = (0.1, 1.0, 10.0)
    #X = np.array([oof_preds1,oof_preds2,oof_preds3,oof_preds4,oof_preds5,oof_preds6,oof_preds7,oof_preds8]).T
    X = np.array([oof_preds5,oof_preds6,oof_preds7,oof_preds8]).T
    y = oof_true5
    #if PRINT_CORR_HEATMAP:
        #sns_plot = sns.heatmap(pd.concat([X, y], axis=1).corr(), annot=True)
        #sns_plot.savefig("corr_w_gt.png")

    reg = RidgeCV(alphas = RIDGE_ALPHAS, normalize=True).fit(X, y)
    if PRINT_RIDGE_WEIGHTS:
        print("## Ridge Coefficients")
        print(f'Sum of coefficients: {sum(reg.coef_)}')
        for ww, ss in zip(reg.coef_, model_lists):
            print(f'{ss} has weight {ww:.2f}')
    #X = subs.iloc[:, :len(submission_paths)]
    #X = prepare_X(X)
    #y_pred = reg.predict(X)
    #y_pred = y_pred.T[0]
    #y_pred = np.clip(y_pred, 0, None)
    #y_pred = np.expm1(y_pred)
    return reg.coef_


# In[60]:


from sklearn.linear_model import RidgeCV
coeff = ridgecv_predict()


# In[61]:


#final_oof_pred =(coeff[0]*oof_preds1+coeff[1]*oof_preds2+coeff[2]*oof_preds3+coeff[3]*oof_preds4+
                #coeff[4]*oof_preds5+coeff[5]*oof_preds6+coeff[6]*oof_preds7+coeff[7]*oof_preds8)
final_oof_pred =(coeff[0]*oof_preds5+coeff[1]*oof_preds6+coeff[2]*oof_preds7+coeff[3]*oof_preds8)


# In[62]:


#final_pred = (coeff[0]*cat_preds1+coeff[1]*cat_preds2+coeff[2]*cat_preds3+coeff[3]*cat_preds4+
              #coeff[4]*lgbm_preds1+coeff[5]*lgbm_preds2+coeff[6]*lgbm_preds3+coeff[7]*lgbm_preds4)
final_pred = (coeff[0]*lgbm_preds1+coeff[1]*lgbm_preds2+coeff[2]*lgbm_preds3+coeff[3]*lgbm_preds4)


# In[63]:


#np.corrcoef([oof_preds1,oof_preds2,oof_preds3,oof_preds4,oof_preds5,oof_preds6,oof_preds7,oof_preds8,final_oof_pred])
np.corrcoef([oof_preds5,oof_preds6,oof_preds7,oof_preds8,final_oof_pred])


# In[64]:


#np.corrcoef([cat_preds1,cat_preds2,cat_preds3,cat_preds4,lgbm_preds1,lgbm_preds2,lgbm_preds3,lgbm_preds4,final_pred])
np.corrcoef([lgbm_preds1,lgbm_preds2,lgbm_preds3,lgbm_preds4,final_pred])


# In[65]:


'''
optR = OptimizedRounder()
optR.fit(final_oof_pred, oof_true5)
coefficients = optR.coefficients()
print("New coefs = ", coefficients)
opt_preds = optR.predict(final_oof_pred, coefficients)
print("New train cappa rounding= ", qwk(oof_true5, opt_preds))
_,train_rounding_origin,_ = eval_qwk_lgb_regr(oof_true5, final_oof_pred)
print("Train cappa origin ", train_rounding_origin)
final_pred = optR.predict(final_pred, coefficients)
'''


# In[66]:


dist = Counter(y_train)
for k in dist:
    dist[k] /= len(y_train)
pd.DataFrame(y_train).hist()

acum = 0
bound = {}
for i in range(3):
    acum += dist[i]
    bound[i] = np.percentile(final_pred, acum * 100)
print(bound)



def classify(x):
    if x <= bound[0]:
        return 0
    elif x <= bound[1]:
        return 1
    elif x <= bound[2]:
        return 2
    else:
        return 3
    
final_pred = np.array(list(map(classify, final_pred)))


# In[67]:


print(final_pred.shape)


# In[68]:


sample_submission = pd.read_csv(
    input_dir / "sample_submission.csv")
sample_submission["accuracy_group"] = final_pred.astype(int)
sample_submission.to_csv('submission.csv', index=False)


# In[69]:


sample_submission['accuracy_group'].value_counts(normalize=True)

