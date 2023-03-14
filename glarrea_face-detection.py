#!/usr/bin/env python
# coding: utf-8



import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')




df = pd.read_csv('../input/training/training.csv')
df_test=pd.read_csv('../input/test/test.csv')

df.dropna(inplace=True)
df.shape




from joblib import Parallel, delayed

def format_img(x):
    return np.asarray([int(e) for e in x.split(' ')], dtype=np.uint8).reshape(96,96)

with Parallel(n_jobs=10, verbose=1, prefer='threads') as ex:
    x = ex(delayed(format_img)(e) for e in df.Image)
with Parallel(n_jobs=10, verbose=1, prefer='threads') as ex:
    test = ex(delayed(format_img)(e) for e in df_test.Image)
test = np.stack(test)[..., None]
x = np.stack(x)[..., None]
x.shape, test.shape




plt.imshow(x[3,:,:,0])




y = df.iloc[:, :-1].values
y.shape




y[1,:]




def show(x, y=None):
    plt.imshow(x[..., 0], 'gray')
    if y is not None:
        points = np.vstack(np.split(y, 15)).T
        plt.plot(points[0], points[1], 'o', color='red')
        
    plt.axis('off')

sample_idx = np.random.choice(len(x))    
show(x[sample_idx], y[sample_idx])




from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
x_train.shape, x_val.shape




# Normalizar las imágenes (1pt) 

#x_train = x_train.reshape([1712, 96*96])/255
#x_val = x_val.reshape([428, 96*96])/255
#x_train[0] #valores entre 0 y 1, usara una capa de batchnormalization en la red

#Se realizó esto en iteraciones previas, el resultado fue peor, se decide no scalar a [0,1] ni utilizar batch normalization




x_train.shape, x_val.shape




# Definir correctamente la red neuronal (5 pts)
from keras.models import Sequential 
from keras.layers import GlobalAveragePooling2D, Dense, Flatten,BatchNormalization, Dropout, Conv2D, MaxPool2D
from keras.optimizers import Adam, SGD
from keras import regularizers

lr = 0.01
bs = 256
nb = math.ceil(len(x_train)/bs)

final_model  = Sequential([
    Conv2D(32, 3, activation='relu', input_shape=(96,96,1)),
    MaxPool2D(),
    Conv2D(16, 3, activation='relu'),
    GlobalAveragePooling2D(),
    Dense(256, activation='relu', kernel_initializer='glorot_normal'),
    Dropout(0.7),
    Dense(128, activation='relu', kernel_initializer='glorot_normal'),
    Dense(64, activation='relu', kernel_initializer='glorot_normal'),
    Dense(30) #no se utiliza función de activación porque se requiere hacer una regresión para cada coordenada
])
final_model .compile(Adam(lr), loss='mse', metrics=['mae'])
final_model .summary()




log = final_model.fit(x_train, y_train, batch_size=100, epochs=100,validation_data=[x_val, y_val])




# Resultado del entrenamiento
# - mae entre 10 y 15 (3 pts)
# - mae entre 8 y 11 (5 pts)
# - mae entre 5 y 8 (7 pts)
# - mae menor o igual a 4.0 (9 pts)

print(f'MAE final: {final_model.evaluate(x_val, y_val)[1]}')




# Ver la perdida en el entrenamiento
def show_results(*logs):
    trn_loss, val_loss, trn_acc, val_acc = [], [], [], []
    
    for log in logs:
        trn_loss += log.history['loss']
        val_loss += log.history['val_loss']
    
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(trn_loss, label='train')
    ax.plot(val_loss, label='validation')
    ax.set_xlabel('epoch'); ax.set_ylabel('loss')
    ax.legend()
    
show_results(log)




# Función para visualizar un resultado
def show_pred(x, y_real, y_pred):
    fig, axes = plt.subplots(1, 2, figsize=(10,5))
    for ax in axes:
        ax.imshow(x[0, ..., 0], 'gray')
        ax.axis('off')
        
    points_real = np.vstack(np.split(y_real[0], 15)).T
    points_pred = np.vstack(np.split(y_pred[0], 15)).T
    axes[0].plot(points_pred[0], points_pred[1], 'o', color='red')
    axes[0].set_title('Predictions', size=16)
    axes[1].plot(points_real[0], points_real[1], 'o', color='green')
    axes[1].plot(points_pred[0], points_pred[1], 'o', color='red', alpha=0.5)
    axes[1].set_title('Real', size=16)




x_val[0,None].shape




sample_x = x_val[0, None]
sample_y = y_val[0, None]
pred = final_model.predict(sample_x)
show_pred(sample_x, sample_y, pred)




# Mostrar 5 resultados aleatorios del set de validación (1 pt)
for it in range(5):
    #idx = np.random.choice(len(x_val))
    #sample_x, sample_y = x_val[idx], y_val[idx]
    #pred = final_model.predict(sample_x)
   #show_pred(sample_x, sample_y, pred)
    idx = np.random.choice(len(x_val))
    sample_x = x_val[idx, None]
    sample_y = y_val[idx, None]
    pred = final_model.predict(sample_x)
    show_pred(sample_x, sample_y, pred)




# Mostrar las 5 mejores predicciones del set de validación (1 pt)
#Calculamos el MAE
diff= np.absolute(final_model.predict(x_val) - y_val)
ranking = np.average(diff,1) #MAE 
indices = np.argsort(ranking)
indices[0:5] # indice de las cinco mejores predicciones
for idx in indices[0:5]:
    sample_x = x_val[idx, None]
    sample_y = y_val[idx, None]
    pred = final_model.predict(sample_x)
    show_pred(sample_x, sample_y, pred)




# Mostrar las 5 peores predicciones del set de validación (1 pt)
indices[-5:] # indice de las cinco peores predicciones
for idx in indices[-5:]:
    sample_x = x_val[idx, None]
    sample_y = y_val[idx, None]
    pred = final_model.predict(sample_x)
    show_pred(sample_x, sample_y, pred)




sample_x_test = test[235, None]
sample_y_test = y_val[255, None]
pred = final_model.predict(sample_x)
show_pred(sample_x_test, sample_y_test, pred)




#labels_area=[['left_eye_center', 'right_eye_center', 'left_eye_inner_corner', 
        'left_eye_outer_corner', 'right_eye_inner_corner',
         'right_eye_outer_corner', 'left_eyebrow_inner_end',
        'left_eyebrow_outer_end', 'right_eyebrow_inner_end', 'right_eyebrow_outer_end', 'nose_tip',
        'mouth_left_corner', 'mouth_right_corner', 'mouth_center_top_lip',
         'mouth_center_bottom_lip']];
#labels_area=np.repeat(labels_area,2)
#labels_area=np.repeat([labels_area],1783,axis=0).flatten()




#labels_axis =np.array([['_x','_y']])
#labels_axis = np.repeat(labels_axis,26745,axis=0).flatten()
#labels_axis.shape




#labels= np.core.defchararray.add(labels_area, labels_axis)
#labels.shape




#ImageId = np.arange(1,1784)
#ImageId =np.repeat(ImageId, 30)
#ImageId.shape




#RowId=np.int32(np.arange(1,53491))
#RowId.shape




results=final_model.predict(test)
results.shape




#sub = np.array([RowId,ImageId,labels,results])
#sub = np.swapaxes(sub,0,1)
#sub.shape




#sub_df = pd.DataFrame(data=sub,columns=['RowId','ImageId','FeatureName','Location'])
#sub_df.ImageId = pd.to_numeric(sub_df.ImageId)




lookup = pd.read_csv('../input/IdLookupTable.csv')




#sub_df[(sub_df['FeatureName'] == 'left_eye_center_x') & (sub_df['ImageId'] == 1)]




lookid_list = list(lookup['FeatureName'])
imageID = list(lookup['ImageId']-1)
pre_list = list(results)




rowid = lookup['RowId']
rowid=list(rowid)
len(rowid)




feature = []
for f in list(lookup['FeatureName']):
    feature.append(lookid_list.index(f))




lookid_list




preded = []
for x,y in zip(imageID,feature):
    preded.append(results[x][y])




rowid = pd.Series(rowid,name = 'RowId')




loc = pd.Series(preded,name = 'Location')




submission = pd.concat([rowid,loc],axis = 1)




submission.to_csv('submission2.csv',index = False)











