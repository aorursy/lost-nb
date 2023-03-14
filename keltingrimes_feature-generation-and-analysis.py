#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from itertools import combinations
import cv2
import os
import random
from time import time
from math import sqrt
from tqdm import tqdm
from imageio import imread
from skimage.transform import resize
from keras.models import load_model
from scipy.spatial import distance
import matplotlib.pyplot as plt
from seaborn import kdeplot




prerun_data = pd.read_csv("../input/calculating-dlib-facial-landmarks-for-rfiw-images/facial_landmark_coordinates.csv")
prerun_data.head()




point_names = ["jaw"+str(i) for i in range(17)]+["browL"+str(i) for i in range(5)]+            ["browR"+str(i) for i in range(5)]+["nRidge"+str(i) for i in range(4)]+            ["nTip"+str(i) for i in range(5)]+["eyeL"+str(i) for i in range(6)]+            ["eyeR"+str(i) for i in range(6)]+["lipOut"+str(i) for i in range(12)]+            ["lipIn"+str(i) for i in range(8)]

for i in point_names:
    x_mean = prerun_data[i +'X'].mean()
    y_mean = prerun_data[i +'Y'].mean()
    col_name = "d2Avg(" + i + ")"
    prerun_data[col_name] = np.sqrt(((prerun_data[i+'X']-x_mean)**2)+((prerun_data[i+'Y']-y_mean)**2))
    
prerun_data.iloc[0:5,137:205]




features = ['jaw0','jaw8','jaw16','browL0','browL4','browR0',
            'browR4','nRidge3','nTip0','nTip2','nTip4',
            'eyeL0','eyeL3','eyeR0','eyeR3','lipOut0','lipOut3',
            'lipOut6','lipOut9']
features = list(combinations(features, 2))

for i in range(0, len(features)):
    name1 = features[i][0]
    name2 = features[i][1]
    col_name = "dist(" + name1 + "," + name2 + ")"
    
    x1 = prerun_data[name1+'X']
    y1 = prerun_data[name1+'Y']
    x2 = prerun_data[name2+'X']
    y2 = prerun_data[name2+'Y']
    
    prerun_data[col_name] = np.sqrt(((x1-x2)**2) + ((y1-y2)**2))
    
prerun_data.iloc[0:5,205:376]




model_path = '../input/facenet-keras/facenet_keras.h5'
facenet_model = load_model(model_path)




def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def load_and_align_images(filepaths, image_size = 160):
    
    aligned_images = []
    for filepath in filepaths:
        img = imread(filepath)
        aligned = resize(img, (image_size, image_size), mode='reflect')
        aligned_images.append(aligned)
            
    return np.array(aligned_images)

def calc_embs(filepaths, margin=10, batch_size=512):
    pd = []
    for start in tqdm(range(0, len(filepaths), batch_size)):
        aligned_images = prewhiten(load_and_align_images(filepaths[start:start+batch_size]))
        pd.append(facenet_model.predict_on_batch(aligned_images))
    embs = l2_normalize(np.concatenate(pd))

    return embs

print("Helper functions compiled succesfully.")




facenet_embs = calc_embs(prerun_data["path"])
facenet_embs = pd.DataFrame(facenet_embs, columns=["facenet_"+str(i) for i in range(128)])
prerun_data = pd.concat([prerun_data, facenet_embs], axis=1)
prerun_data.iloc[0:5,376:504]




def calc_distance(embs_img1, embs_img2):
    dists = distance.euclidean(embs_img1, embs_img2)
    return dists




get_ipython().system('pip install git+https://github.com/rcmalli/keras-vggface.git')




from keras_vggface.vggface import VGGFace
vggface_model = VGGFace(include_top=False, input_shape=(160, 160, 3), pooling='avg')




def calc_embs_vggface(filepaths, margin=10, batch_size=64):
    pd = []
    for start in tqdm(range(0, len(filepaths), batch_size)):
        aligned_images = prewhiten(load_and_align_images(filepaths[start:start+batch_size]))
        pd.append(vggface_model.predict_on_batch(aligned_images))
    embs = l2_normalize(np.concatenate(pd))

    return embs

print("Helper functions successfully compiled.")




vggface_embs = calc_embs_vggface(prerun_data["path"])
vggface_embs = pd.DataFrame(vggface_embs, columns=["vggface_"+str(i) for i in range(512)])
prerun_data = pd.concat([prerun_data, vggface_embs], axis=1)
prerun_data.iloc[0:5,505:1017]




f = plt.figure(figsize=(13,3))
ax1 = f.add_subplot(131)
ax2 = f.add_subplot(132)
ax3 = f.add_subplot(133)

ax1.hist(prerun_data['browL2X'].dropna(), 25, alpha=0.75)
ax1.set_title('Left Brow, Point 2, X Coord')
ax2.hist(prerun_data['dist(jaw0,jaw8)'].dropna(),25, alpha=0.75)
ax2.set_title('Dist. from Jaw0 to Jaw8')
ax3.hist(prerun_data['facenet_12'].dropna(), 25, alpha=0.75)
ax3.set_title('Facenet Embedding 12')
plt.show()




scaler = preprocessing.StandardScaler()

scaled_data = scaler.fit_transform(prerun_data.iloc[:,1:])
scaled_data = pd.DataFrame(scaled_data, columns = prerun_data.columns[1:])
scaled_data.index = prerun_data.index
scaled_data.insert(loc=0, column="path", value=prerun_data["path"])




random.seed(6242)

train_folder = '../input/recognizing-faces-in-the-wild/train/'
family_names = sorted(os.listdir(train_folder))
kin_pairs = pd.read_csv('../input/recognizing-faces-in-the-wild/train_relationships.csv')

kin_files = []
for i in range(0, len(kin_pairs.index)):
    if os.path.exists(train_folder+kin_pairs.iloc[i,0]) and os.path.exists(train_folder+kin_pairs.iloc[i,1]):
        pair = [kin_pairs.iloc[i,0], kin_pairs.iloc[i,1], 1]
        kin_files.append(pair)
for i in range(0, len(kin_files), 2):
    x = kin_files[i][0]
    kin_files[i][0] = kin_files[i][1]
    kin_files[i][1] = x
        
random_files = []
for i in range(len(kin_files)):
    x, y = 0, 0
    while x == y:
        x = random.randint(0,469)
        y = random.randint(0,469)

    fam1 = family_names[x]
    fam2 = family_names[y]

    fam1_folder = train_folder + fam1 + '/'
    fam1_members = sorted(os.listdir(fam1_folder))
    m = random.randint(0,len(fam1_members)-1)
    pers1 = fam1_members[m]

    fam2_folder = train_folder + fam2 + '/'
    fam2_members = sorted(os.listdir(fam2_folder))
    n = random.randint(0,len(fam2_members)-1)
    pers2 = fam2_members[n]

    pair = [fam1+'/' +pers1, fam2+'/'+pers2, 0]
    random_files.append(pair)

image_files = kin_files + random_files
print("There are ", len(image_files), " pairs of individuals in the training data")




print("There are", pd.isnull(scaled_data).sum()[1], "incorrectly detected images.")  




col_names = []
images_per_person = 3
landmark_data = pd.DataFrame(
    columns = [name + "_A" for name in scaled_data.columns[1:]] + \
        [name + "_B" for name in scaled_data.columns[1:]] + \
        ["dist(Facenet)", "dist(VGGFace)", "Kin"])
emb_cols = ["facenet_" + str(i) for i in range(128)]
emb_cols_vggface = ["vggface_" + str(i) for i in range(512)]

for i in range(len(image_files)):
    rowEntry = []
    pers1 = image_files[i][0]
    pers2 = image_files[i][1]
    images1 = sorted(os.listdir(train_folder+pers1+'/'))
    images2 = sorted(os.listdir(train_folder+pers2+'/'))
    num_imgs1 = len(images1)
    num_imgs2 = len(images2)
    
    if num_imgs1 == 0:
        continue
    elif num_imgs1 > images_per_person:
        num_imgs1 = images_per_person
        
    if num_imgs2 == 0:
        continue
    elif num_imgs2 > images_per_person:
        num_imgs2 = images_per_person
    
    for j in range(0,num_imgs1):
        for k in range(j, num_imgs2):
            img1 = train_folder+pers1+'/'+images1[j]
            img2 = train_folder+pers2+'/'+images2[k]
            points1 = scaled_data.loc[scaled_data['path'] == img1]
            points2 = scaled_data.loc[scaled_data['path'] == img2]
            points1 = points1.values.tolist()[0][1:]
            points2 = points2.values.tolist()[0][1:]
            
            embs_img1 = prerun_data.loc[prerun_data["path"]==img1,emb_cols]
            embs_img2 = prerun_data.loc[prerun_data["path"]==img2,emb_cols]
            emb_dist = calc_distance(embs_img1,embs_img2)
            
            embs_img1_vgg = prerun_data.loc[prerun_data["path"] == img1, emb_cols_vggface]
            embs_img2_vgg = prerun_data.loc[prerun_data["path"] == img2, emb_cols_vggface]
            emb_dist_vggface = calc_distance(embs_img1_vgg, embs_img2_vgg)
            
            rowEntry = points1 + points2 + [emb_dist, emb_dist_vggface] + [image_files[i][2]]
            if len(rowEntry) == len(landmark_data.columns):
                landmark_data.loc[len(landmark_data)] = rowEntry

dist_scaler = preprocessing.StandardScaler()
distance_col = np.asarray(landmark_data["dist(Facenet)"].tolist()).reshape(-1,1)
scaled_dists = dist_scaler.fit_transform(distance_col)
landmark_data["dist(Facenet)"] = scaled_dists

vggface_dist_scaler = preprocessing.StandardScaler()
distance_col = np.asarray(landmark_data["dist(VGGFace)"].tolist()).reshape(-1,1)
scaled_dists = vggface_dist_scaler.fit_transform(distance_col)
landmark_data["dist(VGGFace)"] = scaled_dists

print("We now have", len(landmark_data.index), "pairs of faces.")




lm_size = len(landmark_data.index)
landmark_data = landmark_data.dropna()
print("There are now", len(landmark_data.index), "pairs left, a loss of", lm_size-len(landmark_data.index))




start = time()
logReg = LogisticRegression(penalty="l1", solver="liblinear", random_state=8119)
logReg.fit(landmark_data.drop("Kin", axis=1), landmark_data["Kin"].astype("int"))
end = time()

all_coord_time = end - start




print("That took about", int(all_coord_time/60), "minutes.", sum(logReg.coef_.reshape(len(landmark_data.columns)-1,) == 0), "coefficients were reduced to zero.")
print("Lets train it again with only 250 features and see if we can reduce that time.")




coefficients = logReg.coef_.reshape(len(landmark_data.columns)-1,)
k = len(coefficients) - 250
drop_cols = np.argpartition(coefficients, k)
drop_cols = drop_cols[:k]

top250_data = landmark_data.drop(landmark_data.columns[(drop_cols)], axis=1)

start = time()
logReg250 = LogisticRegression(penalty="l1", solver="liblinear", random_state=8119)
logReg250.fit(top250_data.drop("Kin", axis=1), top250_data["Kin"].astype("int"))
end = time()

coord_time_250 = end - start




print("That took about", int(coord_time_250/60), "minutes, much faster. Let's see how it scores.")




test_pairs = pd.read_csv('../input/recognizing-faces-in-the-wild/sample_submission.csv')
test_pairs_t250 = test_pairs.copy()

test_folder = '../input/recognizing-faces-in-the-wild/test/'
detection_errors = 0

for i in range(len(test_pairs)):
    
    rowEntry = []
    missing_vals = 0
    
    pair = test_pairs.iloc[i,0]
    
    pers1 = pair.split('-')[0]
    pers2 = pair.split('-')[1]
    
    pers1 = test_folder+pers1
    points1 = scaled_data.loc[scaled_data['path'] == pers1]
    points1 = points1.drop('path', axis=1)
    points1.reset_index(drop=True, inplace=True)
    if points1.isnull().values.any():
        missing_vals = 1
    
    pers2 = test_folder+pers2
    points2 = scaled_data.loc[scaled_data['path'] == pers2]
    points2 = points2.drop('path', axis=1)
    points2.reset_index(drop=True, inplace=True)
    if points2.isnull().values.any():
        missing_vals = 1
    
    embs_img1 = prerun_data.loc[prerun_data["path"] == pers1, emb_cols]
    embs_img2 = prerun_data.loc[prerun_data["path"] == pers2, emb_cols]
    emb_dist = calc_distance(embs_img1, embs_img2)
    emb_dist = (emb_dist - dist_scaler.mean_[0])/dist_scaler.scale_[0]
    emb_dist = pd.DataFrame([emb_dist], columns=["dist(Facenet)"])
    
    embs_img1_vgg = prerun_data.loc[prerun_data["path"] == img1, emb_cols_vggface]
    embs_img2_vgg = prerun_data.loc[prerun_data["path"] == img2, emb_cols_vggface]
    emb_dist_vggface = calc_distance(embs_img1_vgg, embs_img2_vgg)
    emb_dist_vggface = (emb_dist_vggface - vggface_dist_scaler.mean_[0])/vggface_dist_scaler.scale_[0]
    emb_dist_vggface = pd.DataFrame([emb_dist_vggface], columns=["dist(VGGFace)"])
    
    rowEntry = pd.concat([points1, points2, emb_dist, emb_dist_vggface], axis=1, ignore_index = True)
    rowEntry.columns = [name + "_A" for name in scaled_data.columns[1:]] +                         [name + "_B" for name in scaled_data.columns[1:]] +                         ["dist(Facenet)", "dist(VGGFace)"]
    
    rowEntry_t250 = rowEntry.drop(rowEntry.columns[(drop_cols)], axis=1)
        
    if missing_vals == 0:
        test_pairs.iloc[i,1] = logReg.predict_proba(rowEntry)[0][0]
        test_pairs_t250.iloc[i,1] = logReg250.predict_proba(rowEntry_t250)[0][0]
    else:
        detection_errors += 1
        test_pairs.iloc[i,1] = 0.5
        test_pairs_t250.iloc[i,1] = 0.5
    
if(logReg.classes_[0] == 0): # Probabilities depend on the order of the classes in the model
    test_pairs["is_related"] = [1-x for x in test_pairs["is_related"]]
    test_pairs_t250["is_related"] = [1-x for x in test_pairs_t250["is_related"]]
    
print(detection_errors, "pairs have been lost due to errors with dLib's facial detection.")
print("All missing cases (", round(detection_errors/len(test_pairs.index),3),"%) have been set to 0.5.")




test_pairs.to_csv('submission_file.csv', index=False)
test_pairs_t250.to_csv('submission_file_t250.csv', index=False)




feature_coeffs = pd.DataFrame()
feature_coeffs["feature"] = landmark_data.columns[:-1]
feature_coeffs["coefficient"] = logReg.coef_[0]
feature_coeffs.to_csv('feature_coefficients.csv', index=False)

sorted_coeffs = feature_coeffs.reindex(feature_coeffs.coefficient.abs().sort_values(ascending=False).index)

zero_coeffs = feature_coeffs[feature_coeffs["coefficient"]==0]
best_coeffs = sorted_coeffs.iloc[0:250,:].reset_index(drop=True)
middle_coeffs = sorted_coeffs.iloc[250:len(feature_coeffs.index)-len(zero_coeffs.index),:]




objects = ("Zero", "Middle", "Top-250")

f = plt.figure(figsize=(13,8))
ax = f.add_subplot(231)
ax2 = f.add_subplot(232)
ax3 = f.add_subplot(233)
ax4 = f.add_subplot(234)
ax5 = f.add_subplot(235)

data = [sum(["X_" in name or "Y_" in name for name in zero_coeffs["feature"]]),
        sum(["X_" in name or "Y_" in name for name in middle_coeffs["feature"]]),
        sum(["X_" in name or "Y_" in name for name in best_coeffs["feature"]])]
ax.bar(objects, data, align='center', alpha=0.75)
ax.set_title('Raw Coordinates')

data = [sum(["d2Avg(" in name for name in zero_coeffs["feature"]]),
        sum(["d2Avg(" in name for name in middle_coeffs["feature"]]),
        sum(["d2Avg(" in name for name in best_coeffs["feature"]])]
ax2.bar(objects, data, align='center', alpha=0.75)
ax2.set_title('Dist. to Average Point')

data = [sum(["dist(" in name for name in zero_coeffs["feature"]]),
        sum(["dist(" in name for name in middle_coeffs["feature"]]),
        sum(["dist(" in name for name in best_coeffs["feature"]])]
ax3.bar(objects, data, align='center', alpha=0.75)
ax3.set_title('Point to Point Dist.')

data = [sum(["facenet_" in name for name in zero_coeffs["feature"]]),
        sum(["facenet_" in name for name in middle_coeffs["feature"]]),
        sum(["facenet_" in name for name in best_coeffs["feature"]])]
ax4.bar(objects, data, align='center', alpha=0.75)
ax4.set_title('Facenet Embeddings')

data = [sum(["vggface_" in name for name in zero_coeffs["feature"]]),
        sum(["vggface_" in name for name in middle_coeffs["feature"]]),
        sum(["vggface_" in name for name in best_coeffs["feature"]])]
ax5.bar(objects, data, align='center', alpha=0.75)
ax5.set_title('VGGFace Embeddings')




feature_coeffs.iloc[2030:2032,:]




best_coeffs[abs(best_coeffs["coefficient"]) > 1]

