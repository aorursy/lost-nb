#!/usr/bin/env python
# coding: utf-8



get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.io import imread, imshow
import cv2
import os
from glob import glob
from subprocess import check_output


TRAIN_DATA = "../input/train"
type_1_files = glob(os.path.join(TRAIN_DATA, "Type_1", "*.jpg"))
type_1_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_1"))+1:-4] for s in type_1_files])
type_2_files = glob(os.path.join(TRAIN_DATA, "Type_2", "*.jpg"))
type_2_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_2"))+1:-4] for s in type_2_files])
type_3_files = glob(os.path.join(TRAIN_DATA, "Type_3", "*.jpg"))
type_3_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_3"))+1:-4] for s in type_3_files])



TEST_DATA = "../input/test"
test_files = glob(os.path.join(TEST_DATA, "*.jpg"))
test_ids = np.array([s[len(TEST_DATA)+1:-4] for s in test_files])


ADDITIONAL_DATA = "../input/additional"
additional_type_1_files = glob(os.path.join(ADDITIONAL_DATA, "Type_1", "*.jpg"))
additional_type_1_ids = np.array([s[len(os.path.join(ADDITIONAL_DATA, "Type_1"))+1:-4] for s in additional_type_1_files])
additional_type_2_files = glob(os.path.join(ADDITIONAL_DATA, "Type_2", "*.jpg"))
additional_type_2_ids = np.array([s[len(os.path.join(ADDITIONAL_DATA, "Type_2"))+1:-4] for s in additional_type_2_files])
additional_type_3_files = glob(os.path.join(ADDITIONAL_DATA, "Type_3", "*.jpg"))
additional_type_3_ids = np.array([s[len(os.path.join(ADDITIONAL_DATA, "Type_3"))+1:-4] for s in additional_type_3_files])
"""
CROP_DATA = "../input/to_crop"
crop_type_1_files = glob(os.path.join(CROP_DATA, "Type_1", "*.jpg"))
crop_type_1_ids = np.array([s[len(os.path.join(CROP_DATA, "Type_1"))+1:-4] for s in crop_type_1_files])
crop_type_2_files = glob(os.path.join(CROP_DATA, "Type_2", "*.jpg"))
crop_type_2_ids = np.array([s[len(os.path.join(CROP_DATA, "Type_2"))+1:-4] for s in crop_type_2_files])
crop_type_3_files = glob(os.path.join(CROP_DATA, "Type_3", "*.jpg"))
crop_type_3_ids = np.array([s[len(os.path.join(CROP_DATA, "Type_3"))+1:-4] for s in crop_type_3_files])

CROP_EDGE_DATA = "../input/to_crop_edge"
crop_edge_type_1_files = glob(os.path.join(CROP_EDGE_DATA, "Type_1", "*.jpg"))
crop_edge_type_1_ids = np.array([s[len(os.path.join(CROP_EDGE_DATA, "Type_1"))+1:-4] for s in crop_edge_type_1_files])
crop_edge_type_2_files = glob(os.path.join(CROP_EDGE_DATA, "Type_2", "*.jpg"))
crop_edge_type_2_ids = np.array([s[len(os.path.join(CROP_EDGE_DATA, "Type_2"))+1:-4] for s in crop_edge_type_2_files])
crop_edge_type_3_files = glob(os.path.join(CROP_EDGE_DATA, "Type_3", "*.jpg"))
crop_edge_type_3_ids = np.array([s[len(os.path.join(CROP_EDGE_DATA, "Type_3"))+1:-4] for s in crop_edge_type_3_files])


Only do the first 20 for computational constraint reasons
"""
type_1_ids = type_1_ids[:20]
#crop_type_1_ids = crop_type_1_ids[:30]

def get_filename(image_id, image_type):
    """
    Method to get image file path from its id and type   
    """
    if image_type == "Type_1" or         image_type == "Type_2" or         image_type == "Type_3":
        data_path = os.path.join(TRAIN_DATA, image_type)
    elif image_type == "Test":
        data_path = TEST_DATA
    elif image_type == "AType_1" or           image_type == "AType_2" or           image_type == "AType_3":
        data_path = os.path.join(ADDITIONAL_DATA, image_type)
    else:
        raise Exception("Image type '%s' is not recognized" % image_type)

    ext = 'jpg'
    return os.path.join(data_path, "{}.{}".format(image_id, ext))

def get_image_data(image_id, image_type):
    """
    Method to get image data as np.array specifying image id and type
    """
    fname = get_filename(image_id, image_type)
    img = cv2.imread(fname)
    assert img is not None, "Failed to read image : %s, %s" % (image_id, image_type)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
"""
crop_type_1_ids = ['176' '810' '102' '1468' '1464' '1251' '1202' '1226' '991' '846' '873'
 '539' '1014' '1393' '764' '643' '1427' '1019' '551' '769' '891' '484'
 '160' '471' '802' '578' '205' '928' '677' '1027' '787' '523' '809' '1194'
 '759' '262' '763' '342' '649' '387' '1061' '48' '1239' '536' '446' '311'
 '1326' '791' '663' '12' '518' '1070' '1190' '338' '379' '685' '895' '739'
 '972' '7' '245' '1314' '1273' '401' '441' '265' '558' '906' '55' '732'
 '1026' '1134' '1279' '138' '1285' '965' '814' '1384' '593' '1093' '645'
 '334' '1223' '623' '1131' '516' '230' '1105' '984' '576' '1056' '497'
 '1077' '1229' '930' '239' '104' '109' '349' '709' '908' '668' '376' '683'
 '454' '333' '215' '144' '1033' '469' '1100' '532' '1245' '1136' '1199'
 '1389' '1040' '879' '356' '880' '725' '267' '1024' '1308' '35' '836' '298'
 '281' '1220' '745' '620' '855' '817' '1288' '1281' '478' '1320' '553'
 '142' '653' '596' '208' '619' '47' '1274' '560' '1154' '580' '624' '1059'
 '308' '57' '982' '434' '1324' '1071' '481' '470' '821' '1161' '1437' '562'
 '579' '10' '887' '1440' '416' '1346' '1204' '667' '783' '14' '605' '1414'
 '0' '708' '713' '779' '1013' '396' '1179' '805' '1174' '1344' '641' '765'
 '248' '41' '254' '34' '660' '513' '463' '901' '1473' '27' '700']
 """
def plt_st(l1,l2):
    plt.figure(figsize=(l1,l2))

#tile_size = (256, 256)
#tile_size=(54, 54) # last_good
tile_size=(34,34)
n = 15

print(check_output(["ls", "../input/train"]).decode("utf8"))
print(len(additional_type_1_files), len(additional_type_2_files), len(additional_type_2_files))
print("Type 1", additional_type_1_ids[:10])
print("Type 2", additional_type_2_ids[:10])
print("Type 3", additional_type_3_ids[:10])
#print(crop_type_1_ids) # from manually selecting images that are not solely the cervix




def mask_black_bkgd(img):
    #Invert the image to be white on black for compatibility with findContours function.

    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Binarize the image and call it thresh.
    ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)

    #Find all the contours in thresh. In your case the 3 and the additional strike
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #Calculate bounding rectangles for each contour.
    rects = [cv2.boundingRect(cnt) for cnt in contours]

    #Calculate the combined bounding rectangle points.
    top_x = min([x for (x, y, w, h) in rects])
    top_y = min([y for (x, y, w, h) in rects])
    bottom_x = max([x+w for (x, y, w, h) in rects])
    bottom_y = max([y+h for (x, y, w, h) in rects])

    #Draw the rectangle on the image
    out = cv2.rectangle(img, (top_x, top_y), (bottom_x, bottom_y), (0, 255, 0), 2)
    crop = img[top_y:bottom_y,top_x:bottom_x]
    return crop #thresh
    
complete_images = []
for k, type_ids in enumerate([type_1_ids]): #, type_2_ids, type_3_ids]):
    m = int(np.floor(len(type_ids) / n))
    complete_image = np.zeros((m*(tile_size[0]+2), n*(tile_size[1]+2), 3), dtype=np.uint8)
    train_ids = sorted(type_ids)
    counter = 0
    for i in range(m):
        ys = i*(tile_size[1] + 2)
        ye = ys + tile_size[1]
        for j in range(n):
            xs = j*(tile_size[0] + 2)
            xe = xs + tile_size[0]
            image_id = train_ids[counter]; counter+=1
            img = get_image_data(image_id, 'Type_%i' % (k+1))
            img = cv2.resize(img, dsize=tile_size)
            #img = cv2.putText(img, image_id, (5,img.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), thickness=3)
            complete_image[ys:ye, xs:xe] = cv2.resize(mask_black_bkgd(img[:,:,:]), dsize=tile_size)
    complete_images.append(complete_image)

plt_st(20, 20)
plt.imshow(complete_images[0])
plt.title("Training dataset of type %i" % (1))

complete_images = []
for k, type_ids in enumerate([type_1_ids]): #, type_2_ids, type_3_ids]):
    m = int(np.floor(len(type_ids) / n))
    complete_image = np.zeros((m*(tile_size[0]+2), n*(tile_size[1]+2), 3), dtype=np.uint8)
    train_ids = sorted(type_ids)
    counter = 0
    for i in range(m):
        ys = i*(tile_size[1] + 2)
        ye = ys + tile_size[1]
        for j in range(n):
            xs = j*(tile_size[0] + 2)
            xe = xs + tile_size[0]
            image_id = train_ids[counter]; counter+=1
            img = get_image_data(image_id, 'Type_%i' % (k+1))
            img = cv2.resize(img, dsize=tile_size)
            #img = cv2.putText(img, image_id, (5,img.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), thickness=3)
            complete_image[ys:ye, xs:xe] = img[:,:,:]
    complete_images.append(complete_image)

plt_st(20, 20)
plt.imshow(complete_images[0])
plt.title("Training dataset of type %i" % (1))

