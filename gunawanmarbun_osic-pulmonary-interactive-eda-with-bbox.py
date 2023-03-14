#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
import math
import random
import functools
import warnings

import scipy
import numpy as np
import pandas as pd

import pydicom
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import ipywidgets as widgets
from ipywidgets import interact, fixed
from tqdm.auto import tqdm


# In[2]:


def distplot_numerical(data, cols_num, col_target=None, grid_c=3, w=15, h_factor=3, **kwargs):
    """
    Distplot numerical column attributes in small multiple grid.

    Parameters
    ----------
    data : pandas.DataFrame
        dataframe without infinite values. will drop null values while plotting.
    cols_num : list of str
        interval or ratio column in data
    col_target : str, optional
        the target variable we want to distingusih the cols_num distributino
    grid_c : int, default=3
        number of grid columns
    w : int, default=15
        figsize witdh arguments
    h_factor : float, default=3.5
        height of small plot
    """
    n = math.ceil(len(cols_num) / grid_c)
    fig, ax = plt.subplots(n, grid_c, figsize=(w, h_factor*n))
    if type(ax) != np.ndarray:
        ax = np.array([ax])
    sorted_cols_num = sorted(cols_num)  # we wnat it sorted for easier search

    if col_target is None:
        for col, a in zip(sorted_cols_num, ax.reshape(-1)):
            sns.distplot(data[col], ax=a, **kwargs)
            a.set_xlabel(col)
    else:
        sorted_cols_target = sorted(data[col_target].unique())
        if len(sorted_cols_target) > 1 and len(sorted_cols_target) <= 5:  # > 5 will be too crowded
            for col, a in zip(sorted_cols_num, ax.reshape(-1)):
                for t in sorted_cols_target:
                    sns.distplot(data[data[col_target] == t][col].dropna(), ax=a, **kwargs)
                a.legend(sorted_cols_target)
                a.set_xlabel(col)
        else:  # most probably regression analysis
            for col, a in zip(sorted_cols_num, ax.reshape(-1)):
                sns.distplot(data[col], ax=a, **kwargs)
                a.set_xlabel(col)
    plt.tight_layout()
        
def distplot_categorical(data, cols_cat, col_target=None, normalize=True, grid_c=3, w=15,
                         h_factor=3, sort=False, kind='bar'):
    """
    Distplot categorical column attributes in small multiple grid.

    Parameters
    ----------
    data : pandas.DataFrame
        dataframe without infinite values. will drop null values while plotting.
    cols_cat : list of str
        categorical column in data
    col_target : str, optional
        the target variable we want to distingusih the cols_num distributino
    normalize : bool, default=True
        wether to normalize the count or not
    grid_c : int, default=3
        number of grid columns
    w : int, default=15
        figsize witdh arguments
    h_factor : float, default=3.5
        height of small plot
    sort : bool, default=False
        prevent sorting based on counts, will fallback to .cat.categories if the series is having
        category dtype
    kind : str, default='bar'
        matplotlib plot kind, really recommend to do bar plot, alternative would be 'barh'
    """
    n = math.ceil(len(cols_cat) / grid_c)
    fig, ax = plt.subplots(n, grid_c, figsize=(w, h_factor*n))
    if type(ax) != np.ndarray:
        ax = np.array([ax])
    sorted_cols_cat = sorted(cols_cat)  # we want it sorted for easier search

    if col_target is None:
        for col, a in zip(sorted_cols_cat, ax.reshape(-1)):
            data[col].value_counts(normalize=normalize, sort=sort).plot(ax=a, kind=kind)
            xlabels = [x.get_text()[:15]+'...' if (len(x.get_text()) > 15) else x for x in a.get_xticklabels()]
            a.set_xticklabels(xlabels, rotation=30, ha='right')
            a.set_xlabel(col)
    else:
        sorted_cols_target = sorted(data[col_target].unique())
        if len(sorted_cols_target) > 1 and len(sorted_cols_target) <= 6:  # > 5 will be too crowded
            for col, a in zip(sorted_cols_cat, ax.reshape(-1)):
                data.groupby(col_target)[col].value_counts(normalize=normalize, sort=sort).unstack(0).plot(ax=a, kind=kind)
                xlabels = [x.get_text()[:15]+'...' if (len(x.get_text()) > 15) else x for x in a.get_xticklabels()]
                a.set_xticklabels(xlabels, rotation=30, ha='right')
        else:  # most probably regression analysis
            for col, a in zip(sorted_cols_cat, ax.reshape(-1)):
                data[col].value_counts(normalize=normalize, sort=sort).plot(ax=a, kind=kind)
                xlabels = [x.get_text()[:15]+'...' if (len(x.get_text()) > 15) else x for x in a.get_xticklabels()]
                a.set_xticklabels(xlabels, rotation=30, ha='right')
                a.set_xlabel(col)
    plt.tight_layout()
    
    
def plot_slices_data(slices_data, n_cols=10, cmap='gray', **kwargs):
    n_rows = math.ceil(slices_data.shape[0] / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows*1.5))
    for img, ax in tqdm(zip(slices_data, axes.reshape(-1)), leave=False, total=slices_data.shape[0]):
        ax.imshow(img, cmap=cmap, **kwargs)
        ax.axis('off')
    
    missing_image_cnt = (n_rows * n_cols) - slices_data.shape[0]
    if missing_image_cnt > 0:
        for ax in axes.reshape(-1)[::-1][:-missing_image_cnt]:
            ax.axis('off')


# In[3]:


def add_first_last_FVC(df_groupby):
    min_obs, max_obs = df_groupby['Weeks'].agg(["min", "max"])
    min_obs_FVC = df_groupby.loc[df_groupby['Weeks'] == min_obs, 'FVC'].values[0]
    max_obs_FVC = df_groupby.loc[df_groupby['Weeks'] == max_obs, 'FVC'].values[0]
    is_decline = max_obs_FVC < min_obs_FVC
    df_groupby['is_decline'] = is_decline
    df_groupby['min_obs_FVC'] = min_obs_FVC
    df_groupby['max_obs_FVC'] = max_obs_FVC
    df_groupby['diff_FVC'] = (max_obs_FVC-min_obs_FVC)
    df_groupby['diff_pct_FVC'] = (max_obs_FVC-min_obs_FVC) / min_obs_FVC
    return df_groupby

def filter_min_max_obs(df_groupby):
    min_obs, max_obs = df_groupby['Weeks'].agg(['min', 'max'])
    return df_groupby[df_groupby['Weeks'].isin([min_obs, max_obs])]

def add_first_last_weeks(df_groupby):
    min_obs, max_obs = df_groupby['Weeks'].agg(["min", "max"])
    df_groupby['min_week'] = min_obs
    df_groupby['max_week'] = max_obs
    df_groupby['diff_weeks'] = max_obs - min_obs
    df_groupby['num_obs'] = df_groupby['Patient'].count()
    df_groupby['rate_obs'] = df_groupby['num_obs'] / df_groupby['diff_weeks']
    return df_groupby


# In[4]:


import re

def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s
    
def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


# In[5]:


basepath = "../input/osic-pulmonary-fibrosis-progression/"
train_df = pd.read_csv(f"{basepath}train.csv")
test_df = pd.read_csv(f"{basepath}test.csv")
submission_df = pd.read_csv(f"{basepath}sample_submission.csv")
print(train_df.shape, test_df.shape, submission_df.shape)


# In[6]:


cols_num = ['Weeks', 'FVC', 'Percent', 'Age']
cols_cat = ['Sex', 'SmokingStatus']


# In[7]:


distplot_numerical(train_df, cols_num, grid_c=4)
distplot_categorical(train_df, cols_cat, grid_c=2)


# In[8]:


temp_df = train_df    .drop_duplicates(subset=['Patient', 'Weeks'], keep='first')    .groupby('Patient')    .apply(add_first_last_FVC)    .drop_duplicates(subset='Patient', keep='first')    .loc[:, ['Patient', 'is_decline', 'min_obs_FVC', 'max_obs_FVC', 'diff_FVC', 'diff_pct_FVC']]
temp_df    .groupby('is_decline')[['Patient']].agg(['count'])    .join(temp_df.groupby('is_decline').agg(['mean', 'std']))


# In[9]:


train_df    .drop_duplicates(subset=['Patient', 'Weeks'], keep='first')    .groupby('Patient')    .apply(add_first_last_weeks)    .drop_duplicates(subset=['Patient'])    .loc[:, ['diff_weeks', 'num_obs', 'rate_obs']]    .agg(['min', 'max', 'mean', 'std'])


# In[10]:


class DICOMImages:
    DOUBLE_IDS = ['ID00078637202199415319443']
    """Wrapper for multiple slices of a patient CT-Scan results."""
    def __init__(self, id, dirpath='../input/osic-pulmonary-fibrosis-progression/train/'):
        self.id = id
        self.basepath = os.path.join(dirpath, self.id)
        self.filepaths = glob.glob(os.path.join(self.basepath, "*.dcm"))
        if self.id in self.DOUBLE_IDS:
            self.filepaths = self.filepaths[:len(self.filepaths)//2]
        sort_nicely(self.filepaths)
        
    def __iter__(self):
        for filepath in self.filepaths:
            yield pydicom.dcmread(filepath)

    def __len__(self):
        return len(self.filepaths)
    
    @property
    def image_type(self):
        """
        Infer dicom image type by its first slice metadata.
        Categories:
            - 'zero' : Rescale Intercept value is 0
            - 'not-zero': Rescale Intercept value is either -1000 or -1024
        """
        mapper = {0: 'zero'}
        rescale_intercept = self.get_dicom_metadata(self.get_slice(index=0))['Rescale Intercept']
        return {
            'name': mapper.get(rescale_intercept, 'not-zero'),
            'rescale_intercept': rescale_intercept
        }
        
    @property
    def slices(self):
        return list(self.__iter__())
    
    def get_slice(self, index):
        return pydicom.dcmread(self.filepaths[index])
    
    @property
    def df(self):
        return pd.DataFrame(
            [self.get_dicom_metadata(slice) for slice in self.__iter__()]
        )
    
    @staticmethod
    def get_dicom_metadata(slice):
        dict_res = {}
        for x in slice.values():
            if isinstance(x, pydicom.dataelem.RawDataElement):
                metadata = pydicom.dataelem.DataElement_from_raw(x)
            else:
                metadata = x
            if metadata.name == 'Pixel Data':
                continue
            dict_res.update({
                f"{metadata.name}": metadata.value
            })
        return dict_res
    
    @property
    def slices_data(self):
        return np.stack([self._to_HU(slice) for slice in self.__iter__()])
    
    @property
    def middle_slice_data(self):
        mid_slice_index = (len(self.filepaths)-1) // 2
        return self._to_HU(pydicom.dcmread(self.filepaths[mid_slice_index]))
        
    def sampled_slices_data(self, n_samples=30):
        if len(self.filepaths) < n_samples:
            msg = f"Total slices is less than number of samples: {len(self.filepaths)} < {n_samples}."
            msg += " Number of samples default to total slices."
            warnings.warn(msg, UserWarning)
            n_samples = len(self.filepaths)
        sample_indexes = np.linspace(0, len(self.slices)-1, n_samples).astype(int)
        sampled_slices = np.array(self.slices)[sample_indexes]
        return np.stack([self._to_HU(slice) for slice in sampled_slices])

    @staticmethod
    def _to_HU(slice):
        intercept, slope = slice.RescaleIntercept, slice.RescaleSlope
        
        slice_data = slice.pixel_array.astype(np.int16)
        slice_data[slice_data <= -1000] = 0
        
        if slope != 1:
            slice_data = slope * slice_data.astype(np.float64)
            slice_data = slice_data.astype(np.int16)
            
        slice_data += np.int16(intercept)
        return slice_data


# In[11]:


cols_image_related = ['Rows', 'Columns', 'Pixel Spacing',
                      'Bits Allocated', 'High Bit', 'Pixel Representation',
                      'Rescale Intercept', 'Rescale Slope']
all_patient_ids = train_df.Patient.unique()
all_dicoms = [DICOMImages(id) for id in all_patient_ids]


# In[12]:


def get_unique_dict(df):
    dict_unique = {'Patient': df['Patient ID'].unique()[0]}
    for col in df.columns:
        try:
            dict_unique.update( {f"{col}": df[col].nunique()} )
        except TypeError:
            dict_unique.update( {f"{col}": df[col].astype(str).nunique()} )
    return dict_unique 


# In[13]:


uniqued_dicom_df = pd.DataFrame([get_unique_dict(dicom.df) for dicom in tqdm(all_dicoms, leave=False)])
uniqued_dicom_df.to_csv("uniqued_dicom_df.csv", header=True, index=False)


# In[14]:


print((uniqued_dicom_df[cols_image_related] == 1).sum(axis=0))


# In[15]:


uniqued_dicom_df[uniqued_dicom_df['Pixel Spacing'] != 1][['Patient'] + cols_image_related]


# In[16]:


investigate_id = 'ID00099637202206203080121'
investigate_df = DICOMImages(investigate_id).df
investigate_df['Pixel Spacing'] = investigate_df['Pixel Spacing'].astype(str)
investigate_df[cols_image_related].drop_duplicates(subset=['Pixel Spacing'])


# In[17]:


investigate_slices_data = DICOMImages(investigate_id).slices_data
plot_slices_data(investigate_slices_data)


# In[18]:


fig, ax = plt.subplots(2, 1, figsize=(16, 6))
sns.distplot(uniqued_dicom_df['SOP Instance UID'], kde=False, bins=100, ax=ax[0])
sns.distplot(uniqued_dicom_df[uniqued_dicom_df['SOP Instance UID'] < 200]['SOP Instance UID'], kde=False, bins=100, ax=ax[1])
plt.show()


# In[19]:


all_dicoms_df = pd.DataFrame([
    # We can take the first slice only since the value is unique
    # for the columns that we'd like to investigate
    DICOMImages.get_dicom_metadata(dicom.get_slice(0)) for dicom in tqdm(all_dicoms, leave=False)
])
print(all_dicoms_df.shape)


# In[20]:


cols_pixel_spacing_extra = ['Pixel Spacing (row)', 'Pixel Spacing (col)']
all_dicoms_df[cols_pixel_spacing_extra] = pd.DataFrame(
    all_dicoms_df['Pixel Spacing'].tolist(), columns=cols_pixel_spacing_extra
)


# In[21]:


# Update our `cols_image_related` to include new columns
cols_image_related = list(set(cols_image_related + cols_pixel_spacing_extra))
cols_image_related.remove('Pixel Spacing')


# In[22]:


cols_image_related_num = cols_pixel_spacing_extra
cols_image_related_cat = [c for c in cols_image_related if c not in cols_image_related_num]
distplot_numerical(all_dicoms_df, cols_image_related_num, grid_c=2)
distplot_categorical(all_dicoms_df, cols_image_related_cat, grid_c=4)


# In[23]:


r_intercept_0_mask = all_dicoms_df['Rescale Intercept'] == 0.0


# In[24]:


def plot_xy_scatter_sized(df, col_x, col_y):    
    dff = df.groupby(col_x)[col_y].value_counts()
    x, y = zip(*dff.index.values)
    s = dff.to_numpy()
    
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, s=10*s)
    plt.show()


# In[25]:


plot_xy_scatter_sized(all_dicoms_df, 'Rows', 'Columns')


# In[26]:


not_square_mask = all_dicoms_df['Rows'] != all_dicoms_df['Columns']
not_square_patient = all_dicoms_df[not_square_mask]['Patient ID'].values[6]
plot_slices_data(DICOMImages(not_square_patient).sampled_slices_data(24))


# In[27]:


sample_id_0_intercept = all_dicoms_df[r_intercept_0_mask]['Patient ID'].values[0]
sample_id_not0_intercept = all_dicoms_df[~r_intercept_0_mask]['Patient ID'].values[0]

print(f"Patient ID (0 Intercept): {sample_id_0_intercept}")
print(f"Patient ID (Not-0 Intercept): {sample_id_not0_intercept}")

sampled_slices_data_0_intercept = DICOMImages(sample_id_0_intercept).sampled_slices_data(n_samples=30)
sampled_slices_data_not0_intercept = DICOMImages(sample_id_not0_intercept).sampled_slices_data(n_samples=30)


# In[28]:


import cv2
from skimage import measure, morphology, segmentation


def threshold_slices_data(slices_data, low=-1000, high=-400):
    copy = slices_data.copy()
    copy[copy < low] = low
    copy[copy > high] = high
    return copy


# In[29]:


plot_slices_data(sampled_slices_data_0_intercept)
plot_slices_data(sampled_slices_data_not0_intercept)


# In[30]:


plot_slices_data(threshold_slices_data(sampled_slices_data_0_intercept, low=-1000, high=-400))
plot_slices_data(threshold_slices_data(sampled_slices_data_not0_intercept, low=-1000, high=-400))


# In[31]:


def watershed_separate_lungs(image, threshold_low=-1000, output_shape=(512, 512), **kwargs):
    """
    Segment lung image using watershed algorithm
    
    Parameters
    ----------
    image : numpy.ndarray
        hounsfield units (HU) transformed image
    threshold_low : int, default=-1000
        lower HU threshold for image
    output_shape : tuple, default=(512, 512)
        desired output masked segmented lung image
    kwargs
        kwarg for generate_markers()
        
    Returns
    -------
    segmented : numpy.ndarray(shape=output_shape)
        segmented lung image
    """
    #Creation of the markers as shown above:
    marker_internal, marker_external, marker_watershed = generate_markers(
        image,
        output_shape=output_shape,
        **kwargs
    )
    
    #Creation of the Sobel-Gradient
    sobel_filtered_dx = scipy.ndimage.sobel(image, 1)
    sobel_filtered_dy = scipy.ndimage.sobel(image, 0)
    sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
    sobel_gradient *= 255.0 / np.max(sobel_gradient)
    
    #Watershed algorithm
    watershed = morphology.watershed(sobel_gradient, marker_watershed)
    
    #Reducing the image created by the Watershed algorithm to its outline
    outline = scipy.ndimage.morphological_gradient(watershed, size=(3,3))
    outline = outline.astype(bool)
    
    #Performing Black-Tophat Morphology for reinclusion
    #Creation of the disk-kernel and increasing its size a bit
    blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0, 0]]
    blackhat_struct = scipy.ndimage.iterate_structure(blackhat_struct, 8)
    #Perform the Black-Hat
    outline += scipy.ndimage.black_tophat(outline, structure=blackhat_struct)
    
    #Use the internal marker and the Outline that was just created to generate the lungfilter
    lungfilter = np.bitwise_or(marker_internal, outline)
    #Close holes in the lungfilter
    #fill_holes is not used here, since in some slices the heart would be reincluded by accident
    lungfilter = scipy.ndimage.morphology.binary_closing(lungfilter, structure=np.ones((5,5)), iterations=3)
    
    #Apply the lungfilter (note the filtered areas being assigned to specified threshold HU)
    segmented = np.where(lungfilter == 1,
                         image,
                         threshold_low*np.ones(output_shape))
    
    return segmented

def generate_markers(image, threshold=-600, output_shape=(512, 512)):
    """
    Create watershed marker matrix.
    
    Parameters
    ----------
    image : numpy.ndarray
        hounsfield units (HU) transformed image
    threshold : int, default=-600
        threshold of internal marker, defaulting -600 for lung segmentation
    output_shape : tuple, default=(512, 512)
        desired output shape of marker_watershed
        
    Returns
    -------
    marker_internal : numpy.ndarray
    marker_external : numpy.ndarray
    watershed_marker : numpy.ndarray
    """
    marker_internal = image < threshold
    marker_internal = segmentation.clear_border(marker_internal)
    marker_internal_labels = measure.label(marker_internal)
    areas = [r.area for r in measure.regionprops(marker_internal_labels)]
    areas.sort()
    if len(areas) > 2:
        for region in measure.regionprops(marker_internal_labels):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                       marker_internal_labels[coordinates[0], coordinates[1]] = 0
    marker_internal = marker_internal_labels > 0
    
    external_a = scipy.ndimage.binary_dilation(marker_internal, iterations=10)
    external_b = scipy.ndimage.binary_dilation(marker_internal, iterations=55)
    marker_external = external_b ^ external_a
    
    marker_watershed = np.zeros(output_shape, dtype=np.int)
    marker_watershed += marker_internal * 255
    marker_watershed += marker_external * 128
    
    return marker_internal, marker_external, marker_watershed


def plot_watershed_segmentation(slices_data, cmap='Blues_r'):
    cnt = slices_data.shape[0]
    rows = cnt // 10
    fig, axes = plt.subplots(rows, 10, figsize=(14, rows*2))
    pbar = tqdm(zip(slices_data, axes.reshape(-1)), leave=False, total=cnt)
    for img, ax in pbar:
        segmented_img = watershed_separate_lungs(img,
                                                 threshold_low=-2000,
                                                 output_shape=(512, 512),
                                                 threshold=-400)
        ax.imshow(segmented_img, cmap=cmap)
        ax.axis('off')


# In[32]:


get_ipython().run_cell_magic('time', '', 'plot_watershed_segmentation(sampled_slices_data_0_intercept)\nplot_watershed_segmentation(sampled_slices_data_not0_intercept)')


# In[33]:


from skimage.filters import threshold_otsu, median
from skimage.segmentation import clear_border
from skimage import morphology
from scipy.ndimage import binary_fill_holes


def lung_segment(img, display=False):
    thresh = threshold_otsu(img)
    binary = img <= thresh

    lungs = median(clear_border(binary))
    lungs = morphology.binary_closing(lungs, selem=morphology.disk(7))
    lungs = binary_fill_holes(lungs)

    final = lungs*img
    final[final == 0] = np.min(img)
    
    if display:
        fig, ax = plt.subplots(1, 4, figsize=(15, 15))

        ax[0].set_title('HU Image')
        ax[0].imshow(img, cmap='gray')
        ax[0].axis('off')

        ax[1].set_title('Thresholded Image')
        ax[1].imshow(binary, cmap='gray')
        ax[1].axis('off')

        ax[2].set_title('Lungs Mask')
        ax[2].imshow(lungs, cmap='gray')
        ax[2].axis('off')

        ax[3].set_title('Final Image')
        ax[3].imshow(final, cmap='gray')
        ax[3].axis('off')
    
    return final, lungs


def plot_morphological_closing_segmentation(slices_data):
    cnt = slices_data.shape[0]
    rows = cnt // 10
    fig, axes = plt.subplots(rows, 10, figsize=(14, rows*2))
    pbar = tqdm(zip(slices_data, axes.reshape(-1)), leave=False, total=cnt)
    for img, ax in pbar:
        segmented_img, segmented_lungs = lung_segment(img)
        ax.imshow(segmented_img, cmap='Blues_r')
        ax.axis('off')


# In[34]:


get_ipython().run_cell_magic('time', '', 'plot_morphological_closing_segmentation(sampled_slices_data_0_intercept)\nplot_morphological_closing_segmentation(sampled_slices_data_not0_intercept)')


# In[35]:


get_ipython().run_cell_magic('time', '', 'plot_morphological_closing_segmentation(threshold_slices_data(sampled_slices_data_0_intercept))\nplot_morphological_closing_segmentation(threshold_slices_data(sampled_slices_data_not0_intercept))')


# In[36]:


def allunia_final_segment(slice, hu_max=-320):
    binary_image = np.array(slice > hu_max, dtype=np.int8)+1
    labels = measure.label(binary_image)

    background_label_1 = labels[0,0]
    background_label_2 = labels[0,-1]
    background_label_3 = labels[-1,0]
    background_label_4 = labels[-1,-1]

    #Fill the air around the person
    binary_image[background_label_1 == labels] = 2
    binary_image[background_label_2 == labels] = 2
    binary_image[background_label_3 == labels] = 2
    binary_image[background_label_4 == labels] = 2

    #We have a lot of remaining small signals outside of the lungs that need to be removed. 
    #In our competition closing is superior to fill_lungs 
    selem = morphology.disk(4)
    binary_image = morphology.closing(binary_image, selem)

    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    return binary_image


def plot_allunia_segmentation(slices_data, cmap='Blues_r'):
    cnt = slices_data.shape[0]
    rows = cnt // 10
    fig, axes = plt.subplots(rows, 10, figsize=(14, rows*2))
    pbar = tqdm(zip(slices_data, axes.reshape(-1)), leave=False, total=cnt)
    for img, ax in pbar:
        segmented_img = allunia_final_segment(img)
        ax.imshow(segmented_img, cmap=cmap)
        ax.axis('off')


# In[37]:


get_ipython().run_cell_magic('time', '', 'plot_allunia_segmentation(sampled_slices_data_0_intercept)\nplot_allunia_segmentation(sampled_slices_data_not0_intercept)')


# In[38]:


from sklearn.cluster import KMeans


def raddq_segmentation(img, display=False):
    """
    What changes:
    * Last dilation from (10, 10) to (15, 15)
    """
    row_size = img.shape[0]
    col_size = img.shape[1]
    
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    # Find the average pixel value near the lungs
    # to renormalize washed out images
    middle = img[int(col_size/5):int(col_size/5*4),int(row_size/5):int(row_size/5*4)] 
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
    # To improve threshold finding, I'm moving the 
    # underflow and overflow on the pixel spectrum
    img[img==max]=mean
    img[img==min]=mean
    #
    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    #
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image

    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.  
    # We don't want to accidentally clip the lung.

    eroded = morphology.erosion(thresh_img,np.ones([3,3]))
    dilation = morphology.dilation(eroded,np.ones([8,8]))

    labels = measure.label(dilation) # Different labels are displayed in different colors
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2]-B[0]<row_size/10*9 and B[3]-B[1]<col_size/10*9 and B[0]>row_size/5 and B[2]<col_size/5*4:
            good_labels.append(prop.label)
    mask = np.ndarray([row_size,col_size],dtype=np.int8)
    mask[:] = 0

    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask 
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    mask = morphology.dilation(mask, np.ones([15, 15])) # one last dilation

    if (display):
        fig, ax = plt.subplots(3, 2, figsize=[12, 12])
        ax[0, 0].set_title("Original")
        ax[0, 0].imshow(img, cmap='gray')
        ax[0, 0].axis('off')
        ax[0, 1].set_title("Threshold")
        ax[0, 1].imshow(thresh_img, cmap='gray')
        ax[0, 1].axis('off')
        ax[1, 0].set_title("After Erosion and Dilation")
        ax[1, 0].imshow(dilation, cmap='gray')
        ax[1, 0].axis('off')
        ax[1, 1].set_title("Color Labels")
        ax[1, 1].imshow(labels)
        ax[1, 1].axis('off')
        ax[2, 0].set_title("Final Mask")
        ax[2, 0].imshow(mask, cmap='gray')
        ax[2, 0].axis('off')
        ax[2, 1].set_title("Apply Mask on Original")
        ax[2, 1].imshow(mask*img, cmap='gray')
        ax[2, 1].axis('off')
        
        plt.show()
    return mask*img


def plot_raddq_segmentation(slices_data, cmap='Blues_r'):
    cnt = slices_data.shape[0]
    rows = cnt // 10
    fig, axes = plt.subplots(rows, 10, figsize=(14, rows*2))
    pbar = tqdm(zip(slices_data, axes.reshape(-1)), leave=False, total=cnt)
    for img, ax in pbar:
        segmented_img = raddq_segmentation(img)
        ax.imshow(segmented_img, cmap=cmap)
        ax.axis('off')


# In[39]:


get_ipython().run_cell_magic('time', '', 'plot_raddq_segmentation(sampled_slices_data_0_intercept)\nplot_raddq_segmentation(sampled_slices_data_not0_intercept)')


# In[40]:


get_ipython().run_cell_magic('time', '', 'plot_raddq_segmentation(threshold_slices_data(sampled_slices_data_0_intercept))\nplot_raddq_segmentation(threshold_slices_data(sampled_slices_data_not0_intercept))')


# In[41]:


worst_patient, best_patient = train_df    .loc[train_df['Weeks'] == 0, ['Patient', 'FVC']]    .sort_values(by='FVC')    .iloc[[0, -1]].values
worst_patient, best_patient


# In[42]:


worst_patient_slices_data = DICOMImages(worst_patient[0]).sampled_slices_data(30)
best_patient_slices_data = DICOMImages(best_patient[0]).sampled_slices_data(30)
plot_slices_data(threshold_slices_data(worst_patient_slices_data, low=-1000, high=-400))
plot_slices_data(threshold_slices_data(best_patient_slices_data, low=-1000, high=-400))


# In[43]:


from skimage.transform import resize
from matplotlib.patches import Circle


def crop_to_square(image):
    width, height = image.shape
    if width != height:
        min_ = min(width, height)
        if min_ == width:
            top = (height - width) // 2
            squared = image[ :, top:top+width].copy()
        else:
            left = (width - height) // 2
            squared = image[ left:left+height, :].copy()
    else:
        squared = image
    assert squared.shape[0] == squared.shape[1]
    return squared


def morphological_segmentation(img):
    segmented_img, _ = lung_segment(img)
    return segmented_img


def center_crop_pad(image, pct=0.08):
    """
    Parameters
    ----------
    image : numpy.ndarray
        slice of image
    pct : float or int, default=0.05
        if float, crop pct% of from all image sides, output 
            shape of (1-pct)% of original image
        if int, crop pct pixels from all image sides
    """
    original_width, original_height = image.shape
    if type(pct) == float:
        left = right = int(pct * original_width)
        top = bottom = int(pct * original_height)
    elif type(pct) == int:
        left = right = pct
        top = bottom = pct

    cropped_image = image[ left:original_width-right, top:original_height-bottom ].copy()
    padded_image = np.pad(cropped_image, [(left, right), (top, bottom)], mode='minimum')
    return padded_image


def recenter_image(slice_data):
    copy_ = slice_data.copy()
    width, height = copy_.shape
    min_val = copy_.min()
    copy_[copy_ != min_val] = 1
    copy_[copy_ == min_val] = 0
    cx, cy = scipy.ndimage.measurements.center_of_mass(copy_)
    return cx, cy

def plot_preprocess_steps(slice_data):
    squared = crop_to_square(slice_data)
    segmented = morphological_segmentation(squared)
    crop_pad = center_crop_pad(segmented)
    resized = resize(crop_pad, output_shape=(512, 512))
    cx, cy = recenter_image(resized)
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    axes = axes.reshape(-1)
    axes[0].imshow(slice_data, cmap='Blues_r')
    axes[0].set_title("Original Image")
    axes[1].imshow(squared, cmap='Blues_r')
    axes[1].set_title("Squared Image")
    axes[2].imshow(segmented, cmap='Blues_r')
    axes[2].set_title("Segmented Image")
    axes[3].imshow(crop_pad, cmap='Blues_r')
    axes[3].set_title("Center Crop & Padded Image")
    axes[4].imshow(resized, cmap='Blues_r')
    axes[4].add_patch(Circle((cx, cy), 24, color='y'))
    axes[4].set_title("Resized Image")
    axes[5].imshow(resized, cmap='Blues_r')
    axes[5].set_title("Centered Image")
    plt.show()


# In[44]:


sample_id_0_intercept = DICOMImages(all_dicoms_df[r_intercept_0_mask]['Patient ID'].sample(1).values[0])
sample_id_not0_intercept = DICOMImages(all_dicoms_df[~r_intercept_0_mask]['Patient ID'].sample(1).values[0])
is_not_square = all_dicoms_df['Rows'] != all_dicoms_df['Columns']
sample_id_not_square = DICOMImages(all_dicoms_df[is_not_square]['Patient ID'].sample(1).values[0])

print(f"Patient ID (0 Intercept): {sample_id_0_intercept.id}")
print(f"Patient ID (Not-0 Intercept): {sample_id_not0_intercept.id}")
print(f"Patient ID (Not Square): {sample_id_not_square.id}")

sampled_slices_data_0_intercept = sample_id_0_intercept.sampled_slices_data(n_samples=30)
sampled_slices_data_not0_intercept = sample_id_not0_intercept.sampled_slices_data(n_samples=30)
sampled_slices_data_not_square = sample_id_not_square.sampled_slices_data(n_samples=30)

middle_slice_data_0_intercept = sample_id_0_intercept.middle_slice_data
middle_slice_data_not0_intercept = sample_id_not0_intercept.middle_slice_data
middle_slice_data_not_square = sample_id_not_square.middle_slice_data


# In[45]:


binaried_image = middle_slice_data_0_intercept.copy()
binaried_image[binaried_image == 0] = -1000
plot_preprocess_steps(threshold_slices_data(binaried_image, low=-1000, high=-400))


# In[46]:


plot_preprocess_steps(threshold_slices_data(middle_slice_data_not0_intercept, low=-1000, high=-400))


# In[47]:


binaried_image = middle_slice_data_not_square.copy()
binaried_image[binaried_image == 0] = -1000
plot_preprocess_steps(threshold_slices_data(binaried_image, low=-1000, high=-400))


# In[48]:


import IPython.display as ipd
from matplotlib.patches import Rectangle


def segment_lung(slice_data, image_type, segment_func):
    if image_type == 'zero':
        slice_data[slice_data == 0] = -1000
    segmented_image = segment_func(threshold_slices_data(slice_data, low=-1000, high=-400))
    return segmented_image


def infer_bounding_box(segmented_image):
    y_match, x_match = np.where(segmented_image != -1000)
    y_min, x_min = y_match.min(), x_match.min()
    y_max, x_max = y_match.max(), x_match.max()
    width = abs(x_max - x_min)
    height = abs(y_max - y_min)
    return BoundingBox((x_min, y_min), width, height)


def draw_with_bounding_box(segmented_image, bbox):
    fig, axes = plt.subplots(figsize=(14, 6))
    axes.imshow(segmented_image, cmap='Blues_r')
    bbox_patch = Rectangle(*bbox.attribute_list,
                           fill=False,
                           color='yellow')
    axes.add_patch(bbox_patch)
    plt.show()
    
    
class BoundingBox:
    """Initiation of bbox follows matplotlib Rectangle patch"""
    def __init__(self, xy, width, height):
        self.x, self.y = xy
        self.width = width
        self.height = height
        
    @property
    def attribute_list(self):
        return [(self.x, self.y), self.width, self.height]
    
    def __repr__(self):
        return f"Bbox (bottom left width height): {self.x} {self.y} {self.width} {self.height}"

    
def crop_recenter(image, bbox, pad_value=-1000):
    x, y, width, height = bbox.x, bbox.y, bbox.width, bbox.height
    cropped_image = image[ y:y+height, x:x+width ]
    out_height, out_width = image.shape
    
    padded_image = np.ones(image.shape, dtype=np.int16) * pad_value
    x_start = (out_width - width) // 2
    y_start = (out_height - height) // 2
    padded_image[ y_start:y_start+height, x_start:x_start+width ] = cropped_image
    return padded_image


# In[49]:


from ipywidgets import IntSlider, interact, fixed, interact_manual, Text


def manual_bbox(slice_data, image_type, x=None, y=None, x_max=None, y_max=None):
    width = x_max - x
    height = y_max - y
    segment_func = morphological_segmentation
    segmented_image = segment_lung(slice_data, image_type, segment_func=segment_func)
    bbox = infer_bounding_box(segmented_image)
    print("Inferred bbox (bottom-left), width, height:", bbox.x, bbox.y, bbox.width, bbox.height)
    bbox.x = x or bbox.x
    bbox.y = y or bbox.y
    bbox.width = width or bbox.width
    bbox.height = height or bbox.height

    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    axes[0].imshow(slice_data, cmap='Blues_r')
    axes[0].set_title('Original Image')

    # Segmented image with bbox
    bbox_patch = Rectangle(*bbox.attribute_list,
                           fill=False,
                           color='yellow')
    axes[1].imshow(segmented_image, cmap='Blues_r')
    axes[1].add_patch(bbox_patch)
    axes[1].set_title(f'Segmented ({segment_func.__name__})')

    # Crop centered image according to bbox
    crop_recentered_image = crop_recenter(segmented_image, bbox)
    axes[2].imshow(crop_recentered_image, cmap='Blues_r')
    axes[2].set_title('Bbox Crop & Centered')
    plt.show()


# In[50]:


# Define all the 'unique' ids
varying_pixel_spacing_ids = ['ID00099637202206203080121']
failed_segmentation_ids = ['ID00026637202179561894768']
bad_ids = ['ID00011637202177653955184', 'ID00052637202186188008618']

# Define the filter mask
not_bad_ids = ~all_dicoms_df['Patient ID'].isin(bad_ids)
zero_intercept = all_dicoms_df['Rescale Intercept'] == 0

# Apply the filter mask
all_zero_intercept_ids = all_dicoms_df[not_bad_ids & zero_intercept]['Patient ID']
all_zero_dicoms = [DICOMImages(id) for id in all_zero_intercept_ids]
all_not_zero_intercept_ids = all_dicoms_df[not_bad_ids & ~zero_intercept]['Patient ID']
all_not_zero_dicoms = [DICOMImages(id) for id in all_not_zero_intercept_ids]


# In[51]:


threshold_map_zero_intercept = []
more_broken_ids = []
more_exceptions = []
is_final = 'q'  # change to 'n' to enable interactive mode
i = 1
total = len(all_zero_dicoms)
segment_func = morphological_segmentation
for dicoms in all_zero_dicoms:
    try:
        ipd.clear_output(wait=True)
        middle_slice_data = dicoms.middle_slice_data
        image_type = dicoms.image_type['name']
        segmented_image = segment_lung(middle_slice_data, image_type, segment_func=segment_func)
        bbox = infer_bounding_box(segmented_image)
        manual_bbox(middle_slice_data, image_type, bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height)

        while is_final not in ['y', 'q']:
            print(f"{i}/{total}")
            is_final = input()
            if is_final in ['y', 'q']:
                break

            bbox.x, bbox.y, bbox.width, bbox.height = map(int, input().split())
            print(bbox)
    #         print(f"New bbox (bottom-left), width, height: {bbox.attribute_list}")
            manual_bbox(middle_slice_data, image_type, bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height)

        if is_final == 'q':
            break

        threshold_map_zero_intercept.append({
            "patient": dicoms.id,
            "x": bbox.x,
            "y": bbox.y,
            "width": bbox.width,
            "height": bbox.height
        })
        i += 1
        is_final = 'n'
    except Exception as e:
        more_exceptions.append(e)
        more_broken_ids.append(dicoms.id)


# In[52]:


# df_map_zero = pd.DataFrame(threshold_map_zero_intercept)
# df_map_zero.to_csv('threshold_map_zero.csv', header=True, index=False)

# df_map_zero_exceptions = pd.DataFrame(list(zip(more_broken_ids, more_exceptions)), columns=['id', 'exception'])
# df_map_zero_exceptions.to_csv('threshold_map_zero_exceptions.csv', header=True, index=False)


# In[53]:


threshold_map_not_zero_intercept = []
more_broken_ids = []
more_exceptions = []
is_final = 'q'  # # change to 'n' to enable interactive mode
i = 1
total = len(all_not_zero_dicoms)
segment_func = morphological_segmentation
for dicoms in all_not_zero_dicoms:
    try:
        ipd.clear_output(wait=True)
        middle_slice_data = dicoms.middle_slice_data
        image_type = dicoms.image_type['name']
        segmented_image = segment_lung(middle_slice_data, image_type, segment_func=segment_func)
        bbox = infer_bounding_box(segmented_image)
        manual_bbox(middle_slice_data, image_type, bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height)

        while is_final not in ['y', 'q']:
            print(f"{i}/{total}")
            is_final = input()
            if is_final in ['y', 'q']:
                break

            bbox.x, bbox.y, bbox.width, bbox.height = map(int, input().split())
            print(bbox)
    #         print(f"New bbox (bottom-left), width, height: {bbox.attribute_list}")
            manual_bbox(middle_slice_data, image_type, bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height)

        if is_final == 'q':
            break

        threshold_map_not_zero_intercept.append({
            "patient": dicoms.id,
            "x": bbox.x,
            "y": bbox.y,
            "width": bbox.width,
            "height": bbox.height
        })
        i += 1
        is_final = 'n'
    except Exception as e:
        more_exceptions.append(e)
        more_broken_ids.append(dicoms.id)


# In[54]:


# df_map_not_zero = pd.DataFrame(threshold_map_not_zero_intercept)
# df_map_not_zero.to_csv('threshold_map_not_zero.csv', header=True, index=False)

# df_map_not_zero_exceptions = pd.DataFrame(list(zip(more_broken_ids, more_exceptions)), columns=['id', 'exception'])
# df_map_not_zero_exceptions.to_csv('threshold_map_not_zero_exceptions.csv', header=True, index=False)


# In[55]:


"""Interactive Part"""
sample_dicoms = DICOMImages('ID00376637202297677828573')
middle_slice_data = sample_dicoms.middle_slice_data
image_type = sample_dicoms.image_type['name']
segmented_image = segment_lung(middle_slice_data, image_type, segment_func=segment_func)
bbox = infer_bounding_box(segmented_image)
x_ = IntSlider(min=0, max=middle_slice_data.shape[1], value=bbox.x)
y_ = IntSlider(min=0, max=middle_slice_data.shape[1], value=bbox.y)
x_max_ = IntSlider(min=0, max=middle_slice_data.shape[0], value=bbox.x + bbox.width)
y_max_ = IntSlider(min=0, max=middle_slice_data.shape[0], value=bbox.y + bbox.height)
values = interact_manual(
    manual_bbox,
    slice_data=fixed(middle_slice_data),
    image_type=fixed(image_type),
    x=x_,
    y=y_,
    x_max=x_max_,
    y_max=y_max_,
)

