#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import timedelta
import numpy as np
import pandas as pd
from pathlib import Path
import pydicom
from scipy.ndimage.interpolation import zoom
from scipy.stats import skew, kurtosis
import scipy.ndimage as ndimage
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from skimage import measure, morphology, segmentation
from time import time, sleep
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import trange
import warnings
import matplotlib.pyplot as plt
from matplotlib import cm
import pytest
import multiprocessing as mp
from time import time
from datetime import timedelta
from functools import partial
from tqdm import tqdm


# In[2]:


class CTScansDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.patients = [p for p in self.root_dir.glob('*') if p.is_dir()]
        self.transform = transform

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image, metadata, msbs = self.load_scan(self.patients[idx])
        sample = {'image': image, 'metadata': metadata, 'msbs': msbs}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def save(self, path):
        t0 = time()
        Path(path).mkdir(exist_ok=True, parents=True)
        print('Saving pre-processed dataset to disk')
        sleep(1)
        cum = 0
        var = 0

        bar = trange(len(self))
        for i in bar:
            sample = self[i]
            image, data = sample['image'], sample['metadata']
            cum += torch.mean(image).item()
            var += torch.var(image).item()

            bar.set_description(f'Saving CT scan {data.PatientID}')
            fname = Path(path) / f'{data.PatientID}.pt'
            torch.save(image, fname)

        sleep(1)
        bar.close()
        print(f'Done! Time {timedelta(seconds=time() - t0)}\n'
              f'Mean value: {cum / len(self)}\n'
              f'Std value: {(var / len(self)) ** (1/2)}')

    def get_patient(self, patient_id):
        patient_ids = [str(p.stem) for p in self.patients]
        return self.__getitem__(patient_ids.index(patient_id))

    @staticmethod
    def load_scan(path):
        raw_slices = [pydicom.read_file(p) for p in path.glob('*.dcm')]
        final_slices = [s for s in raw_slices
                        if hasattr(s, 'ImagePositionPatient')]

        if len(final_slices) > 0:
            final_slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
            msbs = 0
            for i in range(len(final_slices) - 1):
                a = final_slices[i].ImagePositionPatient[2]
                b = final_slices[i + 1].ImagePositionPatient[2]
                msbs += abs(a - b)

            msbs /= len(final_slices)
            image = np.stack([s.pixel_array.astype(float) for s in final_slices])
            metadata = final_slices[0]

        else:
            # Guess based on eyballing other mean spaces between slices:
            msbs = None
            if raw_slices[0].PatientID == 'ID00132637202222178761324':
                msbs = 0.7
            elif raw_slices[0].PatientID == 'ID00128637202219474716089':
                msbs = 5.

            warnings.warn(f'Patient {raw_slices[0].PatientID} CT scan does '
                          f'not have "ImagePositionPatient". Assuming '
                          f'filenames in the right scan order. Also, assuming'
                          f'mean space between slices of {msbs}')

            image = np.stack([s.pixel_array.astype(float) for s in raw_slices])
            metadata = raw_slices[0]

        return image, metadata, msbs


# In[3]:


class CropBoundingBox:
    @staticmethod
    def bounding_box(img3d: np.array):
        mid_img = img3d[int(img3d.shape[0] / 2)]
        same_first_row = (mid_img[0, :] == mid_img[0, 0]).all()
        same_first_col = (mid_img[:, 0] == mid_img[0, 0]).all()
        if same_first_col and same_first_row:
            return True
        else:
            return False

    def __call__(self, sample):
        image, data = sample['image'], sample['metadata']
        if not self.bounding_box(image):
            return sample

        mid_img = image[int(image.shape[0] / 2)]
        r_min, r_max = None, None
        c_min, c_max = None, None
        for row in range(mid_img.shape[0]):
            if not (mid_img[row, :] == mid_img[0, 0]).all() and r_min is None:
                r_min = row
            if (mid_img[row, :] == mid_img[0, 0]).all() and r_max is None                     and r_min is not None:
                r_max = row
                break

        for col in range(mid_img.shape[1]):
            if not (mid_img[:, col] == mid_img[0, 0]).all() and c_min is None:
                c_min = col
            if (mid_img[:, col] == mid_img[0, 0]).all() and c_max is None                     and c_min is not None:
                c_max = col
                break

        image = image[:, r_min:r_max, c_min:c_max]
        return {'image': image, 'metadata': data, 'msbs': sample['msbs']}


# In[4]:


class ConvertToHU:
    def __call__(self, sample):
        image, data = sample['image'], sample['metadata']

        img_type = data.ImageType
        is_hu = img_type[0] == 'ORIGINAL' and not (img_type[2] == 'LOCALIZER')
        # if not is_hu:
        #     warnings.warn(f'Patient {data.PatientID} CT Scan not cannot be'
        #                   f'converted to Hounsfield Units (HU).')

        intercept = data.RescaleIntercept
        slope = data.RescaleSlope
        image = (image * slope + intercept).astype(np.int16)
        return {'image': image, 'metadata': data, 'msbs': sample['msbs']}


# In[5]:


class Resample:
    def __init__(self, new_spacing=(1, 1, 1)):
        """new_spacing means now much every pixel represent in mm, in each
        dimension. E.g. 2, 2, 2 means every pixel represent 2mm in  every
        dimension.
        """
        assert isinstance(new_spacing, tuple)
        self.new_spacing = new_spacing

    def __call__(self, sample):
        image, data, msbs = sample['image'], sample['metadata'], sample['msbs']

        spacing = np.array([msbs] + list(data.PixelSpacing), dtype=np.float32)
        resize_factor = spacing / self.new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape

        image = zoom(image, real_resize_factor, mode='nearest')
        return {'image': image, 'metadata': data}


# In[6]:


class Clip:
    def __init__(self, bounds=(-1000, 500)):
        self.min = min(bounds)
        self.max = max(bounds)

    def __call__(self, sample):
        image, data = sample['image'], sample['metadata']
        image[image < self.min] = self.min
        image[image > self.max] = self.max
        return {'image': image, 'metadata': data}


# In[7]:


class MaskWatershed:
    def __init__(self, min_hu, iterations, show_tqdm):
        self.min_hu = min_hu
        self.iterations = iterations
        self.show_tqdm = show_tqdm

    def __call__(self, sample):
        image, data = sample['image'], sample['metadata']

        stack = []
        if self.show_tqdm:
            bar = trange(image.shape[0])
            bar.set_description(f'Masking CT scan {data.PatientID}')
        else:
            bar = range(image.shape[0])
        for slice_idx in bar:
            sliced = image[slice_idx]
            stack.append(self.seperate_lungs(sliced, self.min_hu,
                                             self.iterations))

        return {
            'image': np.stack(stack),
            'metadata': sample['metadata']
        }

    @staticmethod
    def seperate_lungs(image, min_hu, iterations):
        h, w = image.shape[0], image.shape[1]

        marker_internal, marker_external, marker_watershed = MaskWatershed.generate_markers(image)

        # Sobel-Gradient
        sobel_filtered_dx = ndimage.sobel(image, 1)
        sobel_filtered_dy = ndimage.sobel(image, 0)
        sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
        sobel_gradient *= 255.0 / np.max(sobel_gradient)

        watershed = morphology.watershed(sobel_gradient, marker_watershed)

        outline = ndimage.morphological_gradient(watershed, size=(3,3))
        outline = outline.astype(bool)

        # Structuring element used for the filter
        blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],
                           [0, 1, 1, 1, 1, 1, 0],
                           [1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1],
                           [0, 1, 1, 1, 1, 1, 0],
                           [0, 0, 1, 1, 1, 0, 0]]

        blackhat_struct = ndimage.iterate_structure(blackhat_struct, iterations)

        # Perform Black Top-hat filter
        outline += ndimage.black_tophat(outline, structure=blackhat_struct)

        lungfilter = np.bitwise_or(marker_internal, outline)
        lungfilter = ndimage.morphology.binary_closing(lungfilter, structure=np.ones((5,5)), iterations=3)

        segmented = np.where(lungfilter == 1, image, min_hu * np.ones((h, w)))

        return segmented

    @staticmethod
    def generate_markers(image, threshold=-400):
        h, w = image.shape[0], image.shape[1]

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

        # Creation of the External Marker
        external_a = ndimage.binary_dilation(marker_internal, iterations=10)
        external_b = ndimage.binary_dilation(marker_internal, iterations=55)
        marker_external = external_b ^ external_a

        # Creation of the Watershed Marker
        marker_watershed = np.zeros((h, w), dtype=np.int)
        marker_watershed += marker_internal * 255
        marker_watershed += marker_external * 128

        return marker_internal, marker_external, marker_watershed


# In[8]:


class LungVolume:
    def __init__(self, unit, spacing, min_HU):
        self.unit = unit
        self.spacing = spacing
        self.min_HU = min_HU

    def __call__(self, sample):
        image = sample['image']

        volume = (image > self.min_HU).sum()
        volumes_in_perc = volume / np.prod(image.shape)
        volumes_in_litr = volumes_in_perc * np.prod(image.shape) *                           np.prod(self.spacing) / 10**6

        if self.unit == '%':
            sample['volume'] = volumes_in_perc
        elif self.unit == 'l':
            sample['volume'] = volumes_in_litr

        return sample


# In[9]:


class ChestCircumference:
    def __init__(self, spacing, min_HU):
        self.spacing = spacing
        self.min_HU = min_HU

    def ellipsis_perimeter(self, img):
        # trim slice, removing zero rows and cols
        img = img[~(img == self.min_HU).all(1)].T
        img = img[~(img == self.min_HU).all(1)]
        # ellipsis perimeter approximation 2
        # https://www.mathsisfun.com/geometry/ellipse-perimeter.html
        a, b = img.shape
        a /= 2
        b /= 2
        p = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))
        return p

    def __call__(self, sample):
        image = sample['image']

        perimeters = []
        for slice_idx in range(image.shape[0]):
            slc = image[slice_idx]
            if slc.max() != self.min_HU:  # There's something in the slice
                perimeters.append(self.ellipsis_perimeter(slc))
            else:
                perimeters.append(0)

        # Transform in mm
        cc_in_mm = max(perimeters) * self.spacing[1] * self.spacing[2]
        sample['chest'] = int(cc_in_mm)
        return sample


# In[10]:


class LungHeight:
    def __init__(self, spacing):
        self.spacing = spacing

    def __call__(self, sample):
        image = sample['image']
        sample['height'] = image.shape[0] * self.spacing[0]
        return sample


# In[11]:


class HUHistogram:
    def __init__(self, bounds, bins):
        self.bounds = bounds
        self.bins = bins

    def __call__(self, sample):
        image = sample['image']
        pixels = image[image != min(self.bounds)]

        if pixels.shape[0] > 0:
            sample['mean'] = pixels.mean()
            sample['std'] = pixels.std()
            sample['skew'] = skew(pixels)
            sample['kurtosis'] = kurtosis(pixels)

            hist, _ = np.histogram(pixels, bins=self.bins, range=self.bounds)
            sample['hist'] = hist / pixels.size  # Normalized
            # Density = True in np.histogram doesn't work as expected
        else:
            sample['mean'] = 0
            sample['std'] = 0
            sample['skew'] = 0
            sample['kurtosis'] = 0
            sample['hist'] = np.zeros(self.bins)

        return sample


# In[12]:


def gennerate_row(dataset, i):
    sample = dataset[i]
    df = pd.DataFrame(columns=np.arange(8).tolist())
    df.loc[0] = [sample['metadata'].PatientID, sample["volume"],
                 sample["chest"], sample["height"], sample["mean"],
                 sample["std"], sample["skew"], sample["kurtosis"]]

    dfhist = pd.DataFrame()
    dfhist[0] = sample['hist']
    dfhist = dfhist.T
    df = pd.concat([df, dfhist], axis=1, ignore_index=True)

    columns = ['PatientID', 'Volume', 'Chest', 'Height', 'Mean',
               'Std', 'Skew', 'Kurtosis']
    columns += [f'bin{b + 1}' for b in range(len(dfhist.columns))]
    df.columns = columns

    return df


# In[13]:


t0 = time()
dataset = CTScansDataset(
    root_dir='../input/osic-pulmonary-fibrosis-progression/test',
    transform=transforms.Compose([
        CropBoundingBox(),
        ConvertToHU(),
        Resample(new_spacing=(1, 1, 1)),
        Clip(bounds=(-1000, 400)),
        MaskWatershed(min_hu=-1000, iterations=1, show_tqdm=False),
        # Features
        LungVolume(unit='l',
                   spacing=(1, 1, 1),
                   min_HU=-1000),
        ChestCircumference(spacing=(1, 1, 1), min_HU=-1000),
        LungHeight(spacing=(1, 1, 1)),
        HUHistogram(bounds=(-1000, 400), bins=14)
    ]))

f = partial(gennerate_row, dataset)
with mp.Pool(processes=mp.cpu_count()) as pool:
    results = list(tqdm(pool.imap_unordered(f, range(len(dataset))),
                        total=len(dataset)))

features = pd.concat(results, ignore_index=True)
features.to_csv('features.csv', index=False)

print(f'\nDone! Time: {timedelta(seconds=time() - t0)}')


# In[14]:


features


# In[15]:


pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv').to_csv('submission.csv', index=False)

