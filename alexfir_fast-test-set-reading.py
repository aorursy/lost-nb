#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import os.path
import time

COLUMN_TO_TYPE = {
    'object_id': np.int32,
    'mjd': np.float32,
    'passband': np.int8,
    'flux': np.float32,
    'flux_err': np.float32,
    'detected': np.int8
}

part1_directory = r'../input/test-set-columns-part-1'
part2_directory = r'../input/test-set-columns-part-2'

COLUMN_TO_FOLDER = {
    'object_id': part2_directory,
    'mjd': part2_directory,
    'passband': part2_directory,
    'flux': part1_directory,
    'flux_err': part1_directory,
    'detected': part1_directory
}


def init_reading():
    info = {}
    object_range_file_path = os.path.join(COLUMN_TO_FOLDER['object_id'], 'object_id_range.h5')
    print('reading {}'.format(object_range_file_path))
    object_id_to_range = pd.read_hdf(object_range_file_path, 'data')
    info['object_id_to_range'] = object_id_to_range
    id_to_range = object_id_to_range.set_index('object_id')
    info['object_id_start'] = id_to_range['start'].to_dict()
    info['object_id_end'] = id_to_range['end'].to_dict()

    records_number = object_id_to_range['end'].max()

    mmaps = {}
    for column, dtype in COLUMN_TO_TYPE.items():
        directory = COLUMN_TO_FOLDER[column]
        file_path = os.path.join(directory, 'test_set_{}.bin'.format(column))
        mmap = np.memmap(file_path, dtype=COLUMN_TO_TYPE[column], mode='r', shape=(records_number,))
        mmaps[column] = mmap

    info['mmaps'] = mmaps

    return info


def read_object_info(info, object_id, as_pandas=True, columns=None):
    start = info['object_id_start'][object_id]
    end = info['object_id_end'][object_id]

    data = read_object_by_index_range(info, start, end, as_pandas, columns)
    return data


def read_object_by_index_range(info, start, end, as_pandas=True, columns=None):
    data = {}
    for column, mmap in info['mmaps'].items():
        if columns is None or column in columns:
            data[column] = mmap[start: end]

    if as_pandas:
        data = pd.DataFrame(data)

    return data


def get_chunks(info, chunk_size=1000):
    object_id_to_range = info['object_id_to_range']
    end_of_file_offset = object_id_to_range['end'].max()
    start_offsets = object_id_to_range['start'].values[::chunk_size]
    end_offsets = object_id_to_range['end'].values[(chunk_size - 1)::chunk_size]

    end_offsets = list(end_offsets) + [end_of_file_offset]

    chunks = pd.DataFrame({'start': start_offsets, 'end': end_offsets})
    chunks = chunks.values.tolist()

    return chunks




info = init_reading()




# single object read as pandas object, first object
object_info13 = read_object_info(info, 13)
object_info13.head()




# last object from test_set
object_info104853812 = read_object_info(info, 104853812)
object_info104853812.tail()




object_info104853812 = read_object_info(info, 104853812, as_pandas=False, columns=['flux', 'flux_err'])
object_info104853812['flux'][-5:]




object_ids = info['object_id_to_range']['object_id'].values.tolist()
start_time = time.time()
records_read = 0
for object_id in object_ids:
    object_info = read_object_info(info, object_id, columns=['flux'], as_pandas=False)
    flux = object_info['flux']
    records_read += flux.shape[0]
    max = flux.max()

print("Single field reading took {:6.4f} secs, records = {}".format((time.time() - start_time), records_read))




start_time = time.time()
records_read = 0
chunks = get_chunks(info, 10_000)
for index_start, index_end in chunks:
    data = read_object_by_index_range(info, index_start, index_end)
    flux = data['flux']
    max = flux.max()
    records_read += data.shape[0]

print("Chunks reading took {:6.4f} secs, records = {}".format((time.time() - start_time), records_read))

