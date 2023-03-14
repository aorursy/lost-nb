#!/usr/bin/env python
# coding: utf-8



# For testing, multiprocessing and chaining dictionaries
import numpy as np
import multiprocessing
from collections import ChainMap
import matplotlib.pyplot as plt




# Default RLenc
def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  # list of run lengths
    r = 0  # the current run length
    pos = 1  # count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs += [pos, r]
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs += [pos, r]
        pos += r
        r = 0

    return runs

# RLE encoding, as suggested by Tadeusz Hupało
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths




class Consumer(multiprocessing.Process):
    """Consumer for performing a specific task."""

    def __init__(self, task_queue, result_queue):
        """Initialize consumer, it has a task and result queues."""
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        """Actual run of the consumer."""
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                self.task_queue.task_done()
                break
            # Fetch answer from task
            answer = next_task()
            self.task_queue.task_done()
            # Put into result queue
            self.result_queue.put(answer)
        return


class RleTask_Suggested(object):
    """Wrap the RLE Encoder into a Task."""

    def __init__(self, idx, img):
        """Save image to self."""
        self.idx = idx
        self.img = img

    def __call__(self):
        """When object is called, encode."""
        return {self.idx: rle_encoding(self.img)}

class RleTask(object):
    """Wrap the RLE Encoder into a Task."""

    def __init__(self, idx, img):
        """Save image to self."""
        self.idx = idx
        self.img = img

    def __call__(self):
        """When object is called, encode."""
        return {self.idx: RLenc(self.img)}

class MultiOriginal(object):
    """Perform RLE in paralell."""

    def __init__(self, num_consumers=2):
        """Initialize class."""
        self._tasks = multiprocessing.JoinableQueue()
        self._results = multiprocessing.Queue()
        self._n_consumers = num_consumers
        self._add_count = 0

        # Initialize consumers
        self._consumers = [Consumer(self._tasks, self._results) for i in range(self._n_consumers)]
        for w in self._consumers:
            w.start()

    def add(self, idx, img):
        """Add a task to perform."""
        self._add_count += 1
        self._tasks.put(RleTask(idx, img))

    def get_results(self):
        """Close all tasks."""
        # Provide poison pill
        [self._tasks.put(None) for _ in range(self._n_consumers)]
        # Wait for finish
        self._tasks.join()
        # Return results
        singles = []
        for _ in range(self._add_count):
            singles.append(self._results.get())
        return dict(ChainMap({}, *singles))

class MultiSuggested(object):
    """Perform RLE in paralell."""

    def __init__(self, num_consumers=2):
        """Initialize class."""
        self._tasks = multiprocessing.JoinableQueue()
        self._results = multiprocessing.Queue()
        self._n_consumers = num_consumers
        self._add_count = 0

        # Initialize consumers
        self._consumers = [Consumer(self._tasks, self._results) for i in range(self._n_consumers)]
        for w in self._consumers:
            w.start()

    def add(self, idx, img):
        """Add a task to perform."""
        self._add_count += 1
        self._tasks.put(RleTask_Suggested(idx, img))

    def get_results(self):
        """Close all tasks."""
        # Provide poison pill
        [self._tasks.put(None) for _ in range(self._n_consumers)]
        # Wait for finish
        self._tasks.join()
        # Return results
        singles = []
        for _ in range(self._add_count):
            singles.append(self._results.get())
        return dict(ChainMap({}, *singles))




example_batch = np.random.uniform(0, 1, size=(100, 101, 101)) > 0.5

# Wrap the FastRle class into a method so we measure the time
def original(array):
    results = {}
    for i, arr in enumerate(array):
        results['%d' % i] = RLenc(arr)
    return results

def multi_original(array):
    rle = MultiOriginal(4)
    for i, arr in enumerate(array):
        rle.add('%d' % i, arr)
    return rle.get_results()

def suggested(array):
    results = {}
    for i, arr in enumerate(array):
        results['%d' % i] = rle_encoding(arr)
    return results

def multi_suggested(array):
    rle = MultiSuggested(4)
    for i, arr in enumerate(array):
        rle.add('%d' % i, arr)
    return rle.get_results()
    
# Measure the time
get_ipython().run_line_magic('timeit', '-n1 original(example_batch)')
get_ipython().run_line_magic('timeit', '-n1 multi_original(example_batch)')
get_ipython().run_line_magic('timeit', '-n1 suggested(example_batch)')
get_ipython().run_line_magic('timeit', '-n1 multi_suggested(example_batch)')




# Create a loop, collect time info for different methods
sample_sizes = [100, 250, 500, 1000, 2000, 5000, 10000]
org, morg, sug, msug = ([], [], [], [])
for n_samples in sample_sizes:
    example_batch = np.random.uniform(0, 1, size=(n_samples, 101, 101)) > 0.5
    result_org = get_ipython().run_line_magic('timeit', '-n1 -o original(example_batch)')
    result_multi_org = get_ipython().run_line_magic('timeit', '-n1 -o multi_original(example_batch)')
    result_sug = get_ipython().run_line_magic('timeit', '-n1 -o suggested(example_batch)')
    result_multi_sug = get_ipython().run_line_magic('timeit', '-n1 -o multi_suggested(example_batch)')
    
    org.append(result_org.average)
    morg.append(result_multi_org.average)
    sug.append(result_sug.average)
    msug.append(result_multi_sug.average)




# Plot the results
plt.figure(dpi=150)
ax = plt.axes()
ax.plot(sample_sizes, org, label='original');
ax.plot(sample_sizes, morg, label='multi-original');
ax.plot(sample_sizes, sug, label='suggested');
ax.plot(sample_sizes, msug, label='multi-suggested');
plt.legend()




example_batch = np.random.uniform(0, 1, size=(10, 101, 101)) > 0.5
a = original(example_batch)
b = multi_original(example_batch)
c = suggested(example_batch)
d = multi_suggested(example_batch)
# Make sure they are the same
for key in a:
    if a[key] != b[key]:
        print("Multi processed original differs from original!")
    if a[key] != c[key]:
        print("Suggested differs from original!")
    if a[key] != d[key]:
        print("Multi processed suggested differs from original!")











