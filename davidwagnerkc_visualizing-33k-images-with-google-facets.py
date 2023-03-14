#!/usr/bin/env python
# coding: utf-8



from IPython.core.display import display, HTML
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageDraw




# Bounding boxes from this kernel (@suicaokhoailang ran Martin Piotte's model on the current competition dataset)
# https://www.kaggle.com/suicaokhoailang/generating-whale-bounding-boxes
bb_df = pd.read_csv('../input/boundingbox/bounding_boxes.csv')




DATA_DIR = Path('/kaggle/input/humpback-whale-identification/')
TRAIN_DIR = DATA_DIR / 'train'
TEST_DIR =  DATA_DIR / 'test'

train_df = pd.read_csv(DATA_DIR / 'train.csv')
test_df = pd.read_csv(DATA_DIR / 'sample_submission.csv')




w, h = (bb_df.x1 - bb_df.x0), (bb_df.y1 - bb_df.y0)
bb_df['crop_size'] = w * h
bb_df['crop_ratio'] = w / h




bb_df.head()




train_df['freq'] = train_df.groupby('Id')['Id'].transform('count')
train_df['set'] = 'train'
train_df = train_df.sort_values('freq', ascending=False)




test_df['Id'] = 'unknown'
test_df['freq'] = 1
test_df['set'] = 'test'




df = pd.concat([train_df, test_df]).reset_index(drop=True)




df = pd.merge(df, bb_df, on='Image')




# Add image ratio data
def ratio(row):
    im_path = TRAIN_DIR / row.Image if 'train' in row.set else TEST_DIR / row.Image
    im = Image.open(im_path)
    return im.width / im.height

def total_size(row):
    im_path = TRAIN_DIR / row.Image if 'train' in row.set else TEST_DIR / row.Image
    im = Image.open(im_path)
    return im.width * im.height




def draw_bb(row):
    im_path = TRAIN_DIR / row.Image if 'train' in row.set else TEST_DIR / row.Image
    im = Image.open(im_path)
    bb = row.x0, row.y0, row.x1, row.y1
    draw = ImageDraw.Draw(im) 
    draw.rectangle(bb, outline=255)
    im.save(Path('/kaggle/working/draw_crops/') / row.Image)
    return True




p = Pool()




get_ipython().run_cell_magic('time', '', "df['ratio'] = p.map(ratio, [x[1] for x in list(df.iterrows())]) #df.apply(ratio, axis=1)\ndf['total_size'] = p.map(ratio, [x[1] for x in list(df.iterrows())]) #df.apply(total_size, axis=1)")




df['crop_perc'] = df.crop_size / df.total_size




df[::3000]




# !mkdir draw_crops




# %%time
# p.map(draw_bb, [x[1] for x in list(df.iterrows())])




df = df.drop(['x0', 'x1', 'y0', 'y1'], axis=1)




df['x_rand'] = np.random.random(len(df))
df['y_rand'] = np.random.random(len(df))




get_ipython().system('git clone https://github.com/PAIR-code/facets.git')




# If anybody is interested in building Facets themselves this might be useful. Turns out I didn't need to build Atlasmaker since it is just three Python modules.

# !pip install -r facets/facets_atlasmaker/requirements.txt
# !apt-get install -y pkg-config zip g++ zlib1g-dev unzip python
# !curl -LOk https://github.com/bazelbuild/bazel/releases/download/0.21.0/bazel-0.21.0-installer-linux-x86_64.sh
# !chmod +x bazel-0.21.0-installer-linux-x86_64.sh
# !bash bazel-0.21.0-installer-linux-x86_64.sh

# cd /kaggle/working/facets/facets_atlasmaker/
# %%time
# !bazel build :atlasmaker

# cd /kaggle/working/facets/bazel-bin/facets_atlasmaker/




cd /kaggle/working/facets/facets_atlasmaker/




# Does anybody use Python 2 anymore?
get_ipython().system("sed -i 's/from urlparse import urlparse/from urllib.parse import urlparse/g' atlasmaker_io.py")
# Let's pretend tensorflow isn't available to avoid another Python 2 problem 
get_ipython().system("sed -i 's/import tensorflow as tf/import tensorflop/g' atlasmaker_io.py")




#df.apply(lambda x: str(Path('/kaggle/working/draw_crops/') / x.Image), axis=1).to_csv('absolute_paths.csv', index=False)
df.apply(lambda x: str(TRAIN_DIR / x.Image) if 'train' in x.set else str(TEST_DIR / x.Image), axis=1).to_csv('absolute_paths.csv', index=False)




df.ratio.mean()




get_ipython().run_cell_magic('time', '', '!python atlasmaker.py --sourcelist=absolute_paths.csv --image_width=58 --image_height=29 --output_dir=/kaggle/working/')




#Image.open('/kaggle/working/spriteatlas.png').convert('L').save('/kaggle/working/spriteatlas.png', optimize=True)




cd /kaggle/working/




sprite_width, sprite_height = 58, 29
atlas_path = 'spriteatlas.png'
jsonstr = df.to_json(orient='records')
html = f"""<link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/master/facets-dist/facets-jupyter.html">
           <facets-dive atlas-url="{atlas_path}" fit-grid-aspect-ratio-to-viewport="true" sprite-image-width="{sprite_width}" sprite-image-height="{sprite_height}" height="800" id="elem"></facets-dive>
           <script>document.querySelector("#elem").data = {jsonstr};</script>"""
display(HTML(html))




html = f"""<link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/master/facets-dist/facets-jupyter.html">
           <facets-dive atlas-url="{atlas_path}" fit-grid-aspect-ratio-to-viewport="true" cross-origin="anonymous" sprite-image-width="{sprite_width}" sprite-image-height="{sprite_height}" id="elem"></facets-dive>
           <script>document.querySelector("#elem").data = {jsonstr};</script>"""




with open('facets_static.html', 'w') as out_file:
    out_file.write(html)




get_ipython().system('(jupyter notebook list | grep http | awk \'{printf $1}\'; printf "files/facets_static.html") | sed "s/http:\\/\\/localhost:8888/https:\\/\\/www\\.kaggleusercontent\\.com/"')




get_ipython().system('rm -rf facets/')
get_ipython().system('rm -rf draw_crops/')
get_ipython().system('rm im_paths.csv')
get_ipython().system('rm mani')

