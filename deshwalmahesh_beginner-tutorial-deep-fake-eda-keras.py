#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install mtcnn


# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cv2
from IPython.display import HTML
from base64 import b64encode
from tqdm import tqdm
from skimage.transform import resize
from skimage.metrics import structural_similarity
from keras.layers import Dense,Dropout,Conv2D,Conv3D,LSTM,Embedding,BatchNormalization,Input,LeakyReLU,ELU,GlobalMaxPooling2D,GlobalMaxPooling3D
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from mtcnn.mtcnn import MTCNN
from matplotlib.patches import Rectangle
from tensorflow import random as tf_rnd

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[3]:


INPUT_PATH = '/kaggle/input/deepfake-detection-challenge/'
TEST_PATH = 'test_videos/'
TRAIN_PATH = 'train_sample_videos/'
PAD = 50 # padding for the copped face so that a little bit of around of face can be visible
SIZE = 128
BATCH_SIZE = 32
CASCADE_PATH = cv2.data.haarcascades # with this line, you do not need to download the XML files from web. Later.


# In[4]:


SEED = 13
np.random.seed(SEED) # set random seed to get reproducible results
tf_rnd.set_seed(SEED) # tensor flow randomness remover
plt.style.use('seaborn-whitegrid') # just some fancy un useful stuff


# In[5]:


# iterate through the directory to get all the file names and save them as a DataFrame. No need to pay attention to
train_files = []
ext = []
for _, _, filenames in os.walk(INPUT_PATH+TRAIN_PATH): # iterate within the directory
    for filename in filenames: # get all the files inside directory
        splitted = filename.split('.') # split the files as a . such .exe, .deb, .txt, .csv
        train_files.append(splitted[0]) # first part is name of file
        ext.append(splitted[1]) # second one is extension type

files_df = pd.DataFrame({'filename':train_files, 'type':ext})
files_df.head()


# In[6]:


files_df.shape # 401 files


# In[7]:


files_df['type'].value_counts() # 400 mp4 files and 1 json file


# In[8]:


meta_df = pd.read_json(INPUT_PATH+TRAIN_PATH+'metadata.json') # We have Transpose the Df
meta_df.head()


# In[9]:


meta_df = meta_df.T
meta_df.head()


# In[10]:


meta_df.reset_index(inplace=True) # set the index as 0,1,2....
meta_df.rename(columns={'index':'names'},inplace=True) 
# rename the column which was first index but is currently named as 'index'
meta_df.head()


# In[11]:


meta_df.isna().sum() # 77 original files are missing


# In[12]:


meta_df['label'].value_counts().plot(kind='pie',autopct='%1.1f%%',label='Real Vs Fake')


# In[13]:


class VideoFeatures():
    '''
    Class for working with features related to videos such getting frames, plotting frames, playing videos etc
    '''
    
    def get_properties(self,filepath):
        '''
        returns the properties of a video file
        args:
            filepath: path of the video file
        out:
            num_frames: total number of frames in a video
            frame_rate: frames played per second
        '''
        cap = cv2.VideoCapture(filepath)
        num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        return num_frames, frame_rate
        
    
    def get_frames(self,filepath,first_only=False, show=False):
        '''
        method for getting the frames from a video file
        args: 
            filepath: exact path of the video file
            first_only: whether to detect the first frame only or all of the frames
        out:
            frame: first frame in form of numpy array 
        '''
    
        cap = cv2.VideoCapture(filepath) 
        # captures the video. Think of it as if life is a movie so we ask the method to focus on patricular event
        # that is our video in this case. It will concentrate on the video
        
        
        if not first_only: # whether to get all the frames or not
            all_frames = []
            while(cap.isOpened()): # as long as all the frames have been traversed
                ret, frame = cap.read()
                # capture the frame. Again, if life is a movie, this function acts as camera
                
                if ret==True:
                    all_frames.append(frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'): # break in between by pressing the key given
                        break
                else:
                    break
                    
        else:
            ret, all_frames = cap.read()
            if show:
                plt.imshow(cv2.cvtColor(all_frames, cv2.COLOR_BGR2RGB))
                # plot the image but the cv2 changes thge ordering to Blue,Green,Red than RGB so it converts the 
                # metrices to proper ordering
        
        
        cap.release()
        # release whatever was held by the method for say, resources and the video itself
        return all_frames
        
        
    def play_video(self, filepath):
        '''
        Method that uses the HTML inside Python to put a code in the Kernal so that there is a HTML page
        like code where the supported video can be played
        args:
            filepath: path of the file which you want to play
        '''
        
        video = open(filepath,'rb').read() # read video file
        dec_data = "data:video/mp4;base64," + b64encode(video).decode()
        # decode the video data in form of a sting. Funny! Video is now a string
        
        return HTML("""<video width=350 controls><source src="%s" type="video/mp4"></video>""" % dec_data)
        # embed the string as <video> tag in HTML Kernal so that it can be understood by HTML and can be played 
    
    


# In[14]:


class FrameProcessor():
    '''
    class to process the images such as resizing, changing colors, detect faces from frames etc
    '''

    def __init__(self):
        '''
        Constructor where the data from OpenCV is used directly to find the Faces. 
        '''
        self.face_cascade=cv2.CascadeClassifier(CASCADE_PATH+'haarcascade_frontalface_default.xml')
        # XML file which has code for Frontal Face
        self.eye_cascade=cv2.CascadeClassifier(CASCADE_PATH+'haarcascade_eye.xml')
        # it extracts eyes
        
    
    def detect_face_eye(self,img,scaleFactor=1.3, minNeighbors=5, minSize=(50,50),get_cropped_face=False):
        '''
        Method to detect face and eye from the image
        args:
            img: image in the form of numpy array pixels
            scaleFactor: scale the image in proportion. indicates how much the image size is 
                         reduced at each image scale. A lower value uses a smaller step for downscaling.
            minNeighbors: int, number of Neighbors to select from. You know that the pixels at eyes are correlated 
                            with surrounding with pixels around the eye but not the 1000 pixels away at feet
            minSize: tuple. Smaller the face in the image, it is best to adjust the minSize value lower
            get_zoomed_face: Bin. Wheter to return the zoomed face only rather than the full image
        out:
            image with detected faces
        '''
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # convert image to Grayscale to make better use of resources
        
        faces = self.face_cascade.detectMultiScale(gray,scaleFactor=scaleFactor,
                                                   minNeighbors=minNeighbors,
                                                  minSize=minSize)
        # Return the face rectangle from the image
        
        if get_cropped_face:
            for (x,y,w,h) in faces:
                cropped_img = img[y-PAD:y+h+PAD, x-PAD:x+w+PAD] # slice the array to-from where the face(s) have been found
            return cropped_img
            
        
        for (x,y,w,h) in faces:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
            # draw a rectangle around the face with (0,0,255= Blue) color
        
            eyes = self.eye_cascade.detectMultiScale(gray,minSize=(minSize[0]//2,minSize[1]//2),
                                                     minNeighbors=minNeighbors)
            # eyes will always be inside a front profile. So it will reduce the TruePositive of finding other eyes
            
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),3)
                # draw a rectangle around the eyes with Green color (0,255,0)
        
        return img
        
        
        
    def plot_frame(self,img):
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
    
    def resize_frame(self,frame,res_w=256,preserve_aspect=True,anti_aliasing=True):
        '''
        resize the images according to desired width and height
        param:
            frame: numpy image pixels array
            rew_w: resize width default to 256
            preserve_aspect: preserve the aspect ratio in the frame. If not, the output will be a square matrix
            anti_aliasing: whether to apply or not
        out:
            resized numpy array
        '''
        
        res_h = res_w
        if preserve_aspect: # protect the aspect ratio even after the resizing
            aspect_ratio = frame.shape[0]/frame.shape[1]  # get aspect ratio
            res_h = res_w*aspect_ratio # set resulting height according to ratio
            
        return resize(frame,(res_h,res_w),anti_aliasing=anti_aliasing)
        
    
    def frames_similarity(self,frames,full=True, multichannel=True):
        '''
        Find the similarity between the consecutive frames based on a common scale
        param:
            frames: list of numpy pixel arrays
            full: whether to return full  structural similarity 
            multichannel: Bool. IF the images are Grayscale or RGB
            with_resize: Bool. Default True. whether to resize the frames before finding similarity
        '''
        sim_scores = []
        for i in tqdm(range(1, len(frames))): # tqdm shows a progress bar
            curr_frame = frames[i]
            prev_frame = frames[i-1]

            if curr_frame.shape[0] != prev_frame.shape[0]: 
                # different sizes of same images will be seen as two different images so we have to deal with this
                # so just resize the bigger image as the smaller one
                if  curr_frame.shape[0] > prev_frame.shape[0]:
                    curr_frame = curr_frame[:prev_frame.shape[0], :prev_frame.shape[0], :]
                else:
                    prev_frame = prev_frame[:curr_frame.shape[0], :curr_frame.shape[0], :]


            mean_ssim,_ = structural_similarity(curr_frame, prev_frame, full=full,multichannel=multichannel)
            # get mean similarity scores of the images 
            sim_scores.append(mean_ssim)
        
        return sim_scores
        


# In[15]:


vf =  VideoFeatures() # instantiate both the classes to use later
fp = FrameProcessor()


# In[16]:


vf.get_properties(INPUT_PATH+TRAIN_PATH+'cwrtyzndpx.mp4') # get properties of video


# In[17]:


vf.play_video(INPUT_PATH+TRAIN_PATH+'cwrtyzndpx.mp4') # see the magic of HTML with Python


# In[18]:


img = vf.get_frames(INPUT_PATH+TRAIN_PATH+'cwrtyzndpx.mp4',first_only=True,show=True)
# get first frame from image and display it too


# In[19]:


detected_face = fp.detect_face_eye(img,minNeighbors=5,scaleFactor=1.3,minSize=(50,50))
# detect the faces form the image. Tweak the parameters to get the face if ace is not found
# it is difficult to tweak the parameters for every image and this is one the reasons there is need of MTCNN
fp.plot_frame(detected_face)


# In[20]:


frames = vf.get_frames(INPUT_PATH+TRAIN_PATH+'cwrtyzndpx.mp4') # get all the frames
fp.plot_frame(frames[54]) # plot a random frame from the frames 


# In[21]:


# plt.plot(range(1,len(frames)),fp.frames_similarity(frames))


# In[22]:


zoomed_face = fp.detect_face_eye(frames[13],get_cropped_face=True,) # get cropped image array which has pixels of face
fp.plot_frame(zoomed_face)


# In[23]:


class MTCNNWrapper():
    '''
    Detect and show faces using MTCNN
    '''
    
    def get_face(self,mtcnn_obj,img):
        '''
        method to get face from an image
        args:
            img: image as numpy array
        out:
            rect: coordinates of rectangle(s) for multiple face(s)
        '''
        faces = mtcnn_obj.detect_faces(img)
        # dectect_faces returns a list of dicts of all the faces
        x, y, width, height = faces[0]['box'] 
        # faces return a list of dicts so [0] means first faces out of all the faces
        return faces
    
    
    def show_faces(self,img):
        '''
        Show faces on the original image as red boxes
        args:
            img: image as numpy array
        out: 
            None: plot the original image with faces inside red boxes
        '''
        
        faces = self.get_face(img)   # get the list of faces dict
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # plot the image and next modify the image
        ax = plt.gca() # get the context for drawing boxes
        # Get the current Axes instance on the current figure matching the given keyword args, or create one

        for result in faces: # faces returns a list of dicts of all the faces
            x, y, width, height = result['box'] # get coordinates of each box found
            rect = Rectangle((x, y), width, height, fill=False, color='red') # form rectangle at the given coordinates
            ax.add_patch(rect) # add that box to the axis/ current image
        plt.show() # plot the extra rectangles
        
        
    def get_cropped(self,img,show_only=False):
        '''
        get the cropped image only from detected face
        args:
            img: numpy image array
            show_only: whether to return cropped array or just plot the image. Default False
        out:
            numpy array of cropped image at the face
        '''
        faces = self.get_face(img)
        x, y, width, height = faces[0]['box'] # first face. Will add logic later to find the most significant face
        if show_only:
            plt.imshow(cv2.cvtColor(img[y-PAD:y+height+PAD, x-PAD:x+width+PAD], cv2.COLOR_BGR2RGB))
            return None
        else:
            return img[y-PAD:y+height+PAD, x-PAD:x+width+PAD]


# In[24]:


img = VideoFeatures().get_frames(INPUT_PATH+TRAIN_PATH+'cwrtyzndpx.mp4',first_only=True)
mt_wrapper = MTCNNWrapper()
mt_wrapper.show_faces(img)


# In[25]:


mt_wrapper.get_cropped(img,show_only=True) # show only. it does not return anything


# In[26]:


#os.makedirs('train_1') # run this line of code only once.
import shutil 
# shutil.rmtree('train_1') # just in case you want to delete the directory


# In[27]:


x = '''
mt_wrapper = MTCNNWrapper()
frames_names = []
frames_labels = []
for i in tqdm(range(meta_df.shape[0])):
    filepath = INPUT_PATH+TRAIN_PATH+meta_df['names'][i]
    label = meta_df['label'][i]
    cap = cv2.VideoCapture(filepath)
    framerate = cap.get(5) # 5 means to get the framerarate, 3 means get width, 4 for height and so on
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
    # https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get
    count=0
    while(cap.isOpened()):
        frame_no = cap.get(1) # get frame number of current frame
        ret,frame = cap.read()
        if ret!=True:
            break
        if count%10==0:
            filename ='train_1/' + filepath.split('/')[-1].split('.')[-2]+"_%d.jpg" % count
            #frame = mt_wrapper.get_cropped(img)
            cv2.imwrite(filename, frame)
            frames_names.append(filename)
            frames_labels.append(label)
        count+=1

images_csv = pd.DataFrame({'image_name':frames_names,'label':frames_labels})
images_csv.to_csv('video_faces.csv')
'''


# In[28]:


images_df = pd.read_csv('video_faces.csv')
#images_df[] = images_df['label'].apply(lambda x: x.split('/')[-1])
images_df.drop('Unnamed: 0',axis=1,inplace=True)
images_df['label'] = images_df['label'].apply(lambda x: x.split('/')[-1])
images_df.head()
images_df.to_csv('video_faces.csv')


# In[ ]:




