
# coding: utf-8

# In[1]:


# Put these at the top of every notebook, to get automatic reloading and inline plotting
# %reload_ext autoreload
# %autoreload 2
# %matplotlib notebook


# In[2]:


import fastai
from fastai import * 
from fastai.vision import *
from fastai.metrics import accuracy
from fastai.basic_data import *

import pandas as pd
import numpy as np
import os

fastai.version.__version__


# In[3]:


# make sure CUDA is available and enabled
print('CUDA enabled:',torch.cuda.is_available()) 
print('CUDNN enabled:', torch.backends.cudnn.enabled)


# In[4]:


def recreate_directory(directory):
    get_ipython().system('rm -R {directory} 2>nul')
    get_ipython().system('mkdir {directory}')


# # Dataset preprocessing

# In[5]:


current_dir = os.getcwd()
input_path =f'{current_dir}'
train_dir = f"{input_path}/train"
train_labels = f"{input_path}/train.csv"
test_dir = f"{input_path}/test"
model_dir = f'{current_dir}/models'


# In[6]:


# TODO adjust!!!!
# labels_df = pd.read_csv(train_labels)
# print(labels_df.shape)
# labels_df = labels_df.sample(frac=0.01)
# print(labels_df.shape)


# In[7]:


# TODO adjust!!!!
# labels_df.to_csv(workdir_train_labels, index=False)


# In[8]:


# TODO adjust!!!!
# recreate_directory(workdir_train)
# for img in labels_df['Image']:
#     !cp {train_dir}/{img} {workdir_train}/{img}


# ## Train model

# In[9]:


SZ = 224
BS = 1
NUM_WORKERS = 0
SEED=0
arch = models.resnet50


# ### TEST -------

# In[10]:


df = pd.read_csv(train_labels)
df = df[df['Id']!='new_whale']
df = df.sample(frac=0.01).reset_index()

print(df.shape)
print(df.head())


# In[11]:


grouped_df = df.groupby('Id')
grouped_counted = grouped_df.count().sort_values(by=['Image'], ascending=False)
grouped_counted = grouped_counted[(grouped_counted['Image']>5) & (grouped_counted['Image']<1000)]
print(len(grouped_counted))
print(grouped_counted.sum().Image, 'of', len(df))
print(grouped_counted.head())


# In[12]:


valid_pct = 0.2

valid_filenames = pd.DataFrame(columns=df.columns)

for name, group in enumerate(grouped_df):
    sub_df = group[1]
#     if group[0] != 'new_whale' and (len(sub_df)>5):
    sample = sub_df.sample(frac=valid_pct)
    valid_filenames = valid_filenames.append(sample, ignore_index=True)


# In[13]:


valid_filenames.drop(labels=['index'], axis=1, inplace=True, errors='ignore')

print(valid_filenames.shape)
print(valid_filenames.head())


# In[14]:


fn2label = {row[1].Image: row[1].Id for row in df.iterrows()}
path2fn = lambda path: re.search('\w*\.jpg$', path).group(0)


# In[15]:


valid_files = ItemList.from_df(df=valid_filenames, path=train_dir, cols=['Image'])


# In[16]:


test_files = ImageItemList.from_folder(test_dir)


# In[17]:


# TODO label from df?
data = (
    ImageItemList
        .from_df(df, train_dir, cols=['Image'])
        .no_split()
#         .split_by_files(valid_files)
        .label_from_func(lambda path: fn2label[path2fn(path)])
        .add_test(test_files)
        .transform(get_transforms(do_flip=False), size=SZ, resize_method=ResizeMethod.SQUISH)
        .databunch(bs=BS, num_workers=NUM_WORKERS, path=input_path)
        .normalize(imagenet_stats)
)


# In[18]:


# data.show_batch(rows=3, fig_size=(SZ, SZ))


# # Learning rate

# In[19]:


# learn = create_cnn(data, arch, metrics=accuracy, model_dir=f"{work_dir}/model")


# In[20]:


# learn.lr_find()


# In[21]:


# learn.recorder.plot()


# # Precompute

# In[ ]:


learn = create_cnn(data, arch, metrics=accuracy)


# In[ ]:


# learn.save('abc')
# learn.load('/kaggle/input/whales-fast-ai/model2/abc')


# In[ ]:


# modelx_path = '/kaggle/input/whalesmodeltrainstage1/model-train-stage-1'
# !cp /kaggle/working/model2/abc.pth /kaggle/working/abc.pth
# learn2_model_path = f"{work_dir}/model3"
# learn2 = create_cnn(data, arch, metrics=accuracy, model_dir=learn2_model_path)
# learn2.load(f"{learn2_model_path}/abc")


# In[ ]:


# learn.fit_one_cycle(1, 1e-2)


# In[ ]:


# learn few epochs with unfreeze
# learn.unfreeze()


# In[ ]:


# lr_rate = 1e-3
# learn.fit_one_cycle(1, [lr_rate/100, lr_rate/10, lr_rate])


# # Prediction & Summition

# In[ ]:


# classes = learn.data.classes + ["new_whale"]
# print(len(classes))


# In[ ]:


# log_preds,y = learn.TTA()


# In[ ]:


# preds = torch.cat((log_preds, torch.ones_like(log_preds[:, :1])), 1)


# In[ ]:


# preds.shape


# In[ ]:


# submittion_df = pd.DataFrame(columns=["Image", "Id"])


# In[ ]:


# for idx, val in enumerate(os.listdir(test_dir)):
#     class_ids = preds[idx].argsort()[-5:]
#     class_1 = classes[class_ids[0]]
#     class_2 = classes[class_ids[1]]
#     class_3 = classes[class_ids[2]]
#     class_4 = classes[class_ids[3]]
#     class_5 = classes[class_ids[4]]
#     prediction_row = f'{class_1} {class_2} {class_3} {class_4} {class_5}'
#     submittion_df = submittion_df.append({'Image' : val.split(".")[0], 'Id': prediction_row}, ignore_index=True)


# In[ ]:


# print(submittion_df.shape)
# submittion_df.head()


# In[ ]:


# submittion_df.to_csv('submission2.csv', index=False)


# In[ ]:


# print(submission.head())
# print(submission.shape)

