import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearnex import patch_sklearn
import shutil

reference_path = r'./qualified_reference_img'
unqualified_reference_path = r'./unqualified_reference_img'
AIGenerated_path = r'./ai_generated'
num_qualified_reference_image = 0
for path in os.listdir(reference_path):
    if os.path.isfile(os.path.join(reference_path, path)):
        num_qualified_reference_image += 1
num_ai_generated_image = 0
for path in os.listdir(AIGenerated_path):
    if os.path.isfile(os.path.join(AIGenerated_path, path)):
        num_ai_generated_image += 1

patch_sklearn()

Categories = ['qualified_reference_img','unqualified_reference_img']
flat_data_arr = []
target_arr = []

for i in Categories:
    print(f'loading... category : {i}')
    path = i
    for img in os.listdir(path):
        img_array = imread(os.path.join(path, img))
        img_resized = resize(img_array, (150, 150, 3))
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(Categories.index(i))
    print(f'loaded category:{i} successfully')
flat_data = np.array(flat_data_arr)
target = np.array(target_arr)


# dataframe
df = pd.DataFrame(flat_data)
df['Target'] = target
df.shape


# input data
x = df.iloc[:, :-1]

# output data
y = df.iloc[:, -1]

# Defining the parameters grid for GridSearchCV
param_grid={'C':[0.1,1,10,100],
            'gamma':[0.0001,0.001,0.1,1],
            'kernel':['rbf']}

# Creating a support vector classifier
svc=svm.SVC(probability=False)

  
# Creating a model using GridSearchCV with the parameters grid
model=GridSearchCV(svc,param_grid)

sc = StandardScaler()
x = sc.fit_transform(x)

# Training the model using the training data
model.fit(x, y)


apple_list = os.listdir(reference_path)
orange_list = os.listdir(unqualified_reference_path)

os.mkdir("qualified_ai_generated_images_svm")

for name in apple_list:
    fname = reference_path + "/" + name
    img = imread(fname)
    img_resize = resize(img, (150, 150, 3))
    l = [img_resize.flatten()]
    if Categories[model.predict(l)[0]] != "unqualified_reference_img":
        # this is a qualified image
        shutil.copy(fname, "qualified_ai_generated_images_svm")

# assume there are 20 AI-generated orange images

idx = 0

for name in orange_list:
    idx += 1
    if idx == 20:
        break
    fname = unqualified_reference_path + "/" + name
    img = imread(fname)
    img_resize = resize(img, (150, 150, 3))
    l = [img_resize.flatten()]
    if Categories[model.predict(l)[0]] == "qualified_reference_img":
        shutil.copy(fname, "qualified_ai_generated_images_svm")