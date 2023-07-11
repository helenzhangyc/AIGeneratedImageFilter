from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import shutil
from sklearn.metrics import pairwise_distances_argmin_min


# number of clusters
k = 5
# threshold for distance
threshold = 150

reference_path = r'./qualified_reference_img'
AIGenerated_path = r'./ai_generated'
num_reference_image = 0
for path in os.listdir(reference_path):
    if os.path.isfile(os.path.join(reference_path, path)):
        num_reference_image += 1
num_ai_generated_image = 0
for path in os.listdir(AIGenerated_path):
    if os.path.isfile(os.path.join(AIGenerated_path, path)):
        num_ai_generated_image += 1

def image_feature(direc):
    model = InceptionV3(weights='imagenet', include_top=False)
    features = []
    img_name = []
    for i in tqdm(direc):
        fname='qualified_reference_img'+'/'+i
        print(fname)
        img=image.load_img(fname,target_size=(224,224))
        x = img_to_array(img)
        x=np.expand_dims(x,axis=0)
        x=preprocess_input(x)
        feat=model.predict(x)
        feat=feat.flatten()
        features.append(feat)
        img_name.append(i)
    return features,img_name

img_path=os.listdir("qualified_reference_img")
img_features,img_name=image_feature(img_path)

clusters = KMeans(k, random_state = 40)

clusters.fit(img_features)
m_clusters = clusters.labels_.tolist()

all_data = [i for i in range(num_reference_image)]
tf_matrix = np.random.random((num_reference_image, num_reference_image))


centers = np.array(clusters.cluster_centers_)
closest_data = []
for i in range(k):
    center_vec = centers[i]
    data_idx_within_i_cluster = [idx for idx, clu_num in enumerate(m_clusters) if clu_num == i]

    one_cluster_tf_matrix = np.zeros((len(data_idx_within_i_cluster), centers.shape[1]))
    for row_num, data_idx in enumerate(data_idx_within_i_cluster):
        one_row = img_features[data_idx]
        one_cluster_tf_matrix[row_num] = one_row
    
    tmp = []
    tmp.append(center_vec)
    closest, _ = pairwise_distances_argmin_min(tmp, one_cluster_tf_matrix)
    closest_idx_in_one_cluster_tf_matrix = closest[0]
    closest_data_row_num = data_idx_within_i_cluster[closest_idx_in_one_cluster_tf_matrix]
    data_id = all_data[closest_data_row_num]

    closest_data.append(data_id)

closest_data = list(set(closest_data))

centroid_img_name = []
for item in closest_data:
    centroid_img_name.append(img_name[item])


image_cluster = pd.DataFrame(img_name,columns=['image'])
image_cluster["clusterid"] = clusters.labels_
image_cluster


for i in range(k):
    os.mkdir('cluster' + str(i ))
for i in range(len(image_cluster)):
    id = image_cluster['clusterid'][i]
    shutil.copy(os.path.join('qualified_reference_img', image_cluster['image'][i]), 'cluster' + str(id))

# compare each AI-generated image to the centroids
def image_feature2(direc):
    model = InceptionV3(weights='imagenet', include_top=False)
    features = []
    img_name = []
    for i in tqdm(direc):
        fname='ai_generated'+'/'+i
        print(fname)
        img=image.load_img(fname,target_size=(224,224))
        x = img_to_array(img)
        x=np.expand_dims(x,axis=0)
        x=preprocess_input(x)
        feat=model.predict(x)
        feat=feat.flatten()
        features.append(feat)
        img_name.append(i)
    return features,img_name



ai_img_path=os.listdir("ai_generated")
img_features,img_name=image_feature2(ai_img_path)
model = InceptionV3(weights='imagenet', include_top=False)
centroid_feature = []
for i in range(k):
    imgname = centroid_img_name[i]
    fname = "qualified_reference_img/" + imgname
    img = image.load_img(fname, target_size=(224,224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x = preprocess_input(x)
    feat = model.predict(x)
    feat = feat.flatten()
    centroid_feature.append(feat)

qualified_img = []

for feat in centroid_feature:
    dist = np.linalg.norm(img_features - feat, axis=1)
    qualified_img_idx = np.nonzero(dist < threshold)
    for id in qualified_img_idx[0]:
        qualified_img.append(img_name[id])
qualified_img = list(set(qualified_img))    
os.mkdir('qualified_ai_generated_images')
for name in qualified_img:
    shutil.copy(os.path.join('ai_generated', name), 'qualified_ai_generated_images')

# delete clusters
for i in range(k):
    shutil.rmtree("./cluster" + str(i))