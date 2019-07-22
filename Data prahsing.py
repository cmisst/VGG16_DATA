import numpy as np
import time
import sys
import os
import random
from skimage import io
import pandas as pd
from matplotlib import pyplot as plt
from shutil import copyfile

import cv2
import tensorflow as tf

base_path = '/Users/junzhejiang/Desktop/data_prahsing'
images_boxable_fname = 'train-images-boxable.csv'
annotations_bbox_fname = 'train-annotations-bbox.csv'
class_descriptions_fname = 'class-descriptions-boxable.csv'

images_boxable = pd.read_csv(os.path.join(base_path, images_boxable_fname))
images_boxable.head()

annotations_bbox = pd.read_csv(os.path.join(base_path, annotations_bbox_fname))
annotations_bbox.head()

class_descriptions = pd.read_csv(os.path.join(base_path, class_descriptions_fname))
class_descriptions.head()

traffic_light = class_descriptions.loc[class_descriptions['Tortoise']=='Traffic light'].index
traffic_sign = class_descriptions.loc[class_descriptions['Tortoise']=='Traffic sign'].index
print('traffic_light info: {0}\ntraffic_sign info: {1}'.format(traffic_light, traffic_sign))

# retrueve the LabelName of 'Car'
traffic_light_in = class_descriptions.iloc[22]
traffic_sign_in = class_descriptions.iloc[94]

# or equally: class_descriptions['/m/011k07'][569]
print(traffic_light_in)
print(traffic_sign_in)

traffic_light_images = annotations_bbox[annotations_bbox["LabelName"]=="/m/015qff"]
traffic_sign_images = annotations_bbox[annotations_bbox["LabelName"]=="/m/01mqdt"]
print("len(traffic_light): {0}\nlen(traffic_sign): {1}".format( len(traffic_light_images), len(traffic_sign_images)))

img_obj = images_boxable[images_boxable['image_name']=='00006bdb1eb5cd74.jpg']
print("image name: %s" % (img_obj['image_name'][1690734]))
print("image url: %s" % (img_obj['image_url'][1690734]))

img = io.imread(img_obj['image_url'][1690734])
height, width, _ = img.shape
print(img.shape)
plt.figure(figsize=(15,10))
plt.subplot(1,2,1)
plt.title('Original Image')
plt.imshow(img)
img_name = img_obj['image_name'][1690734]
img_id = img_name[:16]
bboxs = annotations_bbox[annotations_bbox['ImageID']==img_id]
img_bbox = img.copy()
for index, row in bboxs.iterrows():
    xmin = row['XMin']
    xmax = row['XMax']
    ymin = row['YMin']
    ymax = row['YMax']
    xmin = int(xmin*width)
    xmax = int(xmax*width)
    ymin = int(ymin*height)
    ymax = int(ymax*height)
    label_name = row['LabelName']
    class_series = class_descriptions[class_descriptions['/m/011k07']==label_name]
    class_name = class_series['Tortoise'].values[0]
    cv2.rectangle(img_bbox,(xmin,ymin),(xmax,ymax),(0,255,0),2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_bbox,class_name,(xmin,ymin-10), font, 1,(0,255,0),2)
plt.subplot(1,2,2)
plt.title('Image with Bounding Box')
plt.imshow(img_bbox)
plt.show()

traffic_light = class_descriptions['/m/011k07'][22]
traffic_sign = class_descriptions['/m/011k07'][94]


traffic_light_image_id = np.unique(traffic_light_images['ImageID'])
traffic_sign_image_id = np.unique(traffic_sign_images['ImageID'])


print("#traffic_light images: {0}\n#traffic_sign images: {1}".format( len(traffic_light_image_id), len(traffic_sign_image_id)))

# shuffle the unique images ids and pick the first 1000 ids 
# Shuffle the ids and pick the first 1000 ids
copy_traffic_light_id = traffic_light_image_id.copy()
random.seed(1)
random.shuffle(copy_traffic_light_id)

copy_traffic_sign_id = traffic_sign_image_id.copy()
random.seed(1)
random.shuffle(copy_traffic_sign_id)


n = 1700
subtraffic_light_img_id = [str(i) for i in copy_traffic_light_id[:n]]
subtraffic_sign_img_id = [str(i) for i in copy_traffic_sign_id[:n]]

subtraffic_light_img_url = [images_boxable[images_boxable['image_name']==name+'.jpg'] for name in subtraffic_light_img_id]
subtraffic_sign_img_url = [images_boxable[images_boxable['image_name']==name+'.jpg'] for name in subtraffic_sign_img_id]

subset_base_path = r"/Users/junzhejiang/Desktop/data_prahsing/data_get"

subtraffic_light_pd = pd.DataFrame()
subtraffic_sign_pd = pd.DataFrame()

for i in range(len(subtraffic_light_img_url)):
    subtraffic_light_pd = subtraffic_light_pd.append(subtraffic_light_img_url[i], ignore_index = True)
    subtraffic_sign_pd = subtraffic_sign_pd.append(subtraffic_sign_img_url[i], ignore_index = True)
subtraffic_light_pd.to_csv(os.path.join(subset_base_path, 'subtraffic_light_img_url.csv'))
subtraffic_sign_pd.to_csv(os.path.join(subset_base_path, 'subtraffic_sign_img_url.csv'))

subset_base_path = r"/Users/junzhejiang/Desktop/data_prahsing/data_get"

subtraffic_light_pd = pd.read_csv(os.path.join(subset_base_path, 'subtraffic_light_img_url.csv'))
subtraffic_sign_pd = pd.read_csv(os.path.join(subset_base_path, 'subtraffic_sign_img_url.csv'))


subtraffic_light_img_url = subtraffic_light_pd['image_url'].values
subtraffic_sign_img_url = subtraffic_sign_pd['image_url'].values

urls = [subtraffic_light_img_url, subtraffic_sign_img_url]

# labels of truck, car, person
label_names = ["/m/015qff", "/m/01mqdt"]
classes = ["traffic_light", "traffic_sign"]

traffic_light_df = pd.DataFrame(columns=["FileName", "XMin", "XMax", "YMin", "YMax", "ClassName"])
traffic_sign_df = pd.DataFrame(columns=["FileName", "XMin", "XMax", "YMin", "YMax", "ClassName"])


all_img = []

# Truck-box dataframe
for i in range(len(subtraffic_light_pd)):
    img_name = subtraffic_light_pd["image_name"][i]
    img_id = img_name[:16]
    tmp_df = annotations_bbox[annotations_bbox["ImageID"] == img_id]
    for index, row in tmp_df.iterrows():
        label_name = row["LabelName"]
        if label_name == label_names[0]:
            traffic_light_df = traffic_light_df.append({"FileName": img_name, 
                                            "XMin": row["XMin"], 
                                            "XMax": row["XMax"], 
                                            "YMin": row["YMin"], 
                                            "YMax": row["YMax"], 
                                       "ClassName": classes[0]}, 
                                        ignore_index=True)

# Car-box dataframe
for i in range(len(subtraffic_sign_pd)):
    img_name = subtraffic_sign_pd["image_name"][i]
    img_id = img_name[:16]
    tmp_df = annotations_bbox[annotations_bbox["ImageID"] == img_id]
    for index, row in tmp_df.iterrows():
        label_name = row["LabelName"]
        if label_name == label_names[1]:
            traffic_sign_df = traffic_sign_df.append({"FileName": img_name, 
                                        "XMin": row["XMin"], 
                                        "XMax": row["XMax"], 
                                        "YMin": row["YMin"], 
                                        "YMax": row["YMax"], 
                                   "ClassName": classes[1]}, 
                                    ignore_index=True)

traffic_light_df.to_csv(os.path.join(subset_base_path, 'traffic_light.csv'))
traffic_sign_df.to_csv(os.path.join(subset_base_path, 'traffic_sign.csv'))
