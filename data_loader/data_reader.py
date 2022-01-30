import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import cv2
import numpy as np
import os
import torchvision.io as tv

def Reader(img_dir, mask_dir, size = 256, display = False):
    '''
    Reads image and mask from directory. Draws mask on image
    '''
    img = plt.imread(img_dir)
    img =  np.array (img[:, :, 0:3])
    if display:
        plt.subplot(1, 2, 1)
        plt.imshow(img)
    with open(mask_dir, 'r') as f: 
        data = f.read() 
    Bs = BeautifulSoup(data, "html.parser")
    xMax = int (Bs.find('xmax').text)
    xMin = int (Bs.find('xmin').text)
    yMax = int (Bs.find('ymax').text)
    yMin = int (Bs.find('ymin').text)
    image =  cv2.resize(img, (size, size))
    imgBoxed = cv2.rectangle(img, (xMin, yMin), (xMax, yMax), (0 ,255, 0), 2)
    mask = cv2.resize(imgBoxed, (size, size))
    if display:
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.resize(imgBoxed, (size,size)))
    return (image,mask)


def Iterator(img_dir, mask_dir):
    dataset = {'image':[], 'mask':[]}
    images = os.listdir(img_dir)
    masks = os.listdir(mask_dir)
    for i,img_path in enumerate(images):
        image,mask = Reader(img_dir+img_path, mask_dir+masks[i])
        dataset['image'].append(image)
        dataset['mask'].append(mask)
    return dataset

    return perf_counter() - start
def ShowPrediction(predicted_mask, true_mask):
    plt.figure(figsize=(20,10))
    filter = np.array([[-1, -1, -1], [-1, 8.99, -1], [-1, -1, -1]]) 
    imgSharpen = cv2.filter2D(predicted_mask,-1,filter)
    plt.subplot(1,2,1)
    plt.imshow(imgSharpen)
    plt.title('Predicted Mask Position')
    plt.subplot(1,2,2)
    plt.imshow(true_mask)
    plt.title('Actual Mask Position')  


    