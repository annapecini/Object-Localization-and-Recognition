"""
@authors: Atakan Serbes
          Ana Pecini
          Endi Merkuri
"""

# -*- coding: utf-8 -*-
import numpy as np
import PIL
from PIL import Image
import cv2
import torch
import torch.nn as nn
import sys
import os
import sklearn
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import glob
import torchvision
from sklearn.metrics import confusion_matrix
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset

class MyCustomDataset(Dataset):
    def __init__(self, a , b ):
        self.inputs = a 
        self.labels = b
    
    def __getitem__(self, index):
        return (self.inputs[index], classesDict[self.labels[index]])
    
    def __len__(self):
        return len(self.inputs)


class Feedforward(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(2048, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 10)
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Identity(nn.Module):
    def _init_(self):
        super()._init_()

    def forward(self, x):
        return x

# input is of type PIL Image
def preprocessImage(image):
    # Read RGB image 
    image_arr = np.asarray(image) # convert to a numpy array
    org_size = image_arr.shape
    max_dim = max(org_size[0], org_size[1])
    min_dim = min(org_size[0], org_size[1])
    
    # Create pads to concatenate
    hpad = np.zeros(( (int)(max_dim), (int)((max_dim - min_dim) /2) , 3))
    vpad = np.zeros(( (int)((max_dim - min_dim) /2 ), (int)(max_dim), 3))
    
    # Part 3: Pad the image
    if max_dim == org_size[0]:
        res_im = np.hstack((hpad, image_arr, hpad))
    else:
        res_im = np.vstack((vpad, image_arr, vpad))
    
    # Resize image
    res_im = cv2.resize(res_im, dsize=(224, 224))
    
#    plt.figure()
#    plt.imshow(np.uint8(res_im))
    
    # Normalize
    res_im = np.asarray(res_im)
    res_im = res_im /255
    subs_vals = np.array([0.485, 0.456, 0.406])
    res_im = res_im - subs_vals
    div_vals =  np.array([0.229, 0.224, 0.225])
    res_im = res_im / div_vals
    
    # Convert to float 32 for the model
    res_im = res_im.astype(np.float32)

#    plt.imsave("padding/withoutnorm/)
#    plt.figure()
#    plt.imshow(res_im)
#    plt.show()
    
    #Feature extraction
    # append a dimension to indicate batch_size, which is one
    res_im = np.reshape(res_im, [1, 224, 224, 3])
    # model accepts input images of size [batch_size, 3, im_height, im_width]
    res_im = np.transpose(res_im, [0, 3, 1, 2])
    # convert the Numpy image to torch.FloatTensor
    res_im = torch.from_numpy(res_im)
    # extract features
    
    feature_vector = model(res_im)
    # convert the features of type torch.FloatTensor to a Numpy array
    # so that you can either work with them within the sklearn environment
    # or save them as .mat files
    feature_vector = feature_vector.detach().numpy().ravel()
    return feature_vector


    
def load_images(folder):
    imgs = []
    for filename in os.listdir(folder):
        if any([filename.endswith('.JPEG')]):
            img = cv2.imread(os.path.join(folder, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            feats_img = preprocessImage(img)
            if feats_img is not None:
                imgs.append(feats_img)
                inputs.append(feats_img)
#            labels.append()
                labels.append(folder.split('/')[2])
    return imgs


def intersection(predicted_box, ground_truth):
    width_max = max(predicted_box[0], ground_truth[0])
    width_min = min(predicted_box[2], ground_truth[2])
    width_diff = width_min - width_max
    
    height_max = max(predicted_box[1], ground_truth[1])
    height_min = min(predicted_box[3], ground_truth[3])
    height_diff = height_min - height_max
    
    area = width_diff * height_diff
    # If there is no intersection return 0
    if area <= 0:
        return 0
    else:
        return area
    
def union(predicted_box, ground_truth):
    predicted_area = (predicted_box[2]-predicted_box[0])*(predicted_box[3]-predicted_box[1])
    ground_truth_area = (ground_truth[2]-ground_truth[0])*(ground_truth[3]-ground_truth[1])
    total_area = predicted_area + ground_truth_area
    union = total_area - intersection(predicted_box, ground_truth)
    return union
    
def iou(predicted_box, ground_truth):
    return intersection(predicted_box, ground_truth)/ union(predicted_box, ground_truth)


sys.path.insert(0, 'data/')

#Pretrained network for feature extraction
model = models.resnet50(pretrained = True)

#Gives out model summary
model.eval()

#Delete last layer of the model
#Set the models last layer to identity
model.fc = Identity()
#%%

# The sample train data, this must be converted to the original directory after all process finished
rootDir = 'data/train/'
model.eval()

sys.path.insert(0, 'data/')
rootDir = "data/train/"

train_data = []
train_label = []
inputs = []
labels = []

classes = ['n01615121', 'n02099601', 'n02123159', 'n02129604', 'n02317335', 'n02391049', 'n02410509', 'n02422699', 'n02481823', 'n02504458']
classesDict = {}
for i,j in enumerate(classes,0):
    classesDict[j] = i

for folder in classes:
    images = load_images('data/train/'+folder)
    train_data.append(images)
    
#Tensor transformation
input2 = []
for j in range(len(inputs)):
    j = torch.from_numpy(inputs[j])
    input2.append(j)
    
label2 = []
for j in range(len(labels)):
    label2.append(classesDict[labels[j]])

#%%
#PART 4 - Training the classifier
    
#Classifier    
classifier = Feedforward(500)

#Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)
trainset = MyCustomDataset(a = input2 , b = labels)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                          shuffle=True, num_workers=2)

# Train the classifier
for epoch in range(30):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = classifier(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 5 == 4:    # print every 5 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 5))
            running_loss = 0.0
   
print('Finished Training')

#Part 5
test_file_names =  glob.glob("data/test/images/*.JPEG")
# Sort the file names
test_file_names = sorted(test_file_names,key=lambda x: int(os.path.splitext(x)[0].split('/')[-1]))

test_data = []
for i in range(len(test_file_names)):
    image = cv2.imread(test_file_names[i])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    test_data.append(image)

#%%
#Part5.1 Extracting Candidate Windows
    
actual_classes = []
actual_boxes = []
with open('data/test/bounding_box.txt') as f:
    lines=f.readlines()
    for line in lines:
        temp = []
        actual_classes.append(line.split(',')[0])
        temp.append(int(line.split(',')[1]))
        temp.append(int(line.split(',')[2]))
        temp.append(int(line.split(',')[3]))
        temp.append(int(line.split(',')[4]))
        actual_boxes.append(temp)

#Faster progress using multithreads
cv2.setUseOptimized(True)
cv2.setNumThreads(8)

classNames = ['eagle', 'dog', 'cat', 'tiger', 'star', 'zebra', 'bison', 'antelope', 'chimpanzee', 'elephant']

pred_classes = []
pred_classes_scores = []
pred_boxes = []
allowed_rects = []
for i in range(100):
    print(str(i))
    ssearch = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ssearch.setBaseImage(test_data[i])
    
#    ssearch.switchToSelectiveSearchQuality()
    ssearch.switchToSelectiveSearchFast()
    
    rects = ssearch.process()
    nb_rects = 1000
    
    maxBox = 0
    maxBoxValue = -10000000
    maxClass = ''
    wimg = test_data[i].copy()
    
    for j in range(len(rects)):
        x, y, w, h = rects[j]
        if (w*h) > 5000:
            allowed_rects.append(rects[j])
            boxFeats = preprocessImage(test_data[i][y:y+h, x:x+w])
            boxPred = classifier(torch.from_numpy(boxFeats))
            
            if max(boxPred).item() > maxBoxValue:
                maxBox = j
                maxBoxValue = max(boxPred).item()
                maxClass = classes[(boxPred == max(boxPred)).nonzero().item()]
                print(maxBox, end=": ")
                print(maxBoxValue)
                
    pred_classes.append(maxClass)
    pred_classes_scores.append(maxBoxValue)
    pred_boxes.append(rects[maxBox])
    print(rects[maxBox])
    x, y, w, h = rects[maxBox]
    # Color the ground truth and prection in different colors
    # GT in green
    # Prediction in magenta
    
    #Prediction
    cv2.rectangle(wimg, (x, y), (x+w, y+h), (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(wimg, classNames[classesDict[maxClass]],
                (int((x+w)/2), int((y+h)/2)), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,0,255),2)
    # Ground Truth
    cv2.rectangle(wimg, (actual_boxes[i][0], actual_boxes[i][1]),
              (actual_boxes[i][2], actual_boxes[i][3]), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(wimg, classNames[classesDict[actual_classes[i]]],
                (actual_boxes[i][0], int(3 * (actual_boxes[i][3]-actual_boxes[i][1] )/ 4)), 
                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0),2)
    plt.imsave('results/'+str(i)+'.png',wimg)
    
#%%
# 6 Evaluating the results
fp = [0 for i in range(10)]
fn = [0 for i in range(10)]
tp = [0 for i in range(10)]
tn = [0 for i in range(10)]
# Compute confusion matrix
conf1 = np.zeros((len(classes),len(classes)))
for i in range(len(actual_classes)):
    conf1[classesDict[pred_classes[i]], classesDict[actual_classes[i]]] += 1

# Compute evaluation metrics
for i in range(10):
    tp[i] = conf1[i][i]
    fp[i] = sum(conf1[i]) - tp[i]
    fn[i] = sum(conf1[:,i]) - tp[i]
    tn[i] = len(actual_classes) - sum(conf1[i]) - sum(conf1[:,i]) + tp[i]
    
    
actual_boxes_arr = [ np.array(actual_boxes[i]) for i in range(100)]
class_locs = {}
pred_boxes1 = [np.array([x,y,x+w,y+h]) for x, y, w, h in pred_boxes]

# Calculate the bounding box overlap ratio for each image
class_loc_i = []
for i in range(100):
    class_loc_i.append(iou(actual_boxes_arr[i],pred_boxes1[i]))
    if (i+1) % 10 == 0:
        class_locs[classNames[classesDict[actual_classes[i]]]] = class_loc_i
        class_loc_i = []

# Compute the precision and recall for each class
precision = [0 for i in range(10)]
recall = [0 for i in range(10)]

for i in range(10):
    precision[i] = tp[i] / sum(conf1[i])
    recall[i] = tp[i] / sum(conf1[:,i])
    
# Compute overall percentage of localization accuracy
test_pics_no = 100
correct_count = 0
for i in range(100):
    if iou(actual_boxes_arr[i], pred_boxes1[i]) >= 0.5:
        correct_count += 1

loc_acc = correct_count / test_pics_no    
loc_accs = {}
for i in class_locs.keys():
    loc_accs[i] = len([j for j in class_locs[i] if j >= 0.5]) / 10

# Compute average localization accuracy for each class
loc_acs_per_class = {}
for i in class_locs.keys():
    loc_acs_per_class[i] = sum(class_locs[i]) / 10