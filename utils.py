import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import os
import skimage.transform as skiTransf
from progressBar import printProgressBar
import scipy.io as sio
import pdb
import time
from os.path import isfile, join
import statistics
from PIL import Image
from medpy.metric.binary import dc, hd, asd, assd
import scipy.spatial
from PIL import Image
from matplotlib import pyplot as plt 

# from scipy.spatial.distance import directed_hausdorff


labels = {0: 'Background', 1: 'Foreground'}


def computeDSC(pred, gt):
    dscAll = []
    for i_b in range(pred.shape[0]):
        pred_id = pred[i_b, 1, :]
        gt_id = gt[i_b, 0, :]
        dscAll.append(dc(pred_id.cpu().data.numpy(), gt_id.cpu().data.numpy()))

    DSC = np.asarray(dscAll)

    return DSC.mean()


def getImageImageList(imagesFolder):
    if os.path.exists(imagesFolder):
        imageNames = [f for f in os.listdir(imagesFolder) if isfile(join(imagesFolder, f))]

    imageNames.sort()

    return imageNames


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def DicesToDice(Dices):
    sums = Dices.sum(dim=0)
    return (2 * sums[0] + 1e-8) / (sums[1] + 1e-8)


def predToSegmentation(pred):
    Max = pred.max(dim=1, keepdim=True)[0]
    x = pred / Max
    # pdb.set_trace()
    return (x == 1).float()


def getTargetSegmentation(batch):
    # input is 1-channel of values between 0 and 1
    # values are as follows : 0, 0.33333334, 0.6666667 and 0.94117647
    # output is 1 channel of discrete values : 0, 1, 2 and 3

    denom = 0.33333334  # for ACDC this value
    return (batch / denom).round().long().squeeze()


from scipy import ndimage


def inference(net, img_batch, CE_loss, epoch):
    total = len(img_batch)
    net.eval()

    softMax = nn.Softmax().cuda()
    CE_loss = CE_loss.cuda()
    DSC = []
    HD = []
    MSD = []
    losses = []
    for i, data in enumerate(img_batch):

        printProgressBar(i, total, prefix="[Inference] Getting segmentations...", length=30)
        images, labels= data

        images = to_var(images)
        labels = to_var(labels)
        
        net_predictions = net(images)
        s_pred = softMax(net_predictions)
        
        masks = torch.argmax(s_pred, dim=1)
        segmentation_classes = getTargetSegmentation(labels)
        seg_one_hot = F.one_hot(segmentation_classes ,num_classes=4).permute(0, 3, 1, 2).float()
       
        
        CE_loss_value = CE_loss(net_predictions ,segmentation_classes, softmax=True)
        path = os.path.join('./Results/Images/', "model1", str(epoch))
        if not os.path.exists(path):
            os.makedirs(path)

        torchvision.utils.save_image(
            torch.cat([images.data, labels.data, masks.view(labels.shape[0], 1, 256, 256).data / 3.0]),
            os.path.join(path, str(i) + '.png'), padding=0)
        
        with torch.no_grad():
            d = evaluation(s_pred, seg_one_hot)
        DSC.append(d)
        #HD.append(h)
        #MSD.append(m)    
        losses.append(CE_loss_value.cpu().data.numpy())
        

    losses = np.asarray(losses)
    dice_mean = np.mean(DSC, axis=0)
    #hd_mean = np.mean(HD, axis=0)
    #msd_mean = np.mean(MSD, axis=0)
                      
    printProgressBar(total, total, done="[Inference] Segmentation Done DiceLoss {:.4f}, Dice: {}, Dice Mean: {}".format(losses.mean(), dice_mean,np.mean(dice_mean)))
    return losses.mean()

def evaluation(pred,gt):
    DSC = []
    pred = (pred > 0.5).float()
    for c_i in range(1,4):
        pred_c = pred[:,c_i,:,:]
        gt_c = gt[:,c_i,:,:]

        nPred = np.array(pred_c.cpu())
        nGt = np.array(gt_c.cpu())

        DSC.append(dc(nPred,nGt))
        
    return DSC


def evaluationTest(pred,gt):
    DSC = []
    HD = []
    ASD = []
    pred = (pred > 0.5).float()
    for c_i in range(1,4):
        pred_c = pred[c_i,:,:]
        gt_c = gt[c_i,:,:]

        nPred = np.array(pred_c.cpu())
        nGt = np.array(gt_c.cpu())

        DSC.append(dc(nPred,nGt))
        HD.append(0)
        ASD.append(0)

    return DSC,HD,ASD


def inferenceTest(net, img_batch, modelName):
    total = len(img_batch)
    net.eval()

    softMax = nn.Softmax().cuda()
    DSC = []
    HD = []
    ASD = []
   
    for i, data in enumerate(img_batch):

        printProgressBar(i, total, prefix="[Inference] Getting segmentations...", length=30)
        images, labels = data

        images = to_var(images)
        labels = to_var(labels)

        net_predictions = net(images)
       
        pred_y = softMax(net_predictions)
        masks = torch.argmax(pred_y, dim=1)
        segmentation_classes = getTargetSegmentation(labels)
        seg_one_hot = F.one_hot(segmentation_classes ,num_classes=4).permute(2,1,0).float
        path = os.path.join('./Results/Images/', modelName)

        if not os.path.exists(path):
            os.makedirs(path)
        
        torchvision.utils.save_image(
            torch.cat([masks.view(labels.shape[0], 1, 256, 256).data / 3.0]),
            os.path.join(path, str(i) + '.png'), padding=0)
    
    printProgressBar(total, total, done="[Inference] Segmentation Done !")


    
    
def evaluate(pred_path):
    path_GT = './Data/test/GT'
    path_pred = pred_path
    
    GT_names = getImageImageList(path_GT)
    Pred_names = getImageImageList(path_pred)

    GT_names.sort()
    Pred_names.sort()

    numClasses = 4
    DSC = np.zeros((len(Pred_names), numClasses))
    HD = np.zeros((len(Pred_names), numClasses))
    
    pixelClasse = [0 ,85, 170, 255]
    
    for s_i in range(len(Pred_names)):
        path_Subj_GT = path_GT + '/' + GT_names[s_i]
        path_Subj_pred = path_pred + '/' + Pred_names[s_i]
    
        imageDataGT = np.asarray(Image.open(path_Subj_GT))
        imageDataCNN = np.asarray(Image.open(path_Subj_pred))
        imageDataCNN = imageDataCNN[:,:,1]
        
        
        for c_i in range(numClasses):
            label_GT = np.zeros(imageDataGT.shape, dtype=np.int8)
            label_CNN = np.zeros(imageDataCNN.shape, dtype=np.int8)

            idx_GT = np.where(imageDataGT == pixelClasse[c_i])
            label_GT[idx_GT] = 1

            idx_CNN = np.where(imageDataCNN == pixelClasse[c_i])
            label_CNN[idx_CNN] = 1


            DSC[s_i,c_i] = dc(label_GT,label_CNN)
            HD[s_i,c_i] = 0
            
    return [DSC]    

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).float()

