import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.models 
from dataLoader import *
from blocks import *
from model import *

case_num = 200
## loss function
def loss_dice(y_pred, y_true, eps=1e-6):
    '''
    y_pred, y_true -> [N, D, H, W]
    '''
    numerator = torch.sum(y_true*y_pred, dim=(1,2)) * 2
    denominator = torch.sum(y_true, dim=(1,2)) + torch.sum(y_pred, dim=(1,2)) + eps
    return torch.mean(1. - (numerator / denominator))

def plot_bland_altman(data1, data2):
    mean = (data1 + data2) / 2
    diff = data1 - data2                  
    md = np.mean(diff)                   
    sd = np.std(diff)            
    plt.figure()
    plt.scatter(mean, diff,  s=5)
    plt.axhline(md, color='black', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    plt.xlabel('mean of two methods')
    plt.ylabel('difference of two methods')
    plt.title('Bland-Altman Plot')
    plt.show() 
    
def calcF1(predict_array, true_array):
    tp = np.sum(np.logical_and(predict_array==1,true_array==1)*1)
    fp = np.sum(np.logical_and(predict_array==1,true_array==0)*1)
    fn= np.sum(np.logical_and(predict_array==0,true_array==1)*1)
    F1 = tp/(tp+1/2*(fp+fn))
    return F1
    
    
def main(EPOCH, threshold):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print('begin segmentation, threshold = %.1f:\n' % threshold)
    val_set = SegDataset('dataset70-200.h5',False)
    
    model_seg_consensus = torch.load('segmentation_voting')
    model_seg_random = torch.load('segmentation_random')
    model_clas = torch.load('classification')
    
    acc_list, acc_b_list, acc_r_list = [], [], []
    if threshold == 0.5:
        loss_list, loss_b_list, loss_r_list = [], [], []
        F1_list, F1_b_list, F1_r_list = [], [], []
    
    seg1_list, seg2_list, result_list = [], [], []
    for epoch in range(EPOCH): 
        val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=int(case_num*0.2), 
        shuffle=True)
           
        model_seg_random.eval()
        model_seg_consensus.eval()
        model_clas.eval()
        for step, (frame, label1, label2) in enumerate(val_loader): 
            segmentation = model_seg_consensus(frame)
            classification = ((torch.sigmoid(model_clas(frame))>threshold)*1).unsqueeze(dim=2)
            result = classification*segmentation
            segmentation_r = model_seg_random(frame)
            
            loss = loss_dice(result, label2)            
            result_array = ((result.cpu().detach().numpy()).flatten()>0.5)*1
            true_array = (label2.cpu().detach().numpy()).flatten()
            F1 = calcF1(result_array, true_array)
            accuracy = np.sum((result_array==true_array)*1)/(true_array.shape)        
            acc_list.append(accuracy[0])
            if threshold == 0.5:
                loss_list.append(loss.item())
                F1_list.append(F1)
            
            loss_b = loss_dice(segmentation, label2)        
            result_b_array = ((segmentation.cpu().detach().numpy()).flatten()>0.5)*1
            F1_b = calcF1(result_b_array, true_array)
            accuracy_b = np.sum((result_b_array==true_array)*1)/(true_array.shape)    
            acc_b_list.append(accuracy_b[0])
            if threshold == 0.5:
                loss_b_list.append(loss_b.item())
                F1_b_list.append(F1_b)
            
            loss_r = loss_dice(segmentation_r, label2)        
            result_r_array = ((segmentation_r.cpu().detach().numpy()).flatten()>0.5)*1
            F1_r = calcF1(result_r_array, true_array)
            accuracy_r = np.sum((result_r_array==true_array)*1)/(true_array.shape)    
            acc_r_list.append(accuracy_r[0])
            if threshold == 0.5:
                loss_r_list.append(loss_r.item())
                F1_r_list.append(F1_r)
            
        print('Trial %d \n without classification: \n  loss: %.5f F1 score : %.5f accuracy: %.5f \n with classification:\n  loss: %.5f F1 score: %.5f accuracy: %.5f' \
              % (epoch+1,loss_b.item(),F1_b,accuracy_b,loss.item(),F1,accuracy))
        
        x = frame[0,0,:,:].cpu().detach().numpy()
        y1 = label1[0,:,:].cpu().detach().numpy()
        y2 = label2[0,:,:].cpu().detach().numpy()
        y1_pre = ((segmentation_r[0,:,:].cpu().detach().numpy())>0.5)*1
        y2_pre = ((segmentation[0,:,:].cpu().detach().numpy())>0.5)*1
        y3_pre = ((result[0,:,:].cpu().detach().numpy())>0.5)*1
        plt.figure(figsize=(15,15))
        plt.subplot(141)
        plt.imshow(x,cmap='gray')
        plt.imshow(y2, cmap='jet', alpha=0.5) 
        plt.title('ground-truth')
        plt.axis('off')
        plt.subplot(142)
        plt.imshow(x,cmap='gray')
        plt.imshow(y1_pre, cmap='jet', alpha=0.5) 
        plt.title('random prediction')
        plt.axis('off')
        plt.subplot(143)
        plt.imshow(x,cmap='gray')
        plt.imshow(y2_pre, cmap='jet', alpha=0.5) 
        plt.axis('off')
        plt.title('consensus prediction')
        plt.subplot(144)
        plt.imshow(x,cmap='gray')
        plt.imshow(y3_pre, cmap='jet', alpha=0.5) 
        plt.axis('off')
        plt.title('consensus prediction with classification')
        plt.show()
        
        seg1 = segmentation_r.cpu().detach().numpy().flatten()
        seg2 = segmentation.cpu().detach().numpy().flatten()
        result = result.cpu().detach().numpy().flatten()
        seg1_list.append(list(seg1))
        seg2_list.append(list(seg2))
        result_list.append(list(result))
        
    
    plot_bland_altman(np.array(seg1_list), np.array(seg2_list))
    plot_bland_altman(np.array(seg2_list), np.array(result_list))
        
    print('\n')
    
    if threshold == 0.5:  
        return np.array(acc_list), np.array(acc_b_list), np.array(acc_r_list), \
           loss_list, loss_b_list, loss_r_list, \
           F1_list, F1_b_list, F1_r_list    
    else:
        return np.array(acc_list), np.array(acc_b_list), np.array(acc_r_list)

def plotting(accuracy, accuracy_b):
    try:
        acc_mean = np.mean(accuracy,axis=1)
        acc_b_mean = np.mean(accuracy_b,axis=1)
    except:
        acc_mean = accuracy
        acc_b_mean = accuracy_b
    plt.figure()
    plt.plot(np.arange(1,10)/10,acc_mean, label='with classification')
    plt.plot(np.arange(1,10)/10,acc_b_mean, label='without classification')
    plt.legend()
    plt.xlabel('threshold')
    plt.ylabel('accuracy')
    plt.title('segmentation accuracy with respect to classification threshold')

if __name__ == '__main__':
    trial_num = 100
    acc_array, acc_b_array, acc_r_array = np.zeros((9,trial_num)), np.zeros((9,trial_num)), np.zeros((9,trial_num))
    for threshold in np.arange(1,10):
        if threshold == 5:
            accuracy, accuracy_b, accuracy_r, loss, loss_b, loss_r, F1, F1_b, F1_r = main(trial_num, threshold/10)
        else:
            accuracy, accuracy_b, accuracy_r = main(trial_num, threshold/10)
        
        acc_array[threshold-1,:] = accuracy
        acc_b_array[threshold-1,:] =  accuracy_b
        acc_r_array[threshold-1,:] =  accuracy_r

    plotting(acc_array, acc_b_array)
    print(np.mean(acc_array,axis=1)[4], sum(loss)/trial_num, sum(F1)/trial_num)
    print(np.mean(acc_b_array,axis=1)[4], sum(loss_b)/trial_num, sum(F1_b)/trial_num)
    print(np.mean(acc_r_array,axis=1)[4], sum(loss_r)/trial_num, sum(F1_r)/trial_num)
    