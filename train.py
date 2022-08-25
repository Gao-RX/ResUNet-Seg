import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.models 
from dataLoader import *
from blocks import *
from model import *

case_num = 200
batch_size = 4


def loss_dice(y_pred, y_true, eps=1e-6):
    numerator = torch.sum(y_true*y_pred, dim=(1,2)) * 2
    denominator = torch.sum(y_true, dim=(1,2)) + torch.sum(y_pred, dim=(1,2)) + eps
    return torch.mean(1. - (numerator / denominator))

class StableBCELoss(nn.modules.Module):
    def __init__(self):
        super(StableBCELoss, self).__init__()
    def forward(self, input, target):
        neg_abs = - input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()

def bce_loss(input, target):
    BCELoss = StableBCELoss()
    loss = BCELoss(input, target)
    return loss

def discriminator_loss(logits_real, logits_fake):
    loss = bce_loss(logits_real, torch.ones(batch_size)) + bce_loss(logits_fake, torch.zeros(batch_size))
    return loss

def generator_loss(logits_fake):
    loss = bce_loss(logits_fake, torch.ones(batch_size))
    return loss

def plot_bland_altman(data1, data2):
    mean = (data1 + data2) / 2
    diff = data1 - data2                  
    md = np.mean(diff)                   
    sd = np.std(diff)            
    plt.figure()
    plt.scatter(mean, diff)
    plt.axhline(md, color='black', linestyle='-')
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

def DenseNet121(num_init_features, pretrained: bool = False, progress: bool = True, **kwargs):
    return torchvision.models.densenet._densenet('densenet201', 32, (6, 12, 24, 16), num_init_features, pretrained, progress, **kwargs)
    
def main(task, EPOCH):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if task == 'segmentation':
        print('train segmentation network:')
        train_set = SegDataset('dataset70-200.h5',True)
        val_set = SegDataset('dataset70-200.h5',False)
        
#        model1 = ResUnetPlus_3L([1,64,64,48,32,24,1], p=0.2).to(device)
#        model2 = ResUnetPlus_3L([1,64,64,48,32,24,1], p=0.2).to(device)
        
        model1 = ResUnetPlus_2L([1,32,32,16,8,1], p=0.1).to(device)
        model2 = ResUnetPlus_2L([1,32,32,16,8,1], p=0.1).to(device)
        
        D1 = Discriminator().to(device)
        D2 = Discriminator().to(device)
        
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-4)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-4)
        D_solver1 = torch.optim.Adam(D1.parameters(), lr=1e-4, betas=(0.5,0.999))
        D_solver2 = torch.optim.Adam(D2.parameters(), lr=1e-4, betas=(0.5,0.999))
        
        for epoch in range(EPOCH):
            train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size, 
            shuffle=True)
            
            val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=int(case_num*0.2), 
            shuffle=True)
            
            loss1_list, loss2_list = list(), list()
            model1.train()
            model2.train()
            for step, (frame, label1, label2) in enumerate(train_loader):  
                frame = frame.to(device)
                label1 = label1.to(device)
                label2 = label2.to(device)
                    
                ''' adversarial
                # model 1
                D_solver1.zero_grad()
                predict1 = model1(frame).detach()
                logits_fake = D1(2*(frame-0.5),2*(predict1.view(batch_size, 1, 58, 52)-0.5))
                logits_real = D1(2*(frame-0.5),2*(label1.view(batch_size, 1, 58, 52)-0.5))
                D_error = discriminator_loss(logits_real, logits_fake)
                D_error.backward()
                D_solver1.step()
            
                optimizer1.zero_grad()
                predict1 = model1(frame)
                loss1 = loss_dice(predict1, label1) 
                logits_fake = D1(2*(frame-0.5),2*(predict1.view(batch_size, 1, 58, 52)-0.5))
                G_error = generator_loss(logits_fake)
                loss1 = loss1 + 0.01*G_error
                loss1.backward()
                optimizer1.step()
                loss1 = loss1 - 0.01*G_error

                # model 2
                D_solver2.zero_grad()
                predict2 = model2(frame).detach()
                logits_fake = D2(2*(frame-0.5),2*(predict2.view(batch_size, 1, 58, 52)-0.5))
                logits_real = D2(2*(frame-0.5),2*(label2.view(batch_size, 1, 58, 52)-0.5))
                D_error = discriminator_loss(logits_real, logits_fake)
                D_error.backward()
                D_solver2.step()
                
                optimizer2.zero_grad()
                predict2 = model2(frame)
                loss2 = loss_dice(predict2, label2) 
                logits_fake = D2(2*(frame-0.5),2*(predict2.view(batch_size, 1, 58, 52)-0.5))
                G_error = generator_loss(logits_fake)
                loss2 = loss2 + 0.01*G_error
                loss2.backward()
                optimizer2.step()
                loss2 = loss2 - 0.01*G_error
                '''
                
                # model 1
                optimizer1.zero_grad()
                predict1 = model1(frame)
                loss1 = loss_dice(predict1, label1) 
                loss1.backward()
                optimizer1.step()
                
                # model 2
                optimizer2.zero_grad()
                predict2 = model2(frame)
                loss2 = loss_dice(predict2, label2) 
                loss2.backward()
                optimizer2.step()
                
                loss1 = loss_dice(predict1, label2) 
                loss1_list.append((loss1*label1.size(0)).item())
                loss2_list.append((loss2*label2.size(0)).item())
                
            loss1 = sum(loss1_list)/(case_num*0.8*2)
            loss2 = sum(loss2_list)/(case_num*0.8*2)
            
            model1.eval()
            model2.eval()
            for step, (frame, label1, label2) in enumerate(val_loader):   
                predict1 = model1(frame)
                loss1_val = loss_dice(predict1, label2)
                
                predict2 = model2(frame)
                loss2_val = loss_dice(predict2, label2) 
                
                predict1_array = ((predict1.cpu().detach().numpy()).flatten()>0.5)*1
                predict2_array = ((predict2.cpu().detach().numpy()).flatten()>0.5)*1
                true_array = (label2.cpu().detach().numpy()).flatten()
                
                # metrics
                acc_1 = np.sum((true_array==predict1_array)*1)/(true_array.shape)
                acc_2 = np.sum((true_array==predict2_array)*1)/(true_array.shape)
                
                F1_1 = calcF1(predict1_array, true_array)
                F1_2 = calcF1(predict2_array, true_array)
                
                
            print('Epoch %d \n  train loss1: %.5f train loss2: %.5f \n  val loss1: %.5f val loss2: %.5f \n  val acc_1: %.5f val acc_2: %.5f \n  val F1_1: %.5f val F1_2: %.5f' \
                  % (epoch+1,loss1,loss2,loss1_val.item(),loss2_val.item(),acc_1,acc_2,F1_1,F1_2))
            
#            y1 = label1[0,:,:].cpu().detach().numpy()
#            y2 = label2[0,:,:].cpu().detach().numpy()
#            y1_pre = ((predict1[0,:,:].cpu().detach().numpy())>0.5)*1
#            y2_pre = ((predict2[0,:,:].cpu().detach().numpy())>0.5)*1
#            plt.figure(figsize=(15,15))
#            plt.subplot(131)
#            plt.imshow(y2,cmap='gray')
#            plt.title('ground-truth')
#            plt.subplot(132)
#            plt.imshow(y1_pre,cmap='gray')
#            plt.title('random prediction')
#            plt.subplot(133)
#            plt.imshow(y2_pre,cmap='gray')
#            plt.title('consensus prediction')
#            plt.show()     
        print('\n')   
        
        torch.save(model1, 'segmentation_random')
        torch.save(model2, 'segmentation_voting')
        
        data1 = predict1.cpu().detach().numpy().flatten()
        data2 = predict2.cpu().detach().numpy().flatten()
        plot_bland_altman(data1, data2)
                
    
    else:
        print('train classification network:')
        train_set = ClasDataset('dataset70-200.h5',True)
        val_set = ClasDataset('dataset70-200.h5',False)
        
        init_feature = 64
        model = DenseNet121(num_classes=1, num_init_features=init_feature).to(device)
        model.features[0] = nn.Conv2d(1, init_feature, kernel_size=7, stride=2, padding=3, bias=False)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        
        for epoch in range(EPOCH):
            train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size*4, 
            shuffle=True)
            
            val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=int(case_num*0.2), 
            shuffle=True)
            
            loss_list = list()
            model.train()
            for step, (frame, label) in enumerate(train_loader): 
                frame = frame.to(device)
                label = label.to(device)
                
                optimizer.zero_grad()
                predict = model(frame).squeeze()
                loss = criterion(predict.float(), label.float())
                loss.backward()
                optimizer.step()
                
                loss_list.append((loss*label.size(0)).item())
                
            loss = sum(loss_list)/(case_num*0.8*2)
            
            model.eval()
            for step, (frame, label) in enumerate(val_loader):   
                predict = model(frame).squeeze()
                loss_val = criterion(predict.float(), label.float())
                
            print('Epoch %d \n  train loss: %.5f val loss: %.5f' \
                  % (epoch+1,loss,loss_val.item()))
        print('\n')
        
        torch.save(model, 'classification')


if __name__ == '__main__':
    main('segmentation', 100)
    main('classification', 100)
