import h5py
import numpy as np
import torch


class SegDataset(torch.utils.data.Dataset):
    def __init__(self, filename, train=True):
        self.h5_file = h5py.File(filename, 'r')
        self.num_subjects = len(set([k.split('_')[1] for k in self.h5_file.keys()]))
        self.num_frames = np.zeros(self.num_subjects)
        for key in self.h5_file.keys():
            info = key.split('_')
            if info[0]=='frame':
                self.num_frames[int(info[1])] = int(info[2]) + 1
        self.train = train
                
    def __len__(self):
        if self.train == True:
            return int(self.num_subjects*0.8)*2
        else:
            return int(self.num_subjects*0.2)

    def __getitem__(self, index):
        if self.train == True:
            index = index % int(self.num_subjects*0.8)
            size = index // int(self.num_subjects*0.8) + 1
            idx_frame = np.random.randint(self.num_frames[index], size=size)
            gamma = np.random.rand(size)
            gamma = gamma / np.sum(gamma)
            idx_label = np.random.randint(3, size=size)
        else:
            index = index + int(self.num_subjects*0.8)
            idx_frame = np.random.randint(self.num_frames[index], size=1)
            gamma = np.array([1])
            idx_label = np.random.randint(3, size=1)
            
        frame, label1, label2 = torch.zeros((1,58,52)), torch.zeros((58,52)), torch.zeros((58,52))
        
        for i in range(gamma.shape[0]):
            frame += self.__getframe__(index, idx_frame[i])*gamma[i]
            label1 += self.__random__(index, idx_frame[i], idx_label[i])*gamma[i]
            label2 += self.__vote__(index, idx_frame[i])*gamma[i]
        
        label1 = (label1>0.5)*1
        label2 = (label2>0.5)*1
        
        return (frame, label1, label2)
    
    def __getframe__(self, index, idx_frame): 
        frame = torch.unsqueeze(torch.tensor(self.h5_file['frame_%04d_%03d' % (index, idx_frame)][()].astype('float32')), dim=0)/255
        return frame
    
    def __random__(self, index, idx_frame, idx_label): 
        label1 = torch.squeeze(torch.tensor(self.h5_file['label_%04d_%03d_%02d' % (index, idx_frame, idx_label)][()].astype('int64')))
        return label1
    
    def __vote__(self, index, idx_frame):
        label2 = torch.squeeze(torch.tensor(self.h5_file['label_%04d_%03d_00' % (index, idx_frame)][()].astype('int64'))) \
                 + torch.squeeze(torch.tensor(self.h5_file['label_%04d_%03d_01' % (index, idx_frame)][()].astype('int64'))) \
                 + torch.squeeze(torch.tensor(self.h5_file['label_%04d_%03d_02' % (index, idx_frame)][()].astype('int64')))
        label2 = (label2>=2)*1     
        return label2



class ClasDataset(torch.utils.data.Dataset):
    def __init__(self, filename, train):
        self.h5_file = h5py.File(filename, 'r')
        self.num_subjects = len(set([k.split('_')[1] for k in self.h5_file.keys()]))
        self.num_frames = np.zeros(self.num_subjects)
        for key in self.h5_file.keys():
            inf = key.split('_')
            if inf[0]=='frame':
                self.num_frames[int(inf[1])] = int(inf[2]) + 1
        self.train = train
                
    def __len__(self):
        if self.train == True:
            return int(self.num_subjects*0.8)*2
        else:
            return int(self.num_subjects*0.2)

    def __getitem__(self, index):
        if self.train == True:
            index = index % int(self.num_subjects*0.8)
            size = index // int(self.num_subjects*0.8) + 1
            idx_frame = np.random.randint(self.num_frames[index], size=size)
            gamma = np.random.rand(size)
            gamma = gamma / np.sum(gamma)
            idx_label = np.random.randint(3, size=size)
        else:
            index = index + int(self.num_subjects*0.8)
            idx_frame = np.random.randint(self.num_frames[index], size=1)
            gamma = np.array([1])
            idx_label = np.random.randint(3, size=1)
            
        frame, label = torch.zeros((1,58,52)), 0
        
        for i in range(gamma.shape[0]):
            frame += self.__getframe__(index, idx_frame[i])*gamma[i]
            label += self.__vote__(index, idx_frame[i])*gamma[i]
        
        label = (label>0.5)*1
        
        return (frame, label)

    def __getframe__(self, index, idx_frame): 
        frame = torch.unsqueeze(torch.tensor(self.h5_file['frame_%04d_%03d' % (index, idx_frame)][()].astype('float32')), dim=0)/255
        return frame
    
    def __vote__(self, index, idx_frame):
        labels = torch.zeros(3)
        for idx_label in range(3):
            label = torch.squeeze(torch.tensor(self.h5_file['label_%04d_%03d_%02d' % (index, idx_frame, idx_label)][()].astype('int64')))
            labels[idx_label] = (torch.sum(label)>0)*1
            
        label = (torch.sum(labels)>=2)*1  
        return label


