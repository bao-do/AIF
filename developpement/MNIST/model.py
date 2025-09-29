#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MNISTNet, self).__init__(*args, **kwargs)
        layers=[]
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(5,5),
                                stride=1, padding='same', padding_mode='zeros')
        self.norm1 = nn.BatchNorm2d(num_features=8)
        
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(5,5),
                                stride=1, padding='same', padding_mode='zeros')
        self.norm2 = nn.BatchNorm2d(num_features=16)
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=2)

        self.fc1 = nn.Linear(7**2 * 16,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,10)

    def forward(self, x):
        if not torch.jit.is_tracing():
            assert x.shape[-1] == 28, "The input should be 28x28 images"
            assert x.shape[-3] == 1, "The input should be black and white image"
            
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x= self.norm2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.fc1(x.flatten(1,-1))
        x=F.relu(x)
        x = self.fc2(x)
        x=F.relu(x)
        x = self.fc3(x)
        x=F.relu(x)

        return x
    
    def get_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 7**2)
        return x


        
        
# %%
