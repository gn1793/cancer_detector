import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)  
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1) 
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)       
        self.gamma = nn.Parameter(torch.zeros(1))  
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Q, K, V 변환 및 행렬 곱을 위한 reshape
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)  
        key = self.key(x).view(batch_size, -1, height * width)                        
        value = self.value(x).view(batch_size, -1, height * width)                   
        
        # Attention 점수 계산
        attention = torch.bmm(query, key)  
        attention = F.softmax(attention, dim=-1)  # 각 위치별 attention 점수 정규화
        
        # Attention &  Value 결합
        out = torch.bmm(value, attention.permute(0, 2, 1))  
        out = out.view(batch_size, channels, height, width)  
        
        # Residual connection
        return self.gamma * out + x
