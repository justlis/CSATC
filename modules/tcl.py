import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalCasualLearning(nn.Module):
    def __init__(self,in_channels, n_div=8, inplace=False):
        super(TemporalShift, self).__init__()

        self.fold_div = n_div
        self.inplace = inplace
        if inplace:
            print('=> Using in-place shift...')

        self.in_channels = in_channels
        self.fold = self.in_channels // n_div
        self.causal_shift = nn.Conv1d(
                                    self.in_channels, self.in_channels,
                                    kernel_size=3, padding=0, groups=self.in_channels,dilation=3,
                                    bias=False)
        self.trans = nn.Conv1d(
                                    self.in_channels*2, self.in_channels,
                                    kernel_size=3, padding=1,
                                    bias=False)
       
        self._init_weight(self.causal_shift)



    def _init_weight(self, layer):
        layer.weight.requires_grad = True
        layer.weight.data.zero_()
        fold = self.fold
        layer.weight.data[:fold, 0, 2] = 1  # shift left
        layer.weight.data[fold: 2 * fold, 0, 0] = 1  # shift right
        if 2 * fold < self.in_channels:
            layer.weight.data[2 * fold:, 0, 1] = 1  # fixed

    def forward(self,x):
        n,c, t, h, w = x.size()

        x_shift = x.permute([0, 3, 4, 1, 2])
        x_shift = x_shift.contiguous().view(n*h*w, c, t)

        xz_b  = nn.functional.pad(x_shift, (0, 6)) 
        xz_b = self.action_shift(xz_b)
        
        xz_f =  nn.functional.pad(x_shift, (6,0))
        xz_f = self.action_shift(xz_f)
        x_combined = torch.cat((xz_b, xz_f), dim=1)  # 
        x_combined = self.trans(x_combined)

        x_combined = x_combined.view(n, h, w, c, t)
        x_combined = x_combined.permute([0, 3, 4, 1, 2])


        return x_combined

#x = torch.randn(2,512,20,7,7)
#ts = TemporalCasualLearning(512,8)
#y = tcl(x)
#print(y.shape)
