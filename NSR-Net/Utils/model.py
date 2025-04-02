import torch
import torch.nn as nn
device = torch.device('cuda:0')

def soft_vale(r_, lam_):
    # l_1 norm based
    # implement a soft threshold function y=sign(r)*max(0,abs(r)-lam)
    R = torch.sign(r_) * torch.clamp(torch.abs(r_) - lam_, 0)
    return R

class Block(torch.nn.Module):
    def __init__(self, D_sum, win_size):
        super(Block, self).__init__()
        self.D_snum = D_sum
        self.win_size = win_size
        self.lambda_step = nn.Parameter(torch.Tensor([0.5])).to(device)
        self.soft_thr = nn.Parameter(torch.Tensor([0.01])).to(device)

        self.encoder_conv1 = nn.Conv2d(self.D_snum, 128, 3, padding=1, bias=False)
        self.encoder_conv2 = nn.Conv2d(128, 64, 5, padding=2, bias=False)
        self.decoder_conv1 = nn.Conv2d(64, 128, 5, padding=2, bias=False)
        self.decoder_conv2 = nn.Conv2d(128, self.D_snum, 3, padding=1, bias=False)

        self.active = nn.ReLU()
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=self.D_snum, affine=False)
        self.norm2 = nn.GroupNorm(num_groups=1, num_channels=32, affine=False)

    def forward(self, Y, R0, We, S):
        """
        :param Y:  batch * bands * 1
        :param X0: batch * dnums * 1
        :param D:  bands * dnums
        :return:
        """
        R0 = self.norm1(R0)
        Xk = self.encoder_conv1(R0)
        Xk = self.active(Xk)
        Xk_forward = self.encoder_conv2(Xk)

        Xk = soft_vale(Xk_forward, self.soft_thr)

        Xk = self.decoder_conv1(Xk)
        Xk = self.active(Xk)
        Xk_backword = self.decoder_conv2(Xk)

        Xk_backword = self.active(Xk_backword)
        R1 = S(Xk_backword) + We(Y)

        Xk = self.decoder_conv1(Xk_forward)
        Xk = self.active(Xk)
        Xk_est = self.decoder_conv2(Xk)
        sy_loss = Xk_est - R0
        return [Xk_backword, torch.mean(torch.pow(sy_loss, 2)), R1]


class NsrNet(torch.nn.Module):
    def __init__(self, bands, win_size, dictionary_size, itive_num, classes):
        super(NsrNet, self).__init__()
        self.bands = bands
        self.win_size = win_size
        self.D_snum = dictionary_size
        self.itive_num = itive_num
        self.classes = classes
        layers = []
        for i in range(itive_num):
            layers.append(Block(self.D_snum, self.win_size))
        self.callayes = nn.ModuleList(layers)

        self.We = nn.Conv2d(bands, self.D_snum, 1, 1, padding=0, bias=False)
        self.S = nn.Conv2d(self.D_snum, self.D_snum, 1, 1, padding=0, bias=False)

        self.sub_Dict_list = []
        for i in range(classes):
            self.sub_Dict_list.append(nn.Conv2d(dictionary_size, bands, 1, 1,padding=0, bias=False))
        self.sub_Dict_list = nn.ModuleList(self.sub_Dict_list)

        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=self.D_snum, affine=False)
        self.bn = nn.BatchNorm2d(bands)

    def forward(self, Y):
        Y = self.bn(Y)
        # Parameter initialization
        R0 = self.We(Y)
        for i in range(self.itive_num):
            [X0, sy_loss, R1] = self.callayes[i](Y, R0, self.We, self.S)
            R0 = R1
            if i == 0:
                sy_loss_list = sy_loss.unsqueeze(0)
            else:
                sy_loss_list = torch.cat((sy_loss_list, sy_loss.unsqueeze(0)), dim=0)
        # Reconstruction based on subdictionary
        Y_res_list = []
        for i in range(self.classes):
            Y_res_i = self.sub_Dict_list[i](X0)- Y
            Y_res_i = torch.pow(Y_res_i, 2).flatten(1).mean(1)
            if i == 0:
                Y_res_list = Y_res_i.unsqueeze(-1)
            else:
                Y_res_list = torch.cat((Y_res_list, Y_res_i.unsqueeze(-1)), dim=1)
        return [Y_res_list, sy_loss_list]