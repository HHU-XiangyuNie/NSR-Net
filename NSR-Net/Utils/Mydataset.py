import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class MyDataset(Dataset):
    def __init__(self, data_tensor, target_tensor,  win_size, transforms=None):
        super(MyDataset, self).__init__()
        self.win_size = win_size
        dim = int(win_size/2)
        # self.data_tensor = F.pad(data_tensor, [dim, dim, dim, dim], "constant", value=0)
        self.data_tensor = F.pad(data_tensor, [dim, dim, dim, dim], "reflect")
        self.target_tensor = F.pad(target_tensor, [dim, dim, dim, dim], "constant", value=0)
        self.transforms = transforms
        self.n1, self.n2 = target_tensor.shape
        self.labelnozero = torch.nonzero(target_tensor)
        print()

    def __len__(self):
        return self.labelnozero.shape[0]

    def __getitem__(self, index):
        half_ws = int(self.win_size / 2)
        row_num = self.labelnozero[index][0] + half_ws
        colum_num = self.labelnozero[index][1] + half_ws
        if self.win_size % 2 == 1:
            start_row, end_row, start_cl, end_cl = int(row_num-half_ws), \
                                                   int(row_num+half_ws), \
                                                   int(colum_num-half_ws), \
                                                   int(colum_num+half_ws)
        else:
            start_row, end_row, start_cl, end_cl = int(row_num - half_ws+1), \
                                                   int(row_num + half_ws), \
                                                   int(colum_num - half_ws+1), \
                                                   int(colum_num + half_ws)
        data = self.data_tensor[:, start_row:end_row+1, start_cl:end_cl+1]
        label = self.target_tensor[row_num, colum_num]
        if self.transforms is not None:
            data = self.transforms(data)
        return data, label





