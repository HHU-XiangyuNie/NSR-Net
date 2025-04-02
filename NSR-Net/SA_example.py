"""
Example of using NSR-Net to classify the SA HSI dataset.
"""
import torch
from Utils.Mydataset import MyDataset
from torch.utils.data import DataLoader
import numpy as np
import scipy.io as scio
from Utils.Funs import  spilt_train, kappa_statistic, color_picture
from  Utils.model import NsrNet as MyModel
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
device = torch.device('cuda:0')


# 01-Parameter settings
select_rate = 0.5/100
batch_size = 32
learning_rate = 1e-4 # 0.000001
epochs = 1000
loss_balance = torch.Tensor([0.001]).to(device)

class_num = 16  #
pca_bands = 30
win_size = 17

# 02-Data Preprocessing
data_hsi = scio.loadmat("./HSI_data/Salinas_corrected.mat")["salinas_corrected"]
data_gt = scio.loadmat("./HSI_data/Salinas_gt.mat")["salinas_gt"]
[n1,n2,bands] = np.shape(data_hsi)
if pca_bands > 0:
    pca_fun = PCA(n_components=pca_bands)
    data_hsi = pca_fun.fit_transform(data_hsi.reshape(n1*n2, bands))
    data_hsi = data_hsi.reshape(n1,n2,pca_bands)
    bands = pca_bands
data_hsi = torch.from_numpy(data_hsi.astype('float32'))
data_hsi = torch.permute(data_hsi, (2,0,1))/ data_hsi.max()
data_gt = torch.from_numpy(data_gt.astype('float32')).squeeze()

# 03-Training set
train_label, train_index = spilt_train(data_gt, rate=select_rate)
traning_data = MyDataset(data_hsi, train_label, win_size)
test_data = MyDataset(data_hsi, data_gt+1, win_size)
train_loader = DataLoader(dataset=traning_data, batch_size=batch_size,   shuffle=True,  num_workers=0, drop_last=True)
test_loader = DataLoader(dataset=test_data,    batch_size=batch_size*8, shuffle=False, num_workers=0)

# 04-Loading the NSR-Net model
net = MyModel(bands, win_size, dictionary_size=100, itive_num=1, classes=class_num).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
loss_function = torch.nn.CrossEntropyLoss().to(device)

# 05-Training phase
print('___________Training_________________')
break_point = 0.
best_loss = 1e10
for epoch in range(epochs):
    net.train()
    correct, total, total_loss = 0., 0., 0.
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        labels = torch.as_tensor(labels-1, dtype=torch.long)
        [Y_res_list, sy_loss_list] = net(images.to(device))

        loss_re = loss_function(Y_res_list, labels.long().to(device))
        loss_I = torch.mean(sy_loss_list)
        loss_all = loss_re + loss_balance*loss_I

        correct += torch.sum(torch.argmax(Y_res_list, dim=1) == labels.long().to(device)).item()
        total += len(labels)
        accuracy = correct / total
        total_loss += loss_all
        loss_all.backward()
        optimizer.step()
    print('F-loss: {}'.format(total_loss.item()))
    if best_loss > total_loss:
        best_loss = total_loss
        weights_save_name = './Weight/data_' + 'sa' + '_epoch=' + str(epoch) + '.pth'
        torch.save(net.state_dict(), weights_save_name)

# 05-Classification phase
net.eval()
net.load_state_dict(torch.load(weights_save_name))
with torch.no_grad():
    pred_label_list = []
    gt_list = []
    for images, labels in test_loader:
        [Y_res_list, sy_loss_list] = net(images.to(device))
        pred_labels = torch.argmax(Y_res_list, dim=1).tolist()
        pred_label_list.append(pred_labels)

last_epoch = np.array(pred_label_list[-1]).flatten()
pred_labels = np.array(pred_label_list[:-1]).flatten()
pred_label_list = np.concatenate((pred_labels, last_epoch)) + 1

# 06-Accuracy evaluation
data_gt[train_index[:,0], train_index[:,1]] = 0
Ground_Truth = data_gt.flatten().detach().cpu().numpy()
no_back_index = np.where(Ground_Truth>0)
Ground_Truth = Ground_Truth[no_back_index]
Pred_labels = pred_label_list[no_back_index]
result = kappa_statistic(Ground_Truth, Pred_labels)
print(result)

# 07-Classification map
map = pred_label_list.reshape(n1,n2)
map_color = color_picture(map, num_class=class_num)
plt.imshow(map_color)
plt.show()


