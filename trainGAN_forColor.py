from torchvision import models
from torchsummary import summary

import os
import glob # 文件搜索
import time # 時間相關函數
import numpy as np # 運算相關
from PIL import Image # 圖片處理
from pathlib import Path
from tqdm import tqdm # 進度條
import matplotlib.pyplot as plt # 繪製結果
from skimage.color import rgb2lab, lab2rgb # 通道轉換

import torch
from torch import nn, optim
from torchvision import transforms # 可視化工具
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader # 資料預處理
# GPU可用的話使用GPU
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)
print(device)

# 設定資料路徑(這邊是256*256的漫畫)
path = r"D:\DataSet\color" # 彩色圖片
path2 = r"D:\DataSet\black" # 黑白圖片
path3 = r"D:\DataSet\sketch_9000up_cv2" # 線條圖片
### 資料預處理部分 ###
# 抓取資料路徑下的所有PNG圖片路徑
paths = glob.glob(path + "/*.jpg") # 取得資料夾底下所有.jpg圖片的名稱
paths2 = glob.glob(path2 + "/*.jpg") # 取得資料夾底下所有.jpg圖片的名稱
paths3 = glob.glob(path3 + "/*.jpg") # 取得資料夾底下所有.jpg圖片的名稱
# 設置seed，每一次取出都相同
np.random.seed(123)
# 從 path中隨機取1000個做為子集合
paths_subset = np.array(paths)[:10000]
paths_subset2 = np.array(paths2)[:10000]
paths_subset3 = np.array(paths3)[:10000]
#paths_subset = np.random.choice(paths, 100, replace=False)  # choosing 1000 images randomly
#paths_subset2 = np.random.choice(paths2, 100, replace=False)  # choosing 1000 images randomly

# 隨機打散100張圖片
rand_idxs = np.random.permutation(10000)
train_idxs = rand_idxs[:8000]  # 選擇前X張圖片當作訓練集
val_idxs = rand_idxs[8000:]  # 選擇後Y張圖片當作測試集
'''for i in range(len(val_idxs)):
    print(val_idxs[i])'''
train_paths = paths_subset[train_idxs] # 存所有訓練集的路徑
train_paths2 = paths_subset2[train_idxs] # 存所有訓練集的路徑
train_paths3 = paths_subset3[train_idxs] # 存所有訓練集的路徑
val_paths = paths_subset[val_idxs] # 存所有測試集的路徑
val_paths2 = paths_subset2[val_idxs] # 存所有測試集的路徑
val_paths3 = paths_subset3[val_idxs] # 存所有測試集的路徑
print(len(train_paths), len(val_paths))
print(len(train_paths2), len(val_paths2))
print(len(train_paths3), len(val_paths3))
# 顯示圖片測試用
_, axes = plt.subplots(4, 4, figsize=(10, 10))
for ax, img_path in zip(axes.flatten(), train_paths):
    ax.imshow(Image.open(img_path))
    ax.axis("off")

# 設定處理時的圖片大小為256X256
SIZE = 256

class ColorizationDataset(Dataset):
    # paths：目標圖片，paths2：輸入圖片，split：判斷是train或val，cuda：是否使用GPU訓練，size：圖片大小
    def __init__(self, paths, paths2, split='train', cuda=False, size=SIZE):
        self.cuda = cuda
        self.split = split
        self.size = size
        self.paths = paths
        self.paths2 = paths2
        if split == 'train':
            # 進行數據增強，使用雙三次插值的方式，將圖片縮放成256X256
            self.transforms = transforms.Compose([
                transforms.Resize((SIZE, SIZE), Image.BICUBIC),
                #transforms.RandomHorizontalFlip(),  # A little data augmentation!
            ])
        elif split == 'val':
            # 進行數據增強，使用雙三次插值的方式，將圖片縮放成256X256
            self.transforms = transforms.Resize((SIZE, SIZE), Image.BICUBIC)
    # 返回數據，obj[index]相當於obj.__getitem__(index)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB") # 使用RGB模式開啟指定路徑的圖片
        img2 = Image.open(self.paths2[idx]).convert("RGB") # 使用RGB模式開啟指定路徑的圖片

        img = self.transforms(img) # 進行數據增強
        img2 = self.transforms(img2) # 進行數據增強
        img = np.array(img) # 轉換成ndarray
        img2 = np.array(img2) # 轉換成ndarray
        img_lab = rgb2lab(img).astype("float32") # RGB轉成LAB
        img_lab2 = rgb2lab(img2).astype("float32") # RGB轉成LAB
        img_lab = transforms.ToTensor()(img_lab) # 轉換成Tensor
        img_lab2 = transforms.ToTensor()(img_lab2) # 轉換成Tensor
        #L = img_lab[[0], ...] / 50. - 1.  # Between -1 and 1
        L = img_lab2[[0], ...] / 50. - 1.  # Between -1 and 1，進行正規化
        ab = img_lab[[1, 2], ...] / 110.  # Between -1 and 1，進行正規化
        # 如果可用GPU訓練就搬到GPU上
        if self.cuda:
            L = L.to(device)
            ab = ab.to(device)
        return {'L': L, 'ab': ab}
    # 返回數據的數量，len(obj)相當於obj.__len__().
    def __len__(self):
        return len(self.paths)

# 建立DataLoader
def make_dataloaders(batch_size=8, **kwargs):  # A handy function to make our dataloaders
    dataset = ColorizationDataset(**kwargs, cuda=True) # 建立DataSet
    # batch_size：幾張圖片一個batch，shuffle：是否打亂順序
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=False, shuffle=True)
    return dataloader

train_dl = make_dataloaders(paths=train_paths, paths2=train_paths2, split='train') # 建立訓練集的DataLoader
val_dl = make_dataloaders(paths=val_paths, paths2=val_paths2, split='val') # 建立測試集的DataLoader

data = next(iter(train_dl)) # 從第一筆資料開始迭代
Ls, abs_ = data['L'], data['ab']
print(Ls.shape, abs_.shape)
print(len(train_dl), len(val_dl))

class UnetBlock(nn.Module):
    def __init__(self, nf, ni, submodule=None, input_c=None, dropout=False,
                 innermost=False, outermost=False):
        super().__init__()
        self.outermost = outermost
        if input_c is None: input_c = nf
        downconv = nn.Conv2d(input_c, ni, kernel_size=4,
                             stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(ni)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(nf)

        if outermost:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4,
                                        stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(ni, nf, kernel_size=4,
                                        stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4,
                                        stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if dropout: up += [nn.Dropout(0.5)]
            model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class Unet(nn.Module):
    def __init__(self, input_c=1, output_c=2, n_down=8, num_filters=64):
        super().__init__()
        unet_block = UnetBlock(num_filters * 8, num_filters * 8, innermost=True)
        for _ in range(n_down - 5):
            unet_block = UnetBlock(num_filters * 8, num_filters * 8, submodule=unet_block, dropout=True)
        out_filters = num_filters * 8
        for _ in range(3):
            unet_block = UnetBlock(out_filters // 2, out_filters, submodule=unet_block)
            out_filters //= 2
        self.model = UnetBlock(output_c, out_filters, input_c=input_c, submodule=unet_block, outermost=True)

    def forward(self, x):
        return self.model(x)

class PatchDiscriminator(nn.Module):
    def __init__(self, input_c, num_filters=64, n_down=3):
        super().__init__()
        model = [self.get_layers(input_c, num_filters, norm=False)]
        model += [self.get_layers(num_filters * 2 ** i, num_filters * 2 ** (i + 1), s=1 if i == (n_down - 1) else 2)
                  for i in range(n_down)]  # the 'if' statement is taking care of not using
        # stride of 2 for the last block in this loop
        model += [self.get_layers(num_filters * 2 ** n_down, 1, s=1, norm=False,
                                  act=False)]  # Make sure to not use normalization or
        # activation for the last layer of the model
        self.model = nn.Sequential(*model)


    def get_layers(self, ni, nf, k=4, s=2, p=1, norm=True,
                   act=True):  # when needing to make some repeatitive blocks of layers,
        layers = [
            nn.Conv2d(ni, nf, k, s, p, bias=not norm)]  # it's always helpful to make a separate method for that purpose
        if norm: layers += [nn.BatchNorm2d(nf)]
        if act: layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


discriminator = PatchDiscriminator(3) # 建立輸入通道為3的PatchDiscriminator
discriminator = discriminator.to(device)

#print(summary(discriminator, (3, 256, 256)))
#print(discriminator)

dummy_input = torch.randn(8, 3, 256, 256) # batch_size, channels, size, size
#dummy_input = torch.randn(8, 3, 512, 512) # batch_size, channels, size, size
out = discriminator(dummy_input)
out.shape

class GANLoss(nn.Module):
    def __init__(self, gan_mode='vanilla', real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()

    def get_labels(self, preds, target_is_real):
        if target_is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
        return labels.expand_as(preds)

    def __call__(self, preds, target_is_real):
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss

def init_weights(net, init='norm', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init == 'norm':
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')

            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)

    net.apply(init_func)
    print(f"model initialized with {init} initialization")
    return net

def init_model(model, device):
    model = init_weights(model)
    model = model.to(device)  # 跟前一行交換
    return model

class MainModel(nn.Module):
    def __init__(self, net_G=None, lr_G=2e-4, lr_D=2e-4,
                 beta1=0.5, beta2=0.999, lambda_L1=100.):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1

        if net_G is None:
            self.net_G = init_model(Unet(input_c=1, output_c=2, n_down=8, num_filters=64), self.device)
        else:
            self.net_G = net_G.to(self.device)
        self.net_D = init_model(PatchDiscriminator(input_c=3, n_down=3, num_filters=64), self.device)
        self.GANcriterion = GANLoss(gan_mode='vanilla').to(self.device)
        self.L1criterion = nn.L1Loss().to(self.device)
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))

    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def setup_input(self, data):
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)

    def forward(self):
        self.fake_color = self.net_G(self.L).to(self.device)

    def backward_D(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image.detach())
        self.loss_D_fake = self.GANcriterion(fake_preds, False)
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.net_D(real_image)
        self.loss_D_real = self.GANcriterion(real_preds, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize(self):
        self.forward()
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()

        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count

def create_loss_meters():
    loss_D_fake = AverageMeter()
    loss_D_real = AverageMeter()
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G = AverageMeter()

    return {'loss_D_fake': loss_D_fake,
            'loss_D_real': loss_D_real,
            'loss_D': loss_D,
            'loss_G_GAN': loss_G_GAN,
            'loss_G_L1': loss_G_L1,
            'loss_G': loss_G}

def update_losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)

def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """

    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)

def visualize(model, data, save=True):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    model.net_G.train()
    fake_color = model.fake_color.detach()
    real_color = model.ab
    L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)
    fig = plt.figure(figsize=(15, 8))
    for i in range(5):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(L[i][0].cpu(), cmap='gray')
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 5)
        ax.imshow(fake_imgs[i])
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 10)
        ax.imshow(real_imgs[i])
        ax.axis("off")
    plt.show()
    if save:
        fig.savefig(f"colorization_{time.time()}.png")

def log_results(loss_meter_dict):
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")

from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet34
from fastai.vision.models.unet import DynamicUnet

def build_res_unet(n_input=1, n_output=2, size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body = create_body(resnet34, pretrained=True, n_in=n_input, cut=-2)
    net_G = DynamicUnet(body, n_output, (size, size)).to(device)
    #print(summary(net_G, (1, 256, 256)))
    return net_G

def pretrain_generator(net_G, train_dl, opt, criterion, epochs):
    for e in range(epochs):
        loss_meter = AverageMeter()
        for data in tqdm(train_dl):
            L, ab = data['L'].to(device), data['ab'].to(device)
            preds = net_G(L)
            loss = criterion(preds, ab)
            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_meter.update(loss.item(), L.size(0))

        print(f"Epoch {e + 1}/{epochs}")
        print(f"L1 Loss: {loss_meter.avg:.5f}")
#'''如果已經有預訓練模型就不用這一段
net_G = build_res_unet(n_input=1, n_output=2, size=256) # 建立Resnet34的生成器
#print(net_G)
opt = optim.Adam(net_G.parameters(), lr=1e-4)
criterion = nn.L1Loss()
pretrain_generator(net_G, train_dl, opt, criterion, 20)
torch.save(net_G.state_dict(), "res34-unet.pt")
#'''
def train_model(model, train_dl, epochs, display_every=200):
    data = next(iter(val_dl)) # getting a batch for visualizing the model output after fixed intrvals
    save_num = 0 # 多少epoch就儲存一次
    save_count = 0 # 已經儲存模型多少次
    for e in range(epochs):
        loss_meter_dict = create_loss_meters() # function returing a dictionary of objects to
        i = 0                                  # log the losses of the complete network
        for data in tqdm(train_dl):
            model.setup_input(data)
            model.optimize()
            update_losses(model, loss_meter_dict, count=data['L'].size(0)) # function updating the log objects
            i += 1
            if i % display_every == 0:
                print(f"\nEpoch {e+1}/{epochs}")
                print(f"Iteration {i}/{len(train_dl)}")
                log_results(loss_meter_dict) # function to print out the losses
                visualize(model, data, save=False) # function displaying the model's outputs
        save_num += 1
        save_count += 1
        if save_num == 10:
            #torch.save(model.state_dict(), 'OnePiece0722_temp.pth')
            #f = open('save_count.txt', 'w')
            #f.write(str(save_count))
            save_num = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net_G = build_res_unet(n_input=1, n_output=2, size=256) # 建立Resnet34的生成器

net_G.load_state_dict(torch.load("res34-unet.pt", map_location=device))
model = MainModel(net_G=net_G)
train_model(model, train_dl, 20)
torch.save(model.state_dict(), 'OnePiece00.pth')