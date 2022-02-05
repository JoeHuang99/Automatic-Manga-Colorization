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
paths = glob.glob(path + "/*.jpg") # Grabbing all the image file names
paths2 = glob.glob(path2 + "/*.jpg") # Grabbing all the image file names
paths3 = glob.glob(path3 + "/*.jpg") # Grabbing all the image file names
# 設置seed，每一次取出都相同
np.random.seed(123)
# 從 path中隨機取1000個做為子集合
paths_subset = np.array(paths)[:6250]
paths_subset2 = np.array(paths2)[:6250]
paths_subset3 = np.array(paths3)[:6250]
#paths_subset = np.random.choice(paths, 100, replace=False)  # choosing 1000 images randomly
#paths_subset2 = np.random.choice(paths2, 100, replace=False)  # choosing 1000 images randomly

rand_idxs = np.random.permutation(6250)
train_idxs = rand_idxs[:5000]  # choosing the first 8000 as training set
val_idxs = rand_idxs[5000:]  # choosing last 2000 as validation set
#print(train_idxs)
train_paths = paths_subset[train_idxs]
train_paths2 = paths_subset2[train_idxs]
train_paths3 = paths_subset3[train_idxs]
val_paths = paths_subset[val_idxs]
val_paths2 = paths_subset2[val_idxs]
val_paths3 = paths_subset3[val_idxs]
print(len(train_paths), len(val_paths))
print(len(train_paths2), len(val_paths2))
print(len(train_paths3), len(val_paths3))

# 顯示圖片
_, axes = plt.subplots(4, 4, figsize=(10, 10))
for ax, img_path in zip(axes.flatten(), train_paths):
    ax.imshow(Image.open(img_path))
    ax.axis("off")

SIZE = 256
#SIZE = 512

class ColorizationDataset(Dataset):
    def __init__(self, paths, paths2, split='train', cuda=False, size=SIZE):
        self.cuda = cuda
        self.split = split
        self.size = size
        self.paths = paths
        self.paths2 = paths2
        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((SIZE, SIZE), Image.BICUBIC),
                #transforms.RandomHorizontalFlip(),  # A little data augmentation!
            ])
        elif split == 'val':
            self.transforms = transforms.Resize((SIZE, SIZE), Image.BICUBIC) # 將圖像轉換成Tensor並且標準化到0-1

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img2 = Image.open(self.paths2[idx]).convert("RGB")

        img = self.transforms(img)
        img2 = self.transforms(img2)
        img = np.array(img)
        img2 = np.array(img2)
        img_lab = rgb2lab(img).astype("float32")  # Converting RGB to L*a*b
        img_lab2 = rgb2lab(img2).astype("float32")  # Converting RGB to L*a*b
        img_lab = transforms.ToTensor()(img_lab)
        img_lab2 = transforms.ToTensor()(img_lab2)
        #L = img_lab[[0], ...] / 50. - 1.  # Between -1 and 1
        L = img_lab2[[0], ...] / 50. - 1.  # Between -1 and 1 # 灰階層
        #ab = img_lab[[1, 2], ...] / 110.  # Between -1 and 1
        ab = img_lab[[0], ...] / 50. - 1.  # Between -1 and 1 # 灰階層
        if self.cuda:
            L = L.to(device)
            ab = ab.to(device)
        return {'L': L, 'ab': ab}

    def __len__(self):
        return len(self.paths)

def make_dataloaders(batch_size=8, **kwargs):  # A handy function to make our dataloaders
    dataset = ColorizationDataset(**kwargs, cuda=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=False, shuffle=True)
    return dataloader

train_dl = make_dataloaders(paths=train_paths2, paths2=train_paths3, split='train')
val_dl = make_dataloaders(paths=val_paths2, paths2=val_paths3, split='val')

data = next(iter(train_dl))
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
                  for i in range(n_down)]  # the 'if' statement is taking care of not using # else 2
        # stride of 2 for the last block in this loop
        model += [self.get_layers(num_filters * 2 ** n_down, 1, s=1, norm=False,
                                  act=False)]  # Make sure to not use normalization or
        # activation for the last layer of the model
        self.model = nn.Sequential(*model)

    def get_layers(self, ni, nf, k=4, s=2, p=1, norm=True,
                   act=True):  # when needing to make some repeatitive blocks of layers, s=2
        layers = [
            nn.Conv2d(ni, nf, k, s, p, bias=not norm)]  # it's always helpful to make a separate method for that purpose
        if norm: layers += [nn.BatchNorm2d(nf)]
        if act: layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

discriminator = PatchDiscriminator(1) # 3
dummy_input = torch.randn(8, 1, 256, 256) # batch_size, channels, size, size
#dummy_input = torch.randn(8, 3, 256, 256) # batch_size, channels, size, size
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
            self.net_G = init_model(Unet(input_c=1, output_c=1, n_down=8, num_filters=64), self.device) # output_c=2
        else:
            self.net_G = net_G.to(self.device)
        self.net_D = init_model(PatchDiscriminator(input_c=2, n_down=3, num_filters=64), self.device) # input_c=3
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
        #print(999)
        #print(fake_image.detach().shape) # torch.Size([8, 2, 256, 256])
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

    L = (L + 1.) * 50. # 原本介於-1到1之間
    ab = (ab + 1.) * 50. # 原本介於-1到1之間 torch.Size([16, 1, 256, 256])
    #Lab = (L + ab) // 2 # torch.Size([16, 1, 256, 256]) 將L層做疊加
    Lab = ab
    Lab = Lab.permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = img
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
    #fig = plt.figure(figsize=(15,8))
    fig = plt.figure()
    for i in range(5):
        ax = plt.subplot(3, 5, i + 1)
        #ax = plt.subplot(3, 5, i + 1)
        ax.imshow(L[i][0].cpu(), cmap='gray')
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 5)
        #ax = plt.subplot(3, 1, i + 1 + 5)
        ax.imshow(fake_imgs[i], cmap='gray')
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 10)
        #ax = plt.subplot(3, 1, i + 1 + 10)
        ax.imshow(real_imgs[i])
        ax.axis("off")
        #ax = plt.subplot()'''
        plt.axis('off')
        plt.imshow(fake_imgs[i], cmap='gray')
        #print(len(fake_imgs))
        #fig.savefig(str(i+1) + ".png")
    plt.show()
    if save:
        fig.savefig(f"colorization_{time.time()}.png")

def log_results(loss_meter_dict):
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")

from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet34
from fastai.vision.models.unet import DynamicUnet

def build_res_unet(n_input=1, n_output=1, size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body = create_body(resnet34, pretrained=True, n_in=n_input, cut=-2)
    net_G = DynamicUnet(body, n_output, (size, size)).to(device)
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

'''如果已經有下載預訓練模型就不用這一段
net_G = build_res_unet(n_input=1, n_output=1, size=256)
print(net_G)
opt = optim.Adam(net_G.parameters(), lr=1e-4)
criterion = nn.L1Loss()
pretrain_generator(net_G, train_dl, opt, criterion, 20)
torch.save(net_G.state_dict(), "res34-unet_forBlack.pt")
# '''

def train_model(model, train_dl, epochs, display_every=10):
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
                #visualize(model, data, save=False) # function displaying the model's outputs
        save_num += 1
        save_count += 1
        if save_num == 10:
            #torch.save(net_G.state_dict(), "res34-unet_forBlack" + str(save_count) + ".pt")
            #torch.save(model.state_dict(), 'OnePiece0726_temp.pth')
            #f = open('save_count.txt', 'w')
            #f.write(str(save_count))
            save_num = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net_G = build_res_unet(n_input=1, n_output=1, size=256)
#net_G.load_state_dict(torch.load("res34-unet_forBlack.pt", map_location=device))

model = MainModel(net_G=net_G)
model.load_state_dict(torch.load(r'D:\PythonProject\MangaUNet\OnePiece0802.pth'))
train_model(model, train_dl, 20)
torch.save(model.state_dict(), 'OnePiece0812.pth')
