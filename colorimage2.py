import tkinter as tk
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import glob
import time
import numpy as np
from PIL import Image, ImageTk
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from skimage.color import rgb2lab, lab2rgb

import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet34
from fastai.vision.models.unet import DynamicUnet

def build_res_unet(n_input=1, n_output=2, size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body = create_body(resnet34, pretrained=True, n_in=n_input, cut=-2)
    net_G = DynamicUnet(body, n_output, (size, size)).to(device)
    return net_G

if torch.cuda.is_available():
		dev = "cuda:0"
else:
		dev = "cpu"
device = torch.device(dev)


rootpath = str(Path(__file__).parent)

imgindex = 0

SIZE = 256

class ColorizationDataset(Dataset):
	def __init__(self, paths, split='train', cuda=False, size=SIZE):
		self.cuda = cuda
		self.split = split
		self.size = size
		self.paths = paths
		if split == 'train':
				self.transforms = transforms.Compose([
						transforms.Resize((SIZE, SIZE), Image.BICUBIC),
						#transforms.RandomHorizontalFlip(),  # A little data augmentation!
				])
		elif split == 'val':
				self.transforms = transforms.Resize((SIZE, SIZE), Image.BICUBIC)

	def __getitem__(self, idx):
		img = Image.open(self.paths[idx]).convert("RGB")

		img = self.transforms(img)
		img = np.array(img)
		img_lab = rgb2lab(img).astype("float32")  # Converting RGB to L*a*b
		img_lab = transforms.ToTensor()(img_lab)
		#L = img_lab[[0], ...] / 50. - 1.  # Between -1 and 1

		L = img_lab[[0], ...] / 50. - 1.  # Between -1 and 1
		#RL = img_lab[[0], ...] / 50. - 1.  # Between -1 and 1
		ab = img_lab[[1, 2], ...] / 110.  # Between -1 and 1
		if self.cuda:
				L = L.to(device)
				#RL = RL.to(device)
				ab = ab.to(device)
		return {'L': L, 'ab': ab}

	def __len__(self):
		return len(self.paths)

def make_dataloaders(batch_size=32, **kwargs):  # A handy function to make our dataloaders
		dataset = ColorizationDataset(**kwargs, cuda=True)

		dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=False, shuffle=True)
		return dataloader



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

discriminator = PatchDiscriminator(3)
#dummy_input = torch.randn(testsize, 3, 256, 256) # batch_size, channels, size, size
#out = discriminator(dummy_input)

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
	#print(f"model initialized with {init} initialization")
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

def topath( image ) :

	image.save(rootpath + r"\.imagetemp\gray_temp.jpg")
	return rootpath + r"\.imagetemp\gray_temp.jpg"
	
	

net_G = build_res_unet(n_input=1, n_output=2, size=256)
net_G.load_state_dict(torch.load(rootpath + r"\res34-unet.pt", map_location=device))
model2 = MainModel(net_G=net_G)
model2 = model2.to(device)

model2.load_state_dict(torch.load(rootpath + r"\OnePiece0827_temp180.pth"))


def graytocolor(image):

	global imgindex
	path = [topath(image)]
	data = make_dataloaders(batch_size=1, paths=path, split='train')
	data = next(iter(data))
	
	loader = transforms.Compose([transforms.ToTensor()])
	model2.net_G.eval()
	with torch.no_grad():
		model2.setup_input(data)
		model2.forward()
	model2.net_G.train()
	fake_color = model2.fake_color.detach()
	L = model2.L
	fake_imgs = lab_to_rgb(L, fake_color)

	matplotlib.image.imsave(rootpath + r"\.imagetemp\color_temp.jpg", fake_imgs[0])

	img2 = Image.open(rootpath + r"\.imagetemp\color_temp.jpg")

	return img2