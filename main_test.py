import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import os
import argparse
from models import *
from utils import progress_bar
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import Image
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchmetrics
from skimage import metrics
# import pytorch_msssim
# from pytorch_msssim import ssim

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(epoch, snr, person):
    print('\nEpoch: %d, SNR: %d, Person: %s' % (epoch, snr, person))
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Add noise to the inputs based on the current SNR value
        inputs_noisy = add_noise(inputs, snr, person)
        inputs_noisy = inputs_noisy.to(device)

        optimizer.zero_grad()
        outputs = net(inputs_noisy)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        # acces.append(100.*correct/total)
    
    # wandb.log({"acc": correct/total, "loss": train_loss/(batch_idx+1)})

def add_noise(images, snr, person):
    noise_std = torch.std(images) / (10 ** (snr / 20))
    uniform_noise = torch.rand_like(images)
    
    #Bob
    noise_factor = 1 
    exponential_noise = 1
    if (person == "Eve"):
        noise_factor = 100 
        exponential_noise = -torch.log(1 - uniform_noise) / noise_std
    
    noise = exponential_noise * noise_std * noise_factor
    noisy_images = images + noise
    return noisy_images

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    ssim_score = 0  # Variable to store cumulative SSIM

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            output = outputs.view(100, 3, 32, 32)
            output = output.detach().cpu()
            batch_avg_ssim=0
            for i in range(100):
                org=np.transpose(inputs[i], (1, 2, 0)).numpy()
                denoise=np.transpose(output[i], (1, 2, 0)).numpy()
                batch_avg_ssim+=metrics.structural_similarity(org,denoise,multichannel=True)
            ssim_score += batch_avg_ssim

            # print("Inputs size:", inputs.size())
            # print("Outputs size:", outputs.size())
            # probabilities = F.softmax(outputs, dim=1)
            # print("prob size:", probabilities.size())
            # reshaped_output = probabilities.view(-1, 3, 32, 32)
            # metric = torchmetrics.SSIM(data_range=1.0)
            # ssim_score = metric(inputs, reshaped_output)
            # print("Reshaped size:", reshaped_output.size())
            
            loss = criterion(outputs, targets) 
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # if batch_idx == 0:
            #     print("Example Input Tensor:")
            #     print(inputs[0])
            # if inputs.size() != outputs.size():
            #     print(f"Warning: Dimensions of inputs {inputs.size()} and outputs {outputs.size()} do not match!")


            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | SSIM: %.3f'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total, 100. * batch_avg_ssim / total))

    # Save checkpoint.
    acc = 100.*correct/total
    avg_ssim = 100.*ssim_score/total

    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'ssim': ssim_score,  # Add SSIM to the saved state
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    acces.append(best_acc)
    ssim_values.append(ssim_score)  # You can store SSIM values for further analysis


def tensor_to_image(tensor):
    # Assuming tensor is of shape [batch_size, channels, height, width]
    # Convert the tensor to a PIL Image
    image = transforms.ToPILImage()(tensor.squeeze())
    return image

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
snr = 5
person = "Bob"

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

# Model
print('==> Building model..')
net = DenseNet121()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                    momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# List of accuracy
acces = []
ssim_values = []

for epoch in range(start_epoch, start_epoch+1):
    train(epoch, snr, person)
    test(epoch)
    
    scheduler.step()

file = ('Pytorch-CIFAR10/results/acc_DenseNet121_SNR'+str(snr)+'.csv')
file_ssim = ('Pytorch-CIFAR10/results/ssim_DenseNet121_SNR'+str(snr)+'.csv')
if (person == "Eve"):
    file = ('Pytorch-CIFAR10/results/Eve_acc_DenseNet121_SNR'+str(snr)+'.csv')
    file_ssim = ('Pytorch-CIFAR10/results/Eve_ssim_DenseNet121_SNR'+str(snr)+'.csv')
data = pd.DataFrame(acces)
data.to_csv(file, index=False)
data_ssim = pd.DataFrame(ssim_values)
data_ssim.to_csv(file_ssim, index=False)