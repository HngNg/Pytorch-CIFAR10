'''Train CIFAR10 with PyTorch.'''
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
from skimage.metrics import structural_similarity as ssim
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import Image

# import wandb
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="DenseNet121_CIFAR10",
    
#     # track hyperparameters and run metadata
#     config={
#     "learning_rate": 0.1,
#     "architecture": "CNN",
#     "dataset": "CIFAR-10",
#     "epochs": 100,
#     }
# )

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Training
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
    ssim_sum = 0 # Sum of SSIM values
    count_ssim = 0 # Count of images for which SSIM is calculated

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs) # focus on this to get SSIM 
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Calculate SSIM for each image in the batch
            for i in range(inputs.size(0)):
                img_gt = inputs[i].cpu().numpy()  # Assuming input images are in the range [0, 1]
                img_pred = outputs[i].argmax(dim=0).cpu().numpy()
                img_gt_resized = resize_image(img_gt, img_pred.shape)
                # SSIM calculation
                ssim_value, _ = ssim(img_gt_resized, img_pred, full=True)
                ssim_sum += ssim_value
                count_ssim += 1

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | SSIM: %.3f'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total, ssim_sum/total))


    # Save checkpoint.
    acc = 100.*correct/total
    avg_ssim = ssim_sum / count_ssim if count_ssim > 0 else 0

    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'ssim': avg_ssim,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    acces.append(best_acc)
    ssimes.append(avg_ssim)

def to_pil_image_32_channels(img):
    # Assuming img is a PyTorch tensor
    img_list = [to_pil_image(img_channel) for img_channel in img]
    img_merged = Image.merge('RGB', img_list)
    return img_merged

def resize_image(img, size):
    img_pil = to_pil_image_32_channels(img)
    img_resized = img_pil.resize(size)
    return to_tensor(img_resized)

# List of SNR
# snr_values = [-5, 0, 5, 10, 15, 20]
# snr_values = [-5, 0, 5]
# snr_values = [10, 15, 20]
snr_values = [5, 20]

people = ["Bob", "Eve"]
for person in people:
    for snr in snr_values:
        best_acc = 0  # best test accuracy
        start_epoch = 0  # start from epoch 0 or last checkpoint epoch

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
        # net = VGG('VGG19')
        # net = ResNet18()
        # net = PreActResNet18()
        # net = GoogLeNet()

        net = DenseNet121()
        
        # net = ResNeXt29_2x64d()
        # net = MobileNet()
        # net = MobileNetV2()
        # net = DPN92()
        # net = ShuffleNetG2()
        # net = SENet18()
        # net = ShuffleNetV2(1)
        # net = EfficientNetB0()
        # net = RegNetX_200MF()
        # net = SimpleDLA()
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
        ssimes = []

        for epoch in range(start_epoch, start_epoch+100):
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
        data_ssim = pd.DataFrame(ssimes)
        data_ssim.to_csv(file_ssim, index=False)

# # Finish the wandb run, necessary in notebooks
# wandb.finish()


