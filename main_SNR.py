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
def train(epoch, snr):
    print('\nEpoch: %d, SNR: %d' % (epoch, snr))
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Add noise to the inputs based on the current SNR value
        # snr = np.random.choice(snr_values)
        inputs_noisy = add_noise(inputs, snr)
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

def add_noise(images, snr):
    noise_std = torch.std(images) / (10 ** (snr / 20))
    uniform_noise = torch.rand_like(images)
    # Use for Bob
    noise_factor = 1 
    exponential_noise = 1

    # Use for Eve
    # noise_factor = 100 
    # exponential_noise = -torch.log(1 - uniform_noise) / noise_std
    
    noise = exponential_noise * noise_std * noise_factor
    noisy_images = images + noise
    return noisy_images


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    acces.append(best_acc)
# List of SNR
# snr_values = [-5, 0, 5, 10, 15, 20]
# snr_values = [-5, 0, 5]
# snr_values = [10, 15, 20]
snr_values = [0, 15]

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

    for epoch in range(start_epoch, start_epoch+100):
        train(epoch, snr)
        test(epoch)
        
        scheduler.step()

    file = ('Pytorch-CIFAR10/results/acc_DenseNet121_SNR'+str(snr)+'.csv')
    data = pd.DataFrame(acces)
    data.to_csv(file, index=False)

# # Finish the wandb run, necessary in notebooks
# wandb.finish()


