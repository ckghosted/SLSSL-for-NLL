from __future__ import print_function

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import models.preact_resnet as models

import os
import sys
import time
import argparse
import datetime

from torch.autograd import Variable

import dataloader
import numpy as np
import json

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--lr', default=0.04, type=float, help='learning_rate')
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--lrdecay_epoch', default=100, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--optim_type', default='SGD')
parser.add_argument('--seed', default=7)
parser.add_argument('--gpuid', default=1, type=int)
parser.add_argument('--id', default='ce')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--noise_mode', default='sym', help='sym or asym')
parser.add_argument('--noise_ratio', default=0.5, type=float, help='noise ratio')
parser.add_argument('--data_path', default='./data', type=str, help='path to dataset')
parser.add_argument('--run', default=0, type=int, help='run id (0, 1, 2, 3, or 4) to specify the version of noisy labels')
parser.add_argument('--npy_fname', default='ssl_sym05_run0_M0S4n10rho05w05_full_kl.npy', type=str, help='npy file name of the saved KL statistics (if not found, use the real class transition matrix)')
parser.add_argument('--net_ckpt', default='none', help='model to be transfer from (if not found, train from scratch)')
parser.add_argument('--use_tm', action='store_true', help='use class transition matrix to perform forward loss correction (Patrini 2017)')
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
  
# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    learning_rate = args.lr
    if epoch > args.start_epoch:
        learning_rate=learning_rate/10        
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    print('\n=> %s Training Epoch #%d, LR=%.4f' %(args.id,epoch, learning_rate))
    for batch_idx, (inputs, targets, sample_idx) in enumerate(train_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)               # Forward Propagation
        # print('outputs.size():', outputs.size())
        # print('tm_tensor.size():', tm_tensor.size())
        # if batch_idx == 0:
        #     print('tm_tensor:', tm_tensor)
        outputs_prob = softmax_dim1(outputs)
        outputs_adjust = torch.mm(outputs_prob, tm_tensor)
        outputs_log_prob = torch.log(outputs_adjust)
        loss = nll_loss(outputs_log_prob, targets)
        # print('outputs_adjust.size():', outputs_adjust.size())
        # loss = criterion(outputs_adjust, targets)  # Loss
        loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update

        # train_loss += loss.data[0]
        train_loss += loss.data.item()
        _, predicted = torch.max(outputs_adjust.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        # sys.stdout.write('\r')
        # sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
        #         %(epoch, args.num_epochs, batch_idx+1, (len(train_loader.dataset)//args.batch_size)+1, loss.data[0], 100.*correct/total))
        # sys.stdout.flush()
        # if batch_idx%1000==0:
        #     val(epoch)
        #     net.train()
        if batch_idx%10==0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                %(epoch, args.num_epochs, batch_idx+1, (len(train_loader.dataset)//args.batch_size)+1, loss.data.item(), 100.*correct/total))
        if batch_idx%50==0:
            with torch.no_grad():
                val(epoch)
            net.train()
            
def val(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        # test_loss += loss.data[0]
        test_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    # Save checkpoint when best model
    acc = 100.*correct/total
    # print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, loss.data[0], acc))
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, loss.data.item(), acc))
    # record.write('Validation Acc: %f\n'%acc)
    # record.flush()
    print('Validation Acc: %f\n'%acc)
    if acc > best_acc:
        best_acc = acc
        print('| Saving Best Model ...')
        save_checkpoint({
            'state_dict': net.state_dict(),
        }, save_point) 

def test():
    global test_acc
    test_net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = test_net(inputs)
        loss = criterion(outputs, targets)

        # test_loss += loss.data[0]
        test_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
    acc = 100.*correct/total   
    test_acc = acc
    # record.write('Test Acc: %f\n'%acc)
    print('Test Acc: %f\n'%acc)

if not os.path.exists('./checkpoint'):
    os.mkdir('checkpoint')
# record=open('./checkpoint/'+args.id+'_test.txt','w')
# record.write('learning rate: %f\n'%args.lr)
# record.flush()
print('learning rate: %f\n'%args.lr)

KL_stat_path = './checkpoint/%s' % args.npy_fname
if os.path.exists(KL_stat_path):
    if 'partial' in KL_stat_path:
        kl_from = 'partial'
    elif 'full' in KL_stat_path:
        kl_from = 'full'
    else:
        kl_from = 'other'
else:
    kl_from = 'real'

net_ckpt_path = './checkpoint/%s.pth.tar'%args.net_ckpt
if os.path.exists(net_ckpt_path):
    # '_mlnt', '_wcl', or '_ssl'
    transfer_from = '_' + args.net_ckpt.split('_')[0]
else:
    transfer_from = ''

if args.use_tm:
    used_id = used_id + '_tm'
save_point = './checkpoint/%s%s_%s%s_run%d_%s.pth.tar'%(used_id,
                                                        transfer_from,
                                                        args.noise_mode,
                                                        ''.join(str(args.noise_ratio).split('.')),
                                                        args.run,
                                                        kl_from)
if not args.lr == 0.2:
    save_point = save_point.replace('.pth.tar', '_lr%s.pth.tar' % float_to_filename(args.lr))
if not args.num_epochs == 200:
    save_point = save_point.replace('.pth.tar', '_ep%d.pth.tar' % args.num_epochs)

loader = dataloader.cifar_dataloader(dataset=args.dataset,
                                     noise_ratio=args.noise_ratio,
                                     noise_mode=args.noise_mode,
                                     batch_size=args.batch_size,
                                     num_workers=0,
                                     root_dir=args.data_path,
                                     # log=stats_log,
                                     train_val_split_file='%s/train_val_split.json'%args.data_path,
                                     noise_file='%s/%s%s_run%d.json'%(args.data_path,
                                                                      args.noise_mode,
                                                                      ''.join(str(args.noise_ratio).split('.')),
                                                                      args.run))
train_loader,val_loader,test_loader = loader.run()

best_acc = 0
test_acc = 0
# Model
print('\nModel setup')
print('| Building net')
net = models.PreActResNet32()
test_net = models.PreActResNet32()

net_ckpt_path = './checkpoint/%s.pth.tar'%args.net_ckpt
if os.path.exists(net_ckpt_path):
    print('| load net from %s' % net_ckpt_path)
    net_ckpt_param = torch.load(net_ckpt_path)
    net.load_state_dict(net_ckpt_param['state_dict'])
else:
    print('| %s not found, train net from scratch' % net_ckpt_path)


if use_cuda:
    net.cuda()
    test_net.cuda()
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
softmax_dim1 = nn.Softmax(dim=1)
nll_loss = nn.NLLLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

print('\nTraining model')
print('| Training Epochs = ' + str(args.num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))
print('| Optimizer = ' + str(args.optim_type))

def float_to_filename(num):
    split_result = str(num).split('.')
    if split_result[1] == '0':
        return split_result[0]
    else:
        return ''.join(split_result)

num_class = 10
if args.use_tm:
    if os.path.exists(KL_stat_path):
        print('create estimated transition matrix from the KL statistics in %s' % KL_stat_path)
        tm = np.load('./checkpoint/%s' % args.npy_fname)
        if 'partial' in args.npy_fname:
            print('rescaling diagonal')
            for i in range(tm.shape[0]):
                tm[i][i] = tm[i][i]*(num_class-2)/(num_class-1)
        tm = np.power(tm+1e-10, -20.0)
        row_sums = tm.sum(axis=1, keepdims=True)
        tm = tm / row_sums
    else:
        print('%s does not exists, use real transition matrix (%s%s)' % (KL_stat_path, args.noise_mode, float_to_filename(args.noise_ratio)))
        if args.noise_mode == 'sym':
            tm = np.full((num_class, num_class), fill_value=args.noise_ratio/num_class)
            for i in range(num_class):
                tm[i][i] = tm[i][i] + (1-args.noise_ratio)
        elif args.noise_mode == 'asym' or args.noise_mode == 'unnat':
            if args.noise_mode == 'asym':
                transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8}
            elif args.noise_mode == 'unnat':
                transition = {0:7,1:1,2:2,3:1,4:4,5:5,6:5,7:0,8:2,9:9}
            tm = np.full((num_class, num_class), fill_value=0.0)
            for i in range(num_class):
                tm[i][i] = tm[i][i] + (1-args.noise_ratio)
                tm[i][transition[i]] = tm[i][transition[i]] + args.noise_ratio
        else:
            noise_file = os.path.join(args.data_path, '%s00_run%d.json' % (args.noise_mode, args.run))
            noise_label = json.load(open(noise_file, 'r'))
            targets_train = json.load(open(noise_file.replace('%s00_run%d.json' % (args.noise_mode, args.run), 'sym00_run0.json'), 'r'))
            tm = np.zeros((num_class, num_class))
            for i in range(len(targets_train)):
                tm[targets_train[i]][noise_label[i]] += 1
            row_sums = tm.sum(axis=1, keepdims=True)
            tm = tm / row_sums
else:
    tm = np.eye(num_class)

tm_tensor = torch.Tensor(tm)
tm_tensor = tm_tensor.cuda()
tm_tensor = tm_tensor.detach()
print('tm_tensor:', tm_tensor)

for epoch in range(1, 1+args.num_epochs):
    train(epoch)
    # val(epoch)

print('\nTesting model')
checkpoint = torch.load(save_point)
test_net.load_state_dict(checkpoint['state_dict'])
with torch.no_grad():
    test()

print('* Test results : Acc@1 = %.2f%%' %(test_acc))
# record.write('Test Acc: %.2f\n' %test_acc)
# record.flush()
# record.close()
