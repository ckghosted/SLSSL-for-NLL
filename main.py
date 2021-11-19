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

from collections import OrderedDict
import math
import random
import numpy as np
import re

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 with Noisy Labels')
parser.add_argument('--lr', default=0.2, type=float, help='learning_rate')
parser.add_argument('--fast_lr', default=0.2, type=float, help='meta learning_rate')
parser.add_argument('--num_fast', default=10, type=int, help='number of randomly-perturbed mini-batches') # "... the experiments in this paper are conducted using M = 10, as a trade-off between the training speed and the model’s performance"
parser.add_argument('--perturb_ratio', default=0.5, type=float, help='proportion of randomly-perturbed samples per mini-batch')
parser.add_argument('--num_neighbor', default=10, type=int, help='number of neighbors considered for label transfer')
parser.add_argument('--use_wcl', action='store_true', help='Use weighted consistency loss')
parser.add_argument('--T', default=10.0, type=float, help='Inverse temperature for the weight for WCL (weighted consistency loss)')
parser.add_argument('--init_weight', default=0.5, type=float, help='Default weight for WCL (weighted consistency loss)')
parser.add_argument('--num_ssl', default=10, type=int, help='number of SLSSL-perturbed mini-batches')
parser.add_argument('--num_epochs', default=180, type=int)
parser.add_argument('--lrdecay_epoch', default=180, type=int) # "For each training iteration, we divide the learning rate by 10 after 80 epochs, and train until 120 epochs"
parser.add_argument('--rampup_epoch', default=20, type=int) # "... ramp up eta (meta-learning rate) from 0 to 0.4 during the first 20 epochs"
parser.add_argument('--kl_epoch', default=20, type=int) # Epoch to start to record KL-divergence
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--meta_lr', default=0.4, type=float)
parser.add_argument('--gamma_init', default=0.99, type=float, help='Initial exponential moving average weight for the teacher model')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--id', default='slssl')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--noise_mode', default='sym', help='sym or asym')
parser.add_argument('--noise_ratio', default=0.5, type=float, help='noise ratio')
parser.add_argument('--run', default=0, type=int, help='run id (0, 1, 2, 3, or 4) to specify the version of noisy labels')
parser.add_argument('--data_path', default='./data', type=str, help='path to dataset')
parser.add_argument('--pretrain_ckpt', default='ce_sym05_run0_lr02_ep200')
parser.add_argument('--use_tm', action='store_true', help='Use the estimated class transition matrix to perform forward loss correction (Patrini 2017)')
parser.add_argument('--partial_kl', action='store_true', help='Use partial KL (masked) or full KL (unmasked)')
parser.add_argument('--sharpen', default=20, type=int, help='The power to raise the KL statistics for creating estimated transition matrix')
parser.add_argument('--mentor_ckpt', default='mentor_ckpt')
parser.add_argument('--mentor_ckpt2', default='mentor_ckpt2')
parser.add_argument('--keep_threshold', default=0.5, type=float, help='threshold to determine which samples are filtered out')
args = parser.parse_args()

random.seed(args.seed)
torch.cuda.set_device(args.gpuid)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
use_cuda = torch.cuda.is_available()

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

# Sample filtering
def filtering(keep_threshold):
    keep_idx = []
    print('Sample filtering')
    for batch_idx, (inputs, targets, sample_idx) in enumerate(filtering_loader):
        if use_cuda:
            inputs, targets, sample_idx = inputs.cuda(), targets.cuda(), sample_idx.cuda()
        mentor_outputs = mentor_net(inputs, get_feat=False)
        p_mentor = F.softmax(mentor_outputs, dim=1)
        p_mentor = p_mentor.detach()
        keeps = torch.squeeze(torch.gather(p_mentor, 1, torch.unsqueeze(targets, 1))) > keep_threshold
        keep_idx.extend(sample_idx[keeps].tolist())
    return keep_idx

# Training
def train(epoch, use_mentor=False):
    global init
    net.train()
    tch_net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    learning_rate = args.lr
    if epoch > args.lrdecay_epoch:
        learning_rate=learning_rate/10
        
    if epoch>args.rampup_epoch:
        meta_lr = args.meta_lr
        gamma = 0.999
        tch_r = 0.5
    else:
        u = epoch/args.rampup_epoch
        meta_lr = args.meta_lr*math.exp(-5*(1-u)**2)
        gamma = args.gamma_init
        tch_r = 0.5*math.exp(-5*(1-u)**2)

    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    
    print('\n=> %s Training Epoch #%d, LR=%.6f' %(args.id,epoch, learning_rate))

    tm_tensor = torch.Tensor(tm)
    tm_tensor = tm_tensor.cuda()
    tm_tensor = tm_tensor.detach()
    print('tm_tensor:', tm_tensor)
    
    for batch_idx, (inputs, targets, sample_idx) in enumerate(train_loader):
        # if batch_idx < 5:
        #     print('sample_idx:', sample_idx)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda() 
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs) # Forward Propagation
        outputs_prob = softmax_dim1(outputs)
        outputs_adjust = torch.mm(outputs_prob, tm_tensor)
        outputs_log_prob = torch.log(outputs_adjust)
        class_loss = nll_loss(outputs_log_prob, targets)
        class_loss.backward(retain_graph=True)
        
        if epoch > 2:
            if init:
                init = False
                for param,param_tch in zip(net.parameters(),tch_net.parameters()):
                    param_tch.data.copy_(param.data)
            else:
                for param,param_tch in zip(net.parameters(),tch_net.parameters()):
                    param_tch.data.mul_(gamma).add_((1-gamma), param.data)
            
            _,feats = pretrain_net(inputs,get_feat=True)
            tch_outputs = tch_net(inputs,get_feat=False)
            if use_mentor:
                mentor_outputs = mentor_net(inputs,get_feat=False)
                p_tch = tch_r * F.softmax(tch_outputs, dim=1) + (1 - tch_r) * F.softmax(mentor_outputs, dim=1)
            else:
                p_tch = F.softmax(tch_outputs,dim=1)
            p_tch = p_tch.detach()
            
            # Compute class prototypes
            # if batch_idx < 1:
            #     # Initial prototypes: average the top-k ranked features based on their confidence scores on each class
            #     top_scores = {}
            #     confidence_score, predicted_lb = torch.max(p_tch, dim=1)
            #     print('predicted_lb:', predicted_lb)
            #     print('confidence_score:', confidence_score)
            #     prototypes = torch.cuda.FloatTensor((num_class, top_k)).fill_(0.0)
            
            for i in range(args.num_fast):
                targets_fast = targets.clone()
                randidx = torch.randperm(targets.size(0))
                if args.use_wcl:
                    loss_weights = torch.cuda.FloatTensor(targets.size()).fill_(args.init_weight)
                for n in range(int(targets.size(0)*args.perturb_ratio)):
                    idx = randidx[n]
                    feat = feats[idx]
                    feat.view(1,feat.size(0))
                    feat.data = feat.data.expand(targets.size(0),feat.size(0))
                    dist = torch.sum((feat-feats)**2,dim=1)
                    _, neighbor = torch.topk(dist.data,args.num_neighbor+1,largest=False)
                    targets_fast[idx] = targets[neighbor[random.randint(1,args.num_neighbor)]]
                    if args.use_wcl:
                        neighbor_labels = torch.gather(targets, 0, neighbor)
                        label_histogram = torch.bincount(neighbor_labels)
                        loss_weights[idx] = 2 * args.init_weight * torch.sigmoid(torch.log(torch.true_divide(label_histogram[targets[idx]], label_histogram[targets_fast[idx]])) * args.T)
                    
                fast_loss = criterion(outputs,targets_fast)
                
                grads = torch.autograd.grad(fast_loss, net.parameters(), create_graph=True, retain_graph=True, only_inputs=True)
                # grads = torch.autograd.grad(fast_loss, net.parameters())
                
                # grads_list = list(grads)
                # print('grads_list')
                # print(len(grads_list))
                # for grad in grads_list:
                #     print(grad.shape)
                
                for grad in grads:
                    grad = grad.detach()
                    grad.requires_grad = False
                fast_weights = OrderedDict((name, param - args.fast_lr*grad) for ((name, param), grad) in zip(net.named_parameters(), grads))
                # grads_temp = [grad.detach() for grad in grads]
                # for grad in grads_temp:
                #     grad.requires_grad = False
                # fast_weights = OrderedDict((name, param - args.fast_lr*grad) for ((name, param), grad) in zip(net.named_parameters(), grads_temp))
                
                fast_out = net.forward(inputs,fast_weights)  
                
                logp_fast = F.log_softmax(fast_out,dim=1)
                
                if args.use_wcl:
                    if i == 0:
                        consistent_loss = torch.matmul(torch.mean(consistent_criterion(logp_fast,p_tch), dim=1), loss_weights)
                    else:
                        consistent_loss = consistent_loss + torch.matmul(torch.mean(consistent_criterion(logp_fast,p_tch), dim=1), loss_weights)
                else:
                    if i == 0:
                        consistent_loss = consistent_criterion(logp_fast,p_tch)
                    else:
                        consistent_loss = consistent_loss + consistent_criterion(logp_fast,p_tch)
            if args.num_fast > 0:
                meta_loss = consistent_loss*meta_lr/args.num_fast 
                meta_loss.backward(retain_graph=True)

            ssl_lr = meta_lr
            for i in range(args.num_ssl):
                targets_fast = targets.clone()
                rand_lb_pair = np.random.choice(range(num_class), size=2, replace=True)
                loss_mask = torch.cuda.FloatTensor(num_class).fill_(1.0)
                for idx in rand_lb_pair:
                    loss_mask[idx] = 0.0
                # print('targets:', targets)
                # print('rand_lb_pair:', rand_lb_pair)
                # print('loss_mask:', loss_mask)
                idx0 = [idx for idx in range(targets.size(0)) if targets[idx] == rand_lb_pair[0]]
                # idx1 = [idx for idx in range(targets.size(0)) if targets[idx] == rand_lb_pair[1]]
                # print('idx0:', idx0)
                # print('idx1:', idx1)
                for n in range(targets.size(0)):
                    if n in idx0:
                        targets_fast[n] = rand_lb_pair[1]
                    # elif n in idx1:
                    #     targets_fast[n] = rand_lb_pair[0]
                # print('targets_fast:', targets_fast)
                fast_loss = criterion(outputs,targets_fast)
                grads = torch.autograd.grad(fast_loss, net.parameters(), create_graph=True, retain_graph=True, only_inputs=True)
                for grad in grads:
                    grad = grad.detach()
                    grad.requires_grad = False
                fast_weights = OrderedDict((name, param - args.fast_lr*grad) for ((name, param), grad) in zip(net.named_parameters(), grads))
                fast_out = net.forward(inputs,fast_weights)
                logp_fast = F.log_softmax(fast_out,dim=1)
                
                kl_div_vector = consistent_criterion(logp_fast,p_tch)
                kl_div_masked = torch.matmul(kl_div_vector, loss_mask)
                
                #### choose one from the following two lines (partial KL or full KL?)
                if args.partial_kl:
                    kl_div_reduced = torch.mean(kl_div_masked, dim=0)
                else:
                    kl_div_reduced = torch.mean(kl_div_vector)
                
                # rand_lb_pair_ordered = sorted(rand_lb_pair)
                # rand_lb_pair_tuple = (rand_lb_pair_ordered[0], rand_lb_pair_ordered[1])
                rand_lb_pair_tuple = (rand_lb_pair[0], rand_lb_pair[1])
                if epoch > args.kl_epoch:
                    if rand_lb_pair_tuple in kl_dict.keys():
                        kl_dict[rand_lb_pair_tuple].append(kl_div_reduced.data.item())
                    else:
                        kl_dict[rand_lb_pair_tuple] = [kl_div_reduced.data.item()]
                if i == 0:
                    # ssl_loss = kl_div_reduced
                    ssl_loss = torch.mean(kl_div_masked, dim=0)
                else:
                    # ssl_loss = ssl_loss + kl_div_reduced
                    ssl_loss = ssl_loss + torch.mean(kl_div_masked, dim=0)
            if args.num_ssl > 0:
                meta_loss2 = ssl_loss*ssl_lr/args.num_ssl
                meta_loss2.backward()
            
        optimizer.step() # Optimizer update
        
        # train_loss += class_loss.data[0]
        train_loss += class_loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
        # sys.stdout.write('\r')
        # sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
        #         # %(epoch, args.num_epochs, batch_idx+1, (len(train_loader.dataset)//args.batch_size)+1, class_loss.data[0], 100.*correct/total))
        #         %(epoch, args.num_epochs, batch_idx+1, (len(train_loader.dataset)//args.batch_size)+1, class_loss.data.item(), 100.*correct/total))
        # sys.stdout.flush()
        if batch_idx%10==0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                %(epoch, args.num_epochs, batch_idx+1, (len(train_loader.dataset)//args.batch_size)+1, class_loss.data.item(), 100.*correct/total))
        if batch_idx%50==0:
            with torch.no_grad():
                val(epoch,batch_idx)
                val_tch(epoch,batch_idx)
            net.train()
            tch_net.train()
            
            
def val(epoch,iteration):
    global best
    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        
        # val_loss += loss.data[0]
        val_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
    # Save checkpoint when best model
    acc = 100.*correct/total
    # print("\n| Validation Epoch #%d Batch #%3d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, iteration, loss.data[0], acc))
    print("\n| Validation Epoch #%d Batch #%3d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, iteration, loss.data.item(), acc))
    # record.write('Epoch #%d Batch #%3d  Acc: %.2f' %(epoch,iteration,acc))
    # print('Epoch #%d Batch #%3d  Acc: %.2f' %(epoch,iteration,acc))
    if acc > best:
        best = acc
        print('| Saving Best Model (net)...')
        save_checkpoint({
            'state_dict': net.state_dict(),
            'best_acc': best,
        }, save_point)

def val_tch(epoch,iteration):
    global best
    tch_net.eval()
    val_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = tch_net(inputs)
        loss = criterion(outputs, targets)
        
        # val_loss += loss.data[0]
        val_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
    # Save checkpoint when best model
    acc = 100.*correct/total
    # print("| tch Validation Epoch #%d Batch #%3d\t\t\tLoss: %.4f Acc@1: %.2f%%\n" %(epoch, iteration, loss.data[0], acc))
    print("| tch Validation Epoch #%d Batch #%3d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, iteration, loss.data.item(), acc))
    # record.write(' | tchAcc: %.2f\n' %acc)
    # record.flush()
    # print(' | tchAcc: %.2f\n' %acc)
    if acc > best:
        best = acc
        print('| Saving Best Model (tchnet)...')
        save_checkpoint({
            'state_dict': tch_net.state_dict(),
            'best_acc': best,
        }, save_point)

def test():
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
    test_acc = 100.*correct/total   
    print('* Test results : Acc@1 = %.2f%%' %(test_acc))
    # record.write('\nTest Acc: %f\n'%test_acc)
    # record.flush()
    # print('\nTest Acc: %f\n'%test_acc)
    
def float_to_filename(num):
    split_result = str(num).split('.')
    if split_result[1] == '0':
        return split_result[0]
    else:
        return ''.join(split_result)

# ===============================================
# record=open('./checkpoint/'+args.id+'.txt','w')
# record.write('learning rate: %f\n'%args.lr)
# record.write('batch size: %f\n'%args.batch_size)
# record.write('start iter: %d\n'%args.start_iter)
# record.write('mid iter: %d\n'%args.mid_iter)
# record.flush()
print('learning rate: %f\n'%args.lr)
print('batch size: %d\n'%args.batch_size)
print('number of additional mini-batches: %d\n'%args.num_fast)
print('perturbation ratio: %f\n'%args.perturb_ratio)
print('number of neighbor: %d\n'%args.num_neighbor)
print('ramp-up end epoch of the meta-learning rate: %d\n'%args.rampup_epoch)
print('LR decay epoch: %d\n'%args.lrdecay_epoch)

if os.path.exists('./checkpoint/%s.pth.tar'%args.mentor_ckpt):
    use_mentor = True
    if os.path.exists('./checkpoint/%s.pth.tar'%args.mentor_ckpt2):
        use_mentor2 = True
        keep_thre1 = float(re.search('2nd([0-9]+)', args.mentor_ckpt2).group(1))*0.1
        keep_thre2 = args.keep_threshold
        used_id = args.id + '_2nd%s_3rd%s' % (float_to_filename(keep_thre1), float_to_filename(keep_thre2))
    else:
        use_mentor2 = False
        keep_thre1 = args.keep_threshold
        used_id = args.id + '_2nd%s' % float_to_filename(keep_thre1)
else:
    use_mentor = False
    used_id = args.id
if args.use_tm:
    used_id = used_id + '_tm'
save_point = './checkpoint/%s_%s%s_run%d_M%dS%dn%drho%s.pth.tar'%(used_id,
                                                                  args.noise_mode,
                                                                  float_to_filename(args.noise_ratio),
                                                                  args.run,
                                                                  args.num_fast,
                                                                  args.num_ssl,
                                                                  args.num_neighbor,
                                                                  float_to_filename(args.perturb_ratio))
if args.use_wcl:
    save_point = save_point.replace('.pth.tar', '_w%s.pth.tar' % float_to_filename(args.init_weight))
if not args.lr == 0.2:
    save_point = save_point.replace('.pth.tar', '_lr%s.pth.tar' % float_to_filename(args.lr))
if not args.num_epochs == 180:
    save_point = save_point.replace('.pth.tar', '_ep%d.pth.tar' % args.num_epochs)

best = 0
init = True
# Model
print('\nModel setup')
print('| Building net')
net = models.PreActResNet32()
tch_net = models.PreActResNet32()
pretrain_net = models.PreActResNet32()
test_net = models.PreActResNet32()

print('| load pretrain from ./checkpoint/%s.pth.tar'%args.pretrain_ckpt)
pretrain_ckpt = torch.load('./checkpoint/%s.pth.tar'%args.pretrain_ckpt)
pretrain_net.load_state_dict(pretrain_ckpt['state_dict'])

if use_cuda:
    net.cuda()
    tch_net.cuda()
    pretrain_net.cuda()
    test_net.cuda()
    cudnn.benchmark = True
pretrain_net.eval()

if use_mentor:
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
                                                                          args.run),
                                         sample_filtering=True)
    filtering_loader = loader.run()
    mentor_net = models.PreActResNet32()
    print('| load mentor model from ./checkpoint/%s.pth.tar' % args.mentor_ckpt)
    mentor_ckpt = torch.load('./checkpoint/%s.pth.tar'%args.mentor_ckpt)
    mentor_net.load_state_dict(mentor_ckpt['state_dict'])
    mentor_net.cuda()
    mentor_net.eval()
    for param in mentor_net.parameters():
        param.requires_grad = False
    keep_idx1 = filtering(keep_thre1)
    print('remove %d samples' % len(filtering_loader.dataset)-len(keep_idx1))
    if use_mentor2:
        mentor_net = models.PreActResNet32()
        print('| load mentor model from ./checkpoint/%s.pth.tar' % args.mentor_ckpt2)
        mentor_ckpt = torch.load('./checkpoint/%s.pth.tar'%args.mentor_ckpt2)
        mentor_net.load_state_dict(mentor_ckpt['state_dict'])
        mentor_net.cuda()
        mentor_net.eval()
        for param in mentor_net.parameters():
            param.requires_grad = False
        keep_idx2 = filtering(keep_thre2)
        newly_removed_idx = [i for i in range(filtering_loader.dataset) if ((i in keep_idx1) and (not i in keep_idx2))]
        print('remove %d additional samples' % len(newly_removed_idx))
        keep_idx = [i for i in keep_idx1 if i in keep_idx2]
    else:
        keep_idx = keep_idx1
else:
    print('| no mentor model')
    keep_idx = [] # [Note] empty list: keep all training samples

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
                                                                      args.run),
                                     keep_idx=keep_idx)
train_loader,val_loader,test_loader = loader.run()
print('After sample filtering, len(train_loader.dataset) = %d' % len(train_loader.dataset))

for param in tch_net.parameters():
    param.requires_grad = False
for param in pretrain_net.parameters():
    param.requires_grad = False

criterion = nn.CrossEntropyLoss()
softmax_dim1 = nn.Softmax(dim=1)
nll_loss = nn.NLLLoss()
consistent_criterion = nn.KLDivLoss(reduction='none')
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

print('\nTraining model')
print('| Training Epochs = ' + str(args.num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))

num_class = 10 # [TODO] remove magic number
top_k = 5 # [TODO] remove magic number
tm = np.eye(num_class) # initialize the transition matrix using an identity matrix
for epoch in range(1, 1+args.num_epochs):
    # [TODO] remove magic number
    tm_keep_r = 0.99
    kl_dict = {}
    train(epoch, use_mentor)
    # if epoch%2==0:
    #     print('\nTesting model')
    #     best_model = torch.load(save_point)
    #     test_net.load_state_dict(best_model['state_dict'])
    #     with torch.no_grad():
    #         test()
    if use_tm and epoch > args.kl_epoch:
        tm_from_kl = np.zeros([num_class,num_class])
        for i in range(num_class):
            for j in range(num_class):
                if (i, j) in kl_dict.keys():
                    tm_from_kl[i][j] = np.mean(kl_dict[(i,j)])
        tm_from_kl = np.power(tm_from_kl+1e-10, -float(args.sharpen))
        row_sums = tm_from_kl.sum(axis=1, keepdims=True)
        tm_from_kl = tm_from_kl / row_sums
        tm = tm_keep_r * tm + (1 - tm_keep_r) * tm_from_kl
# kl_dict_sorted = sorted(kl_dict.items(), key=lambda x: np.mean(x[1]), reverse=True)
# for i in kl_dict_sorted:
#     print(i[0], np.mean(i[1]))
# print('kl_dict:')
# for k, v in kl_dict.items():
#     print(k, v)

if use_tm:    
    np.save(save_point.replace('.pth.tar', '_tm.npy'), tm)
    # print(tm)
    print('tm:')
    for i in range(num_class):
        for j in range(num_class):
            print('%.4f' % tm[i][j], end=' ')
        print()

# Run testing only once using the best model
print('\nTesting model')
best_model = torch.load(save_point)
test_net.load_state_dict(best_model['state_dict'])
with torch.no_grad():
    test()


# record.close()
