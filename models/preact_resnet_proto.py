'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        x = self.conv1(F.relu(self.bn1(x)))
        x = self.conv2(F.relu(self.bn2(x)))
        x += shortcut
        return x

class PreActResNet(nn.Module):
    def __init__(self):
        super(PreActResNet, self).__init__()
        self.in_planes = 32

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block=PreActBlock, planes=32, num_blocks=5, stride=1)
        self.layer2 = self._make_layer(block=PreActBlock, planes=64, num_blocks=5, stride=2)
        self.layer3 = self._make_layer(block=PreActBlock, planes=128, num_blocks=5, stride=2)
        self.bn_final = nn.BatchNorm2d(128)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(128, 10)
        # MoPro
        self.l2norm = Normalize(2)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.register_buffer('prototypes', torch.zeros(10,128)) # (num_class, low_dim)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1) # strides = [stride, 1, 1, 1, 1]
        layers = []
        for stride in strides:
            layers.append(block(in_planes=self.in_planes, planes=planes, stride=stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self,
                x,
                y,
                weights=None,
                is_eval=False,
                is_proto=False,
                is_clean=False,
                temperature=0.1,
                alpha=0.5,
                pseudo_th=0.8,
                proto_m=0.999):
        if weights==None:
            x = self.conv1(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = F.relu(self.bn_final(x))
            x = self.avgpool(x)
            feat = x.view(x.size(0), -1)
            out = self.linear(feat)
            # MoPro
            if is_eval:
               return out
            q = F.relu(self.fc1(feat))
            q = self.fc2(q)
            q = self.l2norm(q)
            if is_proto:
                # compute protoypical logits
                prototypes = self.prototypes.clone().detach()
                logits_proto = torch.mm(q, prototypes.t())/temperature
            else:
                logits_proto = 0
            if is_clean:
                # noise cleaning
                soft_label = alpha * F.softmax(out, dim=1) + (1-alpha)*F.softmax(logits_proto, dim=1)
                # keep ground truth label
                gt_score = soft_label[y >= 0,y]
                clean_idx = gt_score > 0.1 # assume num_class==10 (TODO: allow other num_class)
                # assign a new pseudo label
                max_score, hard_label = soft_label.max(1)
                correct_idx = max_score > pseudo_th
                y[correct_idx] = hard_label[correct_idx]
                # confident sample index
                clean_idx = clean_idx | correct_idx
                # update momentum prototypes with pseudo-labels
                for q_, y_ in zip(q[clean_idx], y[clean_idx]):
                    self.prototypes[y_] = proto_m * self.prototypes[y_] + (1 - proto_m) * q_
                # select only the confident samples to return (i.e., we may get a smaller batch)
                out = out[clean_idx]
                y = y[clean_idx]
                feat = feat[clean_idx]
                logits_proto = logits_proto[clean_idx]
            return out,y,feat,logits_proto
        else:
            x = F.conv2d(x, weights['conv1.weight'], stride=1, padding=1)
            #layer 1
            strides = [1, 1, 1, 1, 1]
            for i in range(5):
                if 'layer1.%d.shortcut.0.weight'%i in weights.keys():
                    shortcut = F.conv2d(x, weights['layer1.%d.shortcut.0.weight'%i], stride=strides[i])
                else:
                    shortcut = x
                x = F.batch_norm(x, self.layer1[i].bn1.running_mean, self.layer1[i].bn1.running_var, 
                                 weights['layer1.%d.bn1.weight'%i], weights['layer1.%d.bn1.bias'%i], training=True)
                x = F.threshold(x, 0, 0, inplace=True)
                x = F.conv2d(x, weights['layer1.%d.conv1.weight'%i], stride=strides[i], padding=1)

                x = F.batch_norm(x, self.layer1[i].bn2.running_mean, self.layer1[i].bn2.running_var, 
                                 weights['layer1.%d.bn2.weight'%i], weights['layer1.%d.bn2.bias'%i], training=True)
                x = F.threshold(x, 0, 0, inplace=True)
                x = F.conv2d(x, weights['layer1.%d.conv2.weight'%i], stride=1, padding=1)
                x = x + shortcut
            #layer 2
            strides = [2, 1, 1, 1, 1]
            for i in range(5):
                if 'layer2.%d.shortcut.0.weight'%i in weights.keys():
                    shortcut = F.conv2d(x, weights['layer2.%d.shortcut.0.weight'%i], stride=strides[i])
                else:
                    shortcut = x
                x = F.batch_norm(x, self.layer2[i].bn1.running_mean, self.layer2[i].bn1.running_var, 
                                 weights['layer2.%d.bn1.weight'%i], weights['layer2.%d.bn1.bias'%i], training=True)
                x = F.threshold(x, 0, 0, inplace=True)
                x = F.conv2d(x, weights['layer2.%d.conv1.weight'%i], stride=strides[i], padding=1)

                x = F.batch_norm(x, self.layer2[i].bn2.running_mean, self.layer2[i].bn2.running_var, 
                                 weights['layer2.%d.bn2.weight'%i], weights['layer2.%d.bn2.bias'%i], training=True)
                x = F.threshold(x, 0, 0, inplace=True)
                x = F.conv2d(x, weights['layer2.%d.conv2.weight'%i], stride=1, padding=1)
                x = x + shortcut
            #layer 3
            strides = [2, 1, 1, 1, 1]
            for i in range(5):
                if 'layer3.%d.shortcut.0.weight'%i in weights.keys():
                    shortcut = F.conv2d(x, weights['layer3.%d.shortcut.0.weight'%i], stride=strides[i])
                else:
                    shortcut = x
                x = F.batch_norm(x, self.layer3[i].bn1.running_mean, self.layer3[i].bn1.running_var, 
                                 weights['layer3.%d.bn1.weight'%i], weights['layer3.%d.bn1.bias'%i], training=True)
                x = F.threshold(x, 0, 0, inplace=True)
                x = F.conv2d(x, weights['layer3.%d.conv1.weight'%i], stride=strides[i], padding=1)

                x = F.batch_norm(x, self.layer3[i].bn2.running_mean, self.layer3[i].bn2.running_var, 
                                 weights['layer3.%d.bn2.weight'%i], weights['layer3.%d.bn2.bias'%i], training=True)
                x = F.threshold(x, 0, 0, inplace=True)
                x = F.conv2d(x, weights['layer3.%d.conv2.weight'%i], stride=1, padding=1)
                x = x + shortcut
            x = F.batch_norm(x, self.bn_final.running_mean, self.bn_final.running_var,
                             weights['bn_final.weight'], weights['bn_final.bias'], training=True)            
            x = F.threshold(x, 0, 0, inplace=True)
            x = F.avg_pool2d(x, kernel_size=8, stride=1, padding=0)
            feat = x.view(x.size(0), -1)
            out = F.linear(feat, weights['linear.weight'], weights['linear.bias'])                
            return out


def PreActResNet32():
    return PreActResNet()
