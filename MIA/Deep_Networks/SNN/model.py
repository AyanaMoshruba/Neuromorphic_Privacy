import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")   

class SurrogateBPFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda()
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * 0.3 * F.threshold(1.0 - torch.abs(input), 0, 0)
        return grad


def poisson_gen(inp, rescale_fac=2.0):
    rand_inp = torch.rand_like(inp).cuda()
    return torch.mul(torch.le(rand_inp * rescale_fac, torch.abs(inp)).float(), torch.sign(inp))


class SResnet(nn.Module):
    def __init__(self, n, nFilters, num_steps, leak_mem=0.95, img_size=32,  num_cls=10, boosting=False, poisson_gen=False):
        super(SResnet, self).__init__()

        self.n = n
        self.img_size = img_size
        self.num_cls = num_cls
        self.num_steps = num_steps
        self.spike_fn = SurrogateBPFunction.apply
        self.leak_mem = leak_mem
        self.batch_num = self.num_steps
        self.poisson_gen = poisson_gen
        if boosting:
            self.boost = nn.AvgPool1d(10, 10)
        else:
            self.boost = False

        print(">>>>>>>>>>>>>>>>>>> S-ResNet >>>>>>>>>>>>>>>>>>>>>>")

        affine_flag = True
        bias_flag = False
        self.nFilters = nFilters

        self.conv1 = nn.Conv2d(3, self.nFilters, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt1 = nn.ModuleList(
            [nn.BatchNorm2d(self.nFilters, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])

        self.conv_list = nn.ModuleList([self.conv1])
        self.bntt_list = nn.ModuleList([self.bntt1])

        for block in range(3):
            for layer in range(2*n):
                if block !=0 and layer == 0:
                    stride = 2
                    prev_nFilters = -1
                else:
                    stride = 1
                    prev_nFilters = 0
                self.conv_list.append(nn.Conv2d(self.nFilters*(2**(block + prev_nFilters)), self.nFilters*(2**block), kernel_size=3, stride=stride, padding=1, bias=bias_flag))
                self.bntt_list.append(nn.ModuleList(
                    [nn.BatchNorm2d(self.nFilters*(2**block), eps=1e-4, momentum=0.1, affine=affine_flag) for i in
                     range(self.batch_num)]))

        self.conv_resize_1 = nn.Conv2d(self.nFilters, self.nFilters * 2, kernel_size=1, stride=2, padding=0,
                                       bias=bias_flag)
        self.resize_bn_1 = nn.ModuleList(
                    [nn.BatchNorm2d(self.nFilters*2, eps=1e-4, momentum=0.1, affine=affine_flag) for i in
                     range(self.batch_num)])
        self.conv_resize_2 = nn.Conv2d(self.nFilters * 2, self.nFilters * 4, kernel_size=1, stride=2, padding=0,
                                       bias=bias_flag)
        self.resize_bn_2 = nn.ModuleList(
                    [nn.BatchNorm2d(self.nFilters*4, eps=1e-4, momentum=0.1, affine=affine_flag) for i in
                     range(self.batch_num)])

        self.pool2 = nn.AdaptiveAvgPool2d((1,1))

        if self.boost:
            self.fc = nn.Linear(self.nFilters * 4, self.num_cls * 10, bias=bias_flag)
        else:
            self.fc = nn.Linear(self.nFilters*4, self.num_cls, bias=bias_flag)

        self.conv1x1_list = nn.ModuleList([self.conv_resize_1, self.conv_resize_2])

        self.bn_conv1x1_list = nn.ModuleList([self.resize_bn_1,self.resize_bn_2])

        # Turn off bias of BNTT
        for bn_temp in self.bntt_list:
            bn_temp.bias = None

        # Initialize the firing thresholds of all the layers
        for m in self.modules():
            if (isinstance(m, nn.Conv2d)):
                m.threshold = 1.0
                nn.init.xavier_uniform_(m.weight, gain=2)
            elif (isinstance(m, nn.Linear)):
                m.threshold = 1.0
                nn.init.xavier_uniform_(m.weight, gain=2)

    def forward(self, inp):

        batch_size = inp.size(0)

        mem_conv_list = [torch.zeros(batch_size, self.nFilters, self.img_size, self.img_size).cuda()]

        for block in range(3):
            for layer in range(2*self.n):
                mem_conv_list.append(torch.zeros(batch_size, self.nFilters*(2**block), self.img_size // 2**block,
                                                 self.img_size // 2**block).cuda())

        mem_fc = torch.zeros(batch_size, self.num_cls).cuda()

        for t in range(self.num_steps):
            if self.poisson_gen:
                spike_inp = poisson_gen(inp)
                out_prev = spike_inp
            else:
                out_prev = inp

            index_1x1 = 0
            for i in range(len(self.conv_list)):
                mem_conv_list[i] = self.leak_mem * mem_conv_list[i] + self.bntt_list[i][t](self.conv_list[i](out_prev))
                mem_thr = (mem_conv_list[i] / self.conv_list[
                    i].threshold) - 1.0  # Positive values have surpassed the threshold
                out = self.spike_fn(mem_thr)

                if i>0 and i%2 == 0:  # Add skip conn spikes to the current output spikes
                    if i == 2 + 2 * self.n or i == 2 + 4 * self.n:  # Beggining of block 2 and 3 downsize
                        skip = self.bn_conv1x1_list[index_1x1][t](self.conv1x1_list[index_1x1](skip))  # Connections guided by 1x1 conv instead of 1 to 1 correspondance
                        index_1x1 += 1
                    out = out + skip
                    skip = out.clone()
                elif i == 0:
                    skip = out.clone()

                rst = torch.zeros_like(mem_conv_list[i].cuda())
                rst[mem_thr > 0] = self.conv_list[i].threshold  # Matrix of 0s with Th in activated cells
                mem_conv_list[i] = mem_conv_list[i] - rst  # Reset by subtraction
                out_prev = out.clone()

                if i == len(self.conv_list)-1:
                    out = self.pool2(out_prev)
                    out_prev = out.clone()

            out_prev = out_prev.reshape(batch_size, -1)

            #  Accumulate voltage in the last layer
            if self.boost:
                mem_fc = mem_fc + self.boost(self.fc(out_prev).unsqueeze(1)).squeeze(1)
            else:
                mem_fc = mem_fc + self.fc(out_prev)

        out_voltage = mem_fc / self.num_steps

        return out_voltage


class SResnetNM(nn.Module):
    def __init__(self, n, nFilters, num_steps, leak_mem=0.95, img_size=32,  num_cls=10):
        super(SResnetNM, self).__init__()

        self.n = n
        self.img_size = int(img_size/2)
        self.num_cls = num_cls
        self.num_steps = num_steps
        self.spike_fn = SurrogateBPFunction.apply
        self.leak_mem = leak_mem
        self.batch_num = self.num_steps

        print(">>>>>>>>>>>>>>>>>>> S-ResNet NM >>>>>>>>>>>>>>>>>>>>>>")

        affine_flag = True
        bias_flag = False
        self.nFilters = nFilters

        self.conv1 = nn.Conv2d(2, self.nFilters, kernel_size=3, stride=2, padding=1, bias=bias_flag)
        self.bntt1 = nn.ModuleList(
            [nn.BatchNorm2d(self.nFilters, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])

        self.conv_list = nn.ModuleList([self.conv1])
        self.bntt_list = nn.ModuleList([self.bntt1])

        for block in range(3):
            for layer in range(2*n):
                if block !=0 and layer == 0:
                    stride = 2
                    prev_nFilters = -1
                else:
                    stride = 1
                    prev_nFilters = 0
                self.conv_list.append(nn.Conv2d(self.nFilters*(2**(block + prev_nFilters)), self.nFilters*(2**block), kernel_size=3, stride=stride, padding=1, bias=bias_flag))
                self.bntt_list.append(nn.ModuleList(
                    [nn.BatchNorm2d(self.nFilters*(2**block), eps=1e-4, momentum=0.1, affine=affine_flag) for i in
                     range(self.batch_num)]))

        self.conv_resize_1 = nn.Conv2d(self.nFilters, self.nFilters * 2, kernel_size=1, stride=2, padding=0,
                                       bias=bias_flag)
        self.resize_bn_1 = nn.ModuleList(
                    [nn.BatchNorm2d(self.nFilters*2, eps=1e-4, momentum=0.1, affine=affine_flag) for i in
                     range(self.batch_num)])
        self.conv_resize_2 = nn.Conv2d(self.nFilters * 2, self.nFilters * 4, kernel_size=1, stride=2, padding=0,
                                       bias=bias_flag)
        self.resize_bn_2 = nn.ModuleList(
                    [nn.BatchNorm2d(self.nFilters*4, eps=1e-4, momentum=0.1, affine=affine_flag) for i in
                     range(self.batch_num)])

        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(self.nFilters * 4, self.num_cls, bias=bias_flag)

        self.conv1x1_list = nn.ModuleList([self.conv_resize_1, self.conv_resize_2])

        self.bn_conv1x1_list = nn.ModuleList([self.resize_bn_1, self.resize_bn_2])

        # Turn off bias of BNTT
        for bn_temp in self.bntt_list:
            bn_temp.bias = None

        # Initialize the firing thresholds of all the layers
        for m in self.modules():
            if (isinstance(m, nn.Conv2d)):
                m.threshold = 1.0
                nn.init.xavier_uniform_(m.weight, gain=2)
            elif (isinstance(m, nn.Linear)):
                m.threshold = 1.0
                nn.init.xavier_uniform_(m.weight, gain=2)

    def forward(self, inp):

        inp = inp.permute(1, 0, 2, 3, 4)  # changes to: [T, N, 2, *, *] T=timesteps, N=batch_size

        batch_size = inp.size(1)

        mem_conv_list = [torch.zeros(batch_size, self.nFilters, self.img_size, self.img_size).cuda()]

        for block in range(3):
            for layer in range(2*self.n):
                mem_conv_list.append(torch.zeros(batch_size, self.nFilters*(2**block), self.img_size // 2**block,
                                                 self.img_size // 2**block).cuda())

        mem_fc = torch.zeros(batch_size, self.num_cls).cuda()

        for t in range(inp.size(0)):

            out_prev = inp[t,:]
            out_prev = transforms.Resize([64,64])(out_prev)

            index_1x1 = 0
            for i in range(len(self.conv_list)):
                mem_conv_list[i] = self.leak_mem * mem_conv_list[i] + self.bntt_list[i][t](self.conv_list[i](out_prev))
                mem_thr = (mem_conv_list[i] / self.conv_list[
                    i].threshold) - 1.0  # Positive values have surpassed the threshold
                out = self.spike_fn(mem_thr)

                if i>0 and i%2 == 0:  # Add skip conn spikes to the current output spikes
                    if i == 2 + 2 * self.n or i == 2 + 4 * self.n:  # Beggining of block 2 and 3 downsize
                        skip = self.bn_conv1x1_list[index_1x1][t](self.conv1x1_list[index_1x1](skip))  # Connections guided by 1x1 conv instead of 1 to 1 correspondance
                        index_1x1 += 1
                    out = out + skip
                    skip = out.clone()
                elif i == 0:
                    skip = out.clone()

                rst = torch.zeros_like(mem_conv_list[i].cuda())
                rst[mem_thr > 0] = self.conv_list[i].threshold  # Matrix of 0s with Th in activated cells
                mem_conv_list[i] = mem_conv_list[i] - rst  #  Reset by subtraction
                out_prev = out.clone()

                if i == len(self.conv_list) - 1:
                    out = self.pool2(out_prev)
                    out_prev = out.clone()

            out_prev = out_prev.reshape(batch_size, -1)

            #  Accumulate voltage in the last layer
            mem_fc = mem_fc + self.fc(out_prev)

        out_voltage = mem_fc / self.num_steps

        return out_voltage
        
class gSResnet(nn.Module):
    def __init__(self, n, nFilters, num_steps, leak_mem=0.95, img_size=32, num_cls=10, boosting=False, poisson_gen=False, num_channels=3):
        super(gSResnet, self).__init__()

        self.n = n
        self.img_size = img_size
        self.num_cls = num_cls
        self.num_steps = num_steps
        self.spike_fn = SurrogateBPFunction.apply
        self.leak_mem = leak_mem
        self.batch_num = self.num_steps
        self.poisson_gen = poisson_gen
        if boosting:
            self.boost = nn.AvgPool1d(10, 10)
        else:
            self.boost = False

        print(">>>>>>>>>>>>>>>>>>> gSResNet >>>>>>>>>>>>>>>>>>>>>>")

        affine_flag = True
        bias_flag = False
        self.nFilters = nFilters

        self.conv1 = nn.Conv2d(num_channels, self.nFilters, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt1 = nn.ModuleList(
            [nn.GroupNorm(1, self.nFilters, eps=1e-4, affine=affine_flag) for i in range(self.batch_num)])

        self.conv_list = nn.ModuleList([self.conv1])
        self.bntt_list = nn.ModuleList([self.bntt1])

        for block in range(3):
            for layer in range(2*n):
                if block != 0 and layer == 0:
                    stride = 2
                    prev_nFilters = -1
                else:
                    stride = 1
                    prev_nFilters = 0
                self.conv_list.append(nn.Conv2d(self.nFilters*(2**(block + prev_nFilters)), self.nFilters*(2**block), kernel_size=3, stride=stride, padding=1, bias=bias_flag))

                self.bntt_list.append(nn.ModuleList(
                    [nn.GroupNorm(1, self.nFilters*(2**block), eps=1e-4, affine=affine_flag) for i in
                     range(self.batch_num)]))

        self.conv_resize_1 = nn.Conv2d(self.nFilters, self.nFilters * 2, kernel_size=1, stride=2, padding=0,
                                       bias=bias_flag)
        self.resize_bn_1 = nn.ModuleList(
                    [nn.GroupNorm(1, self.nFilters*2, eps=1e-4, affine=affine_flag) for i in
                     range(self.batch_num)])
        self.conv_resize_2 = nn.Conv2d(self.nFilters * 2, self.nFilters * 4, kernel_size=1, stride=2, padding=0,
                                       bias=bias_flag)
        self.resize_bn_2 = nn.ModuleList(
                    [nn.GroupNorm(1, self.nFilters*4, eps=1e-4,  affine=affine_flag) for i in
                     range(self.batch_num)])

        self.pool2 = nn.AdaptiveAvgPool2d((1,1))

        if self.boost:
            self.fc = nn.Linear(self.nFilters * 4, self.num_cls * 10, bias=bias_flag)
        else:
            self.fc = nn.Linear(self.nFilters*4, self.num_cls, bias=bias_flag)

        self.conv1x1_list = nn.ModuleList([self.conv_resize_1, self.conv_resize_2])

        self.bn_conv1x1_list = nn.ModuleList([self.resize_bn_1,self.resize_bn_2])

        # Turn off bias of BNTT
        for bn_temp in self.bntt_list:
            bn_temp.bias = None

        # Initialize the firing thresholds of all the layers
        for m in self.modules():
            if (isinstance(m, nn.Conv2d)):
                m.threshold = 1.0
                nn.init.xavier_uniform_(m.weight, gain=2)
            elif (isinstance(m, nn.Linear)):
                m.threshold = 1.0
                nn.init.xavier_uniform_(m.weight, gain=2)

    def forward(self, inp):

        batch_size = inp.size(0)

        mem_conv_list = [torch.zeros(batch_size, self.nFilters, self.img_size, self.img_size).cuda()]

        for block in range(3):
            for layer in range(2*self.n):
                mem_conv_list.append(torch.zeros(batch_size, self.nFilters*(2**block), self.img_size // 2**block,
                                                 self.img_size // 2**block).cuda())

        mem_fc = torch.zeros(batch_size, self.num_cls).cuda()

        for t in range(self.num_steps):
            if self.poisson_gen:
                spike_inp = poisson_gen(inp)
                out_prev = spike_inp
            else:
                out_prev = inp

            index_1x1 = 0
            for i in range(len(self.conv_list)):
                mem_conv_list[i] = self.leak_mem * mem_conv_list[i] + self.bntt_list[i][t](self.conv_list[i](out_prev))
                mem_thr = (mem_conv_list[i] / self.conv_list[i].threshold) - 1.0  # Positive values have surpassed the threshold
                out = self.spike_fn(mem_thr)

                if i > 0 and i % 2 == 0:  # Add skip conn spikes to the current output spikes
                    if i == 2 + 2 * self.n or i == 2 + 4 * self.n:  # Beginning of block 2 and 3 downsize
                        skip = self.bn_conv1x1_list[index_1x1][t](self.conv1x1_list[index_1x1](skip))  # Connections guided by 1x1 conv instead of 1 to 1 correspondence
                        index_1x1 += 1
                    out = out + skip
                    skip = out.clone()
                elif i == 0:
                    skip = out.clone()

                rst = torch.zeros_like(mem_conv_list[i]).cuda()
                rst[mem_thr > 0] = self.conv_list[i].threshold  # Matrix of 0s with Th in activated cells
                mem_conv_list[i] = mem_conv_list[i] - rst  # Reset by subtraction
                out_prev = out.clone()

                if i == len(self.conv_list)-1:
                    out = self.pool2(out_prev)
                    out_prev = out.clone()

            out_prev = out_prev.reshape(batch_size, -1)

            #  Accumulate voltage in the last layer
            if self.boost:
                mem_fc = mem_fc + self.boost(self.fc(out_prev).unsqueeze(1)).squeeze(1)
            else:
                mem_fc = mem_fc + self.fc(out_prev)

        out_voltage = mem_fc / self.num_steps

        return out_voltage

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpikingVGG16(nn.Module):
    def __init__(self, num_steps, leak_mem, img_size, num_cls, boosting=False, poisson_gen=False, num_channels=3):
        super(SpikingVGG16, self).__init__()
        self.num_steps = num_steps
        self.leak_mem = leak_mem
        self.img_size = img_size
        self.num_cls = num_cls
        self.boosting = boosting
        self.poisson_gen = poisson_gen
        self.num_channels = num_channels
        self.spike_fn = SurrogateBPFunction.apply

        self.features = self._make_layers()

        # Calculate the size of the feature map before the fully connected layers
        self.feature_size = self._get_feature_size()

        if self.boosting:
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_size, 4096),
                nn.Linear(4096, 4096),
                nn.Linear(4096, self.num_cls * 10)
            )
            self.boost = nn.AvgPool1d(10, 10)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_size, 4096),
                nn.Linear(4096, 4096),
                nn.Linear(4096, self.num_cls)
            )

        self._initialize_weights()

    def _make_layers(self):
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        layers = []
        in_channels = self.num_channels
        for v in cfg:
            if v == 'M':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                layers += [conv2d, nn.ModuleList([nn.BatchNorm2d(v, eps=1e-4, momentum=0.1, affine=True) for _ in range(self.num_steps)])]
                in_channels = v
        return nn.Sequential(*layers)

    def _get_feature_size(self):
        with torch.no_grad():
            x = torch.zeros(1, self.num_channels, self.img_size, self.img_size)
            for layer in self.features:
                if isinstance(layer, nn.Conv2d):
                    x = layer(x)
                elif isinstance(layer, nn.ModuleList):  # BatchNorm
                    x = layer[0](x)  # Use the first BatchNorm in the list
                elif isinstance(layer, nn.AvgPool2d):
                    x = layer(x)
            return x.numel()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                m.threshold = 1.0
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                m.threshold = 1.0

    def reset_membrane_potentials(self, batch_size):
        self.mem_conv = {}
        self.spike_conv = {}
        self.mem_fc = {}
        self.spike_fc = {}

        current_size = self.img_size
        for layer in self.features:
            if isinstance(layer, nn.Conv2d):
                self.mem_conv[layer] = torch.zeros(batch_size, layer.out_channels, current_size, current_size).cuda()
                self.spike_conv[layer] = torch.zeros_like(self.mem_conv[layer])
            elif isinstance(layer, nn.AvgPool2d):
                current_size = current_size // 2  # Update size after pooling

        for fc in self.classifier:
            if isinstance(fc, nn.Linear):
                self.mem_fc[fc] = torch.zeros(batch_size, fc.out_features).cuda()
                self.spike_fc[fc] = torch.zeros_like(self.mem_fc[fc])

    def forward(self, inp):
        batch_size = inp.size(0)
        
        self.reset_membrane_potentials(batch_size)

        for t in range(self.num_steps):
            out_prev = inp if not self.poisson_gen else poisson_gen(inp)

            for layer in self.features:
                if isinstance(layer, nn.Conv2d):
                    self.mem_conv[layer] = self.leak_mem * self.mem_conv[layer] + layer(out_prev)
                    mem_thr = (self.mem_conv[layer] / layer.threshold) - 1.0
                    out = self.spike_fn(mem_thr)
                    rst = torch.zeros_like(self.mem_conv[layer]).cuda()
                    rst[mem_thr > 0] = layer.threshold
                    self.mem_conv[layer] = self.mem_conv[layer] - rst
                    self.spike_conv[layer] = out.clone()
                elif isinstance(layer, nn.ModuleList):  # BatchNorm
                    out = layer[t](out)
                elif isinstance(layer, nn.AvgPool2d):
                    out = layer(out)

                out_prev = out.clone()

            out_prev = out_prev.reshape(batch_size, -1)

            for fc in self.classifier:
                if isinstance(fc, nn.Linear):
                    self.mem_fc[fc] = self.leak_mem * self.mem_fc[fc] + fc(out_prev)
                    mem_thr = (self.mem_fc[fc] / fc.threshold) - 1.0
                    out = self.spike_fn(mem_thr)
                    rst = torch.zeros_like(self.mem_fc[fc]).cuda()
                    rst[mem_thr > 0] = fc.threshold
                    self.mem_fc[fc] = self.mem_fc[fc] - rst
                    self.spike_fc[fc] = out.clone()
                    out_prev = out.clone()

        out_voltage = self.mem_fc[self.classifier[-1]] / self.num_steps

        if self.boosting:
            out_voltage = self.boost(out_voltage.unsqueeze(1)).squeeze(1)

        return out_voltage

