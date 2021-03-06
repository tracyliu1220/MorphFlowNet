import torch
import torch.nn as nn

from networks import FlowNetC
from networks import FlowNetS
from networks import FlowNetSD
from networks import FlowNetFusion

from networks.submodules import *

from networks.resample2d_package.resample2d import Resample2d
from networks.channelnorm_package.channelnorm import ChannelNorm

def morphize_forward_hook(module, input, output):
    return output * module.gate

class FlowNet2(nn.Module):

    def __init__(self, rgb_max=1, batchNorm=False, div_flow = 20.):
        super(FlowNet2,self).__init__()
        self.batchNorm = batchNorm
        self.div_flow = div_flow
        self.rgb_max = rgb_max

        self.channelnorm = ChannelNorm()

        # First Block (FlowNetC)
        self.flownetc = FlowNetC.FlowNetC(batchNorm=self.batchNorm)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.resample1 = Resample2d()

        # Block (FlowNetS1)
        self.flownets_1 = FlowNetS.FlowNetS(batchNorm=self.batchNorm)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.resample2 = Resample2d()

        # Block (FlowNetS2)
        self.flownets_2 = FlowNetS.FlowNetS(batchNorm=self.batchNorm)

        # Block (FlowNetSD)
        self.flownets_d = FlowNetSD.FlowNetSD(batchNorm=self.batchNorm)
        self.upsample3 = nn.Upsample(scale_factor=4, mode='nearest')
        self.upsample4 = nn.Upsample(scale_factor=4, mode='nearest')
        self.resample3 = Resample2d()
        self.resample4 = Resample2d()

        # Block (FLowNetFusion)
        self.flownetfusion = FlowNetFusion.FlowNetFusion(batchNorm=self.batchNorm)

    def morphize(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                module.name = name
                module.gate = nn.Parameter(torch.ones((1, module.out_channels, 1, 1)))
                module.handler = module.register_forward_hook(morphize_forward_hook)

    def demorphize(self):
        tot, cnt = 0, 0
        for name, module in self.named_modules():
            if not isinstance(module, nn.Conv2d) and not isinstance(module, nn.ConvTranspose2d):
                continue
            module.handler.remove()
            weight = module.gate.data.reshape(-1)
            print (name, weight[weight < 0.01].size(0))
            tot += weight.size(0)
            cnt += weight[weight < 0.01].size(0)
            weight[weight < 0.01] = 0
            if isinstance(module, nn.Conv2d):
                weight = weight.reshape(-1, 1, 1, 1)
            if isinstance(module, nn.ConvTranspose2d):
                weight = weight.reshape(1, -1, 1, 1)
            module.weight = nn.Parameter(module.weight * weight)
            if module.bias is not None:
                weight = weight.reshape(-1)
                module.bias = nn.Parameter(module.bias * weight)
        print (cnt, '/', tot)

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))

        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:,:,0,:,:]
        x2 = x[:,:,1,:,:]
        x = torch.cat((x1,x2), dim = 1)

        # flownetc
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2*self.div_flow)

        # warp img1 to img0; magnitude of diff between img0 and and warped_img1,
        resampled_img1 = self.resample1(x[:,3:,:,:], flownetc_flow)
        diff_img0 = x[:,:3,:,:] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag ;
        concat1 = torch.cat((x, resampled_img1, flownetc_flow/self.div_flow, norm_diff_img0), dim=1)

        # flownets1
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2*self.div_flow)

        # warp img1 to img0 using flownets1; magnitude of diff between img0 and and warped_img1
        resampled_img1 = self.resample2(x[:,3:,:,:], flownets1_flow)
        diff_img0 = x[:,:3,:,:] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag
        concat2 = torch.cat((x, resampled_img1, flownets1_flow/self.div_flow, norm_diff_img0), dim=1)

        # flownets2
        flownets2_flow2 = self.flownets_2(concat2)[0]
        flownets2_flow = self.upsample4(flownets2_flow2 * self.div_flow)
        norm_flownets2_flow = self.channelnorm(flownets2_flow)

        diff_flownets2_flow = self.resample4(x[:,3:,:,:], flownets2_flow)

        diff_flownets2_img1 = self.channelnorm((x[:,:3,:,:]-diff_flownets2_flow))

        # flownetsd
        flownetsd_flow2 = self.flownets_d(x)[0]
        flownetsd_flow = self.upsample3(flownetsd_flow2 / self.div_flow)
        norm_flownetsd_flow = self.channelnorm(flownetsd_flow)

        diff_flownetsd_flow = self.resample3(x[:,3:,:,:], flownetsd_flow)

        diff_flownetsd_img1 = self.channelnorm((x[:,:3,:,:]-diff_flownetsd_flow))

        # concat img1 flownetsd, flownets2, norm_flownetsd, norm_flownets2, diff_flownetsd_img1, diff_flownets2_img1
        concat3 = torch.cat((x[:,:3,:,:], flownetsd_flow, flownets2_flow, norm_flownetsd_flow, norm_flownets2_flow, diff_flownetsd_img1, diff_flownets2_img1), dim=1)
        flownetfusion_flow = self.flownetfusion(concat3)

        return flownetfusion_flow

if __name__ == '__main__':
    model = FlowNet2()
    weights = torch.load('checkpoint/FlowNet2_checkpoint.pth.tar')['state_dict']
    model.load_state_dict(weights)
    print (model)
