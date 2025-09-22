import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class LoRAConv2d(nn.Module):
    def __init__(self, orig_conv, r=4, alpha=1.0, gated=False):
        super(LoRAConv2d, self).__init__()
        self.orig_conv = orig_conv
        self.r = r
        self.alpha = alpha / math.sqrt(r)  # More stable scaling factor

        in_channels = orig_conv.in_channels
        out_channels = orig_conv.out_channels
        kernel_size = orig_conv.kernel_size

        # Initialize LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(out_channels, r))
        lora_B = torch.zeros(r, in_channels * kernel_size[0] * kernel_size[1])
        
        # Initialize LoRA parameters
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(lora_B, a=math.sqrt(5))

        # Reshape LoRA B to match convolution weight dimensions
        self.lora_B = nn.Parameter(lora_B.view(r, in_channels, kernel_size[0], kernel_size[1]))

        self.gated = gated

        if gated:
            self.gate = nn.Parameter(torch.tensor(-1.0))

        # Freeze original conv parameters
        for param in self.orig_conv.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.orig_conv(x)

        lora_update = F.conv2d(x, self.lora_B, stride=self.orig_conv.stride, padding=self.orig_conv.padding, 
                               dilation=self.orig_conv.dilation, groups=self.orig_conv.groups)
        lora_update = torch.einsum('bchw,oc->bohw', lora_update, self.lora_A)
        
        if self.gated:
            gate_weight = torch.sigmoid(self.gate)
            return out + gate_weight * self.alpha * lora_update
        else:
            return out + self.alpha * lora_update


class LoRAConvTranspose2d(nn.Module):
    def __init__(self, orig_conv, r=4, alpha=1.0, gated=False):
        super(LoRAConvTranspose2d, self).__init__()
        self.orig_conv = orig_conv
        self.r = r
        self.alpha = alpha / math.sqrt(r)  # More stable scaling factor

        in_channels = orig_conv.in_channels
        out_channels = orig_conv.out_channels
        kernel_size = orig_conv.kernel_size

        # Initialize LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(out_channels, r))
        lora_B = torch.zeros(r, in_channels * kernel_size[0] * kernel_size[1])

        # Initialize LoRA parameters
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(lora_B, a=math.sqrt(5))

        # Reshape LoRA B to match deconvolution weight dimensions
        self.lora_B = nn.Parameter(lora_B.view(in_channels, r, kernel_size[0], kernel_size[1]))

        self.gated = gated
        
        if gated:
            self.gate = nn.Parameter(torch.tensor(-1.0))

        # Freeze original conv parameters
        for param in self.orig_conv.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.orig_conv(x)

        lora_update = F.conv_transpose2d(x, self.lora_B, stride=self.orig_conv.stride, padding=self.orig_conv.padding, output_padding=self.orig_conv.output_padding, 
                                            dilation=self.orig_conv.dilation, groups=self.orig_conv.groups)
        lora_update = torch.einsum('bchw,oc->bohw', lora_update, self.lora_A)
        
        if self.gated:
            gate_weight = torch.sigmoid(self.gate)
            return out + gate_weight * self.alpha * lora_update
        else:
            return out + self.alpha * lora_update


def inject_lora(module, r=4, alpha=1.0, gated=False):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            setattr(module, name, LoRAConv2d(child, r=r, alpha=alpha, gated=gated))
        elif isinstance(child, nn.ConvTranspose2d):
            setattr(module, name, LoRAConvTranspose2d(child, r=r, alpha=alpha, gated=gated))
        else:
            inject_lora(child, r=r, alpha=alpha, gated=gated) 

        