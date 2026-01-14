import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class LoRAConv2d(nn.Module):
    def __init__(self, orig_conv, rank=4, alpha=1.0, gated=False, gated_type="basic"):
        super(LoRAConv2d, self).__init__()
        self.orig_conv = orig_conv
        self.rank = rank
        self.alpha = alpha / math.sqrt(rank)  # More stable scaling factor

        in_channels = orig_conv.in_channels
        out_channels = orig_conv.out_channels
        kernel_size = orig_conv.kernel_size

        # Initialize LoRA matrices
        lora_A = torch.zeros(rank, in_channels * kernel_size[0] * kernel_size[1])
        self.lora_B = nn.Parameter(torch.zeros(out_channels, rank))
        
        # Initialize LoRA parameters
        nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))

        # Reshape LoRA A to match convolution weight dimensions
        self.lora_A = nn.Parameter(lora_A.view(rank, in_channels, kernel_size[0], kernel_size[1]))

        self.gated = gated
        self.gated_type = gated_type

        if gated:
            # Basic gating
            if gated_type == "basic":
                self.gate = nn.Parameter(torch.tensor(-1.0))
            # Input Adaptive Gating
            elif gated_type == "input_ada":
                self.gate_conv = nn.Conv2d(1, 1, kernel_size=1)
            # Input Adaptive Channel Gating
            elif gated_type == "input_ada_channel":
                self.gate_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
            else:
                raise ValueError(
                    f"Unknown gated_type: '{gated_type}'. "
                    f"Supported gated_type values are: basic, input_ada and input_ada_channel") 


        # Freeze original conv parameters
        for param in self.orig_conv.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.orig_conv(x)

        lora_update = F.conv2d(x, self.lora_A, stride=self.orig_conv.stride, padding=self.orig_conv.padding, 
                               dilation=self.orig_conv.dilation, groups=self.orig_conv.groups)
        lora_update = torch.einsum('bchw,oc->bohw', lora_update, self.lora_B)
        
        if self.gated:
            # Basic gating
            if self.gated_type == "basic":
                gate_weight = torch.sigmoid(self.gate)
                return out + gate_weight * self.alpha * lora_update
            # Input Adaptive Gating
            elif self.gated_type == "input_ada":
                s = lora_update.mean(dim=(1, 2, 3), keepdim=True)  # [B, 1, 1, 1]
                gate_weight = torch.sigmoid(self.gate_conv(s))              # 1x1 conv or Linear
                return out + gate_weight * self.alpha * lora_update
            # Input Adaptive Channel Gating
            elif self.gated_type == "input_ada_channel":
                gate_weight = torch.sigmoid(self.gate_conv(lora_update))  # [B, out_channels, H, W]
                return out + gate_weight * self.alpha * lora_update
            else:
                raise ValueError(
                    f"Unknown gated_type: '{self.gated_type}'. "
                    f"Supported gated_type values are: basic, input_ada and input_ada_channel") 

        else:
            return out + self.alpha * lora_update


class LoRAConvTranspose2d(nn.Module):
    def __init__(self, orig_conv, rank=4, alpha=1.0, gated=False, gated_type="basic"):
        super(LoRAConvTranspose2d, self).__init__()
        self.orig_conv = orig_conv
        self.rank = rank
        self.alpha = alpha / math.sqrt(rank)  # More stable scaling factor

        in_channels = orig_conv.in_channels
        out_channels = orig_conv.out_channels
        kernel_size = orig_conv.kernel_size

        # Initialize LoRA matrices
        lora_A = torch.zeros(rank, in_channels * kernel_size[0] * kernel_size[1])
        self.lora_B = nn.Parameter(torch.zeros(out_channels, rank))

        # Initialize LoRA parameters
        nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))

        # Reshape LoRA A to match deconvolution weight dimensions
        self.lora_A = nn.Parameter(lora_A.view(in_channels, rank, kernel_size[0], kernel_size[1]))

        self.gated = gated
        self.gated_type = gated_type
        
        if gated:
            # Basic gating
            if gated_type == "basic":
                self.gate = nn.Parameter(torch.tensor(-1.0))
            # Input Adaptive Gating
            elif gated_type == "input_ada":
                self.gate_conv = nn.Conv2d(1, 1, kernel_size=1)
            # Input Adaptive Channel Gating
            elif gated_type == "input_ada_channel":
                self.gate_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
            else:
                raise ValueError(
                    f"Unknown gated_type: '{gated_type}'. "
                    f"Supported gated_type values are: basic, input_ada and input_ada_channel") 

        # Freeze original conv parameters
        for param in self.orig_conv.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.orig_conv(x)

        lora_update = F.conv_transpose2d(x, self.lora_A, stride=self.orig_conv.stride, padding=self.orig_conv.padding, output_padding=self.orig_conv.output_padding, 
                                            dilation=self.orig_conv.dilation, groups=self.orig_conv.groups)
        lora_update = torch.einsum('bchw,oc->bohw', lora_update, self.lora_B)
        
        if self.gated:
            # Basic gating
            if self.gated_type == "basic":
                gate_weight = torch.sigmoid(self.gate)
                return out + gate_weight * self.alpha * lora_update
            # Input Adaptive Gating
            elif self.gated_type == "input_ada":
                s = lora_update.mean(dim=(1, 2, 3), keepdim=True)  # [B, 1, 1, 1]
                gate_weight = torch.sigmoid(self.gate_conv(s))              # 1x1 conv or Linear
                return out + gate_weight * self.alpha * lora_update
            # Input Adaptive Channel Gating
            elif self.gated_type == "input_ada_channel":
                gate_weight = torch.sigmoid(self.gate_conv(lora_update))  # [B, out_channels, H, W]
                return out + gate_weight * self.alpha * lora_update
            else:
                raise ValueError(
                    f"Unknown gated_type: '{self.gated_type}'. "
                    f"Supported gated_type values are: basic, input_ada and input_ada_channel" )
        else:
            return out + self.alpha * lora_update


class LoRALinear(nn.Module):
    def __init__(self, orig_linear, rank=4, alpha=1.0, gated=False, gated_type="basic"):
        super().__init__()
        self.orig_linear = orig_linear
        self.rank = rank
        # self.alpha = alpha / math.sqrt(rank)
        self.alpha = alpha

        in_features = orig_linear.in_features
        out_features = orig_linear.out_features

        # A: (rank, in)
        # B: (out, rank)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))

        self.gated = gated
        self.gated_type = gated_type

        if gated:
            # Basic gating
            if gated_type == "basic":
                self.gate = nn.Parameter(torch.tensor(-1.0))
            # Input Adaptive Gating
            elif gated_type == "input_ada":
                self.gate_fc = nn.Linear(1, 1)
            # Input Adaptive Channel Gating
            elif gated_type == "input_ada_channel":
                self.gate_fc = nn.Linear(out_features, out_features)
            else:
                raise ValueError(
                    f"Unknown gated_type: '{gated_type}'. "
                    f"Supported gated_type values are: basic, input_ada and input_ada_channel") 

        # Freeze original linear
        for p in self.orig_linear.parameters():
            p.requires_grad = False

    def forward(self, x):
        out = self.orig_linear(x)
        lora_update = x @ self.lora_A.t()
        lora_update = lora_update @ self.lora_B.t()

        if self.gated:
            # Basic gating
            if self.gated_type == "basic":
                return out + torch.sigmoid(self.gate) * self.alpha * lora_update
            # Input Adaptive Gating
            elif self.gated_type == "input_ada":
                s = lora_update.mean(dim=1, keepdim=True)  # [B, 1]
                gate = torch.sigmoid(self.gate_fc(s))      # [B, 1]
                return out + gate * self.alpha * lora_update
            # Input Adaptive Channel Gating
            elif self.gated_type == "input_ada_channel":
                gate_weight = torch.sigmoid(self.gate_fc(lora_update))
                return out + gate_weight * self.alpha * lora_update
            else:
                raise ValueError(
                    f"Unknown gated_type: '{self.gated_type}'. "
                    f"Supported gated_type values are: basic, input_ada and input_ada_channel" )
            
        else:
            return out + self.alpha * lora_update


def inject_lora(module, rank=4, alpha=1.0, gated=False, gated_type="basic", freeze_norm=True):
    """
    Recursively injects LoRA layers into convolutional and transposed convolutional layers
    of the given module, enabling efficient low-rank adaptation for fine-tuning.

    Parameters:
        module (torch.nn.Module):
            The input PyTorch module (e.g., a model or submodule) in which LoRA layers
            will be injected. All nn.Conv2d and nn.ConvTranspose2d layers will be replaced.

        rank (int, default=4):
            The rank of the low-rank decomposition in LoRA layers. Higher rank increases
            the capacity and number of trainable parameters.

        alpha (float, default=1.0):
            Scaling factor for the LoRA update. The effective weight update is scaled by
            alpha / rank for numerical stability.

        gated (bool, default=False):
            Whether to enable a learnable gating mechanism that controls the influence
            of the LoRA update. If True, each LoRA module includes a scalar gate parameter.

        gated_type (str, default="basic"):
            The type of gating mechanism. 
            "basic" denotes a basic gate controlled by a single learnable scalar parameter.
            "input_ada" denotes an input-adaptive gate, where the gating strength is dynamically determined conditioned on the input.
            "input_ada_channel" denotes an input-adaptive channel-wise gate, which learns per-channel gating weights conditioned on the input.

        freeze_norm (bool, default=True):
            If True, normalization layers (BatchNorm, LayerNorm, InstanceNorm, GroupNorm)
            will be frozen, preventing their parameters from updating during training.
    """
    
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            setattr(module, name, LoRAConv2d(child, rank, alpha, gated, gated_type))

        elif isinstance(child, nn.ConvTranspose2d):
            setattr(module, name, LoRAConvTranspose2d(child, rank, alpha, gated, gated_type))

        elif isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank, alpha, gated, gated_type))
        else:
            # If a normalized layer, whether to freeze parameters is controlled by freeze_norm.
            if freeze_norm and isinstance(child, (nn.BatchNorm2d, nn.LayerNorm, nn.InstanceNorm2d, nn.GroupNorm)):
                for param in child.parameters():
                    param.requires_grad = False
        
            inject_lora(child, rank=rank, alpha=alpha, gated=gated, gated_type=gated_type) 
