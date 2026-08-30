import torch
import torch.nn as nn


class SemanticFPNHead(nn.Module):
    """
    Lightweight segmentation head for Wan video model.
    Similar to MagicMotion's segmentation head, but adapted for Wan's architecture.
    """
    def __init__(self, in_channels, out_channels=2, num_groups=16, num_tensors=18, patch_size=(1, 2, 2)):
        """
        Args:
            in_channels (int): Number of input channels for each tensor in the list.
            out_channels (int): Number of output channels for the final tensor. Default is 2.
            num_groups (int): Number of groups for GroupNorm.
            num_tensors (int): Number of input tensors (number of transformer layers).
            patch_size (tuple): Patch size (t, h, w) for upsampling.
        """
        super(SemanticFPNHead, self).__init__()
        hidden_dim = 64
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=min(num_groups, hidden_dim), num_channels=hidden_dim),
                    nn.ReLU(inplace=True),
                ) 
                for i in range(num_tensors)
            ]
        )
        self.conv_out = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)
        # For Wan model, patch_size is (t, h, w), we only need to upsample spatial dimensions
        # nn.Upsample for 2D expects scale_factor as a number or tuple (h, w)
        # Ensure we pass a tuple for spatial dimensions
        if isinstance(patch_size, (tuple, list)) and len(patch_size) >= 3:
            spatial_scale = (patch_size[1], patch_size[2])  # (h, w)
        else:
            # Fallback: assume patch_size is already spatial scale or use default
            spatial_scale = patch_size if isinstance(patch_size, (int, float, tuple)) else (2, 2)
        self.upsample = nn.Upsample(scale_factor=spatial_scale, mode='bilinear', align_corners=False)

    def forward(self, tensor_list):
        """
        Args:
            tensor_list (list of torch.Tensor): List of tensors with shape [B*T, C, H, W].
        
        Returns:
            torch.Tensor: A tensor with shape [B*T, C_out, H*scale, W*scale].
        """
        if len(tensor_list) != len(self.convs):
            raise ValueError(
                f"Number of input tensors ({len(tensor_list)}) must match the number of conv layers ({len(self.convs)})."
            )
        
        convolved_tensors = []
        for conv, tensor in zip(self.convs, tensor_list):
            tensor = conv(tensor)
            convolved_tensors.append(tensor)

        summed_tensor = torch.sum(torch.stack(convolved_tensors), dim=0)
        summed_tensor = self.upsample(summed_tensor)
        output = self.conv_out(summed_tensor)
        return output


