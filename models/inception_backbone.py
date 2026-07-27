import torch
import torch.nn as nn

class InceptionBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_sizes=(3, 7, 15),
                 bottleneck_ratio=0.5, stride=1):
        super().__init__()
        assert all(k > 0 and k % 2 == 1 for k in kernel_sizes), "Use odd kernels."
        num_branches = len(kernel_sizes) + 1  
        assert out_ch % num_branches == 0, "out_ch must be divisible by number of branches."
        branch_ch = out_ch // num_branches

        bottleneck_ch = max(1, int(in_ch * bottleneck_ratio)) if bottleneck_ratio > 0 else in_ch

        self.conv_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_ch, bottleneck_ch, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv1d(bottleneck_ch, branch_ch, kernel_size=k,
                          stride=stride, padding=k // 2, bias=False),
            ) for k in kernel_sizes
        ])

        self.pool = nn.AvgPool1d(kernel_size=3, stride=stride, padding=1)
        self.pool_proj = nn.Conv1d(in_ch, branch_ch, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        ys = [b(x) for b in self.conv_branches]
        y_pool = self.pool_proj(self.pool(x))
        y = torch.cat(ys + [y_pool], dim=1)  
        y = self.bn(y)
        return self.act(y)


class InceptionBackbone1D(nn.Module):
    """
    Pure 1D Inception feature extractor.
    Outputs a feature vector of shape [B, 512] with the new defaults.
    """
    def __init__(self, in_ch=1,
                 channels=(64, 128, 256, 512),      
                 blocks_per_stage=(2, 2, 2, 2),     
                 kernel_sizes=(3, 7, 15),
                 bottleneck_ratio=0.5):
        super().__init__()
        assert len(channels) == len(blocks_per_stage)

        # Stem
        stem_out = channels[0] // 2
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, stem_out, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(stem_out),
            nn.ReLU(inplace=True),
        )
        curr_ch = stem_out

        # Inception Stages
        stages = []
        for si, out_ch in enumerate(channels):
            stage_stride = 2 if si > 0 else 1  
            blocks = []
            for bi in range(blocks_per_stage[si]):
                s = stage_stride if bi == 0 else 1
                blocks.append(
                    InceptionBlock1D(
                        in_ch=curr_ch,
                        out_ch=out_ch,
                        kernel_sizes=kernel_sizes,
                        bottleneck_ratio=bottleneck_ratio,
                        stride=s,
                    )
                )
                curr_ch = out_ch
            stages.append(nn.Sequential(*blocks))
        self.features = nn.Sequential(*stages)

        # Global Average Pooling & Flatten
        self.pool_and_flatten = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # [B, C, 1]
            nn.Flatten(),             # [B, C]
        )
        
        self.out_dim = curr_ch

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

        self.arch = "inception1d_backbone"

    def forward(self, x):  # x: [B, C, T]
        x = self.stem(x)
        x = self.features(x)
        feat = self.pool_and_flatten(x)
        return feat 
