import torch
from torch import nn
import torch.nn.functional as F

class LaneDetectionHead3D(nn.Module):
    def __init__(self, in_channels, bev_h, bev_w, anchors, num_queries, K, num_category=21):
        super().__init__()
        self.bev_h, self.bev_w = bev_h, bev_w
        self.num_anchors = num_queries
        self.K = K
        self.num_category = num_category
        self.out_channels = K * 3 + num_category
        self.anchors = anchors

        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((1, self.num_anchors))
        self.fc = nn.Conv2d(64, self.out_channels, kernel_size=1)

        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv2.bias, 0)
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='linear')
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.bev_h and W == self.bev_w
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.adaptive_pool(x)
        x = self.fc(x).squeeze(2)

        x_offsets = torch.tanh(x[:, 0:self.K, :]) * 10.0
        z_offsets = torch.tanh(x[:, self.K:2*self.K, :]) * 3.0
        vis_logits = x[:, 2*self.K:3*self.K, :]
        category_probs = F.softmax(x[:, 3*self.K:, :], dim=1)
        return torch.cat([x_offsets, z_offsets, vis_logits, category_probs], dim=1)

    def decode_predictions(self, outputs):
        B, _, num_anchors = outputs.shape
        batch_decoded_lanes = []
        batch_confidences = []
        for b in range(B):
            decoded_lanes = []
            confidences = []
            for n in range(num_anchors):
                lane_data = outputs[b, :, n]
                x_coords = lane_data[0:self.K] + self.anchors[n]
                z_coords = lane_data[self.K:2*self.K]
                vis_logits = lane_data[2*self.K:3*self.K]
                category_probs = lane_data[3*self.K:]
                vis_binary = (torch.sigmoid(vis_logits) > 0.5).float()
                lane_line = torch.cat([x_coords, z_coords, vis_binary, category_probs], dim=0)
                decoded_lanes.append(lane_line)
                confidences.append(vis_binary.mean())
            batch_decoded_lanes.append(torch.stack(decoded_lanes))  # [num_anchors, D]
            batch_confidences.append(torch.stack(confidences))      # [num_anchors]
        return torch.stack(batch_decoded_lanes), torch.stack(batch_confidences)
    
    