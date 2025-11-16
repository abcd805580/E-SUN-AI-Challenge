import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class SAGEBlock(nn.Module):
    """
    一個包含 SAGEConv, BatchNorm, ReLU, Dropout 和殘差連接 (Residual Connection)
    的 GraphSAGE 區塊。
    """

    def __init__(self, indim, outdim, drop=0.3):
        """
        Args:
            indim (int): 輸入特徵維度。
            outdim (int): 輸出特徵維度。
            drop (float): Dropout 比例。
        """
        super().__init__()
        self.conv = SAGEConv(indim, outdim)
        self.bn = nn.BatchNorm1d(outdim)
        self.drop = nn.Dropout(drop)
        self.res = None if indim == outdim else nn.Linear(indim, outdim, bias=False)

    def forward(self, x, ei):
        """
        前向傳播。

        Args:
            x (torch.Tensor): 輸入節點特徵。
            ei (torch.Tensor): 邊索引。

        Returns:
            torch.Tensor: 輸出節點特徵。
        """
        h = F.relu(self.bn(self.conv(x, ei)))
        h = self.drop(h)
        if self.res:
            x = self.res(x)
        return h + x


class SAGEModel(nn.Module):
    """
    由三個 SAGEBlock 堆疊而成的主模型，最後接一個 Linear 輸出層。
    """

    def __init__(self, in_dim, hid=512, drop=0.35):
        """
        Args:
            in_dim (int): 原始輸入特徵維度。
            hid (int): 隱藏層維度。
            drop (float): SAGEBlock 中的 Dropout 比例。
        """
        super().__init__()
        self.b1 = SAGEBlock(in_dim, hid, drop)
        self.b2 = SAGEBlock(hid, hid, drop)
        self.b3 = SAGEBlock(hid, hid, drop)
        self.out = nn.Linear(hid, 1)

    def forward(self, x, ei):
        """
        前向傳播。

        Args:
            x (torch.Tensor): 輸入節點特徵。
            ei (torch.Tensor): 邊索引。

        Returns:
            torch.Tensor: 每個節點的原始 logit 輸出 (shape: [N])。
        """
        return self.out(self.b3(self.b2(self.b1(x, ei), ei), ei)).squeeze(-1)
