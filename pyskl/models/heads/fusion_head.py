import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import HEADS
from .base import BaseHead

@HEADS.register_module()
class FusionHead(BaseHead):
    """The classification head for fusion model.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (tuple[int]): Number of channels in input feature.
        loss_cls (dict): Config for building loss. Default: dict(type='CrossEntropyLoss').
        dropout (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initializ the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 loss_components=['cnn', 'gcn'],
                 loss_weights=1.,
                 dropout=0.5,
                 init_std=0.01,
                 **kwargs):

        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        if isinstance(dropout, float):
            dropout = {'cnn': dropout, 'gcn': dropout}
        assert isinstance(dropout, dict)

        self.dropout = dropout
        self.init_std = init_std
        self.in_channels = in_channels

        self.loss_components = loss_components
        if isinstance(loss_weights, float):
            loss_weights = [loss_weights] * len(loss_components)
        assert len(loss_weights) == len(loss_components)
        self.loss_weights = loss_weights

        self.dropout_cnn = nn.Dropout(p=self.dropout['cnn'])
        self.dropout_gcn = nn.Dropout(p=self.dropout['gcn'])

        self.fc_cnn = nn.Linear(in_channels[0], num_classes)
        self.fc_gcn = nn.Linear(in_channels[1], num_classes)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cnn, std=self.init_std)
        normal_init(self.fc_gcn, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """

        x_cnn = x[0]
        x_gcn = x[1]

        ################# CNN #################
        pool_cnn = nn.AdaptiveAvgPool3d(1)
        if isinstance(x_cnn, tuple) or isinstance(x_cnn, list):
            x_cnn = torch.cat(x_cnn, dim=1)
        x_cnn = pool_cnn(x_cnn)
        x_cnn = x_cnn.view(x_cnn.shape[:2])
        #######################################

        ################# GCN #################
        pool_gcn = nn.AdaptiveAvgPool2d(1)
        N, M, C, T, V = x_gcn.shape
        x_gcn = x_gcn.reshape(N * M, C, T, V)
        x_gcn = pool_gcn(x_gcn)
        x_gcn = x_gcn.reshape(N, M, C)
        x_gcn = x_gcn.mean(dim=1)
        #######################################

        x_cnn = self.dropout_cnn(x_cnn)
        x_gcn = self.dropout_gcn(x_gcn)

        cls_scores = {}
        cls_scores['cnn'] = self.fc_cnn(x_cnn)
        cls_scores['gcn'] = self.fc_gcn(x_gcn)

        return cls_scores
