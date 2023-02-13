import numpy as np
import torch

from ..builder import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class FusionRecognizer(BaseRecognizer):
    """Recognizer used for the fused model

    Args:
        backbone_cnn (dict): 3D-CNN backbone modules to extract feature.
        backbone_gcn (dict): GCN backbone modules to extract feature.
        cls_head (dict | None): Classification head to process feature. Default: None.
        train_cfg (dict): Config for training. Default: {}.
        test_cfg (dict): Config for testing. Default: {}.
    """

    def __init__(self,
                 backbone_cnn,
                 backbone_gcn,
                 cls_head=None,
                 train_cfg=dict(),
                 test_cfg=dict()):
        super().__init__()

        # record the source of the backbone
        self.backbone_cnn = builder.build_backbone(backbone_cnn)
        self.backbone_gcn = builder.build_backbone(backbone_gcn)

        self.cls_head = builder.build_head(cls_head) if cls_head else None

        if train_cfg is None:
            train_cfg = dict()
        if test_cfg is None:
            test_cfg = dict()

        assert isinstance(train_cfg, dict)
        assert isinstance(test_cfg, dict)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # max_testing_views should be int
        self.max_testing_views = test_cfg.get('max_testing_views', None)
        self.init_weights()

    def init_weights(self):
        """Initialize the model network weights."""
        self.backbone_cnn.init_weights()
        self.backbone_gcn.init_weights()
        if self.with_cls_head:
            self.cls_head.init_weights()


    def forward_train(self, imgs, keypoint, label, **kwargs):
        """Defines the computation performed at every call when training."""

        labels = label
        assert self.with_cls_head

        # for cnn
        imgs = imgs.reshape((-1,) + imgs.shape[2:])
        # for gcn
        assert keypoint.shape[1] == 1
        keypoint = keypoint[:, 0]

        losses = dict()

        # We extract feats from both backbones
        # We directly use backbone instead of extract_feats
        x_3d = self.backbone_cnn(imgs)
        x_gcn = self.backbone_gcn(keypoint)

        cls_scores = self.cls_head((x_3d, x_gcn))

        gt_labels = labels.squeeze()

        """
                        ##########################################
                        ########## for separate losses ##########
                        ##########################################
                        
        loss_components = self.cls_head.loss_components
        # Weights between 3d and gcn (default=1)
        loss_weights = self.cls_head.loss_weights

        for loss_name, weight in zip(loss_components, loss_weights):
            cls_score = cls_scores[loss_name]
            loss_cls = self.cls_head.loss(cls_score, gt_labels) # from BaseHead
            loss_cls = {loss_name + '_' + k: v for k, v in loss_cls.items()}
            loss_cls[f'{loss_name}_loss_cls'] *= weight
            losses.update(loss_cls)

        """

        loss_cls = self.cls_head.loss(cls_score, gt_label, **kwargs)
        losses.update(loss_cls)

        return losses

    def forward_test(self, imgs, keypoint, **kwargs):
        """Defines the computation performed at every call when evaluation and
        testing."""



        """
                        ##########################################
                        ########## for separate scores ##########
                        ##########################################
        # for cnn
        batches = imgs.shape[0]
        num_segs = imgs.shape[1]
        imgs = imgs.reshape((-1,) + imgs.shape[2:])
        # for gcn
        bs, nc = keypoint.shape[:2]
        keypoint = keypoint.reshape((bs * nc,) + keypoint.shape[2:])
        """


        #feat extraction
        x_3d = self.backbone_cnn(imgs)
        x_gcn = self.backbone_gcn(keypoint)

        # ... for feat_ext,pool_opt,score_ext cases see recognizergcn.py, recognizer3d.py ...

        # should have cls_head if not extracting features
        assert self.with_cls_head
        cls_scores = self.cls_head((x_3d, x_gcn))


        """
                ##########################################
                ########## for separate scores ##########
                ##########################################
        # cnn: cls_score = cls_score.reshape(batches, num_segs, cls_score.shape[-1])
        # gcn: cls_score = cls_score.reshape(bs, nc, cls_score.shape[-1])
        cls_scores[0] = cls_scores[0].reshape(batches, num_segs, cls_scores[0].shape[-1])
        cls_scores[1] = cls_scores[1].reshape(bs, nc, cls_scores[1].shape[-1])

        for k in cls_scores:
            cls_score = self.average_clip(cls_scores[k][None])
            cls_scores[k] = cls_score.data.cpu().numpy()[0]
        
        """

        cls_scores = self.average_clip(cls_scores)
        cls_scores= cls_scores.cpu().numpy()


        return [cls_scores]

    def forward(self, imgs, keypoint, label=None, return_loss=True, **kwargs):
        """Define the computation performed at every call."""
        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            return self.forward_train(imgs, keypoint, label, **kwargs)

        return self.forward_test(imgs, keypoint, **kwargs)

