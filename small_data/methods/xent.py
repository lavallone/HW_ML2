from torch import nn
from typing import Callable

from .common import BasicAugmentation


class CrossEntropyClassifier(BasicAugmentation):
    """ Standard cross-entropy classification as baseline.

    See `BasicAugmentation` for a documentation of the available hyper-parameters.
    """

    def get_loss_function(self) -> Callable:
        # here I can add the label smoothing parameter for regularization
        return nn.CrossEntropyLoss(reduction='mean', label_smoothing=self.hparams['label_smoothing'])

    @staticmethod
    def default_hparams() -> dict:

        return {
            **super(CrossEntropyClassifier, CrossEntropyClassifier).default_hparams(),
            'momentum' : 0.9,
            'nesterov' : False,
            'label_smoothing' : 0.0
        }