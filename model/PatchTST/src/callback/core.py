# Vendored from the official PatchTST implementation
# (https://github.com/yuqinie98/PatchTST), licensed under Apache License 2.0;
# see LICENSE-Apache-2.0. Modified for this repository (see THIRD_PARTY.md).

__all__ = ['Callback', 'SetupLearnerCB', 'GetPredictionsCB', 'GetTestCB' ]


""" 
Callback lists:
    > before_fit
        - before_epoch
            + before_epoch_train                
                ~ before_batch_train
                ~ after_batch_train                
            + after_epoch_train

            + before_epoch_valid                
                ~ before_batch_valid
                ~ after_batch_valid                
            + after_epoch_valid
        - after_epoch
    > after_fit

    - before_predict        
        ~ before_batch_predict
        ~ after_batch_predict          
    - after_predict

"""

from ..basics import *
import torch

DTYPE = torch.float32

class Callback(GetAttr): 
    _default='learner'


class SetupLearnerCB(Callback): 
    def __init__(self):        
        self.device = default_device(use_cuda=True)

    def before_batch_train(self): self._to_device()
    def before_batch_valid(self): self._to_device()
    def before_batch_predict(self): self._to_device()
    def before_batch_test(self): self._to_device()

    def _to_device(self):
        batch = to_device(self.batch, self.device)        
        if self.n_inp > 1: xb, yb = batch
        else: xb, yb = batch, None        
        self.learner.batch = xb, yb
        
    def before_fit(self): 
        "Set model to cuda before training"                
        self.learner.model.to(self.device)
        self.learner.device = self.device                        


class GetPredictionsCB(Callback):
    def __init__(self):
        super().__init__()

    def before_predict(self):
        self.preds = []        
    
    def after_batch_predict(self):
        # append the prediction after each forward batch (offload to CPU so the
        # full test set is not accumulated on the GPU)
        self.preds.append(self.pred.detach().cpu())

    def after_predict(self):
        self.preds = torch.concat(self.preds)

         

class GetTestCB(Callback):
    def __init__(self):
        super().__init__()

    def before_test(self):
        self.preds, self.targets = [], []        
    
    def after_batch_test(self):
        # append the prediction after each forward batch (offload to CPU so the
        # full test set is not accumulated on the GPU)
        self.preds.append(self.pred.detach().cpu())
        self.targets.append(self.yb.detach().cpu())

    def after_test(self):
        self.preds = torch.concat(self.preds)
        self.targets = torch.concat(self.targets)


