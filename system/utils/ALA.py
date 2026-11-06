import numpy as np
import torch
import torch.nn as nn
import copy
import random
from torch.utils.data import DataLoader
from typing import List, Tuple


class ALA:
    def __init__(self,
                 cid: int,
                 loss: nn.Module,
                 train_data: List[Tuple],
                 batch_size: int,
                 rand_percent: int,
                 layer_idx: int = 0,
                 eta: float = 1.0,
                 device: str = 'cpu',
                 threshold: float = 0.1,
                 num_pre_loss: int = 10) -> None:
        """
         Initialize ALA module

        Args:
            cid: Client ID.
            loss: The loss function.
            train_data: The reference of the local training data.
            batch_size: Weight learning batch size.
            rand_percent: The percent of the local training data to sample.
            layer_idx: Control the weight range. By default, all the layers are selected. Default: 0
            eta: Weight learning rate. Default: 1.0
            device: Using cuda or cpu. Default: 'cpu'
            threshold: Train the weight until the standard deviation of the recorded losses is less than a given threshold. Default: 0.1
            num_pre_loss: The number of the recorded losses to be considered to calculate the standard deviation. Default: 10

        Returns:
            None.
        """

        self.cid = cid
        self.loss = loss
        self.train_data = train_data
        self.batch_size = batch_size
        self.rand_percent = rand_percent
        self.layer_idx = layer_idx
        self.eta = eta
        self.threshold = threshold
        self.num_pre_loss = num_pre_loss
        self.device = device

        self.weights = None  # Learnable local aggregation weights.
        self.start_phase = True

    def adaptive_local_aggregation(self,
                                   global_model: nn.Module,
                                   local_model: nn.Module) -> None:
        """
        Generates the Dataloader for the randomly sampled local training data and
        preserves the lower layers of the update.

        Args:
            global_model: The received global/aggregated model.
            local_model: The trained local model.

        Returns:
            None.
        """

        # randomly sample partial local training data
        rand_ratio = self.rand_percent / 100
        rand_num = int(rand_ratio * len(self.train_data))
        rand_idx = random.randint(0, len(self.train_data) - rand_num)
        rand_loader = DataLoader(self.train_data[rand_idx:rand_idx + rand_num], self.batch_size, drop_last=False)

        # obtain the references of the parameters
        params_g = list(global_model.parameters())
        params = list(local_model.parameters())

        # deactivate ALA at the 1st communication iteration
        if torch.sum(params_g[0] - params[0]) == 0:
            return

        # preserve all the updates in the lower layers
        for param, param_g in zip(params[:-self.layer_idx], params_g[:-self.layer_idx]):
            param.data = param_g.data.clone()

        # temp local model only for weight learning
        model_t = copy.deepcopy(local_model)
        params_t = list(model_t.parameters())

        # only consider higher layers
        params_p = params[-self.layer_idx:]
        params_gp = params_g[-self.layer_idx:]
        params_tp = params_t[-self.layer_idx:]

        # frozen the lower layers to reduce computational cost in Pytorch
        for param in params_t[:-self.layer_idx]:
            param.requires_grad = False

        # used to obtain the gradient of higher layers
        # no need to use optimizer.step(), so lr=0
        optimizer = torch.optim.SGD(params_tp, lr=0)

        # initialize the weight to all ones in the beginning
        if self.weights == None:
            self.weights = [torch.ones_like(param.data).to(self.device) for param in params_p]

        # initialize the higher layers in the temp local model
        for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp,
                                                   self.weights):
            param_t.data = param + (param_g - param) * weight

        # weight learning
        losses = []  # record losses
        cnt = 0  # weight training iteration counter
        while True:
            
            for x, y in rand_loader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                output = model_t(x)
                if isinstance(output, tuple):
                    output = output[0]
                loss_value = self.loss(output, y)  # modify according to the local objective
                loss_value.backward()

                # update weight in this batch
                for param_t, param, param_g, weight in zip(params_tp, params_p,
                                                           params_gp, self.weights):
                    weight.data = torch.clamp(
                        weight - self.eta * (param_t.grad * (param_g - param)), 0, 1)

                # update temp local model in this batch
                for param_t, param, param_g, weight in zip(params_tp, params_p,
                                                           params_gp, self.weights):
                    param_t.data = param + (param_g - param) * weight

            losses.append(loss_value.item())
            cnt += 1

            # only train one epoch in the subsequent iterations
            if not self.start_phase:
                break

            # train the weight until convergence
            if len(losses) > self.num_pre_loss and np.std(losses[-self.num_pre_loss:]) < self.threshold:
                print('Client:', self.cid, '\tStd:', np.std(losses[-self.num_pre_loss:]),
                      '\tALA epochs:', cnt)
                break

        self.start_phase = False

        # obtain initialized local model
        for param, param_t in zip(params_p, params_tp):
            param.data = param_t.data.clone()
# import numpy as np
# import torch
# import torch.nn as nn
# import copy
# import random
# from torch.utils.data import DataLoader
# from typing import List, Tuple


# class ALA:
#     def __init__(self,
#                  cid: int,
#                  loss: nn.Module,
#                  train_data: List[Tuple],
#                  batch_size: int,
#                  rand_percent: int,
#                  layer_idx: int = 1,      # Số layer ở encoder muốn cập nhật (thường là 1 để chỉ encoder)
#                  eta: float = 1.0,
#                  device: str = 'cpu',
#                  threshold: float = 0.1,
#                  num_pre_loss: int = 10) -> None:
#         """
#         Initialize ALA module

#         Args:
#             cid: Client ID.
#             loss: The loss function.
#             train_data: The reference of the local training data.
#             batch_size: Weight learning batch size.
#             rand_percent: The percent of the local training data to sample.
#             layer_idx: Number of top layers to update (encoder layers).
#             eta: Weight learning rate.
#             device: 'cpu' or 'cuda'.
#             threshold: Stopping threshold for std of recent losses.
#             num_pre_loss: Number of recent losses to compute std.

#         Returns:
#             None.
#         """
#         self.cid = cid
#         self.loss = loss
#         self.train_data = train_data
#         self.batch_size = batch_size
#         self.rand_percent = rand_percent
#         self.layer_idx = layer_idx
#         self.eta = eta
#         self.threshold = threshold
#         self.num_pre_loss = num_pre_loss
#         self.device = device

#         self.weights = None  # Learnable weights for encoder layers.
#         self.start_phase = True

#     def adaptive_local_aggregation(self,
#                                    global_encoder: nn.Module,
#                                    local_encoder: nn.Module) -> None:
#         """
#         Adaptively learn weights for aggregating encoder parameters.

#         Args:
#             global_encoder: Global encoder model (from server).
#             local_encoder: Local encoder model (client side).

#         Returns:
#             None
#         """
#         # Sample partial local data randomly
#         rand_ratio = self.rand_percent / 100
#         rand_num = max(1, int(rand_ratio * len(self.train_data)))  # at least 1 sample
#         rand_start = random.randint(0, len(self.train_data) - rand_num)
#         rand_loader = DataLoader(self.train_data[rand_start:rand_start + rand_num],
#                                  batch_size=self.batch_size,
#                                  drop_last=False)

#         # Get parameters list of global and local encoder
#         params_g = list(global_encoder.parameters())
#         params_l = list(local_encoder.parameters())

#         # Check if global and local encoder have same device
#         for p in params_g:
#             if p.device != torch.device(self.device):
#                 p.data = p.data.to(self.device)
#         for p in params_l:
#             if p.device != torch.device(self.device):
#                 p.data = p.data.to(self.device)

#         # If global and local encoder params are identical => disable ALA for first round
#         diff = sum(torch.sum((pg - pl).abs()) for pg, pl in zip(params_g, params_l))
#         if diff == 0:
#             return  # Skip ALA in first communication round

#         # Freeze lower layers except the top layer_idx layers (which we update weights)
#         # For encoder, we assume layer_idx is number of last layers to update weights on
#         lower_layers = params_l[:-self.layer_idx] if self.layer_idx > 0 else params_l
#         top_layers_g = params_g[-self.layer_idx:] if self.layer_idx > 0 else []
#         top_layers_l = params_l[-self.layer_idx:] if self.layer_idx > 0 else []

#         # Preserve lower layers by copying global params to local encoder
#         for local_param, global_param in zip(lower_layers, params_g[:len(lower_layers)]):
#             local_param.data = global_param.data.clone()

#         # Create a temporary encoder model to update weights
#         model_temp = copy.deepcopy(local_encoder)
#         params_temp = list(model_temp.parameters())

#         # Freeze all but top layers in temp model
#         for param in params_temp[:-self.layer_idx]:
#             param.requires_grad = False

#         top_layers_temp = params_temp[-self.layer_idx:] if self.layer_idx > 0 else []

#         # Initialize learnable weights if not done yet
#         if self.weights is None:
#             self.weights = [torch.ones_like(p.data, device=self.device) for p in top_layers_l]

#         # Initialize temp top layers using weighted interpolation between local and global
#         for pt, pl, pg, w in zip(top_layers_temp, top_layers_l, top_layers_g, self.weights):
#             pt.data = pl.data + (pg.data - pl.data) * w.data

#         optimizer = torch.optim.SGD(top_layers_temp, lr=0)  # lr=0 because update by weights, not params

#         losses = []
#         iteration = 0

#         while True:
#             for x, y in rand_loader:
#                 if isinstance(x, (list, tuple)):
#                     x = x[0]
#                 x = x.to(self.device)
#                 y = y.to(self.device)

#                 optimizer.zero_grad()
#                 output = model_temp(x)
#                 if isinstance(output, tuple):
#                     output = output[0]

#                 loss = self.loss(output, y)
#                 loss.backward()

#                 # Update weights with gradient descent step + clamp between 0 and 1
#                 for pt, pl, pg, w in zip(top_layers_temp, top_layers_l, top_layers_g, self.weights):
#                     grad = pt.grad.data * (pg.data - pl.data)
#                     w.data = torch.clamp(w.data - self.eta * grad, 0, 1)

#                 # Update temp model params by new weights
#                 for pt, pl, pg, w in zip(top_layers_temp, top_layers_l, top_layers_g, self.weights):
#                     pt.data = pl.data + (pg.data - pl.data) * w.data

#             losses.append(loss.item())
#             iteration += 1

#             if not self.start_phase:
#                 break

#             if len(losses) > self.num_pre_loss and np.std(losses[-self.num_pre_loss:]) < self.threshold:
#                 print(f'Client: {self.cid}\tALA converged at epoch: {iteration}\tStd of loss: {np.std(losses[-self.num_pre_loss:]):.6f}')
#                 break

#         self.start_phase = False

#         # Finally update local encoder's top layers by weighted params from temp model
#         for pl, pt in zip(top_layers_l, top_layers_temp):
#             pl.data = pt.data.clone()

