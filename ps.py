from typing import Any, Callable, Dict, List, Optional, OrderedDict, Tuple, Union
import ray
import torch
import torch.nn.functional as F
import numpy as np
from model import Model
from consistent_hash import ConsistentHashingRing
from data import get_data_loader


@ray.remote
class Worker:
    """SGD worker:
    1. Read a minibatch X,Y
    2. Pull weights from server
    3. Compute gradients
    4. Push gradients to server
    """

    def __init__(self):
        self.model = Model()
        self.data_iterator = iter(get_data_loader()[0])  # train_loader

    def get_weights(self):
        """get weights for back up"""
        return self.model.get_weights()

    def compute_gradients(self, weights):
        self.model.set_weights(weights)
        try:
            data, target = next(self.data_iterator)
        except StopIteration:  # When the epoch ends, start a new epoch.
            self.data_iterator = iter(get_data_loader()[0])
            data, target = next(self.data_iterator)
        self.model.zero_grad()
        output = self.model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        return self.model.get_gradients()


@ray.remote
class ParameterServer:
    """Single parameter server scenario
    Once received gradients from worker,
    update weights: w = w - lr * grad
    """

    def __init__(self, lr):
        self.model = Model()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

    def apply_gradients(self, *gradients):
        stacked_graidents = [
            np.stack(gradient_zip).sum(axis=0) for gradient_zip in zip(*gradients)
        ]
        self.optimizer.zero_grad()
        self.model.set_gradients(stacked_graidents)
        self.optimizer.step()
        return self.model.get_weights()

    def get_weights(self):
        return self.model.get_weights()


@ray.remote
class WorkerMultiple:
    """
    Worker in multiple servers scenario
    """

    def __init__(self):
        self.model = Model()
        self.data_iterator = iter(get_data_loader()[0])  # train_loader

    def pull_weights(
        self,
        indices: List[List[int]],
        weights: List[torch.Tensor],
        layer_shapes: List[Tuple[str, torch.Size]],
    ):
        # reconstruct weights
        num_parameters = sum([len(index) for index in indices])
        flat_weights = torch.Tensor(num_parameters)

        for i, w in zip(indices, weights):
            idx = torch.tensor(i)
            flat_weights.put_(idx, w)

        flat_weights_slices: Dict[str, slice] = {}

        left = 0
        for layer_name, shape in layer_shapes:
            if len(shape) == 1:
                n_parameters = shape[0]
            elif len(shape) == 2:
                n_parameters = shape[0] * shape[1]

            flat_weights_slices[layer_name] = slice(left, left + n_parameters)
            left += n_parameters

        reconstructed_weights = OrderedDict()
        for layer_name, shape in layer_shapes:
            reconstructed_weights[layer_name] = flat_weights[
                flat_weights_slices[layer_name]
            ].reshape(shape)
        # load weights
        self.model.set_weights(reconstructed_weights)

    def compute_gradients(self):
        try:
            data, target = next(self.data_iterator)
        except StopIteration:  # When the epoch ends, start a new epoch.
            self.data_iterator = iter(get_data_loader()[0])
            data, target = next(self.data_iterator)
        self.model.zero_grad()
        output = self.model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        return self.model.get_gradients()

    def get_weights(self):
        return self.model.get_weights()


@ray.remote
class ParameterServerMultiple:
    """Multiple parameter server scenario"""

    def __init__(
        self,
        lr: Union[float, Callable[[], float]],
        weights: Optional[torch.Tensor] = None,
        index: Optional[List[int]] = None,
    ):
        """
        Args:
            lr: learning rate, float or a callable which determine lr
            based on iteration
            weights: partition weights
            index: partition index
        """
        self.lr = lr
        self.weights = weights
        self.index = index

    def apply_gradients(self, gradients: np.ndarray):
        if self.weights == None:
            raise Exception("weights are not initalized")
        lr = self.lr() if callable(self.lr) else self.lr
        self.weights -= lr * torch.tensor(gradients)
        return self.weights

    def get_weights(self):
        return self.weights

    def set_weights(self, weights: torch.Tensor):
        self.weights = weights

    def set_index(self, index: List[int]):
        self.index = index


@ray.remote
class ParameterServerManager:
    def __init__(self, server_ids: List[str]):
        """
        chunk the weights into #servers part and store in servers

        Args:
            server_ids: list of id of servers
            weights: initial weights
        Return:
            parameters_partition: server_id -> array of index of weights
        """
        self.ring = ConsistentHashingRing(server_ids)
        self.server_ids = server_ids

    def split_weights(self, num_parameters):
        # make partition using consistent hash ring
        parameters_partition: Dict[str, List[int]] = {
            server_id: [] for server_id in self.server_ids
        }
        for i in range(num_parameters):
            parameters_partition[self.assign_server(i)].append(i)

        return parameters_partition

    def get_server_ids(self):
        return self.server_ids

    def add_server(self, server_id: str):
        self.ring.add_node(server_id)
        self.server_ids.append(server_id)

    def remove_server(self, server_id: str):
        self.ring.remove_node(server_id)
        del self.server_ids[id]

    def assign_server(self, item: Any):
        return self.ring.get_node(str(item))
