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
class ParameterServerMultiple:
    """Multiple parameter server scenario"""
    def __init__(self, lr, weights):
        self.lr = lr
        self.weights = weights

    def apply_gradients(self, *gradients):
        self.weights -= self.lf * gradients
        return self.weights

    def get_weights(self):
        return self.weights


@ray.remote
class ParameterServerManager:
    def __init__(self, servers: ray.actor.ActorHandle):
        self.servers = {}
        ids = []
        for server in servers:
            id = server._actor_id.hex()
            ids.append(id)
            self.servers[id] = server
        self.ring = ConsistentHashingRing(ids)
    
    def get_weights():
        raise Exception("TODO")

    def get_servers(self):
        return self.servers

    def add_node(self, node: ray.actor.ActorHandle):
        id = node._actor_id.hex()
        if self.servers.get(id, None):
            raise Exception("server actor id collision")
        self.servers[id] = node
        self.ring.add_node(id)

    def remove_node(self, node: ray.actor.ActorHandle):
        id = node._actor_id.hex()
        if not self.servers.get(id, None):
            return
        del self.servers[id]
        self.ring.remove_node(id)

    def assign_server(self, weights):
        return self.ring.get_node(weights)
