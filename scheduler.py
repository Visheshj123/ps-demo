from typing import OrderedDict, Tuple, List
import numpy as np
import ray
import torch
from ps import (
    ParameterServer,
    ParameterServerManager,
    ParameterServerMultiple,
    Worker,
    WorkerMultiple,
)
from config import DATASET_LEN, ITERATIONS, NSERVERS, NWORKERS
from model import Model
from data import get_data_loader


def evaluate(model, test_loader):
    """Evaluates the accuracy of the model on a validation dataset."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # This is only set to finish evaluation faster.
            if batch_idx * len(data) > 1024:
                break
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100.0 * correct / total


def single_server_asynchronous_training(lr):
    ray.init(ignore_reinit_error=True, _enable_object_reconstruction=True)
    server = ParameterServer.remote(lr)

    sample_idx = np.random.choice(DATASET_LEN,DATASET_LEN, replace=False)
    batches = np.array_split(sample_idx,NWORKERS) #splits indicies into evenly sized arrays
    #generate NWORKERS number of evenly split datasets
    workers = [Worker.remote(batches[i]) for i in range(NWORKERS)]


    # for evaluating accuracy
    model = Model()
    #TODO: pass in subset indicies to workers
    _, test_loader = get_data_loader()

    # get initial weights w_0
    # current_weights is a object reference
    # object reference can be passed to ray.get
    # object reference can be passed as an argument to another Ray remote function
    # get_weights task will be executed on the server process
    current_weights = server.get_weights.remote()

    # maintain workers' reference in this dict
    # ray._raylet.ObjectRef -> ray.actor.ActorHandle
    gradients = {}
    for worker in workers:
        gradients[worker.compute_gradients.remote(current_weights, 0)] = worker

    for i in range(ITERATIONS * NWORKERS):
        ready_gradient_list, _ = ray.wait(list(gradients))
        ready_gradient_id = ready_gradient_list[0]
        worker = gradients.pop(ready_gradient_id)

        # Compute and apply gradients
        current_weights = server.apply_gradients.remote(ready_gradient_id)
        gradients[worker.compute_gradients.remote(current_weights, 0)] = worker

        if i % 10 == 0:
            # Evaluate the current model after every 10 updates.
            model.set_weights(ray.get(current_weights))
            accuracy = evaluate(model, test_loader)
            print("Iter {}: \taccuracy is {:.1f}".format(i, accuracy))

    # Clean up Ray resources and processes
    ray.shutdown()


def flatten_weights(
    weights: OrderedDict[str, torch.Tensor], layer_shapes: List[Tuple[str, torch.Size]]
) -> torch.Tensor:
    flat_weights = torch.tensor([])
    for layer_name, _ in layer_shapes:
        flat_weights = torch.cat([flat_weights, weights[layer_name].flatten()], dim=0)
    return flat_weights


def flatten_gradients(gradients: List[np.ndarray]) -> np.ndarray:
    return np.concatenate([g.flatten() for g in gradients], axis=0)


def multiple_server_asynchronous_training(lr):
    ray.init()

    # setup workers and servers
    servers = [ParameterServerMultiple.remote(lr) for _ in range(NSERVERS)]
    servers = {str(server._actor_id.hex()): server for server in servers}
    server_ids = list(servers)
    server_manager = ParameterServerManager.remote(server_ids)
    sample_idx = np.random.choice(DATASET_LEN,DATASET_LEN, replace=False) #indicies of training set
    batches = np.array_split(sample_idx,NWORKERS) #splits indicies into evenly sized arrays
    workers = [WorkerMultiple.remote(batches[i]) for i in range(NWORKERS)]

    # setup initial weights
    model = Model()
    _, test_loader = get_data_loader()

    # initial weights
    current_weights = model.get_weights()
    layer_shapes = [(k, current_weights[k].shape) for k in current_weights]
    flat_weights = flatten_weights(current_weights, layer_shapes)
    num_parameters = len(flat_weights)

    # make partitions on weights
    # partitions[server_id] -> partition
    partitions = ray.get(server_manager.split_weights.remote(num_parameters))

    indices: List[List[int]] = []
    for server_id in server_ids:
        partition = partitions[server_id]
        indices.append(partition)
        parameter = torch.take(flat_weights, torch.tensor(partition))
        servers[server_id].set_weights.remote(parameter)

    # worker pull weights and compute gradients
    weights = ray.get(
        [servers[server_id].get_weights.remote() for server_id in server_ids]
    )

    gradients = {}
    for worker in workers:
        worker.pull_weights.remote(indices, weights, layer_shapes)
        gradients[worker.compute_gradients.remote()] = worker

    for i in range(ITERATIONS * NWORKERS):
        ready_gradient_list, _ = ray.wait(list(gradients))
        ready_gradient_id = ready_gradient_list[0]
        worker = gradients.pop(ready_gradient_id)

        # Compute and apply gradients
        g = ray.get(ready_gradient_id)
        flat_gradients = flatten_gradients(g)
        weights = ray.get(
            [
                servers[server_id].apply_gradients.remote(
                    flat_gradients.take(partitions[server_id])
                )
                for server_id in server_ids
            ]
        )
        # worker pull updated weights
        for w in workers:
            w.pull_weights.remote(indices, weights, layer_shapes)
        gradients[worker.compute_gradients.remote()] = worker

        if i % 10 == 0:
            # Evaluate the current model after every 10 updates.
            current_weights = ray.get(worker.get_weights.remote())
            model.set_weights(current_weights)
            accuracy = evaluate(model, test_loader)
            print("Iter {}: \taccuracy is {:.1f}".format(i, accuracy))

    # Clean up Ray resources and processes
    ray.shutdown()
