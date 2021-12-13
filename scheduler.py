import ray
import torch
from ps import ParameterServer, ParameterServerManager, ParameterServerMultiple, Worker
from config import ITERATIONS, NSERVERS, NWORKERS
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


def signle_server_asynchronous_training(lr):
    ray.init()
    server = ParameterServer.remote(lr)
    workers = [Worker.remote(i) for i in range(NWORKERS)]
    wids = [i for i in range(NWORKERS)]
    # for evaluating accuracy
    model = Model()
    _, test_loader = get_data_loader()

    # get initial weights w_0
    current_weights = server.get_weights.remote()

    #mapping from worker_id to worker
    gradients = {}  # ray._raylet.ObjectRef -> ray.actor.ActorHandle
    for worker,wid in zip(workers,wids):
        succeed = False
        wid = None
        while not succeed:
            try: 
                wid = worker.compute_gradients.remote(current_weights)
                gradients[wid] = worker
                succeed = True
            except (ray.exceptions.RayActorError, ray.exceptions.RayTaskError) as e:
                print("ERRRRRRRRRRRR")
                print(wid)
                new_worker = worker.remote()
                gradients[wid] = new_worker
            
    for i in range(ITERATIONS * NWORKERS):
        #
        ready_gradient_list, _ = ray.wait(list(gradients))
        ready_gradient_id = ready_gradient_list[0]
        worker = gradients.pop(wids.index(ready_gradient_id))

        # Compute and apply gradients
        succeed = False
        while not succeed:
            try:
                current_weights = server.apply_gradients.remote(ready_gradient_id)
                succeed = True
            except (ray.exceptions.RayActorError, ray.exceptions.RayTaskError) as e:
                server = ParameterServer.remote(lr)
                server.apply_gradients.remote(ready_gradient_id)
                succeed = False

        succeed = False
        while not succeed:
            try:
                gradients[worker.compute_gradients.remote(current_weights)] = worker
                succeed = True
            except (ray.exceptions.RayActorError, ray.exceptions.RayTaskError) as e:
                #del gradients[worker.]
                new_worker = worker.remote()
                gradients[worker.compute_gradients.remote(current_weights)] = new_worker
        #for grad in gradients:
        #    print(grad)
        if i % 10 == 0:
            # Evaluate the current model after every 10 updates.
            model.set_weights(ray.get(current_weights))
            accuracy = evaluate(model, test_loader)
            print("Iter {}: \taccuracy is {:.1f}".format(i, accuracy))

    # Clean up Ray resources and processes
    ray.shutdown()


def multiple_server_asynchronous_training(lr):
    ray.init()
    servers = [ParameterServerMultiple.remote() for i in range(NSERVERS)]
    workers = [Worker.remote() for i in range(NWORKERS)]
    server_manager = ParameterServerManager(servers)
    
    # for evaluating accuracy
    model = Model()
    _, test_loader = get_data_loader()

    # get initial weights w_0
    current_weights = [server.get_weights.remote() for server in servers]

    gradients = {}
    for worker in workers:
        gradients[worker.compute_gradients.remote(current_weights)] = worker

    for i in range(ITERATIONS * NWORKERS):
        ready_gradient_list, _ = ray.wait(list(gradients))
        ready_gradient_id = ready_gradient_list[0]
        worker = gradients.pop(ready_gradient_id)

        # Compute and apply gradients
        raise Exception("TODO")

        if i % 10 == 0:
            # Evaluate the current model after every 10 updates.
            model.set_weights(ray.get(current_weights))
            accuracy = evaluate(model, test_loader)
            print("Iter {}: \taccuracy is {:.1f}".format(i, accuracy))

    # Clean up Ray resources and processes
    ray.shutdown()
