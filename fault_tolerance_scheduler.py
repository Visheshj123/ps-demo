import ray
import torch
from ps import ParameterServer, ParameterServerManager, ParameterServerMultiple, Worker
from config import ITERATIONS, NSERVERS, NWORKERS
from model import Model
from data import get_data_loader
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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


def single_fault_tolerant_server_asynchronous_training(lr):
    ray.init()
    timeout = 10
    server = ParameterServer.remote(lr)
    workers = [Worker.remote(i) for i in range(NWORKERS)]
    worker_statuses = [0 for i in range(NWORKERS)]
    model = Model()
    _, test_loader = get_data_loader()

    # get initial weights w_0
    current_weights = server.get_weights.remote()

    tasks = [ i for i in range(ITERATIONS *NWORKERS)]
    active_tasks = []
    gradients = {}
    task_to_worker = {}
    last_heartbeat = 0
    hearbeat_interval = 1
    while len(tasks) > 0:
        print("Checking to assign tasks")
        
        # If workers are available, assign them a task
        while 0 in worker_statuses:
            task, tasks = tasks[0], tasks[1:]
            print("Assigning task {} to worker {}".format(task,worker_statuses.index(0)))
            worker = workers[worker_statuses.index(0)]
            worker_statuses[worker] = 1
            task_ref = worker.compute_gradients.remote(current_weights,task)
            task_to_worker[task_ref] = workers.index(worker)
            active_tasks.append(task_ref)
        
        # On occasion, perform a heartbeat check
        if last_heartbeat + hearbeat_interval < time.time():
            print("Checking Heartbeats")
            # Start remote heartbeat function for each task
            status_refs = [worker.heartbeat.remote() for worker in workers]
            # Wait until each has finished, and get the results
            
            # Problem is, spinning workers may never return
            # Ideally, when a worker it created, the scheduler gets a reference an obj in the object store to act as its heartbeat info
            # then it just updates that every second, still figuring out how to do that though
            statuses = [ray.get(status_id) for status_id in status_refs]
            print(statuses)
            for i,status in enumerate(statuses):
                worker_statuses[i] = status['status']
                
                if status['timestamp']+10 < time.time():
                    # Worker is dead, manually kill it just in case, replace it, and add its task to the queue
                    print("Worker {} is dead".format(i))
                    ray.kill(workers[i])
                    workers[i] = Worker.remote(i)
                    tasks.push(status['task'])
                    
        # Check if any workers are done
        tasks_complete, tasks_in_progress = ray.wait(active_tasks)
        if len(tasks_complete) > 0:
            for task_ref in tasks_complete:
                print("Worker {} has completed their task".format(task_to_worker[task_ref]))
                # Removing it may not be necessary, unsure why yet
                active_tasks.remove(task_ref)
                current_weights = server.apply_gradients.remote(task_ref)
                # Workers automatically get a new task when done, so we just need to push the results from the finished task to the server
        if i % 10 == 0:
            # Evaluate the current model after every 10 updates.
            model.set_weights(ray.get(current_weights))
            accuracy = evaluate(model, test_loader)
            print("Iter {}: \taccuracy is {:.1f}".format(i, accuracy))

        #For debugging
        time.sleep(.1)

    # Clean up Ray resources and processes
    ray.shutdown()
if __name__ == "__main__":
    s = single_fault_tolerant_server_asynchronous_training(1e-2)
