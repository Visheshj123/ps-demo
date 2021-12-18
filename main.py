from scheduler import (
    single_server_asynchronous_training,
    multiple_server_asynchronous_training,
)

if __name__ == "__main__":
    s = multiple_server_asynchronous_training(1e-2)