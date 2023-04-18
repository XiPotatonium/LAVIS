import os
import random
import time
from typing import Any, Dict, Iterator, List, Set
import importlib
import sys


def check_installed(package_name: str):
    if package_name in sys.modules:
        return True
    elif (spec := importlib.util.find_spec(package_name)) is not None:
        # If you choose to perform the actual import ...
        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = module
        spec.loader.exec_module(module)
        return True
    else:
        return False


def alloc(
    preferred_devices: Iterator[List[int]],
    wait_time: float = 240.0,
) -> Iterator[Dict[str, Any]]:
    if check_installed('pynvml'):
        for device in alloc_cuda(preferred_devices, wait_time):
            yield device
    else:
        print("Package pynvml not found, force using cpu.")
        for _ in preferred_devices:
            yield {"device": "cpu"}


def alloc1(
    preferred_devices: List[int],
    wait_time: float = 240.0,
) -> Dict[str, Any]:
    return next(alloc([preferred_devices], wait_time))


def alloc_cuda(
    preferred_devices: Iterator[List[int]],
    wait_time: float = 240.0,
) -> Iterator[Dict[str, Any]]:
    """_summary_

    Args:
        preferred_devices (List[List[int]]): _description_
        wait_time (float, optional): _description_. Defaults to 240.0.

    Yields:
        _type_: _description_
    """
    import pynvml
    pynvml.nvmlInit()
    driver = pynvml.nvmlSystemGetCudaDriverVersion_v2()

    visible_devices = []
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible_devices = list(map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(",")))
    if len(visible_devices) == 0:
        # all_gpu not set by os.environ
        visible_devices = list(range(pynvml.nvmlDeviceGetCount()))

    gpu_queue: List[int] = []
    gpu_just_used: List[int] = []

    for task_preferred_devices in preferred_devices:
        task_preferred_devices: Set[int] = set(task_preferred_devices)
        if len(task_preferred_devices) == 1 and -1 in task_preferred_devices:
            # cpu-only
            yield {"device": "cpu"}
        else:
            # pick one gpu for training
            while len(gpu_queue) == 0:
                # detect free gpu if gpu queue is empty
                available_gpus = set(visible_devices) - set(gpu_just_used)
                if len(task_preferred_devices) == 0:
                    candidate_gpus = list(available_gpus)
                else:
                    # 只选择特定的GPU
                    candidate_gpus = list(task_preferred_devices.intersection(available_gpus))
                for index in candidate_gpus:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mem_used = meminfo.used / 1024 / 1024   # MB
                    if mem_used < 1500:
                        # add free gpu
                        gpu_queue.append(index)
                if len(gpu_queue) == 0:
                    print("Preferred: {}. Waiting for Free GPU ......".format(
                        task_preferred_devices if len(task_preferred_devices) != 0 else available_gpus,
                    ))
                    time.sleep(wait_time)
                    gpu_just_used = []
                else:
                    print("Available device: ", gpu_queue)

            # print("########### Using GPU Normal Training ###########")

            device_id = random.sample(gpu_queue, k=1)[0]
            gpu_just_used.append(device_id)
            gpu_queue.remove(device_id)
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            yield {
                "device": "cuda:{}".format(device_id),
                "desc": str(pynvml.nvmlDeviceGetName(handle)),
                "driver": driver
            }
