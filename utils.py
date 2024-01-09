import psutil
import torch
import GPUtil
import os
import logging

# Set the logging level to INFO
logging.basicConfig(level=logging.INFO)

PROMPT_TYPES = ["Regular-Prompt", "Hint-Prompt", "CoT-Prompt",  "Ablation1", "Ablation2", "Answerability", "Regular-Prompt-CoT", "Hint-Prompt-CoT", "Ablation1-CoT", "Ablation2-CoT", "Answerability-CoT"]
UNANSWERABLE_REPLIES = ["unanswerable", "n/a", "idk", "i don't know", "not known", "answer not in context", "the answer is unknown"]
UNANSWERABLE_REPLIES_EXACT = ['nan', 'unknown', 'no answer', 'it is unknown', "the answer is unknown", 'none of the above choices', 'none of the above']


def get_max_memory():
    """Get the maximum memory available for the currently visible GPU cards GPU, as well as, CPU for loading models."""
    max_memory_dict = dict()

    # GPU
    # Get list of GPUs that are visible to the process
    visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')
    if visible_devices:
        visible_devices = [int(dev) for dev in visible_devices.split(',')]
    else:
        # If CUDA_VISIBLE_DEVICES is not set, consider all GPUs as visible
        visible_devices = list(range(len(GPUtil.getGPUs())))
    
    # Get list of all GPUs (both visible and not visible)
    gpus = GPUtil.getGPUs()

    visible_gpu_cnt = 0
    for i, gpu in enumerate(gpus):
        if i in visible_devices:
            free_in_GB = int(gpu.memoryFree / 1024)
            
            if visible_gpu_cnt == 0: # the first card needs more memory
                if free_in_GB<40:
                    raise Exception("Make sure you first visible GPU card has at least 40GiB available in memory.")
                max_memory = f'{free_in_GB-20}GiB'
            else:
                if free_in_GB<30:
                    raise Exception("Make sure all you visible GPU cards have at least 20GiB available in memory.")
                max_memory = f'{free_in_GB-10}GiB'
                
            max_memory_dict[visible_gpu_cnt] = max_memory
            visible_gpu_cnt+=1
    
    # CPU
    available_memory = psutil.virtual_memory().available

    # Convert bytes to gigabytes for easier reading
    available_memory_gb = available_memory / (1024 ** 3)

    if available_memory_gb<100:
        raise Exception("Make sure there are at least 100GiB available in the CPU memory.")
    max_memory_dict['cpu'] = f"{min(int(available_memory_gb/2), 100)}GiB"

    gpu_max_memory_used_str = "\n".join([f"card {str(visible_devices[gpu_i])}: {max_memory_dict[gpu_i]}" for gpu_i in range(len(max_memory_dict)-1)])
    max_memory_used_str = f"GPU:\n{gpu_max_memory_used_str}\nCPU:\n{max_memory_dict['cpu']}"
    logging.info(f'max memory used:\n{max_memory_used_str}')
    return max_memory_dict