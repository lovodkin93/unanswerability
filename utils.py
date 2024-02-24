import psutil
import torch
import GPUtil
import os
import logging
from typing import List, Dict
import json

# Set the logging level to INFO
logging.basicConfig(level=logging.INFO)

PROMPT_TYPES = ["Regular-Prompt", "Hint-Prompt", "CoT-Prompt",  "Ablation1", "Ablation2", "Answerability", "Regular-Prompt-CoT", "Hint-Prompt-CoT", "Ablation1-CoT", "Ablation2-CoT", "Answerability-CoT"]
UNANSWERABLE_REPLIES = ["unanswerable", "n/a", "idk", "i don't know", "not known", "answer not in context", "the answer is unknown"]
UNANSWERABLE_REPLIES_EXACT = ['nan', 'unknown', 'no answer', 'it is unknown', "the answer is unknown", 'none of the above choices', 'none of the above']

def get_instruction(prompt_dict, prompt_type, prompt_variant):
    if prompt_type in ["Hint-Prompt", "CoT-Prompt", "Hint-Prompt-CoT"]:
        return prompt_dict[f"instructions-{prompt_type.replace('-CoT', '')}-{prompt_variant}"]
    elif prompt_type in ["Ablation1", "Ablation1-CoT"]:
        return prompt_dict[f"instructions-Hint-Prompt-{prompt_variant}"]
    elif prompt_type in ["Ablation2", "Ablation2-CoT"]:
        return prompt_dict["instructions-Regular-Prompt"]
    else:
        return prompt_dict[f"instructions-{prompt_type.replace('-CoT', '')}"]

def get_data_type_instances(raw_data, data_type):
    if data_type == "un-answerable":
        return [elem for elem in raw_data if elem['answerable']=='no']
    elif data_type == "answerable":
        return [elem for elem in raw_data if elem['answerable']=='yes']
    else:
        raise Exception(f"unrecognized data_type: {data_type}")

def make_demo(item, prompt, instruction=None, answer_related_prompts=None, test=False):
    prompt = prompt.replace("{INST}", instruction)
    prompt = prompt.replace("{P}", item["Passage"])
    prompt = prompt.replace("{Q}", item["Question"])
    if not test:
        prompt = prompt.replace("{A}", answer_related_prompts['answer_format'])
        prompt = prompt.replace("{FIN_A}", item['Answer'])
        prompt = prompt.replace("{CoT}", item['CoT'])
        prompt = prompt.replace("{NO_ANSWER}", answer_related_prompts['no-answer-response'])
    else:
        prompt = prompt.replace("{A}", "")
        prompt = f"{prompt}\n {answer_related_prompts['prompt_suffix']}"
    return prompt.strip()

def construct_prompts(prompt_dict : Dict, raw_data: List[Dict], zero_shot: bool, data_type: str, prompt_variant: str, demo_variant: str = None):
    """
    prompt_dict: dictionary with prompt-related information (instructions, demonstrations, etc.)
    raw_data: the actual current instances
    zero_shot: whether zero shot
    data_type: "answerable" or "un-answerable"
    prompt_variant: either one of "variant1", "variant2", "variant3"
    demo_variant: (relevant for the few-shot experiments) which icl examples variant to take (either one of "1", "2", "3")
    """
    # get relevant raw data (either answerable or un-answerable)
    raw_data = get_data_type_instances(raw_data, data_type)
    # get relevant prompt types
    if zero_shot:
        relevant_prompt_types = [p for p in PROMPT_TYPES if not p.endswith("CoT") and not "Ablation" in p] 
    else:
        relevant_prompt_types = [p for p in PROMPT_TYPES if p!="CoT-Prompt"]
    # get relevant prompt_suffix (in few-shot - prompt ends with "Answer:")
    prompt_suffix = "" if zero_shot else "Answer:"
    # get relevant prompt_stucture (in few shot - starts with "Instructions:" whereas in zero shot - directly starts with the actual instructions)
    prompt_stucture = prompt_dict['demo_prompt_zero_shot'] if zero_shot else prompt_dict['demo_prompt_few_shot']
    # get all instances' prompts
    all_prompts = []
    for instance in raw_data:
        instance_item = {'Passage' : instance['context'], 
                         'Question' : instance['question']}
        updated_instance = {'id':instance['id']}
        updated_instance.update(json.loads(instance['additional_data']))
        for prompt_type in relevant_prompt_types:
            # get relevant instructions (the replace - to also treat the CoT versions in the few-shot setting)
            instruction = get_instruction(prompt_dict, prompt_type, prompt_variant)
           
            head_prompt = ""
            answer_related_prompts = {'answer_format':prompt_dict['CoT-format'] if prompt_type.endswith("CoT") else prompt_dict['non-CoT-format'],
                                      'prompt_suffix':prompt_suffix,
                                      'no-answer-response':prompt_dict['no-answer-response'][prompt_variant] if not "Answerability" in prompt_type else 'unanswerable'}
            # Generate the demonstration part (for the few-shot)
            if not zero_shot:
                curr_demos_variant = prompt_dict['demos'][f'demos-v{demo_variant}']
                if "Answerability" in prompt_type: # replace answerable demonstration's answer to "answerable"
                    relevant_demos = [{key:value if key!="Answer" else "answerable." for key,value in curr_demos_variant['answerable-1'].items()}]
                else:
                    relevant_demos = [curr_demos_variant['answerable-1']]
                if any(prompt_type.startswith(p) for p in ['Regular-Prompt', "Ablation1"]): # both demonstrations are answerable
                    relevant_demos.append(curr_demos_variant['answerable-2'])
                else:
                    relevant_demos.append(curr_demos_variant['un-answerable'])
                for demo in relevant_demos:
                    curr_prompt_demo =  make_demo(item=demo,
                                                  prompt=prompt_stucture,
                                                  instruction=instruction,
                                                  answer_related_prompts=answer_related_prompts,
                                                  test=False)
                    
                    head_prompt += curr_prompt_demo
                    head_prompt += prompt_dict["demo_sep"]
            


            # Generate the actual instance part
            head_prompt += make_demo(item=instance_item,
                                     prompt=prompt_stucture,
                                     instruction=instruction,
                                     answer_related_prompts=answer_related_prompts,
                                     test=True)
            
            updated_instance.update({prompt_type:head_prompt})
        all_prompts.append(updated_instance)
    return all_prompts
            




    






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