from accelerate import init_empty_weights, infer_auto_device_map
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoConfig, T5Model
import numpy as np
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
from datetime import datetime
import gc

import json
import os
import pickle
import openai
import argparse
import time
from pathlib import Path
import logging
from constants import *

# Set the logging level to INFO
logging.basicConfig(level=logging.INFO)


def squad_Passage(full_prompt):
    return full_prompt[full_prompt.index("Passage:"):full_prompt.index("Question:")].replace("Passage:", "").strip()

def squad_Question(full_prompt):
    return full_prompt[full_prompt.index("Question:"):].replace("Question:", "").strip()



def get_responses_unanswerable_questions_squad(data_path, request_function, prompt_prefix, n_instances, only_hint_prompts, only_adversarial, batch_size, **kwargs):
    responses = {"ids":[], "Adversarial":[], "Pseudo-Adversarial":[], "CoT-Adversarial":[], "Answerability":[], "Passage":[], "Question":[]}
    # responses = {"ids":[], "Passage":[], "Question":[]}

    with open(data_path) as f:
        data = json.load(f)
    
    if n_instances != None:
        data = data[:n_instances]

    # the control-group doesn't have this parameter
    if "Unanswerablity-Reason" in data[0].keys():
        responses["Unanswerablity-Reason"] = []


    n_batches = int(np.ceil(len(data) / batch_size))
    for batch_i in tqdm(range(n_batches)):
        curr_data = data[batch_i*batch_size:(batch_i+1)*batch_size]
        responses["ids"].extend([sample["id"] for sample in curr_data])

        if "Unanswerablity-Reason" in data[0].keys():
            responses["Unanswerablity-Reason"].extend([sample["Unanswerablity-Reason"] for sample in curr_data])
        if only_hint_prompts:
            responses["Adversarial"].extend([""]*batch_size)
            responses["Answerability"].extend([""]*batch_size)
        elif only_adversarial:
            responses["Adversarial"].extend(request_function([sample['Adversarial'] for sample in curr_data], **kwargs))
            responses["Answerability"].extend([""]*batch_size)
        else:
            responses["Adversarial"].extend(request_function([sample['Adversarial'] for sample in curr_data], **kwargs))
            responses["Answerability"].extend(request_function([sample['Answerability'] for sample in curr_data], **kwargs))
        
        if only_adversarial:
            responses["Pseudo-Adversarial"].extend([""]*batch_size)
            responses["CoT-Adversarial"].extend([""]*batch_size)
        else:
            # pseudo-adversarial response
            responses["Pseudo-Adversarial"].extend(request_function([sample['Pseudo-Adversarial'] for sample in curr_data], **kwargs))
            
            # CoT-adversarial response
            responses["CoT-Adversarial"].extend(request_function([sample['CoT-Adversarial'] for sample in curr_data], **kwargs))

        responses["Passage"].extend([squad_Passage(sample['Adversarial']) for sample in curr_data])
        responses["Question"].extend([squad_Question(sample['Adversarial']) for sample in curr_data])

    return responses



def NQ_Passage(full_prompt):
    return full_prompt[full_prompt.index("Passage:"):full_prompt.index("Question:")].replace("Passage:", "").strip()

def NQ_Question(full_prompt):
    return full_prompt[full_prompt.index("Question:"):].replace("Question:", "").strip()




def get_responses_unanswerable_questions_NQ(data_path, request_function, prompt_prefix, n_instances, only_hint_prompts, only_adversarial, batch_size, **kwargs):
    responses = {"ids":[], "annotation_ids":[], "Adversarial":[], "Pseudo-Adversarial":[], "CoT-Adversarial":[], "Answerability":[], "Passage":[], "Question":[]}

    with open(data_path) as f:
        data = json.load(f)
    
    if n_instances != None:
        data = data[:n_instances]

    n_batches = int(np.ceil(len(data) / batch_size))

    for batch_i in tqdm(range(n_batches)):
        curr_data = data[batch_i*batch_size:(batch_i+1)*batch_size]
        responses["ids"].extend([sample["example_id"] for sample in curr_data])
        responses["annotation_ids"].extend([sample["annotation_id"] for sample in curr_data])

        if only_hint_prompts:
            responses["Adversarial"].extend([""]*batch_size)
            responses["Answerability"].extend([""]*batch_size)
        elif only_adversarial:
            responses["Adversarial"].extend(request_function([sample['Adversarial'] for sample in curr_data], **kwargs))
            responses["Answerability"].extend([""]*batch_size)
        else:
            responses["Adversarial"].extend(request_function([sample['Adversarial'] for sample in curr_data], **kwargs))
            responses["Answerability"].extend(request_function([sample['Answerability'] for sample in curr_data], **kwargs))
        
        if only_adversarial:
            responses["Pseudo-Adversarial"].extend([""]*batch_size)
            responses["CoT-Adversarial"].extend([""]*batch_size)
        else:
            # pseudo-adversarial response
            responses["Pseudo-Adversarial"].extend(request_function([sample['Pseudo-Adversarial'] for sample in curr_data], **kwargs))
            
            # CoT-adversarial response
            responses["CoT-Adversarial"].extend(request_function([sample['CoT-Adversarial'] for sample in curr_data], **kwargs))

        responses["Passage"].extend([NQ_Passage(sample['Adversarial']) for sample in curr_data])
        responses["Question"].extend([NQ_Question(sample['Adversarial']) for sample in curr_data])

    return responses


def musique_Context(full_prompt):
    return full_prompt[full_prompt.index("Context:"):full_prompt.index("Question:")].replace("Context:", "").strip()

def musique_Question(full_prompt):
    return full_prompt[full_prompt.index("Question:"):].replace("Question:", "").strip()



def get_responses_unanswerable_questions_musique(data_path, request_function, prompt_prefix, n_instances, only_hint_prompts, only_adversarial, batch_size, **kwargs):
    responses = {"ids":[], "Adversarial":[], "Pseudo-Adversarial":[], "CoT-Adversarial":[], "Answerability":[], "Context":[], "Question":[]}

    with open(data_path) as f:
        data = json.load(f)
    
    if n_instances != None:
        data = data[:n_instances]

    n_batches = int(np.ceil(len(data) / batch_size))

    for batch_i in tqdm(range(n_batches)):
        curr_data = data[batch_i*batch_size:(batch_i+1)*batch_size]
        responses["ids"].extend([sample["id"] for sample in curr_data])

        if only_hint_prompts:
            responses["Adversarial"].extend([""]*batch_size)
            responses["Answerability"].extend([""]*batch_size)
        elif only_adversarial:
            responses["Adversarial"].extend(request_function([sample['Adversarial'] for sample in curr_data], **kwargs))
            responses["Answerability"].extend([""]*batch_size)
        else:
            responses["Adversarial"].extend(request_function([sample['Adversarial'] for sample in curr_data], **kwargs))
            responses["Answerability"].extend(request_function([sample['Answerability'] for sample in curr_data], **kwargs))
        
        if only_adversarial:
            responses["Pseudo-Adversarial"].extend([""]*batch_size)
            responses["CoT-Adversarial"].extend([""]*batch_size)
        else:
            # pseudo-adversarial response
            responses["Pseudo-Adversarial"].extend(request_function([sample['Pseudo-Adversarial'] for sample in curr_data], **kwargs))
            
            # CoT-adversarial response
            responses["CoT-Adversarial"].extend(request_function([sample['CoT-Adversarial'] for sample in curr_data], **kwargs))

        responses["Context"].extend([musique_Context(sample['Adversarial']) for sample in curr_data])
        responses["Question"].extend([musique_Question(sample['Adversarial']) for sample in curr_data])

    return responses



def chatGPT_request(content, k_beams, num_return_sequences, **kwargs):
    time.sleep(15) 
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content}
        ]
    )


    return response["choices"][0]["message"]["content"]


def HF_request(prompts, k_beams, num_return_sequences, tokenizer, model, lm_head, eraser, only_first_decoding, k, is_multi_choice, output_max_length, prompt_suffix, return_only_generated_text):

    prompts = [f"{p}{prompt_suffix}" for p in prompts]
    input_ids = tokenizer.batch_encode_plus(
        prompts, 
        padding=True,
        truncation=True,
        return_tensors="pt")["input_ids"].to(model.device)

    # Set the model to evaluation mode
    model.eval()

    # Initialize the decoder input tensor
    decoder_input_ids = [[torch.tensor([[tokenizer.pad_token_id]]), 1.0] for _ in range(k_beams)]


    lm_head = lm_head.to('cuda')

    # Generate the output sequence one token at a time
    output_ids, logits_history, last_hidden_embedding = [[0] for _ in range(k_beams)], [[] for _ in range(k_beams)], [[] for _ in range(k_beams)]
    with torch.no_grad():
        for i in range(20):

            all_candidates = []
            for j,sequence in enumerate(decoder_input_ids):
                curr_decoder_input_ids, prob = sequence
                curr_output_ids = output_ids[j]
                curr_logits_history = logits_history[j]
                curr_last_hidden_embedding = last_hidden_embedding[j]


                # Check if the next token is an eos/unk/pad token (if so - no need to generate new beam - keep results and proceed to the next sequence)
                if i>0 and curr_output_ids[-1] in [tokenizer.eos_token_id]:
                    all_candidates.append([curr_output_ids, curr_logits_history, curr_last_hidden_embedding, curr_decoder_input_ids, prob])
                    continue

                # Get the logits for the next token
                embeddings = model(input_ids=input_ids, 
                            attention_mask=torch.ones_like(input_ids), 
                            decoder_input_ids=curr_decoder_input_ids).last_hidden_state
                embeddings = embeddings[:,-1,:]
                
                if eraser != None and (not only_first_decoding or i == 0):
                    embeddings = eraser(embeddings.to("cuda"))

                logits = lm_head(embeddings)
                # Convert the logits to probabilities
                probabilities = torch.softmax(logits, dim=-1)
                
                # take top-k tokens
                next_token_ids = torch.multinomial(probabilities, num_samples=k_beams)[0]

                new_output_ids = [curr_output_ids + [next_token_id.item()] for next_token_id in next_token_ids]
                new_logits_history = [curr_logits_history + [logits.to('cpu')] for _ in range(k_beams)]
                new_last_hidden_embedding = [curr_last_hidden_embedding + [embeddings.to('cpu')] for _ in range(k_beams)]
                new_decoder_input_ids = [torch.cat([curr_decoder_input_ids, torch.tensor([[next_token_id]])], dim=-1) for next_token_id in next_token_ids]
                new_probs = [prob*probabilities[0,next_token_id].item() for next_token_id in next_token_ids]
                all_candidates.extend([[new_output_ids[ind], new_logits_history[ind], new_last_hidden_embedding[ind], new_decoder_input_ids[ind], new_probs[ind]] for ind in range(k_beams)])

                # first step - same "history" for all 3 beams - so enough just the first beam to start generating the beams
                if i == 0:
                    break
            
            # Order all candidates by probability
            ordered_candidates = sorted(all_candidates, key=lambda tup:tup[-1], reverse=True)
        

            # Select k best
            filtered_candidates = ordered_candidates[:k_beams]

            decoder_input_ids = [[cand[3], cand[4]] for cand in filtered_candidates]
            output_ids = [cand[0] for cand in filtered_candidates]
            logits_history = [cand[1] for cand in filtered_candidates]
            last_hidden_embedding = [cand[2] for cand in filtered_candidates]





    output_text = [tokenizer.decode(elem, skip_special_tokens=True) for elem in output_ids]
    all_outputs_ids = pad_sequence([torch.tensor(l) for l in output_ids], batch_first=True, padding_value=0)
    output_logits = [torch.cat(elem, dim=0) for elem in logits_history]   
    output_last_hidden_embedding = [torch.cat(elem, dim=0) for elem in last_hidden_embedding]   

    return_dicts =  [{"outputs":output_text,
                      "all_outputs_ids": all_outputs_ids,
                      "full_logits": output_logits,
                      "last_hidden_embedding": output_last_hidden_embedding}]
    return return_dicts


def get_all_relevant_models(args, model_name):

    models_list = list()
    if model_name == "ChatGPT":
        if args.openAI_key == None:
            with open("openAI_api_key/openAI_api_key.txt", "r") as f1:
                openai.api_key = f1.read()
        else:
                openai.api_key = args.openAI_key
        return {"output_subdir":"chatGPT", "request_function":chatGPT_request, "kwargs":dict()}

    if model_name == "Flan-UL2":
        tokenizer_UL2 = AutoTokenizer.from_pretrained("google/flan-ul2", model_max_length=args.model_max_length)
        # max_memory_dict = {0:'40GiB',1:'40GiB'}
        # max_memory_dict['cpu'] = '300GiB'
        max_memory_dict = {gpu_i:f"{MAX_GPU_MEM}GiB" for gpu_i in range(torch.cuda.device_count())}
        max_memory_dict['cpu'] = f'{MAX_CPU_MEM}GiB'
        model_UL2 = T5Model.from_pretrained("google/flan-ul2",
                            device_map='auto',
                            max_memory=max_memory_dict,
                            torch_dtype=torch.float16)
        model_UL2_head = AutoModelForSeq2SeqLM.from_pretrained("google/flan-ul2",
                            device_map='auto',
                            max_memory={0:'10GiB', 'cpu':'300GiB'},
                            torch_dtype=torch.float16).lm_head
        return {"output_subdir":"UL2_Flan", "request_function":HF_request, "kwargs":dict(tokenizer=tokenizer_UL2, model=model_UL2, lm_head=model_UL2_head, prompt_suffix="")}

    if model_name == "Flan-T5-xxl":
        tokenizer_flan_t5_xxl = AutoTokenizer.from_pretrained("google/flan-t5-xxl", model_max_length=args.model_max_length)
        # max_memory_dict = {0:'5GiB', 1:'20GiB'}
        # max_memory_dict['cpu'] = '300GiB'
        max_memory_dict = {gpu_i:f"{MAX_GPU_MEM}GiB" for gpu_i in range(torch.cuda.device_count())}
        max_memory_dict['cpu'] = f'{MAX_CPU_MEM}GiB'
        model_flan_t5_xxl = T5Model.from_pretrained("google/flan-t5-xxl",
                            device_map='auto',
                            max_memory=max_memory_dict)
        model_flan_t5_xxl_head = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xxl",
                            device_map='auto',
                            max_memory={0:'10GiB', 'cpu':'300GiB'}).lm_head
        return {"output_subdir":"T5_xxl_Flan", "request_function":HF_request, "kwargs":dict(tokenizer=tokenizer_flan_t5_xxl, model=model_flan_t5_xxl, lm_head=model_flan_t5_xxl_head, prompt_suffix="")}



    if model_name == "OPT":

        tokenizer_OPT = AutoTokenizer.from_pretrained("facebook/opt-iml-max-30b", model_max_length=args.model_max_length, padding_side='left')
        # max_memory_dict = {0:'10GiB', 1:'40GiB'}
        # max_memory_dict['cpu'] = '300GiB'
        max_memory_dict = {gpu_i:f"{MAX_GPU_MEM}GiB" for gpu_i in range(torch.cuda.device_count())}
        max_memory_dict['cpu'] = f'{MAX_CPU_MEM}GiB'
        model_OPT = AutoModelForCausalLM.from_pretrained(
                "facebook/opt-iml-max-30b",
                device_map='auto',
                max_memory=max_memory_dict,
                torch_dtype=torch.float16)
        # AVIVSL: add the lm_head of OPT_IML!
        return {"output_subdir":"OPT", "request_function":HF_request, "kwargs":dict(tokenizer=tokenizer_OPT, model=model_OPT, prompt_suffix="\n Answer:")}


    if model_name == "OPT-1-3B":
        tokenizer_OPT_1_3B = AutoTokenizer.from_pretrained("facebook/opt-iml-max-1.3b", model_max_length=args.model_max_length, padding_side='left')
        
        config = AutoConfig.from_pretrained("facebook/opt-iml-max-1.3b")
        with init_empty_weights():
            model_OPT_1_3B = AutoModelForCausalLM.from_config(config)
        model_OPT_1_3B.tie_weights()
        device_map = infer_auto_device_map(model_OPT_1_3B, no_split_module_classes=["OPTDecoderLayer"], dtype="float16")

        model_OPT_1_3B = AutoModelForCausalLM.from_pretrained(
                        "facebook/opt-iml-max-1.3b",
                        offload_folder="offload",
                        offload_state_dict=True,
                        torch_dtype=torch.float16).to(0)
        return {"output_subdir":"OPT_1_3B", "request_function":HF_request, "kwargs":dict(tokenizer=tokenizer_OPT_1_3B, model=model_OPT_1_3B, prompt_suffix="\n Answer:")} 



    # for debugging:
    if model_name == "Flan-T5-small":
        tokenizer_flan_t5_small = AutoTokenizer.from_pretrained("google/flan-t5-small", model_max_length=args.model_max_length)
        config = AutoConfig.from_pretrained("google/flan-t5-small")
        with init_empty_weights():
            model_flan_t5_small = AutoModelForSeq2SeqLM.from_config(config)
        device_map = infer_auto_device_map(model_flan_t5_small, no_split_module_classes=["T5Block"], dtype="float16")
        model_flan_t5_small.tie_weights()

        model_flan_t5_small = T5Model.from_pretrained(
                            "google/flan-t5-small",
                            device_map=device_map,
                            offload_folder="offload",
                            offload_state_dict=True,
                            torch_dtype=torch.float16).to(0)

        model_flan_t5_small_head = AutoModelForSeq2SeqLM.from_pretrained(
                            "google/flan-t5-small",
                            device_map=device_map,
                            offload_folder="offload",
                            offload_state_dict=True,
                            torch_dtype=torch.float16).to(0).lm_head


        models_list.append({"output_subdir":"flan_t5_small", "request_function":HF_request, "kwargs":dict(tokenizer=tokenizer_flan_t5_small, model=model_flan_t5_small, prompt_suffix="")})
        return {"output_subdir":"flan_t5_small", "request_function":HF_request, "kwargs":dict(tokenizer=tokenizer_flan_t5_small, model=model_flan_t5_small, lm_head=model_flan_t5_small_head, prompt_suffix="")}

    raise Exception(f"Incorrect model passed: {model_name}")


    return models_list


def get_all_relevant_datasets(args):
    data_list = list()
    if "squad" in args.datasets:
        if args.adversarial:
            data_list.append({"type": "adversarial", "data_name":"squad", "get_data_function":get_responses_unanswerable_questions_squad, "is_multi_choice":False})
        if args.control_group:
            data_list.append({"type": "control_group", "data_name":"squad", "get_data_function":get_responses_unanswerable_questions_squad, "is_multi_choice":False})

    if "NQ" in args.datasets:
        if args.adversarial:
            data_list.append({"type": "adversarial", "data_name":"NQ", "get_data_function":get_responses_unanswerable_questions_NQ, "is_multi_choice":False})
        if args.control_group:
            data_list.append({"type": "control_group", "data_name":"NQ", "get_data_function":get_responses_unanswerable_questions_NQ, "is_multi_choice":False})

    if "musique" in args.datasets:
        if args.adversarial:
            data_list.append({"type": "adversarial", "data_name":"musique", "get_data_function":get_responses_unanswerable_questions_musique, "is_multi_choice":False})
        if args.control_group:
            data_list.append({"type": "control_group", "data_name":"musique", "get_data_function":get_responses_unanswerable_questions_musique, "is_multi_choice":False})
    return data_list

def create_dir(subdirs):
    full_subdir = ""
    for subdir in subdirs:
        full_subdir = os.path.join(full_subdir, subdir)

        if not os.path.exists(full_subdir):
            os.makedirs(full_subdir)



def main(args):
    # Load the eraser from the file
    if args.no_eraser:
        eraser = None
    else:
        with open(args.eraser_dir, "rb") as file:
            eraser = pickle.load(file).to("cuda")
    

    now = datetime.now()
    now_str = now.strftime("%d-%m-%Y_%H:%M:%S")
    outdir_path = args.outdir if args.outdir else os.path.join("responses_embeddings", "projections", now_str)
    path = Path(outdir_path)
    path.mkdir(parents=True, exist_ok=True)
    logging.info(f'saved to: {outdir_path}')

    datasets_list = get_all_relevant_datasets(args)

    if args.k_beams_grid_search is None:
        k_beams_list = [args.k_beams]
    else:
        k_beams_list = json.loads(args.k_beams_grid_search)
    
    if args.k_steps_grid_search is None:
        k_steps_list = [args.k_steps]
    else:
        k_steps_list = json.loads(args.k_steps_grid_search)


    model = None
    for k_steps in k_steps_list:
        for model_name in args.models:
            if model: # free up memory to enable loading the next model
                del model['kwargs']['model']
                gc.collect()
                torch.cuda.empty_cache()        
            model = get_all_relevant_models(args, model_name)
            for p_variant in args.prompt_variant:
                for k_beams in k_beams_list:
                    for dataset in datasets_list:
                        num_return_sequences = k_beams if args.num_return_sequences == None else args.num_return_sequences
                        print(f"model: {model['output_subdir']} data: {dataset['data_name']} type: {dataset['type']} variant: {p_variant} beam: {k_beams}")
                        
                        create_dir([outdir_path, model['output_subdir'], "zero_shot", f"k_beams_{k_beams}_num_return_seq_{num_return_sequences}", p_variant, f"k_{k_steps}"])

                        if args.trainset:
                            outdir_suffix = "_trainset_filtered"
                        if args.all_instances:
                            outdir_suffix = "_all"
                        elif args.unfiltered_instances:
                            outdir_suffix = "_unfiltered"
                        else:
                            outdir_suffix = ""
                        curr_outdir = os.path.join(outdir_path, model['output_subdir'], "zero_shot", f"k_beams_{k_beams}_num_return_seq_{num_return_sequences}", p_variant, f"k_{k_steps}", f"{dataset['type']}_{dataset['data_name']}_embeddings{outdir_suffix}_k_beams_{k_beams}_num_return_seq_{num_return_sequences}.pt")
                        
                        if os.path.exists(curr_outdir):
                            print(f"{curr_outdir} exists! skipping...")
                            continue
                        
                        if args.trainset:
                            data_adversarial_path = fr"generated_prompts/train_set/filtered/zero_shot/{p_variant}/{dataset['data_name']}_trainset_{dataset['type']}_filtered.json"
                        elif args.all_instances:
                            data_adversarial_path = fr"generated_prompts/all/zero_shot/{p_variant}/{dataset['data_name']}_{dataset['type']}_all.json"
                        elif args.unfiltered_instances:
                            data_adversarial_path = fr"generated_prompts/unfiltered/zero_shot/{p_variant}/{dataset['data_name']}_{dataset['type']}_unfiltered.json"
                        else:
                            data_adversarial_path = fr"generated_prompts/filtered/zero_shot/{p_variant}/{dataset['data_name']}_{dataset['type']}_filtered.json"
                        
                        if model['kwargs']:
                            responses = dataset['get_data_function'](data_path=data_adversarial_path, request_function=model['request_function'], prompt_prefix="", output_max_length=args.output_max_length, n_instances=args.n_instances, k_beams = k_beams, batch_size=args.batch_size, num_return_sequences=num_return_sequences, only_hint_prompts=args.only_hint_prompts, only_adversarial=args.only_adversarial, tokenizer=model['kwargs']['tokenizer'], model=model['kwargs']['model'], lm_head = model['kwargs']['lm_head'], eraser=eraser, only_first_decoding=args.only_first_decoding, k=k_steps, prompt_suffix=model['kwargs']['prompt_suffix'], is_multi_choice=dataset["is_multi_choice"], return_only_generated_text=args.return_only_generated_text)
                        else:
                            responses = dataset['get_data_function'](data_path=data_adversarial_path, request_function=model['request_function'], n_instances=args.n_instances, only_hint_prompts=args.only_hint_prompts, only_adversarial=args.only_adversarial, prompt_prefix="", is_multi_choice=dataset["is_multi_choice"])
                        torch.save(responses, curr_outdir) # and to load it: loaded_dict = torch.load(curr_outdir)





if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument("--ChatGPT", action='store_true', default=False, help="send requests to ChatGPT.")
    argparser.add_argument('--openAI-key', type=str, default=None, help='API key of OpenAI.')
    argparser.add_argument('--outdir', type=str, default=None, help='outdir to save results')
    argparser.add_argument("--models", nargs='+', type=str, default=["Flan-T5-small"], help="which models to send requests to. any from: Flan-UL2, Flan-T5-xxl, OPT, Flan-T5-small, OPT-1-3B and ChatGPT")
    argparser.add_argument("--adversarial", action='store_true', default=False, help="send adversarial requests.")
    argparser.add_argument("--control-group", action='store_true', default=False, help="send control group request.")
    argparser.add_argument("--datasets", nargs='+', type=str, default=["squad"], help="which datasets to work on. any from: squad, NQ, musique")
    argparser.add_argument("--all-instances", action='store_true', default=False, help="take all the instances of the task.")
    argparser.add_argument("--unfiltered-instances", action='store_true', default=False, help="take the unfiltered instances of the task.")
    argparser.add_argument("--n-instances", type=int, default=None, help="number of instances to process")
    argparser.add_argument("--k-beams", type=int, default=1, help="beam size (will also be the number of returned outputs to check \"unanswerable\" from)")
    argparser.add_argument("--k-beams-grid-search", type=str, default=None, help="grid search on the k-beams. Will overrun \"--k-beams\". Need to pass as a list (e.g. --k-beams-grid-search [4,5,6])")
    argparser.add_argument("--num-return-sequences", type=int, default=None, help="number of returned sequences, in which to check if there is unaswerability (if None - will equal k_beam).")
    argparser.add_argument("--prompt-variant", nargs='+', type=str, default=["variant1"], help="prompt variant list (any of variant1, variant2, variant3).")
    argparser.add_argument("--only-hint-prompts", action='store_true', default=False, help="whether to pass only the prompts with hints (pseudo, CoT, NA fourth answers).")
    argparser.add_argument("--only-adversarial", action='store_true', default=False, help="whether to pass only the Adversarial prompts (also for control group).")
    argparser.add_argument("--return-only-generated-text", action='store_true', default=False, help="whether to return only the generated text, without the logits (in cases of OOM)")

    argparser.add_argument("--batch-size", type=int, default=1, help="size of batch.")
    argparser.add_argument("--cuda", action='store_true', default=False, help="whether to run it on the the GPUs or the CPUs.")
    argparser.add_argument("--model-max-length", type=int, default=2048, help="max input length of model (for datasets like NQ where inputs are very long).")
    argparser.add_argument("--output-max-length", type=int, default=100, help="max output length.")
    argparser.add_argument("--trainset", action='store_true', default=False, help="whether the data is the trainset (for classifiers)")
    argparser.add_argument("--eraser-dir", type=str, required=True, help="path to eraser.")
    argparser.add_argument("--k-steps", type=int, default=10, help="number of steps in the direction of the gradient.")
    argparser.add_argument("--k-steps-grid-search", type=str, default=None, help="grid search on the k-step. Will overrun \"--k-steps\". Need to pass as a list (e.g. --k-steps-grid-search [50,60,70])")
    argparser.add_argument("--no-eraser", action='store_true', default=False, help="do not load eraser (for debugging)")
    argparser.add_argument("--only-first-decoding", action='store_true', default=False, help="perform erasure only on first decoding step.")

    args = argparser.parse_args()
    main(args)







