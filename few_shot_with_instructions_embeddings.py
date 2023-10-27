from accelerate import init_empty_weights, infer_auto_device_map
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoConfig
import numpy as np
from tqdm import tqdm
import torch
from datetime import datetime
import gc

import json
import os
import argparse
from multiprocessing import Pool
from pathlib import Path
import logging
from constants import *
# Set the logging level to INFO
logging.basicConfig(level=logging.INFO)



def get_responses_unanswerable_questions_squad(data_path, args, **kwargs):

    def squad_Passage(full_prompt):
        return full_prompt.split("Passage:")[-1].strip().split("Question:")[0].strip()

    def squad_Question(full_prompt):
        return full_prompt.split("Question:")[-1].strip().split("Answer:")[0].strip()
    
    batch_size = args.batch_size

    responses = {"ids":[], 
                    "Adversarial":[], "Adversarial-CoT":[],
                    "Pseudo-Adversarial":[], "Pseudo-Adversarial-CoT":[],
                    "Ablation1":[], "Ablation1-CoT":[],
                    "Ablation2":[], "Ablation2-CoT":[],
                    "Answerability":[], "Answerability-CoT":[],
                    "Passage":[], "Question":[]}


    with open(data_path) as f:
        data = json.load(f)
    
    if args.n_instances != None:
        data = data[:args.n_instances]


    # the control-group doesn't have this parameter
    if "Unanswerablity-Reason" in data[0].keys():
        responses["Unanswerablity-Reason"] = []


    n_batches = int(np.ceil(len(data) / batch_size))
    for batch_i in tqdm(range(n_batches)):
        curr_data = data[batch_i*batch_size:(batch_i+1)*batch_size]
        responses["ids"].extend([sample["id"] for sample in curr_data])

        if "Unanswerablity-Reason" in data[0].keys():
            responses["Unanswerablity-Reason"].extend([sample["Unanswerablity-Reason"] for sample in curr_data])

        responses["Adversarial"].extend(HF_request([sample['Adversarial'] for sample in curr_data], **kwargs))
        responses["Pseudo-Adversarial"].extend(HF_request([sample['Pseudo-Adversarial'] for sample in curr_data], **kwargs))
        responses["Ablation1"].extend(HF_request([sample['Ablation1'] for sample in curr_data], **kwargs))
        responses["Ablation2"].extend(HF_request([sample['Ablation2'] for sample in curr_data], **kwargs))

        # Chain-of-Thought prompts
        if args.CoT_prompt:
            responses["Adversarial-CoT"].extend(HF_request([sample['Adversarial-CoT'] for sample in curr_data], **kwargs))
            responses["Pseudo-Adversarial-CoT"].extend(HF_request([sample['Pseudo-Adversarial-CoT'] for sample in curr_data], **kwargs))
            responses["Ablation1-CoT"].extend(HF_request([sample['Ablation1-CoT'] for sample in curr_data], **kwargs))
            responses["Ablation2-CoT"].extend(HF_request([sample['Ablation2-CoT'] for sample in curr_data], **kwargs))
        else:
            responses["Adversarial-CoT"].extend([""]*batch_size)
            responses["Pseudo-Adversarial-CoT"].extend([""]*batch_size)
            responses["Ablation1-CoT"].extend([""]*batch_size)
            responses["Ablation2-CoT"].extend([""]*batch_size)            

        # Binary Answerability prompts ("Is it answerable?")
        if args.binary_answerability_prompt:
            responses["Answerability"].extend(HF_request([sample['Answerability'] for sample in curr_data], **kwargs))
            
            if args.CoT_prompt:
                responses["Answerability-CoT"].extend(HF_request([sample['Answerability-CoT'] for sample in curr_data], **kwargs))
            else:
                responses["Answerability-CoT"].extend([""]*batch_size)
        else:
            responses["Answerability"].extend([""]*batch_size)
            responses["Answerability-CoT"].extend([""]*batch_size)

        responses["Passage"].extend([squad_Passage(sample['Adversarial']) for sample in curr_data])
        responses["Question"].extend([squad_Question(sample['Adversarial']) for sample in curr_data])

    return responses



def get_responses_unanswerable_questions_NQ(data_path, args, **kwargs):

    def NQ_Passage(full_prompt):
        return full_prompt.split("Passage:")[-1].strip().split("Question:")[0].strip()

    def NQ_Question(full_prompt):
        return full_prompt.split("Question:")[-1].strip().split("Answer:")[0].strip()

    batch_size = args.batch_size

    responses = {"ids":[], "annotation_ids":[], 
                    "Adversarial":[], "Adversarial-CoT":[],
                    "Pseudo-Adversarial":[], "Pseudo-Adversarial-CoT":[],
                    "Ablation1":[], "Ablation1-CoT":[],
                    "Ablation2":[], "Ablation2-CoT":[],
                    "Answerability":[], "Answerability-CoT":[],
                    "Passage":[], "Question":[]}

    with open(data_path) as f:
        data = json.load(f)
    
    if args.n_instances != None:
        data = data[:args.n_instances]

    n_batches = int(np.ceil(len(data) / batch_size))

    for batch_i in tqdm(range(n_batches)):

        curr_data = data[batch_i*batch_size:(batch_i+1)*batch_size]
        responses["ids"].extend([sample["example_id"] for sample in curr_data])
        responses["annotation_ids"].extend([sample["annotation_id"] for sample in curr_data])

        responses["Adversarial"].extend(HF_request([sample['Adversarial'] for sample in curr_data], **kwargs))
        responses["Pseudo-Adversarial"].extend(HF_request([sample['Pseudo-Adversarial'] for sample in curr_data], **kwargs))
        responses["Ablation1"].extend(HF_request([sample['Ablation1'] for sample in curr_data], **kwargs))
        responses["Ablation2"].extend(HF_request([sample['Ablation2'] for sample in curr_data], **kwargs))

        # Chain-of-Thought prompts
        if args.CoT_prompt:
            responses["Adversarial-CoT"].extend(HF_request([sample['Adversarial-CoT'] for sample in curr_data], **kwargs))
            responses["Pseudo-Adversarial-CoT"].extend(HF_request([sample['Pseudo-Adversarial-CoT'] for sample in curr_data], **kwargs))
            responses["Ablation1-CoT"].extend(HF_request([sample['Ablation1-CoT'] for sample in curr_data], **kwargs))
            responses["Ablation2-CoT"].extend(HF_request([sample['Ablation2-CoT'] for sample in curr_data], **kwargs))
        else:
            responses["Adversarial-CoT"].extend([""]*batch_size)
            responses["Pseudo-Adversarial-CoT"].extend([""]*batch_size)
            responses["Ablation1-CoT"].extend([""]*batch_size)
            responses["Ablation2-CoT"].extend([""]*batch_size)


        # Binary Answerability prompts ("Is it answerable?")
        if args.binary_answerability_prompt:
            responses["Answerability"].extend(HF_request([sample['Answerability'] for sample in curr_data], **kwargs))
            
            if args.CoT_prompt:
                responses["Answerability-CoT"].extend(HF_request([sample['Answerability-CoT'] for sample in curr_data], **kwargs))
            else:
                responses["Answerability-CoT"].extend([""]*batch_size)
        else:
            responses["Answerability"].extend([""]*batch_size)
            responses["Answerability-CoT"].extend([""]*batch_size)


        responses["Passage"].extend([NQ_Passage(sample['Adversarial']) for sample in curr_data])
        responses["Question"].extend([NQ_Question(sample['Adversarial']) for sample in curr_data])

    return responses


def get_responses_unanswerable_questions_musique(data_path, args, **kwargs):


    def musique_Context(full_prompt):
        return full_prompt.split("Context:")[-1].strip().split("Question:")[0].strip()

    def musique_Question(full_prompt):
        return full_prompt.split("Question:")[-1].strip().split("Answer:")[0].strip()


    batch_size = args.batch_size

    responses = {"ids":[], 
                    "Adversarial":[], "Adversarial-CoT":[],
                    "Pseudo-Adversarial":[], "Pseudo-Adversarial-CoT":[],
                    "Ablation1":[], "Ablation1-CoT":[],
                    "Ablation2":[], "Ablation2-CoT":[],
                    "Answerability":[], "Answerability-CoT":[],
                    "Context":[], "Question":[]}


    with open(data_path) as f:
        data = json.load(f)
    
    if args.n_instances != None:
        data = data[:args.n_instances]

    n_batches = int(np.ceil(len(data) / batch_size))

    for batch_i in tqdm(range(n_batches)):
        curr_data = data[batch_i*batch_size:(batch_i+1)*batch_size]
        responses["ids"].extend([sample["id"] for sample in curr_data])

        responses["Adversarial"].extend(HF_request([sample['Adversarial'] for sample in curr_data], **kwargs))
        responses["Pseudo-Adversarial"].extend(HF_request([sample['Pseudo-Adversarial'] for sample in curr_data], **kwargs))
        responses["Ablation1"].extend(HF_request([sample['Ablation1'] for sample in curr_data], **kwargs))
        responses["Ablation2"].extend(HF_request([sample['Ablation2'] for sample in curr_data], **kwargs))

        # Chain-of-Thought prompts
        if args.CoT_prompt:
            responses["Adversarial-CoT"].extend(HF_request([sample['Adversarial-CoT'] for sample in curr_data], **kwargs))
            responses["Pseudo-Adversarial-CoT"].extend(HF_request([sample['Pseudo-Adversarial-CoT'] for sample in curr_data], **kwargs))
            responses["Ablation1-CoT"].extend(HF_request([sample['Ablation1-CoT'] for sample in curr_data], **kwargs))
            responses["Ablation2-CoT"].extend(HF_request([sample['Ablation2-CoT'] for sample in curr_data], **kwargs))
        else:
            responses["Adversarial-CoT"].extend([""]*batch_size)
            responses["Pseudo-Adversarial-CoT"].extend([""]*batch_size)
            responses["Ablation1-CoT"].extend([""]*batch_size)
            responses["Ablation2-CoT"].extend([""]*batch_size)

        # Binary Answerability prompts ("Is it answerable?")
        if args.binary_answerability_prompt:
            responses["Answerability"].extend(HF_request([sample['Answerability'] for sample in curr_data], **kwargs))

            if args.CoT_prompt:
                responses["Answerability-CoT"].extend(HF_request([sample['Answerability-CoT'] for sample in curr_data], **kwargs))
            else:
                responses["Answerability-CoT"].extend([""]*batch_size)
        else:
            responses["Answerability"].extend([""]*batch_size)
            responses["Answerability-CoT"].extend([""]*batch_size)

        responses["Context"].extend([musique_Context(sample['Adversarial']) for sample in curr_data])
        responses["Question"].extend([musique_Question(sample['Adversarial']) for sample in curr_data])

    return responses






def HF_request(prompts, k_beams, tokenizer, model, output_max_length, prompt_suffix, return_only_generated_text):
    prompts = [f"{p}{prompt_suffix}" for p in prompts]
    input_ids = tokenizer.batch_encode_plus(prompts, 
                                            padding=True,
                                            truncation=True,
                                            return_tensors="pt")["input_ids"].to(model.device)
    
    outputs = model.generate(input_ids, 
                             num_return_sequences=k_beams, 
                             max_new_tokens=output_max_length, 
                             output_scores=True, 
                             output_hidden_states=True, 
                             return_dict_in_generate=True, 
                             num_beams=k_beams, 
                             early_stopping=True)
    
    outputs_logits = [s.to("cpu") for s in outputs.scores]
    if "decoder_hidden_states" in outputs.keys(): # in the case of encoder-decoder models
        outputs_last_hidden_embeddings = [s[-1][:,-1,:].to("cpu") for s in outputs.decoder_hidden_states]
    else: # in the case of decoder-only models
        outputs_last_hidden_embeddings = [s[-1][:,-1,:].to("cpu") for s in outputs.hidden_states]
    
    if model.config.model_type in ["t5", "bart", "led"]:
        outputs_sequences = outputs.sequences.to("cpu")
    else:
        outputs_sequences = outputs.sequences[:,input_ids.shape[1]:].to("cpu")

    decoded_outputs = tokenizer.batch_decode(outputs_sequences, skip_special_tokens=True)

    batch_size = int(len(decoded_outputs)/k_beams)

    return_dicts = []
    for batch_i in range(batch_size):
        curr_return_dict = {"outputs":decoded_outputs[batch_i*k_beams:(batch_i+1)*k_beams]}
        if not return_only_generated_text:
        
            curr_return_dict["all_outputs_ids"] = outputs_sequences[batch_i*k_beams:(batch_i+1)*k_beams]
            curr_return_dict["full_logits"] = [curr_logits[batch_i*k_beams] for curr_logits in outputs_logits]
            curr_return_dict["last_hidden_embedding"] = [curr_last_hidden_embedding[batch_i*k_beams] for curr_last_hidden_embedding in outputs_last_hidden_embeddings]

            # when we do beam search/batch size>1 - to remove "excess" padding (make all the beams of the size of the longest output)
            if torch.any(curr_return_dict["all_outputs_ids"] == 1): # if this is false - it means all beams reach output_max_length and were cut off before </s> was generated.
                curr_max_sentence = int(torch.max((curr_return_dict["all_outputs_ids"] == 1).nonzero(as_tuple=False)[:, 1]))
                curr_return_dict["all_outputs_ids"] = curr_return_dict["all_outputs_ids"][:,:curr_max_sentence+1]
                curr_return_dict["full_logits"] = curr_return_dict["full_logits"][:curr_max_sentence]     
                curr_return_dict["last_hidden_embedding"] = curr_return_dict["last_hidden_embedding"][:curr_max_sentence] 
                    
        return_dicts.append(curr_return_dict)
    return return_dicts


def get_model(args, model_name):
    
    if model_name == "Flan-UL2":
        tokenizer_UL2 = AutoTokenizer.from_pretrained("google/flan-ul2", model_max_length=args.model_max_length)
        # max_memory_dict = {0:'20GiB', 1:"40GiB"}
        # max_memory_dict['cpu'] = '300GiB'
        max_memory_dict = {gpu_i:f"{MAX_GPU_MEM}GiB" for gpu_i in range(torch.cuda.device_count())}
        max_memory_dict['cpu'] = f'{MAX_CPU_MEM}GiB'
        model_UL2 = AutoModelForSeq2SeqLM.from_pretrained("google/flan-ul2",
                            device_map='auto',
                            max_memory=max_memory_dict,
                            torch_dtype=torch.float16)

        return {"output_subdir":"UL2_Flan", "kwargs":dict(tokenizer=tokenizer_UL2, model=model_UL2, prompt_suffix="")}

    if model_name == "Flan-T5-xxl":
        tokenizer_flan_t5_xxl = AutoTokenizer.from_pretrained("google/flan-t5-xxl", model_max_length=args.model_max_length)
        # max_memory_dict = {0:'20GiB', 1:"40GiB"}
        # max_memory_dict['cpu'] = '300GiB'
        max_memory_dict = {gpu_i:f"{MAX_GPU_MEM}GiB" for gpu_i in range(torch.cuda.device_count())}
        max_memory_dict['cpu'] = f'{MAX_CPU_MEM}GiB'
        model_flan_t5_xxl = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xxl",
                            device_map='auto',
                            max_memory=max_memory_dict)
        return {"output_subdir":"T5_xxl_Flan", "kwargs":dict(tokenizer=tokenizer_flan_t5_xxl, model=model_flan_t5_xxl, prompt_suffix="")}



    if model_name == "OPT":
        tokenizer_OPT = AutoTokenizer.from_pretrained("facebook/opt-iml-max-30b", model_max_length=args.model_max_length, padding_side='left')
        # max_memory_dict = {0:'20GiB', 1:"40GiB"}
        # max_memory_dict['cpu'] = '300GiB'
        max_memory_dict = {gpu_i:f"{MAX_GPU_MEM}GiB" for gpu_i in range(torch.cuda.device_count())}
        max_memory_dict['cpu'] = f'{MAX_CPU_MEM}GiB'
        model_OPT = AutoModelForCausalLM.from_pretrained(
                "facebook/opt-iml-max-30b",
                device_map='auto',
                max_memory=max_memory_dict,
                torch_dtype=torch.float16)
        return {"output_subdir":"OPT", "kwargs":dict(tokenizer=tokenizer_OPT, model=model_OPT, prompt_suffix="\n Answer:")}



    if model_name == "OPT-1-3B":
        tokenizer_OPT_1_3B = AutoTokenizer.from_pretrained("facebook/opt-iml-max-1.3b", model_max_length=args.model_max_length, padding_side='left')
        # max_memory_dict = {0:'20GiB', 1:"40GiB"}
        # max_memory_dict['cpu'] = '300GiB'
        max_memory_dict = {gpu_i:f"{MAX_GPU_MEM}GiB" for gpu_i in range(torch.cuda.device_count())}
        max_memory_dict['cpu'] = f'{MAX_CPU_MEM}GiB'
        model_OPT_1_3B = AutoModelForCausalLM.from_pretrained(
                "facebook/opt-iml-max-1.3b",
                device_map='auto',
                max_memory=max_memory_dict,
                torch_dtype=torch.float16)
        return {"output_subdir":"OPT_1_3B", "kwargs":dict(tokenizer=tokenizer_OPT_1_3B, model=model_OPT_1_3B, prompt_suffix="\n Answer:")} 



    # for debugging:
    if model_name == "Flan-T5-small":

        tokenizer_flan_t5_small = AutoTokenizer.from_pretrained("google/flan-t5-small", model_max_length=args.model_max_length)
        # max_memory_dict = {0:'20GiB', 1:"40GiB"}
        # max_memory_dict['cpu'] = '300GiB'
        max_memory_dict = {gpu_i:f"{MAX_GPU_MEM}GiB" for gpu_i in range(torch.cuda.device_count())}
        max_memory_dict['cpu'] = f'{MAX_CPU_MEM}GiB'
        model_flan_t5_small = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small",
                            device_map='auto',
                            max_memory=max_memory_dict)

        return {"output_subdir":"flan_t5_small", "kwargs":dict(tokenizer=tokenizer_flan_t5_small, model=model_flan_t5_small, prompt_suffix="")}

    raise Exception(f"Incorrect model passed: {model_name}")




def get_all_relevant_datasets(args):
    data_list = list()
    if "squad" in args.datasets:
        if args.adversarial:
            data_list.append({"type": "adversarial", "data_name":"squad", "get_data_function":get_responses_unanswerable_questions_squad})
        if args.control_group:
            data_list.append({"type": "control_group", "data_name":"squad", "get_data_function":get_responses_unanswerable_questions_squad})

    if "NQ" in args.datasets:
        if args.adversarial:
            data_list.append({"type": "adversarial", "data_name":"NQ", "get_data_function":get_responses_unanswerable_questions_NQ})
        if args.control_group:
            data_list.append({"type": "control_group", "data_name":"NQ", "get_data_function":get_responses_unanswerable_questions_NQ})

    if "musique" in args.datasets:
        if args.adversarial:
            data_list.append({"type": "adversarial", "data_name":"musique", "get_data_function":get_responses_unanswerable_questions_musique})
        if args.control_group:
            data_list.append({"type": "control_group", "data_name":"musique", "get_data_function":get_responses_unanswerable_questions_musique})

    return data_list



def main(args):

    now = datetime.now()
    now_str = now.strftime("%d-%m-%Y_%H:%M:%S")
    outdir_path = args.outdir if args.outdir else os.path.join("responses_embeddings", "k-beams", now_str)
    path = Path(outdir_path)
    path.mkdir(parents=True, exist_ok=True)
    logging.info(f'saved to: {outdir_path}')

    datasets_list = get_all_relevant_datasets(args)

    if args.k_beams_grid_search is None:
        k_beams_list = [args.k_beams]
    else:
        k_beams_list = json.loads(args.k_beams_grid_search)
    
    model = None
    for model_name in args.models:
        if model: # free up memory to enable loading the next model
            del model['kwargs']['model']
            gc.collect()
            torch.cuda.empty_cache()
        model = get_model(args, model_name)
        for p_variant in args.prompt_variant:
            for icl_variant in args.icl_examples_variant:
                for k_beams in k_beams_list:
                    for dataset in datasets_list:
                        print(f"model: {model['output_subdir']} data: {dataset['data_name']} type: {dataset['type']} variant: {p_variant} icl_examples_variant: {icl_variant} k_beams: {k_beams}")
                        
                        if args.all_instances:
                            outdir_suffix = "_all"
                        elif args.unfiltered_instances:
                            outdir_suffix = "_unfiltered"
                        else:
                            outdir_suffix = ""

                        curr_outdir = os.path.join(outdir_path, model['output_subdir'], "few_shot_with_instructions", f"k_beams_{k_beams}", p_variant, f"icl_examples_v{icl_variant}")
                        path = Path(curr_outdir)
                        path.mkdir(parents=True, exist_ok=True)
                        curr_outdir = os.path.join(curr_outdir, f"{dataset['type']}_{dataset['data_name']}_embeddings_{outdir_suffix}_k_beams_{k_beams}.pt")
 
                        if os.path.exists(curr_outdir):
                            print(f"{curr_outdir} exists! skipping...")
                            continue
                        
                        if args.all_instances:
                            data_adversarial_path = fr"generated_prompts/all/few_shot_with_instructions/{p_variant}/{dataset['data_name']}_{dataset['type']}_icl_examples_v{icl_variant}_all.json"
                        elif args.unfiltered_instances:
                            data_adversarial_path = fr"generated_prompts/unfiltered/few_shot_with_instructions/{p_variant}/{dataset['data_name']}_{dataset['type']}_unfiltered.json"
                        else:
                            data_adversarial_path = fr"generated_prompts/filtered/few_shot_with_instructions/{p_variant}/{dataset['data_name']}_{dataset['type']}_filtered.json"
                        
                        print(f"data loaded is: {data_adversarial_path}")

                        responses = dataset['get_data_function'](data_path=data_adversarial_path, 
                                                                 args=args,
                                                                 output_max_length=args.output_max_length, 
                                                                 k_beams = k_beams, 
                                                                 tokenizer=model['kwargs']['tokenizer'], 
                                                                 model=model['kwargs']['model'], 
                                                                 prompt_suffix=model['kwargs']['prompt_suffix'], 
                                                                 return_only_generated_text=args.return_only_generated_text)
                        
                        
                        torch.save(responses, curr_outdir) # and to load it: loaded_dict = torch.load(curr_outdir)





if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument('--outdir', type=str, default=None, help='outdir to save results')
    argparser.add_argument("--models", nargs='+', type=str, default=["Flan-T5-small"], help="which models to send requests to. any from: Flan-UL2, Flan-T5-xxl, OPT, Flan-T5-small, and OPT-1-3B.")
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
    argparser.add_argument("--icl-examples-variant", nargs='+', type=str, default=["1"], help="in-context-learning variant list (any of 1, 2, 3).")
    argparser.add_argument("--return-only-generated-text", action='store_true', default=False, help="whether to return only the generated text, without the logits (in cases of OOM)")
    argparser.add_argument("--batch-size", type=int, default=1, help="size of batch.")
    argparser.add_argument("--model-max-length", type=int, default=2048, help="max input length of model (for datasets like NQ where inputs are very long).")
    argparser.add_argument("--output-max-length", type=int, default=100, help="max output length.")
    argparser.add_argument("--CoT-prompt", action='store_true', default=False, help="whether to also send CoT prompt")
    argparser.add_argument("--binary-answerability-prompt", action='store_true', default=False, help="whether to also send the binary answerability prompt ('Is the question answerable by the passage?').")
    args = argparser.parse_args()
    main(args)







