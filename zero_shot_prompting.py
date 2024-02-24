from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import numpy as np
from tqdm import tqdm
import torch
from datetime import datetime
import gc
import json
import os
import argparse
from pathlib import Path
import logging
from utils import *
from post_processing.pt_to_benchmarks_evaluate_format import main as pt_to_evaluate_format_converter

# Set the logging level to INFO
logging.basicConfig(level=logging.INFO)

def get_responses_unanswerable_questions_squad(data_path, p_variant, data_type, args, **kwargs):

    def squad_Passage(full_prompt):
        return full_prompt[full_prompt.index("Passage:"):full_prompt.index("Question:")].replace("Passage:", "").strip()

    def squad_Question(full_prompt):
        return full_prompt[full_prompt.index("Question:"):].replace("Question:", "").strip()

    responses = {"ids":[], 
                 "Regular-Prompt":[], 
                 "Hint-Prompt":[], 
                 "CoT-Prompt":[], 
                 "Answerability":[], 
                 "Passage":[], 
                 "Question":[]}

    # get prompts
    with open("prompts/squad.json", 'r') as f1:
        prompt_dict = json.loads(f1.read())
    with open(f"raw_data/squad/{args.split}.json", 'r') as f1:
        raw_data = json.loads(f1.read())
    data = construct_prompts(prompt_dict=prompt_dict,
                             raw_data=raw_data,
                             zero_shot=True,
                             data_type=data_type,
                             prompt_variant=p_variant)
    
    if args.n_instances != None:
        data = data[:args.n_instances]

    # the answerable instances don't have this parameter
    if "Unanswerablity-Reason" in data[0].keys():
        responses["Unanswerablity-Reason"] = []

    n_batches = int(np.ceil(len(data) / args.batch_size))
    for batch_i in tqdm(range(n_batches)):
        curr_data = data[batch_i*args.batch_size:(batch_i+1)*args.batch_size]
        responses["ids"].extend([sample["id"] for sample in curr_data])

        if "Unanswerablity-Reason" in data[0].keys():
            responses["Unanswerablity-Reason"].extend([sample["Unanswerablity-Reason"] for sample in curr_data])

        responses["Regular-Prompt"].extend(HF_request([sample['Regular-Prompt'] for sample in curr_data], **kwargs))
        responses["Hint-Prompt"].extend(HF_request([sample['Hint-Prompt'] for sample in curr_data], **kwargs))
        
        # CoT-like prompt
        if args.CoT_prompt:
            responses["CoT-Prompt"].extend(HF_request([sample['CoT-Prompt'] for sample in curr_data], **kwargs))
        else:
            responses["CoT-Prompt"].extend([""]*args.batch_size)
        
        # Binary Answerability prompts ("Is it answerable?")
        if args.binary_answerability_prompt:
            responses["Answerability"].extend(HF_request([sample['Answerability'] for sample in curr_data], **kwargs))
        else:
            responses["Answerability"].extend([""]*args.batch_size)

        responses["Passage"].extend([squad_Passage(sample['Regular-Prompt']) for sample in curr_data])
        responses["Question"].extend([squad_Question(sample['Regular-Prompt']) for sample in curr_data])
    return responses

def get_responses_unanswerable_questions_NQ(data_path, p_variant, data_type, args, **kwargs):

    def NQ_Passage(full_prompt):
        return full_prompt[full_prompt.index("Passage:"):full_prompt.index("Question:")].replace("Passage:", "").strip()

    def NQ_Question(full_prompt):
        return full_prompt[full_prompt.index("Question:"):].replace("Question:", "").strip()

    responses = {"ids":[], 
                 "annotation_ids":[], 
                 "Regular-Prompt":[], 
                 "Hint-Prompt":[], 
                 "CoT-Prompt":[], 
                 "Answerability":[], 
                 "Passage":[], 
                 "Question":[]}
    
    # get prompts
    with open("prompts/NQ.json", 'r') as f1:
        prompt_dict = json.loads(f1.read())
    with open(f"raw_data/NQ/{args.split}.json", 'r') as f1:
        raw_data = json.loads(f1.read())
    data = construct_prompts(prompt_dict=prompt_dict,
                             raw_data=raw_data,
                             zero_shot=True,
                             data_type=data_type,
                             prompt_variant=p_variant)


    if args.n_instances != None:
        data = data[:args.n_instances]

    n_batches = int(np.ceil(len(data) / args.batch_size))

    for batch_i in tqdm(range(n_batches)):
        curr_data = data[batch_i*args.batch_size:(batch_i+1)*args.batch_size]
        responses["ids"].extend([sample["id"] for sample in curr_data])
        responses["annotation_ids"].extend([sample["annotation_id"] for sample in curr_data])

        responses["Regular-Prompt"].extend(HF_request([sample['Regular-Prompt'] for sample in curr_data], **kwargs))
        responses["Hint-Prompt"].extend(HF_request([sample['Hint-Prompt'] for sample in curr_data], **kwargs))

        # CoT-like prompt
        if args.CoT_prompt:
            responses["CoT-Prompt"].extend(HF_request([sample['CoT-Prompt'] for sample in curr_data], **kwargs))   
        else:
            responses["CoT-Prompt"].extend([""]*args.batch_size)
        
        # Binary Answerability prompts ("Is it answerable?")
        if args.binary_answerability_prompt:
            responses["Answerability"].extend(HF_request([sample['Answerability'] for sample in curr_data], **kwargs))           
        else:
            responses["Answerability"].extend([""]*args.batch_size)

        responses["Passage"].extend([NQ_Passage(sample['Regular-Prompt']) for sample in curr_data])
        responses["Question"].extend([NQ_Question(sample['Regular-Prompt']) for sample in curr_data])
    return responses

def get_responses_unanswerable_questions_musique(data_path, p_variant, data_type, args, **kwargs):

    def musique_Context(full_prompt):
        return full_prompt[full_prompt.index("Context:"):full_prompt.index("Question:")].replace("Context:", "").strip()

    def musique_Question(full_prompt):
        return full_prompt[full_prompt.index("Question:"):].replace("Question:", "").strip()

    responses = {"ids":[], 
                 "Regular-Prompt":[], 
                 "Hint-Prompt":[], 
                 "CoT-Prompt":[], 
                 "Answerability":[], 
                 "Context":[], 
                 "Question":[]}

    # get prompts
    with open("prompts/musique.json", 'r') as f1:
        prompt_dict = json.loads(f1.read())
    with open(f"raw_data/musique/{args.split}.json", 'r') as f1:
        raw_data = json.loads(f1.read())
    data = construct_prompts(prompt_dict=prompt_dict,
                             raw_data=raw_data,
                             zero_shot=True,
                             data_type=data_type,
                             prompt_variant=p_variant)

    if args.n_instances != None:
        data = data[:args.n_instances]

    n_batches = int(np.ceil(len(data) / args.batch_size))

    for batch_i in tqdm(range(n_batches)):
        curr_data = data[batch_i*args.batch_size:(batch_i+1)*args.batch_size]
        responses["ids"].extend([sample["id"] for sample in curr_data])

        responses["Regular-Prompt"].extend(HF_request([sample['Regular-Prompt'] for sample in curr_data], **kwargs))
        responses["Hint-Prompt"].extend(HF_request([sample['Hint-Prompt'] for sample in curr_data], **kwargs))

        # CoT-like prompt
        if args.CoT_prompt:
            responses["CoT-Prompt"].extend(HF_request([sample['CoT-Prompt'] for sample in curr_data], **kwargs))
        else:       
            responses["CoT-Prompt"].extend([""]*args.batch_size)

        # Binary Answerability prompts ("Is it answerable?")
        if args.binary_answerability_prompt:
            responses["Answerability"].extend(HF_request([sample['Answerability'] for sample in curr_data], **kwargs))        
        else:
            responses["Answerability"].extend([""]*args.batch_size)

        responses["Context"].extend([musique_Context(sample['Regular-Prompt']) for sample in curr_data])
        responses["Question"].extend([musique_Question(sample['Regular-Prompt']) for sample in curr_data])
    return responses

def HF_request(prompts, k_beams, tokenizer, model, output_max_length, prompt_suffix, return_only_generated_text, return_first_layer):
    prompts = [f"{p}{prompt_suffix}" if not p.strip().endswith(prompt_suffix) else p for p in prompts]

    input_ids = tokenizer.batch_encode_plus(
        prompts, 
        padding=True,
        truncation=True,
        return_tensors="pt")["input_ids"].to(model.device)
    
    outputs = model.generate(input_ids, 
                             num_return_sequences=k_beams, 
                             max_new_tokens=output_max_length, 
                             output_scores=True, 
                             return_dict_in_generate=True, 
                             output_hidden_states=True, 
                             num_beams=k_beams, 
                             early_stopping=True)
    
    outputs_logits = [s.to("cpu") for s in outputs.scores]

    if "decoder_hidden_states" in outputs.keys(): # in the case of encoder-decoder models (Flan-T5-xxl and Flun-UL2)
        outputs_last_hidden_embeddings = [s[-1][:,-1,:].to("cpu") for s in outputs.decoder_hidden_states]
        outputs_first_hidden_embeddings = outputs.encoder_hidden_states[0]
    else: # in the case of decoder-only models (OPT-IML)
        outputs_last_hidden_embeddings = [s[-1][:,-1,:].to("cpu") for s in outputs.hidden_states]
        outputs_first_hidden_embeddings = [outputs.hidden_states[0][0][batch_i, :,:] for batch_i in range(len(outputs.hidden_states[0][0]))]

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
            curr_return_dict["full_logits"] = [torch.stack([curr_logits[batch_i+beam_i] for curr_logits in outputs_logits]) for beam_i in range(k_beams)]
            if return_first_layer:
                curr_return_dict["first_hidden_embedding"] = outputs_first_hidden_embeddings[batch_i].mean(dim=0) # currently not supported for beam search
            else:
                curr_return_dict["last_hidden_embedding"] = [torch.stack([curr_last_hidden_embedding[batch_i+beam_i] for curr_last_hidden_embedding in outputs_last_hidden_embeddings]) for beam_i in range(k_beams)]
        return_dicts.append(curr_return_dict)
    return return_dicts

def get_model(args, model_name):
    model_map = {"Flan-UL2" : "google/flan-ul2",
                 "Flan-T5-xxl" : "google/flan-t5-xxl",
                 "Flan-T5-small" : "google/flan-t5-small",
                 "OPT-IML" : "facebook/opt-iml-max-30b"}

    if not model_name in model_map.keys():
        raise Exception(f"Incorrect model passed: {model_name}")
    # max_memory_dict = {0:'20GiB', 1:"40GiB"}
    # max_memory_dict['cpu'] = '300GiB'
    max_memory_dict = get_max_memory() 

    if model_name == "OPT-IML":
        curr_prompt_suffix = "\n Answer:"
        curr_tokenizer = AutoTokenizer.from_pretrained(model_map[model_name], model_max_length=args.model_max_length, padding_side='left')
        curr_model = AutoModelForCausalLM.from_pretrained(model_map[model_name],
                                                          device_map='auto',
                                                          max_memory=max_memory_dict,
                                                          torch_dtype=torch.float16)
    else:
        curr_prompt_suffix = ""
        curr_tokenizer = AutoTokenizer.from_pretrained(model_map[model_name], model_max_length=args.model_max_length)
        if model_name == "Flan-T5-xxl":
            curr_model = AutoModelForSeq2SeqLM.from_pretrained(model_map[model_name],
                                                               device_map='auto',
                                                               max_memory=max_memory_dict)
        else:
            curr_model = AutoModelForSeq2SeqLM.from_pretrained(model_map[model_name],
                                                               device_map='auto',
                                                               max_memory=max_memory_dict,
                                                               torch_dtype=torch.float16)            
    return {"output_subdir" : model_name, "kwargs":dict(tokenizer=curr_tokenizer, model=curr_model, prompt_suffix=curr_prompt_suffix)}

def get_all_relevant_datasets(args):
    data_function_map = {"squad" : get_responses_unanswerable_questions_squad,
                         "NQ" : get_responses_unanswerable_questions_NQ,
                         "musique" : get_responses_unanswerable_questions_musique}
    data_list = list()
    if not args.only_answerable_instances:
        data_list += [{"type": "un-answerable", "data_name":dataset, "get_data_function":data_function_map[dataset]} for dataset in args.datasets]
    if not args.only_unanswerable_instances:
        data_list += [{"type": "answerable", "data_name":dataset, "get_data_function":data_function_map[dataset]} for dataset in args.datasets]
    return data_list

def main(args):
    now = datetime.now()
    now_str = now.strftime("%d-%m-%Y_%H:%M:%S")
    outdir_path = args.outdir if args.outdir else os.path.join("generated_outputs", now_str)
    logging.info(f'saving to: {outdir_path}')
    datasets_list = get_all_relevant_datasets(args)
    k_beams_list = [args.k_beams] if args.k_beams_grid_search is None else json.loads(args.k_beams_grid_search)

    model = None
    for model_name in args.models:
        if model: # free up memory to enable loading the next model
            del model['kwargs']['model']
            gc.collect()
            torch.cuda.empty_cache()        
        model = get_model(args, model_name)
        for dataset in datasets_list:
            for p_variant in args.prompt_variant:
                for k_beams in k_beams_list:
                    print(f"model: {model['output_subdir']} data: {dataset['data_name']} type: {dataset['type']} variant: {p_variant} beam: {k_beams}")
                    # create directory
                    curr_outdir = os.path.join(outdir_path, model['output_subdir'], "zero_shot", f"k_beams_{k_beams}", p_variant)
                    path = Path(curr_outdir)
                    path.mkdir(parents=True, exist_ok=True)
                    curr_outdir = os.path.join(curr_outdir, f"{dataset['type']}_{dataset['data_name']}_{args.split}.pt")
                    
                    if os.path.exists(curr_outdir):
                        print(f"{curr_outdir} exists! skipping...")
                        continue
                    
                    data_path = fr"data/prompts/{dataset['data_name']}/zero_shot/{args.split}.json"    
                    responses = dataset['get_data_function'](data_path=data_path,
                                                             p_variant=p_variant,
                                                             data_type=dataset['type'],
                                                             args=args, 
                                                             output_max_length=args.output_max_length, 
                                                             k_beams = k_beams, 
                                                             return_first_layer=args.return_first_layer, 
                                                             tokenizer=model['kwargs']['tokenizer'], 
                                                             model=model['kwargs']['model'], 
                                                             prompt_suffix=model['kwargs']['prompt_suffix'], 
                                                             return_only_generated_text=args.return_only_generated_text)
                    torch.save(responses, curr_outdir)

    # if not only_answerable_instances and not only_unanswerable_instances - namely we have both answerable and answerable prompts - then convert the pt files to the formats adhering to the evaluation scripts
    if not args.only_answerable_instances and not args.only_unanswerable_instances:
        pt_to_evaluate_format_converter(indirs=[outdir_path], is_beam_experiment=False)

        # if in beams larger than 1 - also run the conversion to the beam relaxation
        if [k for k in k_beams_list if k>1]:
            pt_to_evaluate_format_converter(indirs=[outdir_path], is_beam_experiment=True)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument('--outdir', type=str, default=None, help='outdir to save results')
    argparser.add_argument("--models", nargs='+', type=str, default=["Flan-T5-small"], help="which models to send requests to. any from: Flan-UL2, Flan-T5-xxl, OPT, Flan-T5-small, OPT-1-3B")
    argparser.add_argument("--datasets", nargs='+', type=str, default=["squad"], help="which datasets to work on. any from: squad, NQ, musique")
    argparser.add_argument("--n-instances", type=int, default=None, help="number of instances to process")
    argparser.add_argument("--k-beams", type=int, default=1, help="beam size (will also be the number of returned outputs to check \"unanswerable\" from)")
    argparser.add_argument("--k-beams-grid-search", type=str, default=None, help="grid search on the k-beams. Will overrun \"--k-beams\". Need to pass as a list (e.g. --k-beams-grid-search [4,5,6])")
    argparser.add_argument("--prompt-variant", nargs='+', type=str, default=["variant1"], help="prompt variant list (any of variant1, variant2, variant3).")
    argparser.add_argument("--return-only-generated-text", action='store_true', default=False, help="whether to return only the generated text, without the logits (in cases of OOM)")
    argparser.add_argument("--return-first-layer", action='store_true', default=False, help="whether to also return the first layer's (uncontextualized) embedding.")
    argparser.add_argument("--batch-size", type=int, default=1, help="size of batch.")
    argparser.add_argument("--model-max-length", type=int, default=2048, help="max input length of model (for datasets like NQ where inputs are very long).")
    argparser.add_argument("--output-max-length", type=int, default=100, help="max output length.")
    argparser.add_argument("--split", type=str, default="test", help="which of the data splits to use (train, dev or test).")
    argparser.add_argument("--only-answerable-instances", action='store_true', default=False, help="send only the answerable prompts.")
    argparser.add_argument("--only-unanswerable-instances", action='store_true', default=False, help="send only the un-answerable prompts.")
    argparser.add_argument("--CoT-prompt", action='store_true', default=False, help="whether to also send CoT prompt")
    argparser.add_argument("--binary-answerability-prompt", action='store_true', default=False, help="whether to also send the binary answerability prompt ('Is the question answerable by the passage?').")
    args = argparser.parse_args()
    main(args)







