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

def get_responses_unanswerable_questions_squad(data_path, p_variant, icl_variant, data_type, args, **kwargs):

    def squad_Passage(full_prompt):
        return full_prompt.split("Passage:")[-1].strip().split("Question:")[0].strip()

    def squad_Question(full_prompt):
        return full_prompt.split("Question:")[-1].strip().split("Answer:")[0].strip()
    
    batch_size = args.batch_size

    responses = {"ids":[], 
                 "Regular-Prompt":[], "Regular-Prompt-CoT":[],
                 "Hint-Prompt":[], "Hint-Prompt-CoT":[],
                 "Ablation1":[], "Ablation1-CoT":[],
                 "Ablation2":[], "Ablation2-CoT":[],
                 "Answerability":[], "Answerability-CoT":[],
                 "Passage":[], "Question":[]}

    # get prompts
    with open("prompts/squad.json", 'r') as f1:
        prompt_dict = json.loads(f1.read())
    with open(f"raw_data/squad/test.json", 'r') as f1:
        raw_data = json.loads(f1.read())
    data = construct_prompts(prompt_dict=prompt_dict,
                             raw_data=raw_data,
                             zero_shot=False,
                             data_type=data_type,
                             prompt_variant=p_variant,
                             demo_variant=icl_variant)

    if args.n_instances != None:
        data = data[:args.n_instances]

    # the answerable instances don't have this parameter
    if "Unanswerablity-Reason" in data[0].keys():
        responses["Unanswerablity-Reason"] = []

    n_batches = int(np.ceil(len(data) / batch_size))
    for batch_i in tqdm(range(n_batches)):
        curr_data = data[batch_i*batch_size:(batch_i+1)*batch_size]
        responses["ids"].extend([sample["id"] for sample in curr_data])

        if "Unanswerablity-Reason" in data[0].keys():
            responses["Unanswerablity-Reason"].extend([sample["Unanswerablity-Reason"] for sample in curr_data])

        responses["Regular-Prompt"].extend(HF_request([sample['Regular-Prompt'] for sample in curr_data], **kwargs))
        responses["Hint-Prompt"].extend(HF_request([sample['Hint-Prompt'] for sample in curr_data], **kwargs))
        responses["Ablation1"].extend(HF_request([sample['Ablation1'] for sample in curr_data], **kwargs))
        responses["Ablation2"].extend(HF_request([sample['Ablation2'] for sample in curr_data], **kwargs))

        # Chain-of-Thought prompts
        if args.CoT_prompt:
            responses["Regular-Prompt-CoT"].extend(HF_request([sample['Regular-Prompt-CoT'] for sample in curr_data], **kwargs))
            responses["Hint-Prompt-CoT"].extend(HF_request([sample['Hint-Prompt-CoT'] for sample in curr_data], **kwargs))
            responses["Ablation1-CoT"].extend(HF_request([sample['Ablation1-CoT'] for sample in curr_data], **kwargs))
            responses["Ablation2-CoT"].extend(HF_request([sample['Ablation2-CoT'] for sample in curr_data], **kwargs))
        else:
            responses["Regular-Prompt-CoT"].extend([""]*batch_size)
            responses["Hint-Prompt-CoT"].extend([""]*batch_size)
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

        responses["Passage"].extend([squad_Passage(sample['Regular-Prompt']) for sample in curr_data])
        responses["Question"].extend([squad_Question(sample['Regular-Prompt']) for sample in curr_data])

    return responses

def get_responses_unanswerable_questions_NQ(data_path, p_variant, icl_variant, data_type, args, **kwargs):

    def NQ_Passage(full_prompt):
        return full_prompt.split("Passage:")[-1].strip().split("Question:")[0].strip()

    def NQ_Question(full_prompt):
        return full_prompt.split("Question:")[-1].strip().split("Answer:")[0].strip()

    batch_size = args.batch_size

    responses = {"ids":[], "annotation_ids":[], 
                 "Regular-Prompt":[], "Regular-Prompt-CoT":[],
                 "Hint-Prompt":[], "Hint-Prompt-CoT":[],
                 "Ablation1":[], "Ablation1-CoT":[],
                 "Ablation2":[], "Ablation2-CoT":[],
                 "Answerability":[], "Answerability-CoT":[],
                 "Passage":[], "Question":[]}

    # get prompts
    with open("prompts/NQ.json", 'r') as f1:
        prompt_dict = json.loads(f1.read())
    with open(f"raw_data/NQ/test.json", 'r') as f1:
        raw_data = json.loads(f1.read())
    data = construct_prompts(prompt_dict=prompt_dict,
                             raw_data=raw_data,
                             zero_shot=False,
                             data_type=data_type,
                             prompt_variant=p_variant,
                             demo_variant=icl_variant)
    
    if args.n_instances != None:
        data = data[:args.n_instances]

    n_batches = int(np.ceil(len(data) / batch_size))

    for batch_i in tqdm(range(n_batches)):

        curr_data = data[batch_i*batch_size:(batch_i+1)*batch_size]
        responses["ids"].extend([sample["example_id"] for sample in curr_data])
        responses["annotation_ids"].extend([sample["annotation_id"] for sample in curr_data])

        responses["Regular-Prompt"].extend(HF_request([sample['Regular-Prompt'] for sample in curr_data], **kwargs))
        responses["Hint-Prompt"].extend(HF_request([sample['Hint-Prompt'] for sample in curr_data], **kwargs))
        responses["Ablation1"].extend(HF_request([sample['Ablation1'] for sample in curr_data], **kwargs))
        responses["Ablation2"].extend(HF_request([sample['Ablation2'] for sample in curr_data], **kwargs))

        # Chain-of-Thought prompts
        if args.CoT_prompt:
            responses["Regular-Prompt-CoT"].extend(HF_request([sample['Regular-Prompt-CoT'] for sample in curr_data], **kwargs))
            responses["Hint-Prompt-CoT"].extend(HF_request([sample['Hint-Prompt-CoT'] for sample in curr_data], **kwargs))
            responses["Ablation1-CoT"].extend(HF_request([sample['Ablation1-CoT'] for sample in curr_data], **kwargs))
            responses["Ablation2-CoT"].extend(HF_request([sample['Ablation2-CoT'] for sample in curr_data], **kwargs))
        else:
            responses["Regular-Prompt-CoT"].extend([""]*batch_size)
            responses["Hint-Prompt-CoT"].extend([""]*batch_size)
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

        responses["Passage"].extend([NQ_Passage(sample['Regular-Prompt']) for sample in curr_data])
        responses["Question"].extend([NQ_Question(sample['Regular-Prompt']) for sample in curr_data])

    return responses

def get_responses_unanswerable_questions_musique(data_path, p_variant, icl_variant, data_type, args, **kwargs):

    def musique_Context(full_prompt):
        return full_prompt.split("Context:")[-1].strip().split("Question:")[0].strip()

    def musique_Question(full_prompt):
        return full_prompt.split("Question:")[-1].strip().split("Answer:")[0].strip()

    batch_size = args.batch_size

    responses = {"ids":[], 
                 "Regular-Prompt":[], "Regular-Prompt-CoT":[],
                 "Hint-Prompt":[], "Hint-Prompt-CoT":[],
                 "Ablation1":[], "Ablation1-CoT":[],
                 "Ablation2":[], "Ablation2-CoT":[],
                 "Answerability":[], "Answerability-CoT":[],
                 "Context":[], "Question":[]}

    # get prompts
    with open("prompts/musique.json", 'r') as f1:
        prompt_dict = json.loads(f1.read())
    with open(f"raw_data/musique/test.json", 'r') as f1:
        raw_data = json.loads(f1.read())
    data = construct_prompts(prompt_dict=prompt_dict,
                             raw_data=raw_data,
                             zero_shot=False,
                             data_type=data_type,
                             prompt_variant=p_variant,
                             demo_variant=icl_variant)
    
    if args.n_instances != None:
        data = data[:args.n_instances]

    n_batches = int(np.ceil(len(data) / batch_size))

    for batch_i in tqdm(range(n_batches)):
        curr_data = data[batch_i*batch_size:(batch_i+1)*batch_size]
        responses["ids"].extend([sample["id"] for sample in curr_data])

        responses["Regular-Prompt"].extend(HF_request([sample['Regular-Prompt'] for sample in curr_data], **kwargs))
        responses["Hint-Prompt"].extend(HF_request([sample['Hint-Prompt'] for sample in curr_data], **kwargs))
        responses["Ablation1"].extend(HF_request([sample['Ablation1'] for sample in curr_data], **kwargs))
        responses["Ablation2"].extend(HF_request([sample['Ablation2'] for sample in curr_data], **kwargs))

        # Chain-of-Thought prompts
        if args.CoT_prompt:
            responses["Regular-Prompt-CoT"].extend(HF_request([sample['Regular-Prompt-CoT'] for sample in curr_data], **kwargs))
            responses["Hint-Prompt-CoT"].extend(HF_request([sample['Hint-Prompt-CoT'] for sample in curr_data], **kwargs))
            responses["Ablation1-CoT"].extend(HF_request([sample['Ablation1-CoT'] for sample in curr_data], **kwargs))
            responses["Ablation2-CoT"].extend(HF_request([sample['Ablation2-CoT'] for sample in curr_data], **kwargs))
        else:
            responses["Regular-Prompt-CoT"].extend([""]*batch_size)
            responses["Hint-Prompt-CoT"].extend([""]*batch_size)
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

        responses["Context"].extend([musique_Context(sample['Regular-Prompt']) for sample in curr_data])
        responses["Question"].extend([musique_Question(sample['Regular-Prompt']) for sample in curr_data])

    return responses

def HF_request(prompts, k_beams, tokenizer, model, output_max_length, prompt_suffix, return_only_generated_text):
    prompts = [f"{p}{prompt_suffix}" if not p.strip().endswith(prompt_suffix) else p for p in prompts]
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

    if "decoder_hidden_states" in outputs.keys(): # in the case of encoder-decoder models (Flan-T5-xxl and Flan-UL2)
        outputs_last_hidden_embeddings = [s[-1][:,-1,:].to("cpu") for s in outputs.decoder_hidden_states]
    else: # in the case of decoder-only models (OPT-IML)
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

            if torch.any(curr_return_dict["all_outputs_ids"] == 1): 
                curr_max_sentence = int(torch.max((curr_return_dict["all_outputs_ids"] == 1).nonzero(as_tuple=False)[:, 1]))
                curr_return_dict["all_outputs_ids"] = curr_return_dict["all_outputs_ids"][:,:curr_max_sentence+1]
                curr_return_dict["full_logits"] = curr_return_dict["full_logits"][:curr_max_sentence]     
                curr_return_dict["last_hidden_embedding"] = curr_return_dict["last_hidden_embedding"][:curr_max_sentence] 
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
        data_list += [{"type": "un-answerable", 
                       "data_name":dataset, 
                       "get_data_function":data_function_map[dataset]} for dataset in args.datasets]
        
    if not args.only_unanswerable_instances:
        data_list += [{"type": "answerable", 
                       "data_name":dataset, 
                       "get_data_function":data_function_map[dataset]} for dataset in args.datasets]
    return data_list

def main(args):
    now = datetime.now()
    now_str = now.strftime("%d-%m-%Y_%H:%M:%S")
    outdir_path = args.outdir if args.outdir else os.path.join("generated_outputs", now_str)
    logging.info(f'saving to: {outdir_path}')
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
                        
                        # create directory
                        curr_outdir = os.path.join(outdir_path, model['output_subdir'], "few_shot", f"k_beams_{k_beams}", p_variant, f"icl_examples_v{icl_variant}")
                        path = Path(curr_outdir)
                        path.mkdir(parents=True, exist_ok=True)
                        curr_outdir = os.path.join(curr_outdir, f"{dataset['type']}_{dataset['data_name']}.pt")
 
                        if os.path.exists(curr_outdir):
                            print(f"{curr_outdir} exists! skipping...")
                            continue

                        data_path = fr"data/prompts/{dataset['data_name']}/few_shot/test.json"    
                        responses = dataset['get_data_function'](data_path=data_path, 
                                                                 p_variant=p_variant,
                                                                 icl_variant=icl_variant,
                                                                 data_type=dataset['type'],
                                                                 args=args,
                                                                 output_max_length=args.output_max_length, 
                                                                 k_beams = k_beams, 
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
    argparser.add_argument("--models", nargs='+', type=str, default=["Flan-T5-small"], help="which models to send requests to. any from: Flan-UL2, Flan-T5-xxl, and OPT-IML.")
    argparser.add_argument("--datasets", nargs='+', type=str, default=["squad"], help="which datasets to work on. any from: squad, NQ, musique")
    argparser.add_argument("--n-instances", type=int, default=None, help="number of instances to process")
    argparser.add_argument("--k-beams", type=int, default=1, help="beam size (will also be the number of returned outputs to check \"unanswerable\" from)")
    argparser.add_argument("--k-beams-grid-search", type=str, default=None, help="grid search on the k-beams. Will overrun \"--k-beams\". Need to pass as a list (e.g. --k-beams-grid-search [4,5,6])")
    argparser.add_argument("--prompt-variant", nargs='+', type=str, default=["variant1"], help="prompt variant list (any of variant1, variant2, variant3).")
    argparser.add_argument("--icl-examples-variant", nargs='+', type=str, default=["1"], help="in-context-learning variant list (any of 1, 2, 3).")
    argparser.add_argument("--return-only-generated-text", action='store_true', default=False, help="whether to return only the generated text, without the logits (in cases of OOM)")
    argparser.add_argument("--batch-size", type=int, default=1, help="size of batch.")
    argparser.add_argument("--model-max-length", type=int, default=2048, help="max input length of model (for datasets like NQ where inputs are very long).")
    argparser.add_argument("--output-max-length", type=int, default=100, help="max output length.")
    argparser.add_argument("--only-answerable-instances", action='store_true', default=False, help="send only the answerable prompts.")
    argparser.add_argument("--only-unanswerable-instances", action='store_true', default=False, help="send only the un-answerable prompts.")
    argparser.add_argument("--CoT-prompt", action='store_true', default=False, help="whether to also send CoT prompt")
    argparser.add_argument("--binary-answerability-prompt", action='store_true', default=False, help="whether to also send the binary answerability prompt ('Is the question answerable by the passage?').")
    args = argparser.parse_args()
    main(args)







