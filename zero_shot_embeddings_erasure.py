from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Model
import numpy as np
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
from datetime import datetime
import gc
import json
import os
import pickle
import argparse
from pathlib import Path
import logging
from constants import *
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

    with open(data_path) as f:
        data = json.load(f)
        data = data[p_variant][data_type]
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

    with open(data_path) as f:
        data = json.load(f)
        data = data[p_variant][data_type]
    if args.n_instances != None:
        data = data[:args.n_instances]

    n_batches = int(np.ceil(len(data) / args.batch_size))
    for batch_i in tqdm(range(n_batches)):
        curr_data = data[batch_i*args.batch_size:(batch_i+1)*args.batch_size]
        responses["ids"].extend([sample["example_id"] for sample in curr_data])
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

    with open(data_path) as f:
        data = json.load(f)
        data = data[p_variant][data_type]
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

def HF_request(prompts, k_beams, tokenizer, model, lm_head, eraser, only_first_decoding, prompt_suffix):
    prompts = [f"{p}{prompt_suffix}" for p in prompts]
    input_ids = tokenizer.batch_encode_plus(prompts, 
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

def get_model(args, model_name):
    model_map = {"Flan-UL2" : "google/flan-ul2",
                 "Flan-T5-xxl" : "google/flan-t5-xxl"}

    if not model_name in model_map.keys():
        raise Exception(f"Incorrect model passed: {model_name}")

    # max_memory_dict = {0:'20GiB', 1:"40GiB"}
    # max_memory_dict['cpu'] = '300GiB'
    max_memory_dict = {gpu_i:f"{MAX_GPU_MEM}GiB" for gpu_i in range(torch.cuda.device_count())}
    max_memory_dict['cpu'] = f'{MAX_CPU_MEM}GiB'

    curr_prompt_suffix = ""
    curr_tokenizer = AutoTokenizer.from_pretrained(model_map[model_name], model_max_length=args.model_max_length)
    if model_name == "Flan-T5-xxl":
        curr_model = T5Model.from_pretrained(model_map[model_name],
                                                device_map='auto',
                                                max_memory=max_memory_dict)
        
        curr_model_head = AutoModelForSeq2SeqLM.from_pretrained(model_map[model_name],
                                                                device_map='auto',
                                                                max_memory={0:'10GiB', 'cpu':'300GiB'}).lm_head
    else:
        curr_model = T5Model.from_pretrained(model_map[model_name],
                                                device_map='auto',
                                                max_memory=max_memory_dict,
                                                torch_dtype=torch.float16)
        
        curr_model_head = AutoModelForSeq2SeqLM.from_pretrained(model_map[model_name],
                                                                device_map='auto',
                                                                max_memory={0:'10GiB', 'cpu':'300GiB'},
                                                                torch_dtype=torch.float16).lm_head
        
    return {"output_subdir" : model_name, "kwargs":dict(tokenizer=curr_tokenizer, model=curr_model, lm_head=curr_model_head, prompt_suffix=curr_prompt_suffix)}

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
    # Load the eraser from the file
    if args.no_eraser:
        eraser = None
    else:
        with open(args.eraser_dir, "rb") as file:
            eraser = pickle.load(file).to("cuda")

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
            for k_beams in k_beams_list:
                for dataset in datasets_list:
                    print(f"model: {model['output_subdir']} data: {dataset['data_name']} type: {dataset['type']} variant: {p_variant} beam: {k_beams}")
                    # create directory
                    curr_outdir = os.path.join(outdir_path, model['output_subdir'], "zero_shot", f"k_beams_{k_beams}", p_variant)
                    path = Path(curr_outdir)
                    path.mkdir(parents=True, exist_ok=True)
                    curr_outdir = os.path.join(curr_outdir, f"{dataset['type']}_{dataset['data_name']}_test.pt")

                    if os.path.exists(curr_outdir):
                        print(f"{curr_outdir} exists! skipping...")
                        continue
                    
                    data_path = fr"data/prompts/{dataset['data_name']}/zero_shot/test.json"
                    responses = dataset['get_data_function'](data_path=data_path, 
                                                             p_variant=p_variant,
                                                             data_type=dataset['type'],
                                                             args=args,
                                                             k_beams=k_beams, 
                                                             tokenizer=model['kwargs']['tokenizer'], 
                                                             model=model['kwargs']['model'], 
                                                             lm_head=model['kwargs']['lm_head'], 
                                                             eraser=eraser, 
                                                             only_first_decoding=args.only_first_decoding, 
                                                             prompt_suffix=model['kwargs']['prompt_suffix'])
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
    argparser.add_argument("--models", nargs='+', type=str, default=["Flan-T5-small"], help="which models to send requests to. any from: Flan-UL2, Flan-T5-xxl, OPT-IML, Flan-T5-small, OPT-1-3B and ChatGPT")
    argparser.add_argument("--datasets", nargs='+', type=str, default=["squad"], help="which datasets to work on. any from: squad, NQ, musique")
    argparser.add_argument("--n-instances", type=int, default=None, help="number of instances to process")
    argparser.add_argument("--k-beams", type=int, default=1, help="beam size (will also be the number of returned outputs to check \"unanswerable\" from)")
    argparser.add_argument("--k-beams-grid-search", type=str, default=None, help="grid search on the k-beams. Will overrun \"--k-beams\". Need to pass as a list (e.g. --k-beams-grid-search [4,5,6])")
    argparser.add_argument("--prompt-variant", nargs='+', type=str, default=["variant1"], help="prompt variant list (any of variant1, variant2, variant3).")
    argparser.add_argument("--batch-size", type=int, default=1, help="size of batch.")
    argparser.add_argument("--model-max-length", type=int, default=2048, help="max input length of model (for datasets like NQ where inputs are very long).")
    argparser.add_argument("--eraser-dir", type=str, required=True, help="path to eraser.")
    argparser.add_argument("--no-eraser", action='store_true', default=False, help="do not load eraser (for debugging)")
    argparser.add_argument("--only-first-decoding", action='store_true', default=False, help="perform erasure only on first decoding step.")
    argparser.add_argument("--only-answerable-instances", action='store_true', default=False, help="send only the answerable prompts.")
    argparser.add_argument("--only-unanswerable-instances", action='store_true', default=False, help="send only the un-answerable prompts.")
    argparser.add_argument("--CoT-prompt", action='store_true', default=False, help="whether to also send CoT prompt")
    argparser.add_argument("--binary-answerability-prompt", action='store_true', default=False, help="whether to also send the binary answerability prompt ('Is the question answerable by the passage?').")
    args = argparser.parse_args()
    main(args)







