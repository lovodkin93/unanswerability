import torch
import pandas as pd
from tqdm import tqdm
import os
import argparse
import json
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import PROMPT_TYPES, UNANSWERABLE_REPLIES, UNANSWERABLE_REPLIES_EXACT


def pt_to_csv_non_beam(indirs):
    for indir in tqdm(indirs):
        for subdir, dirs, files in os.walk(indir):
            for file in files:
                if not file.endswith("pt"):
                    continue
                new_subdir = os.path.join(subdir, f"regular_decoding")

                if not os.path.exists(new_subdir):
                    os.makedirs(new_subdir)


                curr_outdir = os.path.join(new_subdir, file.replace("pt", "csv"))
                curr_data = torch.load(os.path.join(subdir, file))
                curr_df_dict = dict()
                for key,value in curr_data.items():
                    if len(value)>0 and type(value[0]) == dict:
                        curr_df_dict[key] = [elem["outputs"][0] for elem in value]
                    elif any(r for r in value): # if all results are empty strings - this the case when only the "hint" prompts were sent, and then the "Regular-Prompt" and "Answerability" weren't sent and can be omitted
                        curr_df_dict[key] = value
                curr_df = pd.DataFrame(curr_df_dict)
                curr_df.to_csv(curr_outdir)

def get_response_beam_relaxation(options):
    for i,option in enumerate(options["outputs"]):
        option_str = str(option).lower().strip()
        if any(option_str==elem1 for elem1 in UNANSWERABLE_REPLIES_EXACT) or any(option_str==f"{elem1}." for elem1 in UNANSWERABLE_REPLIES_EXACT) or any(elem2 in option_str for elem2 in UNANSWERABLE_REPLIES):
            return ("unanswerable", "")
    return (options["outputs"][0], "")

def pt_to_csv_beam(indirs):
    for indir in tqdm(indirs):
        for subdir, dirs, files in os.walk(indir):
            for file in files:
                if not file.endswith("pt"):
                    continue

                new_subdir = os.path.join(subdir, f"beam_relaxation")

                if not os.path.exists(new_subdir):
                    os.makedirs(new_subdir)
                curr_outdir = os.path.join(new_subdir, file.replace("pt", "csv"))
                curr_data = torch.load(os.path.join(subdir, file))
                curr_df_dict = dict()
                for key,value in curr_data.items():
                    if len(value)>0 and type(value[0]) == dict:
                        results = [get_response_beam_relaxation(elem) for elem in value]
                        curr_df_dict[key] = [elem[0] for elem in results]
                    elif any(elem for elem in value): # remove "empty" replies
                        curr_df_dict[key] = value
                curr_df = pd.DataFrame(curr_df_dict)
                curr_df.to_csv(curr_outdir)

def csv_to_benchmark_evaluate_format(indirs, data_name):
    eval_dir = f"{data_name}_QA_task_format"

    for indir in tqdm(indirs):
        for subdir, dirs, files in os.walk(indir):
            json_dicts = {prompt_type:{} for prompt_type in PROMPT_TYPES}
            for file in files:
                if not file.endswith("csv") or not data_name in file:
                    continue
                curr_df = pd.read_csv(os.path.join(subdir, file))
                if data_name == "squad" or file.startswith("answerable"):
                    id_suffix = ""
                else:
                    id_suffix = "-unanswerable" 
                for prompt_type in PROMPT_TYPES:
                    try:
                        if not prompt_type in curr_df.columns:
                            continue
                        if prompt_type == "Answerability":
                            is_answerable_results = ["unanswerable" if "unanswerable" in str(row[prompt_type]).lower() else curr_df.iloc[i]["Regular-Prompt"] for i, row in curr_df.iterrows()]
                            curr_out_dict = {f'{row["ids"]}{id_suffix}'.strip(): is_answerable_results[i] for i, row in curr_df.iterrows()}
                        else:
                            curr_out_dict = {f'{row["ids"]}{id_suffix}'.strip(): row[prompt_type] for _, row in curr_df.iterrows()}
                        json_dicts[prompt_type].update(curr_out_dict)
                    except:
                        print("done")
            for prompt_type in PROMPT_TYPES:
                if not json_dicts[prompt_type]:
                    continue

                curr_subdir = os.path.join(subdir, eval_dir)
                if not os.path.exists(curr_subdir):
                    os.makedirs(curr_subdir)

                out_dict = json_dicts[prompt_type]
                for key,value in out_dict.items():
                    value = str(value).lower().strip()
                    if any(elem1==value for elem1 in UNANSWERABLE_REPLIES_EXACT) or any(f"{elem1}."==value for elem1 in UNANSWERABLE_REPLIES_EXACT) or any(elem2 in value.lower() for elem2 in UNANSWERABLE_REPLIES):
                        out_dict[key] = ""

                with open(os.path.join(curr_subdir, f"{data_name}_{prompt_type}.json"), 'w') as f1:
                    f1.write(json.dumps(out_dict))    

def main(indirs, is_beam_experiment):
    if is_beam_experiment:
        pt_to_csv_beam(indirs)
    else:
        pt_to_csv_non_beam(indirs)
        
    csv_to_benchmark_evaluate_format(indirs, 'squad')
    csv_to_benchmark_evaluate_format(indirs, 'NQ')
    csv_to_benchmark_evaluate_format(indirs, 'musique')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument("--indirs", nargs='+', type=str, required=True, help="path to the indirs where the pt files were saved")
    argparser.add_argument("--is-beam-experiment", action='store_true', default=False, help="Whether this is the beam relaxation experiment or the regular prompt-manipulation experiments.")
    args = argparser.parse_args()
    main(args.indirs, args.is_beam_experiment)
