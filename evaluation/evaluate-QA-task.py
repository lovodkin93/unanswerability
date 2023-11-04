import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm
import argparse
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import PROMPT_TYPES
from utils import *
import subprocess
from pathlib import Path
import shutil

def main(args):

    # create tmp dir inside the outdir path for calculations of each of the prompt variants
    outdir_path = args.outdir if args.outdir else "evaluation_results"
    tmp_outdir = os.path.join(outdir_path, "tmp")
    path = Path(tmp_outdir)
    path.mkdir(parents=True, exist_ok=True)


    for curr_indir in args.indirs:
        for subdir, dirs, files in os.walk(curr_indir):
            curr_full_results_df = dict()
            for filename in files:
                # if not the json file with the generated texts - ignore
                if not any(filename.endswith(f"{prompt_type}.json") for prompt_type in PROMPT_TYPES):
                    continue

                # get the dataset name
                curr_dataset = get_dataset_name(filename)

                # get prompt_type (it is in the json file's name - simply remove the suffix and the dataset's prefix)
                prompt_type = filename.replace(f"{curr_dataset}_", "").replace(".json", "")


                ###### run the relevant evaluation script ######
                # get the script
                eval_script = "evaluate-squad-v2.0.py" if curr_dataset == "squad" else "evaluate-NQ-musique.py"
                eval_script = os.path.join("evaluation", eval_script)

                # get path to gold data
                gold_outputs_suffix = "json" if curr_dataset == "squad" else "jsonl"
                if args.devset:
                    gold_outputs_indir =  os.path.join("data", "gold_outputs", curr_dataset, f"dev_data.{gold_outputs_suffix}")
                else:
                    gold_outputs_indir =  os.path.join("data", "gold_outputs", curr_dataset, f"test_data.{gold_outputs_suffix}")
                
                # get path to generated text
                generated_text_indir = os.path.join(subdir, filename)

                # get temporary outdir for results
                curr_tmp_outdir = os.path.join(tmp_outdir, f"{curr_dataset}_{prompt_type}.json")

                # run the script
                subprocess.run(['python', eval_script, gold_outputs_indir, generated_text_indir, "--out-file", curr_tmp_outdir])

                # read the results and store them to curr_full_results_df
                with open(curr_tmp_outdir, 'r') as f1:
                    curr_results = json.loads(f1.read())
                curr_full_results_df[prompt_type] = curr_results
                
                # remove the temporary file
                if os.path.isfile(curr_tmp_outdir):
                    os.remove(curr_tmp_outdir)
            
            # if empty dataframe, that means no files where found in current subdir
            if not curr_full_results_df:
                continue
            # convert to dataframe
            labels = list(curr_full_results_df.keys())
            columns = curr_full_results_df[labels[0]].keys()
            df_dict = {col:[curr_full_results_df[label][col] for label in labels] for col in columns}

            # rename columns
            df_dict = {QA_TASK_METRICS_MAP[col]:results for col,results in df_dict.items()}

            curr_out_df = pd.DataFrame(df_dict, index=labels)

            # create outdir
            curr_outdir_path = get_evalulation_outdir(subdir, curr_dataset, outdir_path)
            outdir_csv_file = os.path.join(curr_outdir_path, f"QA-task-results.csv")
            curr_out_df.to_csv(outdir_csv_file)

    # remove the temporary folder altogether
    try:
        if os.path.isdir(tmp_outdir):
            shutil.rmtree(tmp_outdir)
    except:
        print(f'Folder {tmp_outdir} not deleted')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument("--indirs", nargs='+', type=str, required=True, help="path to indirs where the generated texts are.")
    argparser.add_argument('--outdir', type=str, default=None, help='outdir to save results')
    argparser.add_argument("--devset", action='store_true', default=False, help="whether the data is the devset (for choosing the best (hinting) variant)")
    args = argparser.parse_args()
    main(args)