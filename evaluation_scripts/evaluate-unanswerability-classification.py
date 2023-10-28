import argparse
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from openpyxl import load_workbook
from tabulate import tabulate
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import PROMPT_TYPES, UNANSWERABLE_REPLIES, UNANSWERABLE_REPLIES_EXACT
from utils import *

def check_if_unanswerable(response):
    value = str(response).lower().strip()
    return any(elem1==value for elem1 in UNANSWERABLE_REPLIES_EXACT) or any(f"{elem1}."==value for elem1 in UNANSWERABLE_REPLIES_EXACT) or any(elem2 in value.lower() for elem2 in UNANSWERABLE_REPLIES)

def calc_TP_TN_FP_FN(unanswerable_lst, answerable_lst):
    unanswerable_elems = {"tp": len([elem for elem in unanswerable_lst if elem]),
                         "fn": len([elem for elem in unanswerable_lst if not elem]),
                         "fp": len([elem for elem in answerable_lst if not elem]),
                         "tn": len([elem for elem in answerable_lst if elem])}
    
    answerable_elems = {"tp": len([elem for elem in answerable_lst if elem]),
                           "fn": len([elem for elem in answerable_lst if not elem]),
                           "fp": len([elem for elem in unanswerable_lst if not elem]),
                           "tn": len([elem for elem in unanswerable_lst if elem])}

    return unanswerable_elems, answerable_elems

def get_all_results(elems):
    elems = {key:float(value) for key,value in elems.items()}
    
    precision = elems['tp']/(elems['tp'] + elems['fp'])
    recall = elems['tp']/(elems['tp'] + elems['fn'])
    f1 = 2*precision*recall/(precision+recall)
    accuracy = (elems['tp'] + elems['tn'])/(elems['tp'] + elems['tn'] + elems['fp'] + elems['fn'])
    support = elems['tp'] + elems['fn']

    return {"P": round(100*precision, 1),
            "R": round(100*recall, 1),
            "F1": round(100*f1, 1),
            "accuracy": round(100*accuracy, 1),
            "support": support
            }

def create_output_tabular_structure(unanswerable_results, answerable_results):
    labels = ['un-answerable', 'answerable', 'accuracy']
    columns = ['precision', 'recall', 'f1-score', 'support']
    data_df = {columns[0] : [unanswerable_results["P"], answerable_results["P"], ''],
               columns[1] : [unanswerable_results["R"], answerable_results["R"], ''],
               columns[2] : [unanswerable_results["F1"], answerable_results["F1"], unanswerable_results["accuracy"]],
               columns[3] : [unanswerable_results["support"], answerable_results["support"], unanswerable_results["support"]+answerable_results["support"]]}

    table_txt = [
        ['',        columns[0],             columns[1],             columns[2],             columns[3]],
        [labels[0], data_df[columns[0]][0], data_df[columns[1]][0], data_df[columns[2]][0], data_df[columns[3]][0]],
        [labels[1], data_df[columns[0]][1], data_df[columns[1]][1], data_df[columns[2]][1], data_df[columns[3]][1]],
        [labels[2], data_df[columns[0]][2], data_df[columns[1]][2], data_df[columns[2]][2], data_df[columns[3]][2]]
    ]

    return table_txt, labels, data_df

def main(args):
    for curr_indir in args.indirs:
        curr_unanswerable, curr_answerable = pd.DataFrame(), pd.DataFrame()

        for subdir, dirs, files in os.walk(curr_indir):
            if not os.path.basename(subdir) in ["num_return_seq_1", "locate_unanswerable_in_beams"]:
                continue
            for filename in files:
                if not filename.endswith(".csv"):
                    continue
                if "un-answerable" in filename:
                    curr_unanswerable = pd.read_csv(os.path.join(subdir, filename))
                elif filename.startwith("answerable"):
                    curr_answerable = pd.read_csv(os.path.join(subdir, filename))
                else:
                    raise Exception(f"invalid csv file: {os.path.join(subdir, filename)}")
                
                # get the dataset name
                curr_dataset = get_dataset_name(os.path.join(subdir, filename))
            
            if curr_answerable.empty or curr_unanswerable.empty:
                raise Exception(f"didn't find two csv's in {subdir}")
            
            # create outdir
            outdir_path = args.outdir if args.outdir else "evaluation_results"
            outdir_path = get_evalulation_outdir(subdir, curr_dataset, outdir_path)
            outdir_excel_file = os.path.join(outdir_path, f"unanswerability_classification_results.xlsx")

            outdir_df_dict = {}
            for prompt_type in PROMPT_TYPES:
                if not prompt_type in curr_unanswerable.columns:
                    continue
                unanswerable_response_unanswerable = [check_if_unanswerable(elem) for elem in curr_unanswerable[prompt_type]]
                answerable_response_not_unanswerable = [not check_if_unanswerable(elem) for elem in curr_answerable[prompt_type]]
                
                unanswerable_elems, answerable_elems = calc_TP_TN_FP_FN(unanswerable_response_unanswerable, answerable_response_not_unanswerable)        

                unanswerable_results = get_all_results(unanswerable_elems)
                answerable_results = get_all_results(answerable_elems)

                output_text, labels, data_df = create_output_tabular_structure(unanswerable_results, answerable_results)
                
                # print results
                if args.print_results:
                    tabulated_scores = tabulate(output_text, headers='firstrow', tablefmt='plain')
                    print(f"\n{prompt_type}:")
                    print(f"\n{tabulated_scores}\n")
                
                # convert to dataframe
                df_scores = pd.DataFrame(data_df, index=labels)

                # save df to the outdir_df_dict
                outdir_df_dict[prompt_type] = df_scores

            # save to excel file (each prompt type - in a separate sheet)
            with pd.ExcelWriter(outdir_excel_file, engine='openpyxl') as writer:  
                for prompt_type,curr_df_scores in outdir_df_dict.items():
                    curr_df_scores.to_excel(writer, sheet_name=prompt_type)






if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument("--indirs", nargs='+', type=str, required=True, help="path to indirs where the generated texts are.")
    argparser.add_argument('--outdir', type=str, default=None, help='outdir to save results.')
    argparser.add_argument("--print-results", action='store_true', default=False, help="whether to also print the results.")
    args = argparser.parse_args()
    main(args)