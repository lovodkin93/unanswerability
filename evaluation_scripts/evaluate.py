import subprocess
import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm
import argparse
import sys

def main(args):
    outdir = args.outdir if args.outdir else "evaluation_results"

    print(f"results are saved under {outdir}")
    print("Calculating performance in unanswerability classification ...")
    # run unanswerability classification
    eval_script = os.path.join("evaluation_scripts", "evaluate-unanswerability-classification.py")
    script_args = ["--indirs"] + args.indirs + ["--outdir"] + [outdir]
    subprocess.run(['python', eval_script] + script_args)

    # run QA task evaluation
    print("Calculating performance on the QA task ...")
    eval_script = os.path.join("evaluation_scripts", "evaluate-QA-task.py")
    script_args = ["--indirs"] + args.indirs + ["--outdir"] + [outdir]
    if args.devset:
        script_args += ["--devset"]
    subprocess.run(['python', eval_script] + script_args)    



if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument("--indirs", nargs='+', type=str, required=True, help="path to indirs where the generated texts are.")
    argparser.add_argument('--outdir', type=str, default=None, help='outdir to save results.')
    argparser.add_argument("--devset", action='store_true', default=False, help="whether the data is the devset (for choosing the best (hinting) variant).")
    args = argparser.parse_args()
    main(args)