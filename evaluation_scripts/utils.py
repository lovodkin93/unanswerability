import os
from pathlib import Path
import logging
# Set the logging level to INFO
logging.basicConfig(level=logging.INFO)

QA_TASK_METRICS_MAP={'exact' : 'EM (all)',
                     'f1' : 'F1 (all)',
                     'total' : 'total (all)',
                     'HasAns_exact' : 'EM (answerable)',
                     'HasAns_f1' : 'F1 (answerable)',
                     'HasAns_total' : 'total (answerable)',
                     'NoAns_exact' : 'EM (un-answerable)',
                     'NoAns_f1' : 'F1 (un-answerable)',
                     'NoAns_total' : 'total (un-answerable)'}


def get_model_name(indir):
    if "Flan-UL2" in indir:
        curr_model = "Flan-UL2"
    elif "Flan-T5-xxl" in indir:
        curr_model = "Flan-T5-xxl"
    elif "OPT-IML" in indir:
        curr_model = "OPT-IML"
    else:
        raise Exception(f"curr model not found in indir: {indir}")
    return curr_model

def get_dataset_name(indir):
    if "squad" in indir:
        return "squad"
    elif "NQ" in indir:
        return "NQ"
    elif "musique" in indir:
        return "musique"
    else:
        raise Exception(f"curr dataset not found in indir: {indir}")

def get_variant(indir):
    if "variant1" in indir:
        return "variant1"
    elif "variant2" in indir:
        return "variant2"
    elif "variant3" in indir:
        return "variant3"
    else:
        raise Exception(f"curr variant not found in indir: {indir}")

def get_num_beams(indir):
    if "k_beams_1" in indir:
        return "k_beams_1"
    if "k_beams_3" in indir:
        return "k_beams_3"
    if "k_beams_5" in indir:
        return "k_beams_5"    
    if "k_beams_7" in indir:
        return "k_beams_7"
    else:
        raise Exception(f"num beams not found in indir: {indir}")

def get_icl_variant(indir):
    if not "icl_examples" in indir:
        return None
    elif "icl_examples_v1" in indir:
        return "icl_examples_v1"
    elif "icl_examples_v2" in indir:
        return "icl_examples_v2"
    elif "icl_examples_v3" in indir:
        return "icl_examples_v3"
    else:
        raise Exception(f"icl_examples found in indir, but no known version was found: {indir}")

def get_evalulation_outdir(subdir, curr_dataset, outdir_path):
    zero_or_few = "zero_shot" if "zero_shot" in subdir else "few_shot_with_instructions"
    model_name = get_model_name(subdir)
    variant = get_variant(subdir)
    num_beams = get_num_beams(subdir)
    icl_variant = get_icl_variant(subdir)

    top_or_all_beams = "locate_unanswerable_in_beams" if "locate_unanswerable_in_beams" in subdir else "num_return_seq_1"

    outdir_path = os.path.join(outdir_path, zero_or_few, model_name, curr_dataset, num_beams, variant, top_or_all_beams)
    if icl_variant != None: # for few-shot there is also icl_example variant
        outdir_path = os.path.join(outdir_path, icl_variant)


    path = Path(outdir_path)
    path.mkdir(parents=True, exist_ok=True)
    # logging.info(f'saving results to: {outdir_path}')

    return outdir_path

