import numpy as np
from tqdm import tqdm
import json
import os
import pickle
import argparse
from sklearn.metrics import classification_report
import torch
from datetime import datetime
from pathlib import Path

SEED = 42

def adapt_hidden_embeddings(instance):
    # if the embeddings of all the generation steps were saved in a single matrix, rather than in a list, separate them
    if len(instance['last_hidden_embedding'][-1].shape) == 2:
        instance['last_hidden_embedding'] = [instance['last_hidden_embedding'][0][i,:] for i in range(instance['last_hidden_embedding'][0].shape[0])]
    # removing the paddings
    # Compare all elements to 1
    matches = instance['all_outputs_ids'][0,:].eq(1)
    # Find the first non-zero element in matches
    indices = matches.nonzero(as_tuple=True)
    # Get the first index where value is 1 (if no 1 then no "padding" and so can take all embeddings)
    filter_index = indices[0][0].item() if indices[0].numel() != 0 else len(instance['last_hidden_embedding'])
    filtered_hidden_embedding = instance['last_hidden_embedding'][:filter_index]
    return filtered_hidden_embedding

def get_model_name(indir):
    if "Flan-T5-xxl" in indir:
        return "Flan-T5-xxl"
    elif "Flan-UL2" in indir:
        return "Flan-UL2"
    elif "OPT-IML" in indir:
        return "OPT-IML"
    else:
        raise Exception("paths of embeddings must have one of \"Flan-T5-xxl\", \"Flan-UL2\", or \"OPT-IML\".")

def get_curr_variant(indir):
    if "variant1" in indir:
        return "variant1"
    elif "variant2" in indir:
        return "variant2"
    elif "variant3" in indir:
        return "variant3"
    else:
        raise Exception("paths of embeddings must have one of \"variant1\", \"variant2\", or \"variant3\".")

def get_data(indir, prompt_type, dataset, aggregation_type, embedding_type):
    data = dict()
    for file_name in os.listdir(indir):
        if not dataset in file_name or not file_name.endswith(".pt"):
            continue
        curr_data = torch.load(os.path.join(indir, file_name), map_location="cpu")
        data_type = "un-answerable" if "un-answerable" in file_name else "answerable"
        data[data_type] = curr_data

    if not "un-answerable" in data.keys() or not "answerable" in data.keys(): # didn't find the dataset's "answerable" or "un-answerable" tensors
        return None, None, None, None
    
    if embedding_type == "first_hidden_embedding":
        unanswerable_instances = [elem[embedding_type].cpu().numpy() for elem in data["un-answerable"][prompt_type]]
        answerable_instances = [elem[embedding_type].cpu().numpy() for elem in data["answerable"][prompt_type]]
    elif aggregation_type == "average":
        unanswerable_instances = [torch.stack(adapt_hidden_embeddings(elem)).mean(dim=0).cpu().numpy() for elem in data["un-answerable"][prompt_type]]
        answerable_instances = [torch.stack(adapt_hidden_embeddings(elem)).mean(dim=0).cpu().numpy() for elem in data["answerable"][prompt_type]]
    elif aggregation_type == "union":
        unanswerable_instances = [emb.cpu().numpy() for elem in data["un-answerable"][prompt_type] for emb in adapt_hidden_embeddings(elem)]
        answerable_instances = [emb.cpu().numpy() for elem in data["answerable"][prompt_type] for emb in adapt_hidden_embeddings(elem)]
    elif aggregation_type == "only_first_tkn":
        unanswerable_instances = [adapt_hidden_embeddings(elem)[0].cpu().numpy() for elem in data["un-answerable"][prompt_type]]
        answerable_instances = [adapt_hidden_embeddings(elem)[0].cpu().numpy() for elem in data["answerable"][prompt_type]]
    else:
        raise Exception("--aggregation-type did not receive a valid option. Only one of 'average', 'union' or 'only_first_tkn'")
    unanswerable_ids = data['un-answerable']['ids']
    answerable_ids = data['answerable']['ids']
    return unanswerable_instances, answerable_instances, unanswerable_ids, answerable_ids

def main(args):
    label_dict = {0:"unanswerable", 1: "answerable"}
    prompt_type = args.prompt_type
    now = datetime.now()
    now_str = now.strftime("%d-%m-%Y_%H:%M:%S")
    outdir_path = args.outdir if args.outdir else os.path.join("classifier_evaluation_results", now_str)
    print(f"saving all data to {outdir_path}")

    if args.dataset != None: # specific dataset
        datasets = [args.dataset]
    else: # otherwise check all datasets
        datasets = ["musique", "squad", "NQ"]

    for indir in tqdm(args.indirs):
        for subdir, dirs, files in os.walk(indir):
            for dataset in datasets:
                unanswerable_instances, answerable_instances, unanswerable_ids, answerable_ids = get_data(subdir, prompt_type, dataset, aggregation_type=args.aggregation_type, embedding_type=args.embedding_type)
                if unanswerable_instances == None: # didn't find any of the dataset's "answerable" or "un-answerable" tensors (no dataset in this folder)
                    continue                

                # Combine the instances and create corresponding labels
                unanswerable_labels = np.zeros(len(unanswerable_instances))
                answerable_labels = np.ones(len(answerable_instances))

                X_test = np.concatenate((unanswerable_instances, answerable_instances))
                y_test = np.concatenate((unanswerable_labels, answerable_labels))

                # Load the classifier from the file
                with open(args.classifier_dir, "rb") as file:
                    clf = pickle.load(file)

                # Evaluate the model on the test set
                y_test_pred = clf.predict(X_test)
                
                # create curr_outdir
                curr_model = get_model_name(subdir)
                curr_variant = get_curr_variant(subdir)
                curr_outdir = os.path.join(outdir_path, curr_model, curr_variant)
                path = Path(curr_outdir)
                path.mkdir(parents=True, exist_ok=True)

                # Print the classification report
                print(f"Classification report for model {curr_model} dataset {dataset} {curr_variant}:")
                print(f'saved to {os.path.join(curr_outdir, f"{dataset}_classification_report.txt")}')
                clf_report = classification_report(y_test, y_test_pred, digits=4)
                print(clf_report)
                print("\n##############################################################\n")

                # save results
                unanswerable_predicts = [label_dict[l] for l in list(y_test_pred[:len(unanswerable_instances)])]
                answerable_predicts = [label_dict[l] for l in list(y_test_pred[len(unanswerable_instances):])]
                results_unanswerable = {id:unanswerable_predicts[i] for i,id in enumerate(unanswerable_ids)}
                results_answerable = {id:answerable_predicts[i] for i,id in enumerate(answerable_ids)}
                results = {"un-answerable_predicts": results_unanswerable,
                            "answerable_predicts": results_answerable}                
                with open(os.path.join(curr_outdir, f"{dataset}_predictions.json"), 'w') as f1:
                    f1.write(json.dumps(results))
                with open(os.path.join(curr_outdir, f"{dataset}_classification_report.txt"), 'w') as f2:
                    f2.write(clf_report)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument("--indirs", nargs='+', type=str, required=True, help="path to datas")
    argparser.add_argument('--classifier-dir', type=str, required=True, help='path to classifier')
    argparser.add_argument('--outdir', type=str, default=None, help='outdir to save results')
    argparser.add_argument('--dataset', type=str, default=None, help='prompt type to classify ("squad", "NQ", "musique")')
    argparser.add_argument('--prompt-type', type=str, default="Regular-Prompt", help='prompt type to classify ("Regular-Prompt", "Hint-Prompt", "CoT-Prompt", "Answerability")')
    argparser.add_argument('--aggregation-type', type=str, default="only_first_tkn", help='how to aggregate all the hidden layers of all the generated tokens of a single instance (choose from "average" to average them, "union" to treat each of them as an instance, and "only_first_tkn" to only take the first token\'s hidden layers).')
    argparser.add_argument('--embedding-type', type=str, default="last_hidden_embedding", help='which layer to take: any one of "last_hidden_embedding" and "first_hidden_embedding"')
    args = argparser.parse_args()
    main(args)


