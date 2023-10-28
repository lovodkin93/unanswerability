import numpy as np
import os
import pickle
import argparse
import logging
# Set the logging level to INFO
logging.basicConfig(level=logging.INFO)
import numpy as np
import torch
import pickle
from concept_erasure import ConceptEraser
from pathlib import Path

SEED = 42

def adapt_hidden_embeddings(instance):
    # Compare all elements to 1
    matches = instance['all_outputs_ids'][0,:].eq(1)

    # Find the first non-zero element in matches
    indices = matches.nonzero(as_tuple=True)

    # Get the first index where value is 1 (if no 1 then no "padding" and so can take all embeddings)
    filter_index = indices[0][0].item() if indices[0].numel() != 0 else len(instance['last_hidden_embedding'])

    filtered_hidden_embedding = instance['last_hidden_embedding'][:filter_index]
    return filtered_hidden_embedding

def get_data(indir, prompt_type, dataset, num_instances, aggregation_type):
    data = dict()
    for file_name in os.listdir(indir):
        if not dataset in file_name or not file_name.endswith(".pt"):
            continue
        curr_data = torch.load(os.path.join(indir, file_name))

        if num_instances != None:
            curr_data = {key:value[:num_instances] for key,value in curr_data.items()}

        data_type = "un-answerable" if "un-answerable" in file_name else "answerable"
        data[data_type] = curr_data

    if aggregation_type == "average":
        unanswerable_instances = [torch.stack(adapt_hidden_embeddings(elem)).mean(dim=0).cpu().numpy() for elem in data["un-answerable"][prompt_type]]
        answerable_instances = [torch.stack(adapt_hidden_embeddings(elem)).mean(dim=0).cpu().numpy() for elem in data["answerable"][prompt_type]]
    elif aggregation_type == "union":
        unanswerable_instances = [emb.cpu().numpy() for elem in data["un-answerable"][prompt_type] for emb in adapt_hidden_embeddings(elem)]
        answerable_instances = [emb.cpu().numpy() for elem in data["answerable"][prompt_type] for emb in adapt_hidden_embeddings(elem)]
    elif aggregation_type == "only_first":
        unanswerable_instances = [adapt_hidden_embeddings(elem)[0].cpu().numpy() for elem in data["un-answerable"][prompt_type]]
        answerable_instances = [adapt_hidden_embeddings(elem)[0].cpu().numpy() for elem in data["answerable"][prompt_type]]
    else:
        raise Exception("--aggregation-type did not receive a valid option. Only one of 'average', 'union' or 'only_first'")

    return unanswerable_instances, answerable_instances

def main(args):
    indir = args.indir
    outdir = args.outdir
    dataset = args.dataset
    prompt_type = args.prompt_type
    outdir = os.path.join(outdir, dataset, prompt_type)

    # create outdir
    folder_path = Path(outdir)
    folder_path.mkdir(parents=True, exist_ok=True)
    print(f"classifier saved to {outdir}")

    unanswerable_instances, answerable_instances = get_data(indir, prompt_type, dataset, args.num_instances, args.aggregation_type)

    # Combine the instances and create corresponding labels
    unanswerable_labels = np.zeros(len(unanswerable_instances))
    answerable_labels = np.ones(len(answerable_instances))

    X = torch.from_numpy(np.concatenate((unanswerable_instances, answerable_instances))).float()
    y = torch.from_numpy(np.concatenate((unanswerable_labels, answerable_labels))).float()

    eraser = ConceptEraser.fit(X, y).to("cuda")

    # save model
    model_filename = os.path.join(outdir, "eraser.pkl")
    with open(model_filename, 'wb') as file:
        pickle.dump(eraser, file)
    print(f"eraser saved to {model_filename}")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument('-i', '--indir', type=str, required=True, help='path to data')
    argparser.add_argument('-o', '--outdir', type=str, required=True, help='path to outdir')
    argparser.add_argument('--dataset', type=str, default="squad", help='prompt type to classify ("squad", "NQ", "musique")')
    argparser.add_argument('--prompt-type', type=str, default="Adversarial", help='prompt type to classify ("Adversarial", "Pseudo-Adversarial", "CoT-Adversarial", "Answerability")')
    argparser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    argparser.add_argument('--batch-size', type=int, default=32, help='batch size of train set.')
    argparser.add_argument('--eval-batch-size', type=int, default=64, help='batch size of dev and test sets.')
    argparser.add_argument('--num-instances', type=int, default=None, help='number of instances to use for training (will take the same amount from the answerable and the un-answerable). If None - will take all.')
    argparser.add_argument('--save-interval', type=int, default=10, help='how frequently to save model')
    argparser.add_argument('--eval-interval', type=int, default=10, help='how frequently to evaluate on the devset (in epochs)')
    argparser.add_argument('--aggregation-type', type=str, default="only_first", help='how to aggregate all the hidden layers of all the generated tokens of a single instance (choose from "average" to average them, "union" to treat each of them as an instance, and "only_first" to only take the first token\'s hidden layers).')
    args = argparser.parse_args()
    main(args)


