import os
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
import torch
import pickle
from pathlib import Path

SEED = 42

def adapt_hidden_embeddings(instance, embedding_type):

    # if the embeddings of all the generation steps were saved in a single matrix, rather than in a list, separate them
    if len(instance[embedding_type][-1].shape) == 2:
        instance[embedding_type] = [instance[embedding_type][0][i,:] for i in range(instance[embedding_type][0].shape[0])]

    # removing the paddings
    # Compare all elements to 1
    matches = instance['all_outputs_ids'][0,:].eq(1)

    # Find the first non-zero element in matches
    indices = matches.nonzero(as_tuple=True)

    # Get the first index where value is 1 (if no 1 then no "padding" and so can take all embeddings)
    filter_index = indices[0][0].item() if indices[0].numel() != 0 else len(instance[embedding_type])

    filtered_hidden_embedding = instance[embedding_type][:filter_index]
    return filtered_hidden_embedding

def get_model_name(indir):
    if "Flan-UL2" in indir:
        curr_model = "Flan-UL2"
    elif "Flan-T5-xxl" in indir:
        curr_model = "Flan-T5-xxl"
    elif "OPT" in indir:
        curr_model = "OPT"
    else:
        raise Exception(f"curr model not found in indir: {indir}")
    return curr_model

def get_data(indir, prompt_type, embedding_type, dataset, num_instances, aggregation_type):
    data = dict()
    for file_name in os.listdir(indir):
        if not dataset in file_name or not file_name.endswith(".pt"):
            continue
        curr_data = torch.load(os.path.join(indir, file_name))

        if num_instances != None:
            curr_data = {key:value[:num_instances] for key,value in curr_data.items()}

        data_type = "un-answerable" if "un-answerable" in file_name else "answerable"
        data[data_type] = curr_data

    if embedding_type == "first_hidden_embedding":
        unanswerable_instances = [elem[embedding_type].mean(dim=0).cpu().numpy() for elem in data["un-answerable"][prompt_type]]
        answerable_instances = [elem[embedding_type].mean(dim=0).cpu().numpy() for elem in data["answerable"][prompt_type]]
    elif aggregation_type == "average":
        unanswerable_instances = [torch.stack(adapt_hidden_embeddings(elem, embedding_type)).mean(dim=0).cpu().numpy() for elem in data["un-answerable"][prompt_type]]
        answerable_instances = [torch.stack(adapt_hidden_embeddings(elem, embedding_type)).mean(dim=0).cpu().numpy() for elem in data["answerable"][prompt_type]]
    elif aggregation_type == "union":
        unanswerable_instances = [emb.cpu().numpy() for elem in data["un-answerable"][prompt_type] for emb in adapt_hidden_embeddings(elem, embedding_type)]
        answerable_instances = [emb.cpu().numpy() for elem in data["answerable"][prompt_type] for emb in adapt_hidden_embeddings(elem, embedding_type)]
    elif aggregation_type == "only_first_tkn":
        unanswerable_instances = [adapt_hidden_embeddings(elem, embedding_type)[0].cpu().numpy() for elem in data["un-answerable"][prompt_type]]
        answerable_instances = [adapt_hidden_embeddings(elem, embedding_type)[0].cpu().numpy() for elem in data["answerable"][prompt_type]]
    else:
        raise Exception("--aggregation-type did not receive a valid option. Only one of 'average', 'union' or 'only_first_tkn'")

    return unanswerable_instances, answerable_instances

def main(args):
    model_name = get_model_name(args.indir)
    outdir = os.path.join(args.outdir, args.dataset, args.embedding_type, args.prompt_type, args.aggregation_type, f"{model_name}_{args.num_instances}N")

    # create outdir
    folder_path = Path(outdir)
    folder_path.mkdir(parents=True, exist_ok=True)
    print(f"classifier saved to {outdir}")

    # get data
    unanswerable_instances, answerable_instances = get_data(indir=args.indir, 
                                                            prompt_type=args.prompt_type, 
                                                            embedding_type=args.embedding_type, 
                                                            dataset=args.dataset, 
                                                            num_instances=args.num_instances, 
                                                            aggregation_type=args.aggregation_type)

    # Combine the instances and create corresponding labels
    unanswerable_labels = np.zeros(len(unanswerable_instances))
    answerable_labels = np.ones(len(answerable_instances))

    X = np.concatenate((unanswerable_instances, answerable_instances))
    y = np.concatenate((unanswerable_labels, answerable_labels))

    # Split data into train, validation, and test sets (60% train, 20% validation, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    # Train a linear classifier using logistic regression
    clf = LogisticRegression(random_state=SEED)

    param_grid = {
            'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear'],
            'max_iter': [1000, 5000]
        }

    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', refit=True)
    grid_search.fit(X_train, y_train)

    # Log the accuracy of different hyperparameters in the GridSearch
    print("GridSearchCV results:")
    for mean_score, std_score, params in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['std_test_score'], grid_search.cv_results_['params']):
        print(f"Mean accuracy: {mean_score:.4f}, Std: {std_score:.4f}, Parameters: {params}")

    # save model
    model_filename = os.path.join(outdir, "best_model.pkl")
    with open(model_filename, 'wb') as file:
        pickle.dump(grid_search.best_estimator_, file)

    print(f"Best model saved to {model_filename}")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument('-i', '--indir', type=str, required=True, help='path to data')
    argparser.add_argument('-o', '--outdir', type=str, required=True, help='path to outdir')
    argparser.add_argument('--dataset', type=str, default="squad", help='prompt type to classify ("squad", "NQ", "musique")')
    argparser.add_argument('--prompt-type', type=str, default="Adversarial", help='prompt type to classify ("Adversarial", "Pseudo-Adversarial", "CoT-Adversarial", "Answerability")')
    argparser.add_argument('--num-instances', type=int, default=None, help='number of instances to use for training (will take the same amount from the answerable and the un-answerable). If None - will take all.')
    argparser.add_argument('--aggregation-type', type=str, default="only_first_tkn", help='how to aggregate all the hidden layers of all the generated tokens of a single instance (choose from "average" to average them, "union" to treat each of them as an instance, and "only_first_tkn" to only take the first token\'s hidden layers).')
    argparser.add_argument('--embedding-type', type=str, default="last_hidden_embedding", help='which layer to take: any one of "last_hidden_embedding" and "first_hidden_embedding"')
    args = argparser.parse_args()
    main(args)


