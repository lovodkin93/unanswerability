import numpy as np
from tqdm import tqdm


import json
import os
import pickle
import argparse
import logging
# Set the logging level to INFO
logging.basicConfig(level=logging.INFO)
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
import torch
import pickle
from copy import deepcopy
from tqdm import tqdm

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




def get_data(indir, prompt_type, dataset, aggregation_type, embedding_type):
    data = dict()
    for file_name in os.listdir(indir):
        if not dataset in file_name or not file_name.endswith(".pt"):
            continue
        curr_data = torch.load(os.path.join(indir, file_name), map_location="cpu")
        data_type = "control_group" if "control_group" in file_name else "adversarial"
        data[data_type] = curr_data
    if not data: # didn't find any of the dataset's tensors (no dataset in this folder)
        return None, None, None, None
    # adversarial_instances = [torch.stack(adapt_hidden_embeddings(elem)).mean(dim=0).cpu().numpy() for elem in data["adversarial"][prompt_type]]
    # control_group_instances = [torch.stack(adapt_hidden_embeddings(elem)).mean(dim=0).cpu().numpy() for elem in data["control_group"][prompt_type]]

    if embedding_type == "first_hidden_embedding":
        adversarial_instances = [elem[embedding_type].cpu().numpy() for elem in data["adversarial"][prompt_type]]
        control_group_instances = [elem[embedding_type].cpu().numpy() for elem in data["control_group"][prompt_type]]
    elif aggregation_type == "average":
        adversarial_instances = [torch.stack(adapt_hidden_embeddings(elem)).mean(dim=0).cpu().numpy() for elem in data["adversarial"][prompt_type]]
        control_group_instances = [torch.stack(adapt_hidden_embeddings(elem)).mean(dim=0).cpu().numpy() for elem in data["control_group"][prompt_type]]
    elif aggregation_type == "union":
        adversarial_instances = [emb.cpu().numpy() for elem in data["adversarial"][prompt_type] for emb in adapt_hidden_embeddings(elem)]
        control_group_instances = [emb.cpu().numpy() for elem in data["control_group"][prompt_type] for emb in adapt_hidden_embeddings(elem)]
    elif aggregation_type == "only_first_tkn":
        adversarial_instances = [adapt_hidden_embeddings(elem)[0].cpu().numpy() for elem in data["adversarial"][prompt_type]]
        control_group_instances = [adapt_hidden_embeddings(elem)[0].cpu().numpy() for elem in data["control_group"][prompt_type]]
    else:
        raise Exception("--aggregation-type did not receive a valid option. Only one of 'average', 'union' or 'only_first_tkn'")





    adversarial_ids = data['adversarial']['ids']
    control_group_ids = data['control_group']['ids']

    return adversarial_instances, control_group_instances, adversarial_ids, control_group_ids

def create_dir(subdirs):
    full_subdir = ""
    for subdir in subdirs:
        full_subdir = os.path.join(full_subdir, subdir)

        if not os.path.exists(full_subdir):
            os.makedirs(full_subdir)

def create_dir(subdirs):
    full_subdir = ""
    for subdir in subdirs:
        full_subdir = os.path.join(full_subdir, subdir)

        if not os.path.exists(full_subdir):
            os.makedirs(full_subdir)




def get_curr_model(indir):
    if "UL2_Flan" in indir:
        curr_model = "UL2_Flan"
    elif "T5_xxl_Flan" in indir:
        curr_model = "T5_xxl_Flan"
    elif "OPT" in indir:
        curr_model = "OPT"
    else:
        raise Exception(f"curr model not found in indir: {indir}")
    return curr_model

def get_curr_classifier_model(classifier_dir):
    if "UL2_Flan" in classifier_dir:
        curr_classifier_model = "UL2_Flan"
    elif "T5_xxl_Flan" in classifier_dir:
        curr_classifier_model = "T5_xxl_Flan"
    elif "OPT" in classifier_dir:
        curr_classifier_model = "OPT"
    else:
        raise Exception(f"curr model not found in indir: {classifier_dir}")
    return curr_classifier_model


def get_curr_classifier_dataset(classifier_dir):
    if "squad" in classifier_dir:
        curr_classifier_dataset = "squad"
    elif "NQ" in classifier_dir:
        curr_classifier_dataset = "NQ"
    elif "musique" in classifier_dir:
        curr_classifier_dataset = "musique"
    else:
        raise Exception(f"curr model not found in indir: {classifier_dir}")
    return curr_classifier_dataset


def get_curr_train_size(classifier_dir):
    if "500N" in classifier_dir:
        curr_train_size = "500N"
    elif "1000N" in classifier_dir:
        curr_train_size = "1000N"
    elif "2000N" in classifier_dir:
        curr_train_size = "2000N"
    else:
        raise Exception(f"curr model not found in indir: {classifier_dir}")
    return curr_train_size

def get_curr_embedding_aggregation(classifier_dir):
    if "averaged" in classifier_dir:
        curr_embedding_aggregation = "averaged"
    elif "not_averaged" in classifier_dir:
        curr_embedding_aggregation = "not_averaged"
    elif "only_first_tkn" in classifier_dir:
        curr_embedding_aggregation = "only_first_tkn"
    else:
        raise Exception(f"curr embedding aggregation not found in indir: {classifier_dir}")
    return curr_embedding_aggregation


def get_curr_prompt_style(classifier_dir):
    if "Pseudo-Adversarial" in classifier_dir:
        return "Pseudo-Adversarial"
    elif "CoT-Adversarial" in classifier_dir:
        return "CoT-Adversarial"
    elif "Answerability" in classifier_dir:
        return "Answerability"
    elif "Ablation1" in classifier_dir:
        return "Ablation1"
    elif "Ablation2" in classifier_dir:
        return "Ablation2"
    elif "Adversarial" in classifier_dir:
        return "Adversarial"
    else:
        raise Exception(f"curr prompt style not found in indir: {classifier_dir}")





def main(args):
    label_dict = {0:"unanswerable", 1: "answerable"}
    prompt_type = args.prompt_type


    if args.indir != None:
        indirs = [args.indir]
    else:
        # indirs = ["responses_embeddings/k-beams/08-06-2023_16:17:06/T5_xxl_Flan/zero_shot/k_beams_1_num_return_seq_1/variant1",
        #           "responses_embeddings/k-beams/08-06-2023_10:27:59/UL2_Flan/zero_shot/k_beams_1_num_return_seq_1/variant1",
        #           "responses_embeddings/k-beams/08-06-2023_16:17:06/OPT/zero_shot/k_beams_1_num_return_seq_1/variant1",
        #           "responses_embeddings/k-beams/08-06-2023_10:15:46/T5_xxl_Flan/zero_shot/k_beams_1_num_return_seq_1/variant1",
        #           "responses_embeddings/k-beams/08-06-2023_10:15:46/UL2_Flan/zero_shot/k_beams_1_num_return_seq_1/variant1",
        #           "responses_embeddings/k-beams/08-06-2023_23:38:09/OPT/zero_shot/k_beams_1_num_return_seq_1/variant1",
        #           "responses_embeddings/k-beams/16-06-2023_08:51:17/T5_xxl_Flan/zero_shot/k_beams_1_num_return_seq_1/variant3",
        #           "responses_embeddings/k-beams/16-06-2023_08:51:17/UL2_Flan/zero_shot/k_beams_1_num_return_seq_1/variant3",
        #           "responses_embeddings/k-beams/16-06-2023_08:51:17/OPT/zero_shot/k_beams_1_num_return_seq_1/variant3"]
        

        # indirs = ["responses_embeddings/k-beams/22-06-2023_12:26:12/T5_xxl_Flan/zero_shot/k_beams_1_num_return_seq_1/variant3",
        #           "responses_embeddings/k-beams/22-06-2023_12:26:12/UL2_Flan/zero_shot/k_beams_1_num_return_seq_1/variant3",
        #           "responses_embeddings/k-beams/22-06-2023_12:26:12/OPT/zero_shot/k_beams_1_num_return_seq_1/variant3",
        #           "responses_embeddings/k-beams/22-06-2023_12:24:16/T5_xxl_Flan/zero_shot/k_beams_1_num_return_seq_1/variant1",
        #           "responses_embeddings/k-beams/22-06-2023_12:24:16/UL2_Flan/zero_shot/k_beams_1_num_return_seq_1/variant1",
        #           "responses_embeddings/k-beams/22-06-2023_12:24:16/OPT/zero_shot/k_beams_1_num_return_seq_1/variant1",
        #           "responses_embeddings/k-beams/22-06-2023_12:22:18/T5_xxl_Flan/zero_shot/k_beams_1_num_return_seq_1/variant1",
        #           "responses_embeddings/k-beams/22-06-2023_12:22:18/UL2_Flan/zero_shot/k_beams_1_num_return_seq_1/variant1",
        #           "responses_embeddings/k-beams/22-06-2023_12:22:18/OPT/zero_shot/k_beams_1_num_return_seq_1/variant1"]

        indirs = ["responses_embeddings/projections/19-06-2023_16:44:13/UL2_Flan/zero_shot/k_beams_3_num_return_seq_3/variant1/k_10",
                  "responses_embeddings/projections/19-06-2023_16:46:08/UL2_Flan/zero_shot/k_beams_3_num_return_seq_3/variant1/k_10"]

    if args.dataset != None:
        datasets = [args.dataset]
    else:
        datasets = ["musique", "squad", "NQ"]

    if args.classifier_dir != None:
        classifier_dirs = [args.classifier_dir]
    else:
        # classifier_dirs = ["trained_classifiers/squad/last_hidden_embedding/Pseudo-Adversarial/only_first_tkn/T5_xxl_Flan_2000N/best_model.pkl",
        #                 "trained_classifiers/squad/last_hidden_embedding/Pseudo-Adversarial/only_first_tkn/UL2_Flan_2000N/best_model.pkl",
        #                 "trained_classifiers/squad/last_hidden_embedding/Pseudo-Adversarial/only_first_tkn/OPT_2000N/best_model.pkl",
        #                 "trained_classifiers/NQ/last_hidden_embedding/Pseudo-Adversarial/only_first_tkn/T5_xxl_Flan_2000N/best_model.pkl",
        #                 "trained_classifiers/NQ/last_hidden_embedding/Pseudo-Adversarial/only_first_tkn/UL2_Flan_2000N/best_model.pkl",
        #                 "trained_classifiers/NQ/last_hidden_embedding/Pseudo-Adversarial/only_first_tkn/OPT_2000N/best_model.pkl",
        #                 "trained_classifiers/musique/last_hidden_embedding/Pseudo-Adversarial/only_first_tkn/T5_xxl_Flan_2000N/best_model.pkl",
        #                 "trained_classifiers/musique/last_hidden_embedding/Pseudo-Adversarial/only_first_tkn/UL2_Flan_2000N/best_model.pkl",
        #                 "trained_classifiers/musique/last_hidden_embedding/Pseudo-Adversarial/only_first_tkn/OPT_2000N/best_model.pkl",
                        
        #                 "trained_classifiers/squad/last_hidden_embedding/Pseudo-Adversarial/only_first_tkn/T5_xxl_Flan_1000N/best_model.pkl",
        #                 "trained_classifiers/squad/last_hidden_embedding/Pseudo-Adversarial/only_first_tkn/UL2_Flan_1000N/best_model.pkl",
        #                 "trained_classifiers/squad/last_hidden_embedding/Pseudo-Adversarial/only_first_tkn/OPT_1000N/best_model.pkl",
        #                 "trained_classifiers/NQ/last_hidden_embedding/Pseudo-Adversarial/only_first_tkn/T5_xxl_Flan_1000N/best_model.pkl",
        #                 "trained_classifiers/NQ/last_hidden_embedding/Pseudo-Adversarial/only_first_tkn/UL2_Flan_1000N/best_model.pkl",
        #                 "trained_classifiers/NQ/last_hidden_embedding/Pseudo-Adversarial/only_first_tkn/OPT_1000N/best_model.pkl",
        #                 "trained_classifiers/musique/last_hidden_embedding/Pseudo-Adversarial/only_first_tkn/T5_xxl_Flan_1000N/best_model.pkl",
        #                 "trained_classifiers/musique/last_hidden_embedding/Pseudo-Adversarial/only_first_tkn/UL2_Flan_1000N/best_model.pkl",
        #                 "trained_classifiers/musique/last_hidden_embedding/Pseudo-Adversarial/only_first_tkn/OPT_1000N/best_model.pkl",
                        
        #                 "trained_classifiers/squad/last_hidden_embedding/Pseudo-Adversarial/only_first_tkn/T5_xxl_Flan_500N/best_model.pkl",
        #                 "trained_classifiers/squad/last_hidden_embedding/Pseudo-Adversarial/only_first_tkn/UL2_Flan_500N/best_model.pkl",
        #                 "trained_classifiers/squad/last_hidden_embedding/Pseudo-Adversarial/only_first_tkn/OPT_500N/best_model.pkl",
        #                 "trained_classifiers/NQ/last_hidden_embedding/Pseudo-Adversarial/only_first_tkn/T5_xxl_Flan_500N/best_model.pkl",
        #                 "trained_classifiers/NQ/last_hidden_embedding/Pseudo-Adversarial/only_first_tkn/UL2_Flan_500N/best_model.pkl",
        #                 "trained_classifiers/NQ/last_hidden_embedding/Pseudo-Adversarial/only_first_tkn/OPT_500N/best_model.pkl",
        #                 "trained_classifiers/musique/last_hidden_embedding/Pseudo-Adversarial/only_first_tkn/T5_xxl_Flan_500N/best_model.pkl",
        #                 "trained_classifiers/musique/last_hidden_embedding/Pseudo-Adversarial/only_first_tkn/UL2_Flan_500N/best_model.pkl",
        #                 "trained_classifiers/musique/last_hidden_embedding/Pseudo-Adversarial/only_first_tkn/OPT_500N/best_model.pkl"]    

        # classifier_dirs = ["trained_classifiers/squad/first_hidden_embedding/Pseudo-Adversarial/only_first_tkn/T5_xxl_Flan_500N/best_model.pkl",
        #                 "trained_classifiers/squad/first_hidden_embedding/Pseudo-Adversarial/only_first_tkn/UL2_Flan_500N/best_model.pkl",
        #                 "trained_classifiers/squad/first_hidden_embedding/Pseudo-Adversarial/only_first_tkn/OPT_500N/best_model.pkl",
        #                 "trained_classifiers/NQ/first_hidden_embedding/Pseudo-Adversarial/only_first_tkn/T5_xxl_Flan_500N/best_model.pkl",
        #                 "trained_classifiers/NQ/first_hidden_embedding/Pseudo-Adversarial/only_first_tkn/UL2_Flan_500N/best_model.pkl",
        #                 "trained_classifiers/NQ/first_hidden_embedding/Pseudo-Adversarial/only_first_tkn/OPT_500N/best_model.pkl",
        #                 "trained_classifiers/musique/first_hidden_embedding/Pseudo-Adversarial/only_first_tkn/T5_xxl_Flan_500N/best_model.pkl",
        #                 "trained_classifiers/musique/first_hidden_embedding/Pseudo-Adversarial/only_first_tkn/UL2_Flan_500N/best_model.pkl",
        #                 "trained_classifiers/musique/first_hidden_embedding/Pseudo-Adversarial/only_first_tkn/OPT_500N/best_model.pkl"]

        classifier_dirs = ["trained_classifiers/squad/last_hidden_embedding/Adversarial/only_first_tkn/UL2_Flan_500N/best_model.pkl"]

    # print(args.zero_or_few_shot)

    for indir in tqdm(indirs):
        for dataset in datasets:
            adversarial_instances, control_group_instances, adversarial_ids, control_group_ids = get_data(indir, prompt_type, dataset, aggregation_type=args.aggregation_type, embedding_type=args.embedding_type)
            if adversarial_instances == None: # didn't find any of the dataset's tensors (no dataset in this folder)
                continue
            for classifier_dir in classifier_dirs:
                
                curr_model = get_curr_model(indir)
                curr_classifier_model = get_curr_classifier_model(classifier_dir)
                
                if curr_classifier_model != curr_model:
                    continue
                
                curr_classifier_dataset = get_curr_classifier_dataset(classifier_dir)
                curr_train_size = get_curr_train_size(classifier_dir)
                curr_embedding_aggregation = get_curr_embedding_aggregation(classifier_dir)
                curr_classifier_prompt_style = get_curr_prompt_style(classifier_dir)

                # Combine the instances and create corresponding labels
                adversarial_labels = np.zeros(len(adversarial_instances))
                control_group_labels = np.ones(len(control_group_instances))

                X_test = np.concatenate((adversarial_instances, control_group_instances))
                y_test = np.concatenate((adversarial_labels, control_group_labels))

                # Load the classifier from the file
                with open(classifier_dir, "rb") as file:
                    clf = pickle.load(file)


                # Evaluate the model on the test set
                y_test_pred = clf.predict(X_test)
                test_accuracy = accuracy_score(y_test, y_test_pred)
                print(f"model: {curr_model} dataset: {dataset} classifier dataset: {curr_classifier_dataset} ({curr_train_size})")
                # print(f"Test accuracy: {test_accuracy:.4f}")

                # Print the classification report
                print("Classification report:")
                clf_report = classification_report(y_test, y_test_pred, digits=4)
                print(clf_report)
                print("\n##############################################################\n")

                outdir_subdirs = ["classifier_predictions", args.embedding_type, f"{prompt_type}_test_{curr_classifier_prompt_style}_classifier", args.zero_or_few_shot, curr_embedding_aggregation, curr_model, f"{curr_classifier_dataset}_classifier", curr_train_size]
                create_dir(outdir_subdirs)

                adversarial_predicts = [label_dict[l] for l in list(y_test_pred[:len(adversarial_instances)])]
                control_group_predicts = [label_dict[l] for l in list(y_test_pred[len(adversarial_instances):])]

                results_adversarial = {id:adversarial_predicts[i] for i,id in enumerate(adversarial_ids)}
                results_control_group = {id:control_group_predicts[i] for i,id in enumerate(control_group_ids)}

                results = {"adversarial_predicts": results_adversarial,
                           "control_group_predicts": results_control_group}
                print(f'save to {os.path.join(*outdir_subdirs, f"{dataset}_classification_report.txt")}')
                with open(os.path.join(*outdir_subdirs, f"{dataset}_predictions.json"), 'w') as f1:
                    f1.write(json.dumps(results))
                with open(os.path.join(*outdir_subdirs, f"{dataset}_classification_report.txt"), 'w') as f2:
                    f2.write(clf_report)










if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument('-i', '--indir', type=str, default=None, help='path to data')
    argparser.add_argument('--classifier-dir', type=str, default=None, help='path to classifier')
    argparser.add_argument('--dataset', type=str, default=None, help='prompt type to classify ("squad", "NQ", "musique")')
    argparser.add_argument('--prompt-type', type=str, default="Adversarial", help='prompt type to classify ("Adversarial", "Pseudo-Adversarial", "CoT-Adversarial", "Answerability")')
    argparser.add_argument('--zero-or-few-shot', type=str, default="zero_shot", help='whether this is zero of few shot (pass as \"zero_shot\" or \"few_shot\").')

    argparser.add_argument('--eval-batch-size', type=int, default=64, help='batch size of dev and test sets.')
    argparser.add_argument('--aggregation-type', type=str, default="only_first_tkn", help='how to aggregate all the hidden layers of all the generated tokens of a single instance (choose from "average" to average them, "union" to treat each of them as an instance, and "only_first_tkn" to only take the first token\'s hidden layers).')
    argparser.add_argument('--embedding-type', type=str, default="last_hidden_embedding", help='which layer to take: any one of "last_hidden_embedding" and "first_hidden_embedding"')
    args = argparser.parse_args()
    main(args)


