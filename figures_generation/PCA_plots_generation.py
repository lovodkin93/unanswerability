import numpy as np
from tqdm import tqdm
import torch

import os
from sklearn.decomposition import PCA
import umap.umap_ as umap
import plotly.graph_objects as go
import argparse
from pathlib import Path





def adapt_hidden_embeddings(instance):
    
    # if the embeddings of all the generation steps were saved in a single matrix, rather than in a list, separate them
    if len(instance['last_hidden_embedding'][-1].shape) == 2:
        instance['last_hidden_embedding'] = [instance['last_hidden_embedding'][0][i,:] for i in range(instance['last_hidden_embedding'][0].shape[0])]


    # removing the paddings
    # Compare all elements to 1
    if "all_outputs_ids" in instance.keys():
        matches = instance['all_outputs_ids'][0,:].eq(1)

        # Find the first non-zero element in matches
        indices = matches.nonzero(as_tuple=True)

        # Get the first index where value is 1 (if no 1 then no "padding" and so can take all embeddings)
        filter_index = indices[0][0].item() if indices[0].numel() != 0 else len(instance['last_hidden_embedding'])
    else:
        filter_index = len(instance['last_hidden_embedding'])

    filtered_hidden_embedding = instance['last_hidden_embedding'][:filter_index]
    return filtered_hidden_embedding

def get_data_name(full_file_path):
    if "squad" in full_file_path:
        return "squad"
    elif "NQ" in full_file_path:
        return "NQ"
    elif "musique" in full_file_path:
        return "musique"
    else:
        raise Exception(f"dataset name not found in {full_file_path}")

def get_model_name(curr_indir):
    if "UL2_Flan" in curr_indir:
        return "UL2"
    elif "T5_xxl_Flan" in curr_indir:
        return "T5"
    elif "OPT" in curr_indir:
        return "OPT"
    else:
        raise Exception(f"curr model not found in indir: {curr_indir}")




def get_response(options):
    unanswerable_replies = ["unanswerable", "n/a", "idk", "i don't know", "not known", "answer not in context"]
    unanswerable_replies_exact = ['nan', 'unknown', 'no answer', 'it is unknown', "none of the above", 'none of the above choices']
    for option in options:
        option = str(option).lower().strip()
        if any(option==elem1 for elem1 in unanswerable_replies_exact) or any(option==f"{elem1}." for elem1 in unanswerable_replies_exact) or any(elem2 in option for elem2 in unanswerable_replies):
            return "unanswerable"
    return options[0]


def get_data(curr_indir, prompt_type, embedding_type):
    full_pt_dicts = dict()

    for subdir, dirs, files in os.walk(curr_indir):
        for file in files:
            if not file.endswith(".pt"):
                continue
            curr_data = torch.load(os.path.join(subdir, file))
            curr_data_name = get_data_name(os.path.join(subdir, file))

            if file.startswith("adversarial"):
                
                full_pt_dicts["adversarial"] = curr_data



                if embedding_type == "first_hidden_embedding":
                    adversarial_all_embeddings = [instance[embedding_type] for instance in curr_data[prompt_type]]
                else:
                    adversarial_all_embeddings = [torch.stack(adapt_hidden_embeddings(instance)) for instance in curr_data[prompt_type]]
            

            elif file.startswith("control_group"):
                full_pt_dicts["control_group"] = curr_data

                if embedding_type == "first_hidden_embedding":
                    control_group_all_embeddings = [instance[embedding_type] for instance in curr_data[prompt_type]]
                else:
                    control_group_all_embeddings = [torch.stack(adapt_hidden_embeddings(instance)) for instance in curr_data[prompt_type]]
            else:
                raise Exception(f"{file} file doesn't start with \"adversarial\" nor with \"control_group\".")
    
    return adversarial_all_embeddings, control_group_all_embeddings, full_pt_dicts, curr_data_name



def create_pca_plot(data_pca, unanswerable_adversarial, answerable_adversarial, unanswerable_control_group, adversarial_embeddings, outdir):
    scatter1 = go.Scatter3d(
        x=data_pca[:len(unanswerable_adversarial), 0],
        y=data_pca[:len(unanswerable_adversarial), 1],
        z=data_pca[:len(unanswerable_adversarial), 2],
        mode='markers',
        marker=dict(
            size=2,
            color='red',  # Set color to blue for first type of instances
        ),
        name='unanswerable queries (identified)'
    )

    scatter2 = go.Scatter3d(
        x=data_pca[len(unanswerable_adversarial):len(unanswerable_adversarial)+len(answerable_adversarial), 0],
        y=data_pca[len(unanswerable_adversarial):len(unanswerable_adversarial)+len(answerable_adversarial), 1],
        z=data_pca[len(unanswerable_adversarial):len(unanswerable_adversarial)+len(answerable_adversarial), 2],
        mode='markers',
        marker=dict(
            size=2,
            color='pink',  # Set color to blue for first type of instances
        ),
        name='unanswerable queries (unidentified)'
    )

    scatter3 = go.Scatter3d(
        x=data_pca[len(adversarial_embeddings):len(adversarial_embeddings)+len(unanswerable_control_group), 0],
        y=data_pca[len(adversarial_embeddings):len(adversarial_embeddings)+len(unanswerable_control_group), 1],
        z=data_pca[len(adversarial_embeddings):len(adversarial_embeddings)+len(unanswerable_control_group), 2],
        mode='markers',
        marker=dict(
            size=2,
            color='green',  # Set color to blue for first type of instances
        ),
        name='answerable queries (unidentified)'
    )

    scatter4 = go.Scatter3d(
        x=data_pca[len(adversarial_embeddings)+len(unanswerable_control_group):, 0],
        y=data_pca[len(adversarial_embeddings)+len(unanswerable_control_group):, 1],
        z=data_pca[len(adversarial_embeddings)+len(unanswerable_control_group):, 2],
        mode='markers',
        marker=dict(
            size=2,
            color='blue',  # Set color to blue for first type of instances
        ),
        name='answerable queries (identified)'
    )

    # ordered as scatter1, scatter2, scatter4, scatter3, so in the legend looks better
    fig = go.Figure(data=[scatter1, scatter2, scatter4, scatter3])

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                tickfont=dict(
                    size=10,
                ),
            ),
            yaxis=dict(
                tickfont=dict(
                    size=10,
                ),
            ),
            zaxis=dict(
                tickfont=dict(
                    size=10,
                ),
            ),
            aspectmode='cube'
        ),
        legend=dict(
            itemclick=False,  # Disable item click
            itemdoubleclick=False,  # Disable item double click
            font=dict(
                size=12,  # Increase text size to 14
            ),
            traceorder="normal",
            itemsizing='constant'  # Increase marker size
        )
    )

    # fig.show()
    fig.write_html(outdir)


def main(args):
    aggregation_type = args.aggregation_type #"only_first_tkn" 
    prompt_type = args.prompt_type # "Pseudo-Adversarial"
    embedding_type = args.embedding_type # "first_hidden_embedding" 
    indirs = args.indirs #["../responses_embeddings/k-beams/22-06-2023_12:26:12/OPT"]

    # create outdir
    outdir_path = os.path.join(args.outdir, embedding_type, aggregation_type, prompt_type)
    outdir_path_cls = Path(outdir_path)
    outdir_path_cls.mkdir(parents=True, exist_ok=True)

    for indir in tqdm(indirs):
        adversarial_all_embeddings, control_group_all_embeddings, full_pt_dicts, curr_data_name = get_data(indir, prompt_type, embedding_type)



        if embedding_type == "first_hidden_embedding":
            adversarial_embeddings = [elem.cpu().numpy() for elem in adversarial_all_embeddings]
            control_group_embeddings = [elem.cpu().numpy() for elem in control_group_all_embeddings]
        elif aggregation_type == "only_first_tkn":
            adversarial_embeddings = [elem.squeeze()[0,:].cpu().numpy()  if len(elem.shape)>2 else elem[0,:].cpu().numpy() for elem in adversarial_all_embeddings]
            control_group_embeddings = [elem.squeeze()[0,:].cpu().numpy()  if len(elem.shape)>2 else elem[0,:].cpu().numpy() for elem in control_group_all_embeddings]
        elif aggregation_type == "average":
            adversarial_embeddings = [elem.mean(dim=0).cpu().numpy() for elem in adversarial_all_embeddings]
            control_group_embeddings = [elem.mean(dim=0).cpu().numpy() for elem in control_group_all_embeddings]
        elif aggregation_type == "aggregated":
            adversarial_instances = [(emb.cpu().numpy(), instance["outputs"][0]) for instance in full_pt_dicts["adversarial"][prompt_type] for emb in adapt_hidden_embeddings(instance)]
            control_group_instances = [(emb.cpu().numpy(), instance["outputs"][0]) for instance in full_pt_dicts["control_group"][prompt_type] for emb in adapt_hidden_embeddings(instance)]

            adversarial_embeddings = [elem[0] for elem in adversarial_instances]
            control_group_embeddings = [elem[0] for elem in control_group_instances]
        else:
            raise Exception(f'aggregation_type can only be any of any of "average", "only_first_tkn" and "aggregated", but got {aggregation_type}')

        # Extracting Actual Text Outputs
        adversarial_outputs = [elem["outputs"][0] for elem in full_pt_dicts["adversarial"][prompt_type]]
        control_group_outputs = [elem["outputs"][0] for elem in full_pt_dicts["control_group"][prompt_type]]


        # separate questions into "unanswerable" replies and other
        unanswerable_adversarial = [adversarial_embeddings[i] for i,txt in enumerate(adversarial_outputs) if get_response([txt])=="unanswerable"]
        answerable_adversarial = [adversarial_embeddings[i] for i,txt in enumerate(adversarial_outputs) if get_response([txt])!="unanswerable"]

        unanswerable_control_group = [control_group_embeddings[i] for i,txt in enumerate(control_group_outputs) if get_response([txt])=="unanswerable"]
        answerable_control_group = [control_group_embeddings[i] for i,txt in enumerate(control_group_outputs) if get_response([txt])!="unanswerable"]

        # Stack all vectors
        combined_data = np.vstack((unanswerable_adversarial, answerable_adversarial, unanswerable_control_group, answerable_control_group))

        # Initialize PCA
        pca = PCA(n_components=3)

        # Fit and transform data to 2D
        data_pca = pca.fit_transform(combined_data)

        # create and save PCA plot
        curr_model_name = get_model_name(indir)

        curr_outdir = os.path.join(outdir_path, f"{curr_model_name}_{curr_data_name}_3D.html")
        print(f"Saving PCA plot of {curr_model_name} on {curr_data_name} to: {curr_outdir}")
        create_pca_plot(data_pca, unanswerable_adversarial, answerable_adversarial, unanswerable_control_group, adversarial_embeddings, curr_outdir)



if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument('-i', '--indirs', nargs='+', type=str, required=True, help='path to data')
    argparser.add_argument('-o', '--outdir', type=str, required=True, help='path to outdir')
    argparser.add_argument('--prompt-type', type=str, default="Adversarial", help='prompt type to classify ("Adversarial" or "Pseudo-Adversarial")')
    argparser.add_argument('--aggregation-type', type=str, default="only_first_tkn", help='how to aggregate all the hidden layers of all the generated tokens of a single instance (choose from "average" to average them, "union" to treat each of them as an instance, and "only_first_tkn" to only take the first token\'s hidden layers).')
    argparser.add_argument('--embedding-type', type=str, default="last_hidden_embedding", help='which layer to take: any one of "last_hidden_embedding" and "first_hidden_embedding"')
    args = argparser.parse_args()
    main(args)