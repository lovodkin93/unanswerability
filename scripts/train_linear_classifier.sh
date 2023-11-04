
session=$1 
# cuda_visible_device=$2

indir="responses_embeddings/k-beams/21-06-2023_17:48:45/OPT/zero_shot/k_beams_1_num_return_seq_1/variant3"
outdir="trained_classifiers"
dataset="musique"
prompt_type="Regular-Prompt"
epochs="100"
batch_size="64"
n_instances="--num-instances 2000" #"--num-instances 500"
aggregation_type="only_first_tkn" #any one of "average", "union" and "only_first_tkn"
embedding_type="first_hidden_embedding" # any of "last_hidden_embedding" and "first_hidden_embedding"
config="--indir ${indir} --outdir ${outdir} --dataset ${dataset} --prompt-type ${prompt_type} --epochs ${epochs} --batch-size ${batch_size} ${n_instances} --aggregation-type ${aggregation_type} --embedding-type ${embedding_type}"
echo $config

command="conda activate adversarial_gpt && cd /home/nlp/sloboda1/projects/unanswerable_adversarial && python train_linear_classifiers.py ${config}" 




tmux new-session -d -s $session
tmux send-keys -t $session "$command" C-m
tmux attach-session -t $session

