
session=$1 
# cuda_visible_device=$2

prompt_type="Adversarial"
zero_or_few_shot="zero_shot"
embedding_type="last_hidden_embedding" # any of "last_hidden_embedding" and "first_hidden_embedding"
aggregation_type="only_first_tkn" #any one of "average", "union" and "only_first_tkn"

config="--prompt-type ${prompt_type} --zero-or-few-shot ${zero_or_few_shot} --aggregation-type ${aggregation_type} --embedding-type ${embedding_type}"
echo $config

command="conda activate adversarial_gpt && cd /home/nlp/sloboda1/projects/unanswerable_adversarial && python eval_linear_classifiers.py ${config}" 




tmux new-session -d -s $session
tmux send-keys -t $session "$command" C-m
tmux attach-session -t $session

