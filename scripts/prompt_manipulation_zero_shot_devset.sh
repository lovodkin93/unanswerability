
session=$1 #"controlled_reduction4"
cuda_visible_device=$2


# ChatGPT="" #"--ChatGPT"
openAI_key="" #"--openAI-key <something>"
models="Flan-UL2 Flan-T5-xxl OPT" #"Flan-UL2 Flan-T5-xxl OPT"
adversarial="--adversarial" #"--adversarial"
control_group="--control-group" #"--control-group"
datasets="squad" #"squad NQ musique"
all_instances="--all-instances" #"--all-instances"
unfiltered_instances="" #"--unfiltered-instances"
n_instances="--n-instances 100" #"--n-instances 50"
k_beams="1"
k_beams_grid_search="" #"--k-beams-grid-search [3,5,7]"
num_return_sequences="" #"--num-return-sequences ${k_beams}" #""
prompt_variant="variant1"
only_hint_prompts="" #"--only-hint-prompts"
batch_size="1"
model_max_length="--model-max-length 700" #""
cuda="" #"--cuda"
trainset="" #"--trainset"
devset="--devset" #"--trainset"
only_adversarial="" #"--only-adversarial"
return_only_generated_text="--return-only-generated-text" #"--return-only-generated-text"
return_first_layer="" #"--return-first-layer"



config="--models ${models} ${adversarial} ${control_group} --datasets ${datasets} ${all_instances} ${unfiltered_instances} ${n_instances} --k-beams ${k_beams} ${k_beams_grid_search} ${num_return_sequences} --prompt-variant ${prompt_variant} ${only_hint_prompts} --batch-size ${batch_size} ${model_max_length} ${cuda} ${trainset} ${devset} ${only_adversarial} ${return_only_generated_text} ${return_first_layer}"
echo $config
config="${config} ${openAI_key}"

command="conda activate adversarial_gpt && cd /home/nlp/sloboda1/projects/unanswerable_adversarial && CUDA_VISIBLE_DEVICES=${cuda_visible_device} python zero_shot_embeddings.py ${config}" 




tmux new-session -d -s $session
tmux send-keys -t $session "$command" C-m
tmux attach-session -t $session

