
session=$1 #"controlled_reduction4"
cuda_visible_device=$2


# ChatGPT="" #"--ChatGPT"
openAI_key="" #"--openAI-key <something>"
# UL2="" #"--UL2"
# PaLM="--PaLM" #"--PaLM"
# OPT="" #"--OPT"
models="Flan-UL2" #"Flan-UL2 Flan-T5-xxl OPT"
adversarial="--adversarial" #"--adversarial"
control_group="--control-group" #"--control-group"
datasets="squad NQ musique"
all_instances="--all-instances" #"--all-instances"
unfiltered_instances="" #"--unfiltered-instances"
n_instances="" #"--n-instances 50"
k_beams="3"
k_beams_grid_search="" #"--k-beams-grid-search [3,5,7]"
num_return_sequences="" #"--num-return-sequences ${k_beams}" #""
prompt_variant="variant1"
only_hint_prompts="" #"--only-hint-prompts"
batch_size="1"
model_max_length="--model-max-length 700" #""
cuda="" #"--cuda"
trainset="" #"--trainset"
only_adversarial="--only-adversarial" #"--only-adversarial"
return_only_generated_text="" #"--return-only-generated-text"
eraser_dir="trained_erasers/squad/Adversarial/UL2_Flan_first_tkn_1000N/eraser.pkl"
no_eraser="" #"--no-eraser"
only_first_decoding="--only-first-decoding" #""

config="--models ${models} ${adversarial} ${control_group} --datasets ${datasets} ${all_instances} ${unfiltered_instances} ${n_instances} --k-beams ${k_beams} ${k_beams_grid_search} ${num_return_sequences} --prompt-variant ${prompt_variant} ${only_hint_prompts} --batch-size ${batch_size} ${model_max_length} ${cuda} ${trainset} ${only_adversarial} ${return_only_generated_text} --eraser-dir ${eraser_dir} ${no_eraser} ${only_first_decoding}" 
echo $config
config="${config} ${openAI_key}"

command="conda activate usb && cd /home/nlp/sloboda1/projects/unanswerable_adversarial && CUDA_VISIBLE_DEVICES=${cuda_visible_device} python zero_shot_embeddings_erasure.py ${config}" 




tmux new-session -d -s $session
tmux send-keys -t $session "$command" C-m
tmux attach-session -t $session
















