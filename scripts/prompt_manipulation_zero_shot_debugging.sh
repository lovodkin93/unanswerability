
session=$1 #"controlled_reduction4"
cuda_visible_device=$2

models="Flan-UL2 Flan-T5-xxl OPT" #"Flan-UL2 Flan-T5-xxl OPT"
only_unanswerable="" #"--only-unanswerable"
only_answerable="" #"--only-answerable"
datasets="squad NQ musique" #"squad NQ musique"
n_instances="--n-instances 3" #"--n-instances 50"
k_beams="1"
k_beams_grid_search="" #"--k-beams-grid-search [3,5,7]"
num_return_sequences="" #"--num-return-sequences ${k_beams}" #""
prompt_variant="variant1 variant2 variant3"
only_hint_prompts="" #"--only-hint-prompts"
batch_size="1"
model_max_length="--model-max-length 700" #""
cuda="" #"--cuda"
trainset="" #"--trainset"
devset="" #"--devset"
only_adversarial="" #"--only-adversarial"
return_only_generated_text="--return-only-generated-text" #"--return-only-generated-text"
return_first_layer="" #"--return-first-layer"
outdir="generated_text_debugging1"


config="--models ${models} ${only_unanswerable} ${only_answerable} --datasets ${datasets} ${n_instances} --k-beams ${k_beams} ${k_beams_grid_search} ${num_return_sequences} --prompt-variant ${prompt_variant} ${only_hint_prompts} --batch-size ${batch_size} ${model_max_length} ${cuda} ${trainset} ${devset} ${only_adversarial} ${return_only_generated_text} ${return_first_layer} --outdir ${outdir}"
echo $config

command="conda activate adversarial_gpt && cd /home/nlp/sloboda1/projects/unanswerable_adversarial && CUDA_VISIBLE_DEVICES=${cuda_visible_device} python zero_shot_embeddings.py ${config}" 




tmux new-session -d -s $session
tmux send-keys -t $session "$command" C-m
tmux attach-session -t $session

