
session=$1 
command="conda activate adversarial_gpt && cd /home/nlp/sloboda1/projects/unanswerable_adversarial && python PCA_plots_generation.py" 




tmux new-session -d -s $session
tmux send-keys -t $session "$command" C-m
tmux attach-session -t $session

