MAX_GPU_MEM=40 # in GB
MAX_CPU_MEM=300 # in GB

PROMPT_TYPES = ["Adversarial", "Pseudo-Adversarial", "CoT-Adversarial",  "Ablation1", "Ablation2", "Answerability"]
UNANSWERABLE_REPLIES = ["unanswerable", "n/a", "idk", "i don't know", "not known", "answer not in context", "the answer is unknown"]
UNANSWERABLE_REPLIES_EXACT = ['nan', 'unknown', 'no answer', 'it is unknown', "the answer is unknown", 'none of the above choices', 'none of the above']