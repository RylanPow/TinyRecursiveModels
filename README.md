# Tiny Recursive Models (TRMs) Adapted for Transfer Learning

This is the repo for the paper Tiny Recursive Models (TRMs) Addapted for Transfer Learning.  We introduce the Semantic Hot-Starting mechansim to replace the TRM's dependency on puzzle_id embeddings.  

### Motivation

The Tiny Recursive Model (TRM) is a recursive reasoning model that achieved amazing performance with only 7M paramters. However, when solving a task, the TRM relies on a embedding of the entire puzzle it is trying to solve.  It is not just given this puzzle_id at training time: it is also given this embedding during test and evaluation time.  Other studies in addition to my own have found the performance substantially worse when the puzzle_id is removed, which is a fair ablation test considering real life scenarios do not come with perfeclty labeled data and puzzles.  The goal of our research is to replace these static puzzle_id tokens with semantically-grounded intuition vectors dervied from a tutor LLM a la transfer learning.  We refer to this model as the "Hot-Started TRM".

### Architecture
Our modified model's architecture consists of 3 critical components: the original TRM core, a frozen LLM tutor, and a "strategy projector".  The frozen LLM tutor, Qwen3-8B in our experiments, is fed an embedding of the puzzle.  We take the 30th hidden layer (deep enough to contain rich task-related reasoning whilst early enough to avoid information focused on linguistic token generation) and use that as the "intuition vector".  The intuition vector is of 4096 dimension, while the TRM takes in puzzle_id of dimension 512.  TO bridge this gap, the strategy projector, implemented as a 3-layer MLP NN, translates the intuition vector into the appropriate dimensions for the TRM ot understand.  This intuition vector fed through the strategy projector entirely replaces the puzzle_id, allowing us to circumvent the retrieval dependency of the original TRM. 
<p align="center">
  <img src="https://github.com/RylanPow/TinyRecursiveModels/blob/main/assets/trm%20vs%20hot%20start.png" style="width: 100%;">
</p>
In the above figure, the left model is the original TRM, and the right model is the TRM with our Hot-Starting mechanism.


### Requirements
CREDIT: much of the setup comes directly fromt he original TRM repository: https://github.com/SamsungSAILMontreal/TinyRecursiveModels/tree/main.  The major differences here are the more powerful GPU, the Qwen3-8B model, and other related dependencies.

These experiments were conducted on an A100 (80GB RAM) GPU on Runpod.  Enough container disk is necessary for Ubuntu, Python, and the installed pip libraries.  The volume disk must be large enough to contain the Qwen3-8B Weights (~16GB, maze data (~1GB), sudoku data (~0.5GB), the maze latents (~8GB), and the sudoku latents (~8GB).  About 50-60GB of VRAM will be used. In case that you run into issues due to library versions, here is the requirements with the exact versions used: [specific_requirements.txt](https://github.com/SamsungSAILMontreal/TinyRecursiveModels/blob/main/specific_requirements.txt).

- Python 3.10 (or similar)
- Cuda 12.6.0 (or similar)

```bash
pip install --upgrade pip wheel setuptools
pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126 
pip install -r requirements_hotstart.txt # install requirements
pip install --no-cache-dir --no-build-isolation adam-atan2
apt update && apt install tmux -y 
wandb login YOUR-LOGIN # login if you want the logger to sync results to your Weights & Biases (https://wandb.ai/)
```

### Dataset Preparation

# Sudoku-Extreme
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000  # 1000 examples, 1000 augments

# Maze-Hard
python dataset/build_maze_dataset.py # 1000 examples, 8 augments
```

## Experiments

### Sudoku-Extreme (with 1 A100 GPU):

```bash
run_name="pretrain_att_sudoku"
python pretrain.py \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True
```

*runtime:* ~20 hours

### Maze-Hard (with 1 A100 GPU):

```bash
run_name="pretrain_att_maze30x30_1gpu"
python pretrain.py \
arch=trm \
data_paths="[data/maze-30x30-hard-1k]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 global_batch_size=128 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True
```
*Runtime:* ~ 30 hours


## Reference

Also cite the original TRM paper:
```bibtex
@misc{jolicoeurmartineau2025morerecursivereasoningtiny,
      title={Less is More: Recursive Reasoning with Tiny Networks}, 
      author={Alexia Jolicoeur-Martineau},
      year={2025},
      eprint={2510.04871},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.04871}, 
}
```
Original TRM repo: https://github.com/SamsungSAILMontreal/TinyRecursiveModels/tree/main
