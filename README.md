# Tiny Recursive Models (TRMs) Adapted for Transfer Learning

This is the repo for the paper Tiny Recursive Models (TRMs) Addapted for Transfer Learning.  We introduce the Semantic Hot-Starting mechansim to replace the TRM's dependency on puzzle_id embeddings.  

### Motivation

The Tiny Recursive Model (TRM) is a recursive reasoning model that achieved amazing performance with only 7M paramters. However, when solving a task, the TRM relies on a embedding of the entire puzzle it is trying to solve.  It is not just given this puzzle_id at training time: it is also given this embedding during test and evaluation time.  Other studies in addition to my own have found the performance substantially worse when the puzzle_id is removed, which is a fair ablation test considering real life scenarios do not come with perfeclty labeled data and puzzles.  The goal of our research is to replace these static puzzle_id tokens with semantically-grounded intuition vectors dervied from a tutor LLM ala transfer learning.  We refer to this model as the "Hot-Started TRM".

### Architecture

<p align="center">
  <img src="[https://github.com/RylanPow/TinyRecursiveModels/assets/trm vs hot start.png](https://github.com/RylanPow/TinyRecursiveModels/blob/main/assets/trm%20vs%20hot%20start.png)" style="width: 30%;">
</p>


### Requirements

Installation should take a few minutes. For the smallest experiments on Sudoku-Extreme (pretrain_mlp_t_sudoku), you need 1 GPU with enough memory. With 1 L40S (48Gb Ram), it takes around 18h to finish. In case that you run into issues due to library versions, here is the requirements with the exact versions used: [specific_requirements.txt](https://github.com/SamsungSAILMontreal/TinyRecursiveModels/blob/main/specific_requirements.txt).

- Python 3.10 (or similar)
- Cuda 12.6.0 (or similar)

```bash
pip install --upgrade pip wheel setuptools
pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126 # install torch based on your cuda version
pip install -r requirements.txt # install requirements
pip install --no-cache-dir --no-build-isolation adam-atan2 
wandb login YOUR-LOGIN # login if you want the logger to sync results to your Weights & Biases (https://wandb.ai/)
```

### Dataset Preparation

# Sudoku-Extreme
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000  # 1000 examples, 1000 augments

# Maze-Hard
python dataset/build_maze_dataset.py # 1000 examples, 8 augments
```

## Experiments

### Sudoku-Extreme (with 1 L40S GPU):

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

### Maze-Hard (with 1 L40S GPUs):

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
asoning Model [code](https://github.com/sapientinc/HRM) and the Hierarchical Reasoning Model Analysis [code](https://github.com/arcprize/hierarchical-reasoning-model-analysis).
