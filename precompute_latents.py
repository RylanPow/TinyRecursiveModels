import os
import json
import numpy as np
import torch
from tqdm import tqdm
from models.tutor.llm_tutor import LLMTutor

def process_split(dataset_path, split_name, batch_size=32):
    print(f"\n--- Processing {split_name} split ---")
    
    base_dir = os.path.join(dataset_path, split_name)
    
    # Read the metadata to get the correct set names (e.g., ["all"])
    metadata_path = os.path.join(base_dir, "dataset.json")
    if not os.path.exists(metadata_path):
        print(f"Metadata not found: {metadata_path}")
        return
        
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
        
    sets = metadata.get("sets", ["all"])
    
    # Process each set
    for set_name in sets:
        print(f"  -> Processing set: {set_name}")
        inputs_path = os.path.join(base_dir, f"{set_name}__inputs.npy")
        latents_path = os.path.join(base_dir, f"{set_name}__latents.npy")
        
        if not os.path.exists(inputs_path):
            print(f"  File not found: {inputs_path}")
            continue

        # Load inputs using mmap
        inputs = np.load(inputs_path, mmap_mode="r")
        num_examples = inputs.shape[0]
        print(f"  Found {num_examples} examples.")

        #Pre-allocate output array in float16
        latents_mmap = np.lib.format.open_memmap(
            latents_path, mode='w+', dtype=np.float16, shape=(num_examples, 4096)
        )


        for i in tqdm(range(0, num_examples, batch_size)):
            end_idx = min(i + batch_size, num_examples)
            batch_inputs = inputs[i:end_idx]
            
            # Get latents from frozen LLM
            latents = tutor.get_strategy_embedding(batch_inputs).to(torch.float32).cpu().numpy().astype(np.float16)
            
            latents_mmap[i:end_idx] = latents
            
        latents_mmap.flush()
        print(f"  Saved latents to {latents_path}")

if __name__ == "__main__":
    tutor = LLMTutor(layer_to_extract=30)
    
    dataset_dir = "data/maze-30x30-hard-1k"
    
    process_split(dataset_dir, "train", batch_size=64)
    process_split(dataset_dir, "test", batch_size=64)
    
    print("\nPrecomputation Complete!")