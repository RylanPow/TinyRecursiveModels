import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class LLMTutor:
    def __init__(self, model_id="Qwen/Qwen3-8B", layer_to_extract=30):
        print(f"Loading {model_id} Tutor in 4-bit...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.layer_to_extract = layer_to_extract

    def get_strategy_embedding(self, input_tensor):
        batch_texts = []
        for grid in input_tensor:
            grid_list = grid.flatten().tolist()
            seq_len = len(grid_list)
            
            if seq_len == 900:
                # --- MAZE 30x30 FORMATTING ---
                # Reverse mapping based on dataset/build_maze_dataset.py
                char_map = {1: '#', 2: ' ', 3: 'S', 4: 'G', 5: 'o', 0: ' '}
                
                # Convert IDs to characters
                ascii_chars = [char_map.get(val, '?') for val in grid_list]
                
                # Chunk into 30 rows of 30 characters
                rows = ["".join(ascii_chars[i:i+30]) for i in range(0, 900, 30)]
                grid_str = "\n".join(rows)
                
                prompt = f"Examine this 30x30 maze where '#' is a wall, 'S' is start, 'G' is goal, and 'o' is path:\n{grid_str}\nThe underlying spatial transformation rule is..."
                
            elif seq_len == 81:
                # --- SUDOKU 9x9 FORMATTING ---
                # Chunk into 9 rows of 9 numbers
                rows = [" ".join(map(str, grid_list[i:i+9])) for i in range(0, 81, 9)]
                grid_str = "\n".join(rows)
                
                prompt = f"Examine this 9x9 Sudoku grid (0 represents empty cells):\n{grid_str}\nThe underlying logical constraint rule is..."
                
            else:
                # --- FALLBACK ---
                grid_str = " ".join(map(str, grid_list))
                prompt = f"Examine this sequence: {grid_str}. The underlying rule is..."
                
            batch_texts.append(prompt)
            
        inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to("cuda")
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
        # Extract the specified layer's hidden state for the LAST token
        return outputs.hidden_states[self.layer_to_extract][:, -1, :]