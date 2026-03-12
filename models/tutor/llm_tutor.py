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
        # extremely basic grid-to-text conversion for the smoke test
        # in the real run, format this specifically for Maze/Sudoku
        batch_texts = []
        for grid in input_tensor:
            flat_grid = " ".join(map(str, grid.flatten().tolist()[:50])) # Just take first 50 to avoid massive prompts in test
            prompt = f"Examine this grid pattern: {flat_grid}. The underlying transformation rule is..."
            batch_texts.append(prompt)
            
        inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to("cuda")
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
        # extract the specified layer's hidden state for the LAST token
        return outputs.hidden_states[self.layer_to_extract][:, -1, :]