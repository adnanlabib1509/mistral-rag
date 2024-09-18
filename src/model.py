from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from src.config import MISTRAL_MODEL, MAX_NEW_TOKENS, TEMPERATURE

class MistralModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MISTRAL_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(MISTRAL_MODEL, torch_dtype=torch.float16)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        
    def generate(self, prompt: str) -> str:
        """
        Generate text based on the given prompt using the Mistral model.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)