import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class LlamaInference:
    def __init__(self, model_name: str, load_in_4bit: bool = True):
        self.model_name = model_name
        
        logger.info(f"Loading tokenizer from {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Loading model {model_name} with 4-bit quantization: {load_in_4bit}")
        
        try:
            # Configure quantization for memory efficiency
            if load_in_4bit:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            if "401" in str(e) or "authentication" in str(e).lower():
                logger.error("="*80)
                logger.error("ERROR: Authentication failed!")
                logger.error("Please make sure you:")
                logger.error("1. Have a Hugging Face account")
                logger.error("2. Have access to the Llama-2 model")
                logger.error("3. Are logged in (run: python setup_auth.py)")
                logger.error("="*80)
            elif "out of memory" in str(e).lower() or "oom" in str(e).lower():
                logger.error("="*80)
                logger.error("ERROR: Out of GPU memory!")
                logger.error("Try:")
                logger.error("1. Enable 4-bit quantization (default)")
                logger.error("2. Reduce batch size")
                logger.error("3. Use a smaller model")
                logger.error("4. Close other GPU applications")
                logger.error("="*80)
            else:
                logger.error(f"ERROR loading model: {str(e)}")
            raise
        
    def get_answer_and_confidence(self, prompt: str, max_new_tokens: int = 50, 
                                 temperature: float = 0.1, do_sample: bool = False) -> Tuple[str, float]:
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=temperature,
                do_sample=do_sample
            )
        
        answer_ids = outputs.sequences[0, inputs.input_ids.shape[1]:]
        answer_text = self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        
        scores = outputs.scores
        log_probs = []
        for i, score in enumerate(scores):
            if i >= len(answer_ids):
                break
            token_id = answer_ids[i].unsqueeze(0)
            log_prob = torch.log_softmax(score, dim=-1)
            token_log_prob = log_prob[0, token_id].item()
            log_probs.append(token_log_prob)
        
        avg_log_prob = sum(log_probs) / len(log_probs) if log_probs else -float('inf')
        
        return answer_text, avg_log_prob
    
    def batch_inference(self, prompts: list, max_new_tokens: int = 50,
                       temperature: float = 0.1, do_sample: bool = False) -> list:
        results = []
        for prompt in prompts:
            answer, confidence = self.get_answer_and_confidence(
                prompt, max_new_tokens, temperature, do_sample
            )
            results.append({
                'prompt': prompt,
                'answer': answer,
                'confidence': confidence
            })
        return results