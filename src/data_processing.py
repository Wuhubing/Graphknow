import pandas as pd
from datasets import load_dataset
from typing import Dict, List, Tuple
import json
import logging

logger = logging.getLogger(__name__)

class HotpotQAProcessor:
    def __init__(self, split: str = "validation", sample_size: int = 100):
        self.split = split
        self.sample_size = sample_size
        self.dataset = None
        self.samples = []
        
    def load_data(self):
        logger.info(f"Loading HotpotQA dataset, split: {self.split}")
        self.dataset = load_dataset("hotpot_qa", "fullwiki", split=self.split)
        
        if self.sample_size:
            indices = list(range(min(self.sample_size, len(self.dataset))))
            self.samples = [self.dataset[i] for i in indices]
        else:
            self.samples = list(self.dataset)
            
        logger.info(f"Loaded {len(self.samples)} samples")
        
    def extract_knowledge_chains(self, sample: Dict) -> Dict:
        question = sample['question']
        answer = sample['answer']
        supporting_facts = sample['supporting_facts']
        context = sample['context']
        
        titles = context['title']
        sentences = context['sentences']
        
        supporting_sentences = []
        for fact_title, fact_sent_id in zip(supporting_facts['title'], supporting_facts['sent_id']):
            for i, title in enumerate(titles):
                if title == fact_title:
                    if fact_sent_id < len(sentences[i]):
                        supporting_sentences.append({
                            'title': title,
                            'sentence': sentences[i][fact_sent_id]
                        })
        
        return {
            'question': question,
            'answer': answer,
            'supporting_facts': supporting_sentences,
            'all_context': context
        }
    
    def prepare_baseline_prompts(self) -> List[Dict]:
        baseline_data = []
        
        for sample in self.samples:
            knowledge_chain = self.extract_knowledge_chains(sample)
            
            prompt = f"Question: {knowledge_chain['question']}\nAnswer:"
            
            baseline_data.append({
                'id': sample['id'],
                'question': knowledge_chain['question'],
                'gold_answer': knowledge_chain['answer'],
                'prompt': prompt,
                'supporting_facts': knowledge_chain['supporting_facts'],
                'all_context': knowledge_chain['all_context']
            })
        
        return baseline_data
    
    @staticmethod
    def evaluate_answer(predicted: str, gold: str) -> bool:
        predicted_lower = predicted.lower().strip()
        gold_lower = gold.lower().strip()
        
        if gold_lower in predicted_lower:
            return True
        
        pred_tokens = set(predicted_lower.split())
        gold_tokens = set(gold_lower.split())
        
        if len(gold_tokens.intersection(pred_tokens)) / len(gold_tokens) > 0.5:
            return True
            
        return False