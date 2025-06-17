import os
import json
import pandas as pd
from tqdm import tqdm
import logging
from pathlib import Path
from datetime import datetime

from .config import MODEL_CONFIG, EXPERIMENT_CONFIG, RESULTS_DIR
from .model_utils import LlamaInference
from .data_processing import HotpotQAProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaselineEvaluator:
    def __init__(self, model_inference: LlamaInference, data_processor: HotpotQAProcessor):
        self.model = model_inference
        self.data_processor = data_processor
        self.results = []
        
    def run_baseline_evaluation(self):
        logger.info("Starting baseline evaluation...")
        
        baseline_prompts = self.data_processor.prepare_baseline_prompts()
        
        for sample in tqdm(baseline_prompts, desc="Evaluating baseline"):
            answer, confidence = self.model.get_answer_and_confidence(
                sample['prompt'], 
                max_new_tokens=EXPERIMENT_CONFIG['max_new_tokens'],
                temperature=EXPERIMENT_CONFIG.get('temperature', 0.1),
                do_sample=EXPERIMENT_CONFIG.get('do_sample', False)
            )
            
            is_correct = HotpotQAProcessor.evaluate_answer(answer, sample['gold_answer'])
            
            result = {
                'id': sample['id'],
                'question': sample['question'],
                'gold_answer': sample['gold_answer'],
                'predicted_answer': answer,
                'confidence': confidence,
                'is_correct': is_correct,
                'knowledge_type': 'bright' if is_correct and confidence > -2.0 else 'dark',
                'supporting_facts': sample['supporting_facts'],
                'timestamp': datetime.now().isoformat()
            }
            
            self.results.append(result)
            
        return self.results
    
    def save_results(self, output_path: Path):
        df = pd.DataFrame(self.results)
        
        df.to_csv(output_path / 'baseline_results.csv', index=False)
        
        with open(output_path / 'baseline_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        stats = {
            'total_samples': int(len(df)),
            'correct_predictions': int(df['is_correct'].sum()),
            'accuracy': float(df['is_correct'].mean()),
            'bright_knowledge_count': int((df['knowledge_type'] == 'bright').sum()),
            'dark_knowledge_count': int((df['knowledge_type'] == 'dark').sum()),
            'avg_confidence': float(df['confidence'].mean()),
            'avg_confidence_correct': float(df[df['is_correct']]['confidence'].mean()) if df['is_correct'].any() else 0,
            'avg_confidence_incorrect': float(df[~df['is_correct']]['confidence'].mean()) if (~df['is_correct']).any() else 0
        }
        
        with open(output_path / 'baseline_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
            
        logger.info(f"Baseline evaluation complete. Accuracy: {stats['accuracy']:.2%}")
        logger.info(f"Bright knowledge: {stats['bright_knowledge_count']}, Dark knowledge: {stats['dark_knowledge_count']}")
        
        return stats

def main():
    output_dir = RESULTS_DIR / 'baseline'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Initializing model...")
    model = LlamaInference(
        model_name=MODEL_CONFIG['model_name'],
        load_in_4bit=MODEL_CONFIG['load_in_4bit']
    )
    
    logger.info("Loading data...")
    data_processor = HotpotQAProcessor(
        split=EXPERIMENT_CONFIG['dataset_split'],
        sample_size=EXPERIMENT_CONFIG['sample_size']
    )
    data_processor.load_data()
    
    evaluator = BaselineEvaluator(model, data_processor)
    results = evaluator.run_baseline_evaluation()
    stats = evaluator.save_results(output_dir)
    
    logger.info("Baseline evaluation completed successfully!")

if __name__ == "__main__":
    main()