import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Dict, Tuple
import random

from .config import PERTURBATION_STRATEGIES, RESULTS_DIR, EXPERIMENT_CONFIG
from .model_utils import LlamaInference
from .knowledge_graph import LocalKnowledgeGraphBuilder
from .data_processing import HotpotQAProcessor
from .semantic_utils import SemanticAnalyzer

logger = logging.getLogger(__name__)

class PerturbationExperiment:
    def __init__(self, model: LlamaInference, graph_builder: LocalKnowledgeGraphBuilder):
        self.model = model
        self.graph_builder = graph_builder
        self.results = []
        self.semantic_analyzer = SemanticAnalyzer()
        
    def generate_perturbation_prompt(self, original_question: str, influencer_node: str, 
                                   influencer_context: str = None) -> str:
        if not influencer_context:
            influencer_context = f"The entity {influencer_node} is relevant to this topic."
        
        prompt = f"""Context: {influencer_context}
Based on the context, now answer the following question:
Question: {original_question}
Answer:"""
        
        return prompt
    
    def find_influencer_context(self, influencer_node: str, all_context: Dict) -> str:
        for title, sentences in zip(all_context['title'], all_context['sentences']):
            for sentence in sentences:
                if influencer_node.lower() in sentence.lower():
                    return sentence
        
        return f"The entity {influencer_node} is relevant to this topic."
    
    def evaluate_influencer_confidence(self, influencer_context: str) -> float:
        """评估模型对干扰知识本身的置信度"""
        # 构造一个简单的问题来测试模型对这个知识的置信度
        prompt = f"""Based on the following information, evaluate if it is true or false:
Information: {influencer_context}
Is this information likely to be true? Answer with just 'Yes' or 'No':"""
        
        answer, confidence = self.model.get_answer_and_confidence(
            prompt,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=False
        )
        
        return confidence
    
    def select_semantic_influencers(self, G, sample: Dict, similarity_type: str = 'high') -> List[Tuple[str, Dict, str]]:
        """基于语义相似度选择干扰节点"""
        # 获取核心事实句子
        core_sentences = [fact['sentence'] for fact in sample['supporting_facts']]
        core_text = ' '.join(core_sentences)
        
        # 获取图中所有节点及其上下文
        all_nodes = list(G.nodes())
        node_contexts = {}
        for node in all_nodes:
            context = self.find_influencer_context(node, sample['all_context'])
            if context and context != f"The entity {node} is relevant to this topic.":
                node_contexts[node] = context
        
        if not node_contexts:
            return []
        
        # 计算语义相似度并分类
        candidate_sentences = list(node_contexts.values())
        categorized = self.semantic_analyzer.categorize_by_similarity(
            core_text, candidate_sentences
        )
        
        # 根据策略选择节点
        selected_nodes = []
        if similarity_type == 'high':
            candidates = categorized.get('high', [])
        elif similarity_type == 'medium':
            candidates = categorized.get('medium', [])
        elif similarity_type == 'low':
            candidates = categorized.get('low', [])
        else:
            candidates = []
        
        # 转换为节点信息
        node_list = list(node_contexts.keys())
        for idx, sentence, similarity in candidates[:5]:
            node = node_list[idx]
            if node in G:
                node_metrics = {
                    'degree': G.degree(node),
                    'distance_to_core': G.nodes[node].get('distance_to_core', -1),
                    'betweenness_centrality': G.nodes[node].get('betweenness_centrality', 0),
                    'semantic_similarity': similarity
                }
                selected_nodes.append((node, node_metrics, sentence))
        
        return selected_nodes
    
    def run_perturbation_on_sample(self, sample: Dict, baseline_result: Dict) -> List[Dict]:
        if not baseline_result['is_correct']:
            return []
        
        G = self.graph_builder.build_local_graph(sample)
        
        core_entities = set()
        for fact in sample['supporting_facts']:
            entities = self.graph_builder.extract_entities(fact['sentence'])
            core_entities.update([e['text'] for e in entities])
        
        metrics = self.graph_builder.compute_graph_metrics(G, core_entities)
        graph_stats = self.graph_builder.visualize_graph_stats(G, metrics)
        
        perturbation_results = []
        
        # 扩展策略列表，包含语义相似度策略
        extended_strategies = PERTURBATION_STRATEGIES + [
            {'name': 'high_semantic_similarity', 'description': '选择语义相似度高的节点'},
            {'name': 'medium_semantic_similarity', 'description': '选择语义相似度中等的节点'},
            {'name': 'low_semantic_similarity', 'description': '选择语义相似度低的节点'}
        ]
        
        for strategy in extended_strategies:
            if 'semantic_similarity' in strategy['name']:
                # 使用语义相似度选择节点
                similarity_type = strategy['name'].split('_')[0]  # high, medium, or low
                influencer_data = self.select_semantic_influencers(G, sample, similarity_type)
                
                for influencer_node, node_metrics, influencer_context in influencer_data[:3]:
                    perturbed_prompt = self.generate_perturbation_prompt(
                        sample['question'], influencer_node, influencer_context
                    )
                    
                    # 评估干扰知识的初始置信度
                    influencer_initial_confidence = self.evaluate_influencer_confidence(influencer_context)
                    
                    perturbed_answer, perturbed_confidence = self.model.get_answer_and_confidence(
                        perturbed_prompt, 
                        max_new_tokens=EXPERIMENT_CONFIG['max_new_tokens'],
                        temperature=EXPERIMENT_CONFIG.get('temperature', 0.1),
                        do_sample=EXPERIMENT_CONFIG.get('do_sample', False)
                    )
                    
                    is_still_correct = HotpotQAProcessor.evaluate_answer(
                        perturbed_answer, sample['gold_answer']
                    )
                    
                    result = {
                        'sample_id': sample['id'],
                        'question': sample['question'],
                        'baseline_answer': baseline_result['predicted_answer'],
                        'baseline_confidence': baseline_result['confidence'],
                        'baseline_correct': baseline_result['is_correct'],
                        'perturbation_strategy': strategy['name'],
                        'influencer_node': influencer_node,
                        'influencer_context': influencer_context,
                        'influencer_degree': node_metrics['degree'],
                        'influencer_distance': node_metrics['distance_to_core'],
                        'influencer_centrality': node_metrics['betweenness_centrality'],
                        'influencer_semantic_similarity': node_metrics.get('semantic_similarity', -1),
                        'influencer_initial_confidence': influencer_initial_confidence,
                        'perturbed_answer': perturbed_answer,
                        'perturbed_confidence': perturbed_confidence,
                        'perturbed_correct': is_still_correct,
                        'confidence_change': perturbed_confidence - baseline_result['confidence'],
                        'answer_changed': perturbed_answer != baseline_result['predicted_answer'],
                        'correctness_flipped': is_still_correct != baseline_result['is_correct'],
                        'graph_stats': graph_stats,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    perturbation_results.append(result)
            else:
                # 使用原有的选择策略
                influencer_nodes = self.graph_builder.select_influencer_nodes(
                    G, metrics, strategy['name']
                )
                
                for influencer_node, node_metrics in influencer_nodes[:3]:
                    influencer_context = self.find_influencer_context(
                        influencer_node, sample['all_context']
                    )
                    
                    perturbed_prompt = self.generate_perturbation_prompt(
                        sample['question'], influencer_node, influencer_context
                    )
                    
                    # 评估干扰知识的初始置信度
                    influencer_initial_confidence = self.evaluate_influencer_confidence(influencer_context)
                    
                    perturbed_answer, perturbed_confidence = self.model.get_answer_and_confidence(
                        perturbed_prompt, 
                        max_new_tokens=EXPERIMENT_CONFIG['max_new_tokens'],
                        temperature=EXPERIMENT_CONFIG.get('temperature', 0.1),
                        do_sample=EXPERIMENT_CONFIG.get('do_sample', False)
                    )
                    
                    is_still_correct = HotpotQAProcessor.evaluate_answer(
                        perturbed_answer, sample['gold_answer']
                    )
                    
                    result = {
                        'sample_id': sample['id'],
                        'question': sample['question'],
                        'baseline_answer': baseline_result['predicted_answer'],
                        'baseline_confidence': baseline_result['confidence'],
                        'baseline_correct': baseline_result['is_correct'],
                        'perturbation_strategy': strategy['name'],
                        'influencer_node': influencer_node,
                        'influencer_context': influencer_context,
                        'influencer_degree': node_metrics['degree'],
                        'influencer_distance': node_metrics['distance_to_core'],
                        'influencer_centrality': node_metrics['betweenness_centrality'],
                        'influencer_semantic_similarity': -1,  # 原策略没有语义相似度
                        'influencer_initial_confidence': influencer_initial_confidence,
                        'perturbed_answer': perturbed_answer,
                        'perturbed_confidence': perturbed_confidence,
                        'perturbed_correct': is_still_correct,
                        'confidence_change': perturbed_confidence - baseline_result['confidence'],
                        'answer_changed': perturbed_answer != baseline_result['predicted_answer'],
                        'correctness_flipped': is_still_correct != baseline_result['is_correct'],
                        'graph_stats': graph_stats,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    perturbation_results.append(result)
        
        return perturbation_results
    
    def run_full_experiment(self, samples: List[Dict], baseline_results: List[Dict]):
        logger.info("Starting perturbation experiments...")
        
        baseline_dict = {r['id']: r for r in baseline_results}
        
        for sample in tqdm(samples, desc="Running perturbations"):
            if sample['id'] not in baseline_dict:
                continue
                
            baseline = baseline_dict[sample['id']]
            
            if baseline['knowledge_type'] == 'bright':
                perturbation_results = self.run_perturbation_on_sample(sample, baseline)
                self.results.extend(perturbation_results)
        
        return self.results
    
    def save_results(self, output_path: Path):
        df = pd.DataFrame(self.results)
        
        df.to_csv(output_path / 'perturbation_results.csv', index=False)
        
        with open(output_path / 'perturbation_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        analysis_stats = self.analyze_results(df)
        
        with open(output_path / 'perturbation_analysis.json', 'w') as f:
            json.dump(analysis_stats, f, indent=2)
        
        logger.info("Perturbation experiment results saved.")
        return analysis_stats
    
    def analyze_results(self, df: pd.DataFrame) -> Dict:
        if len(df) == 0:
            return {"error": "No perturbation results to analyze"}
        
        analysis = {
            'total_perturbations': len(df),
            'total_samples_perturbed': df['sample_id'].nunique(),
            'overall_flip_rate': df['correctness_flipped'].mean(),
            'avg_confidence_change': df['confidence_change'].mean(),
        }
        
        by_strategy = df.groupby('perturbation_strategy').agg({
            'correctness_flipped': 'mean',
            'confidence_change': 'mean',
            'answer_changed': 'mean',
            'influencer_degree': 'mean',
            'influencer_distance': 'mean',
            'influencer_centrality': 'mean'
        }).to_dict()
        
        analysis['by_strategy'] = by_strategy
        
        distance_bins = [0, 1, 2, 3, float('inf')]
        distance_labels = ['0', '1', '2', '3+']
        df['distance_bin'] = pd.cut(df['influencer_distance'], bins=distance_bins, labels=distance_labels)
        
        by_distance = df.groupby('distance_bin').agg({
            'correctness_flipped': 'mean',
            'confidence_change': 'mean'
        }).to_dict()
        
        analysis['by_distance'] = by_distance
        
        degree_threshold = df['influencer_degree'].median()
        high_degree = df[df['influencer_degree'] > degree_threshold]
        low_degree = df[df['influencer_degree'] <= degree_threshold]
        
        analysis['high_vs_low_degree'] = {
            'high_degree_flip_rate': high_degree['correctness_flipped'].mean() if len(high_degree) > 0 else 0,
            'low_degree_flip_rate': low_degree['correctness_flipped'].mean() if len(low_degree) > 0 else 0,
            'high_degree_confidence_change': high_degree['confidence_change'].mean() if len(high_degree) > 0 else 0,
            'low_degree_confidence_change': low_degree['confidence_change'].mean() if len(low_degree) > 0 else 0
        }
        
        return analysis