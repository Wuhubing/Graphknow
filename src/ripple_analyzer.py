import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import stats
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class RippleEffectAnalyzer:
    """分析知识涟漪效应的实验结果"""
    
    def __init__(self, baseline_results: List[Dict], perturbation_results: List[Dict]):
        """初始化分析器
        
        Args:
            baseline_results: 基准评估结果列表
            perturbation_results: 扰动实验结果列表
        """
        self.baseline_df = pd.DataFrame(baseline_results)
        self.perturbation_df = pd.DataFrame(perturbation_results) if perturbation_results else pd.DataFrame()
        self.results = {}
        
    def summarize_experiment(self) -> Dict[str, Any]:
        """生成实验总结"""
        summary = {
            'baseline_statistics': {
                'total_samples': len(self.baseline_df),
                'accuracy': self.baseline_df['is_correct'].mean() if len(self.baseline_df) > 0 else 0,
                'bright_knowledge_count': (self.baseline_df['knowledge_type'] == 'bright').sum() if len(self.baseline_df) > 0 else 0,
                'dark_knowledge_count': (self.baseline_df['knowledge_type'] == 'dark').sum() if len(self.baseline_df) > 0 else 0,
            }
        }
        
        if len(self.perturbation_df) > 0:
            summary['overall_statistics'] = {
                'total_perturbations': len(self.perturbation_df),
                'unique_samples': self.perturbation_df['sample_id'].nunique(),
                'flip_rate': self.perturbation_df['correctness_flipped'].mean(),
                'avg_confidence_change': self.perturbation_df['confidence_change'].mean(),
                'confidence_increase_rate': (self.perturbation_df['confidence_change'] > 0).mean()
            }
            
            # 按策略分组统计
            by_strategy = {}
            for strategy in self.perturbation_df['perturbation_strategy'].unique():
                strategy_df = self.perturbation_df[self.perturbation_df['perturbation_strategy'] == strategy]
                by_strategy[strategy] = {
                    'count': len(strategy_df),
                    'flip_rate': strategy_df['correctness_flipped'].mean(),
                    'avg_confidence_change': strategy_df['confidence_change'].mean(),
                    'confidence_increase_rate': (strategy_df['confidence_change'] > 0).mean(),
                    'avg_influencer_degree': strategy_df['influencer_degree'].mean(),
                    'avg_influencer_distance': strategy_df['influencer_distance'].mean(),
                }
                
                # 如果有语义相似度信息
                if 'influencer_semantic_similarity' in strategy_df.columns:
                    valid_similarities = strategy_df[strategy_df['influencer_semantic_similarity'] >= 0]['influencer_semantic_similarity']
                    if len(valid_similarities) > 0:
                        by_strategy[strategy]['avg_semantic_similarity'] = valid_similarities.mean()
            
            summary['by_strategy'] = by_strategy
        else:
            summary['overall_statistics'] = {
                'total_perturbations': 0,
                'message': 'No perturbation results available'
            }
            summary['by_strategy'] = {}
        
        return summary
    
    def test_hypotheses(self) -> Dict[str, Any]:
        """测试实验假设"""
        results = {}
        
        if len(self.perturbation_df) == 0:
            return {'message': 'No perturbation data available for hypothesis testing'}
        
        # 测试1: 节点度数与扰动效果的关系
        if 'influencer_degree' in self.perturbation_df.columns:
            corr, p_value = stats.pearsonr(
                self.perturbation_df['influencer_degree'],
                self.perturbation_df['correctness_flipped'].astype(int)
            )
            results['degree_vs_flip'] = {
                'correlation': corr,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        # 测试2: 距离与扰动效果的关系
        if 'influencer_distance' in self.perturbation_df.columns:
            corr, p_value = stats.pearsonr(
                self.perturbation_df['influencer_distance'],
                self.perturbation_df['confidence_change']
            )
            results['distance_vs_confidence'] = {
                'correlation': corr,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        # 测试3: 语义相似度与扰动效果的关系（如果有数据）
        if 'influencer_semantic_similarity' in self.perturbation_df.columns:
            valid_data = self.perturbation_df[self.perturbation_df['influencer_semantic_similarity'] >= 0]
            if len(valid_data) > 0:
                corr, p_value = stats.pearsonr(
                    valid_data['influencer_semantic_similarity'],
                    valid_data['correctness_flipped'].astype(int)
                )
                results['semantic_similarity_vs_flip'] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        return results
    
    def plot_ripple_effect_analysis(self, save_path: str = None):
        """生成涟漪效应分析可视化"""
        if len(self.perturbation_df) == 0:
            logger.warning("No perturbation data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 策略效果对比
        ax = axes[0, 0]
        strategy_flip_rates = self.perturbation_df.groupby('perturbation_strategy')['correctness_flipped'].mean()
        strategy_flip_rates.plot(kind='bar', ax=ax)
        ax.set_title('Flip Rate by Strategy')
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Flip Rate')
        ax.tick_params(axis='x', rotation=45)
        
        # 2. 距离与置信度变化
        ax = axes[0, 1]
        distance_groups = self.perturbation_df.groupby('influencer_distance')['confidence_change'].mean()
        distance_groups.plot(kind='bar', ax=ax)
        ax.set_title('Confidence Change by Distance')
        ax.set_xlabel('Distance to Core')
        ax.set_ylabel('Average Confidence Change')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 3. 度数分布与翻转率
        ax = axes[1, 0]
        degree_bins = pd.qcut(self.perturbation_df['influencer_degree'], q=5, duplicates='drop')
        flip_by_degree = self.perturbation_df.groupby(degree_bins)['correctness_flipped'].mean()
        flip_by_degree.plot(kind='bar', ax=ax)
        ax.set_title('Flip Rate by Node Degree Quintiles')
        ax.set_xlabel('Node Degree Quintile')
        ax.set_ylabel('Flip Rate')
        ax.tick_params(axis='x', rotation=45)
        
        # 4. 语义相似度分析（如果有数据）
        ax = axes[1, 1]
        if 'influencer_semantic_similarity' in self.perturbation_df.columns:
            valid_data = self.perturbation_df[self.perturbation_df['influencer_semantic_similarity'] >= 0]
            if len(valid_data) > 0:
                ax.scatter(valid_data['influencer_semantic_similarity'], 
                          valid_data['correctness_flipped'].astype(int),
                          alpha=0.5)
                ax.set_title('Semantic Similarity vs Correctness Flip')
                ax.set_xlabel('Semantic Similarity')
                ax.set_ylabel('Correctness Flipped (0/1)')
            else:
                ax.text(0.5, 0.5, 'No semantic similarity data', 
                       ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No semantic similarity data', 
                   ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        return fig