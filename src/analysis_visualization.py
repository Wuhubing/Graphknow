import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import stats
import logging
from .utils import NumpyJSONEncoder

logger = logging.getLogger(__name__)

class RippleEffectAnalyzer:
    def __init__(self, baseline_path: Path, perturbation_path: Path):
        self.baseline_df = pd.read_csv(baseline_path / 'baseline_results.csv')
        try:
            self.perturbation_df = pd.read_csv(perturbation_path / 'perturbation_results.csv')
        except pd.errors.EmptyDataError:
            # Handle empty perturbation results
            self.perturbation_df = pd.DataFrame()
            logger.warning("No perturbation results found - creating empty dataframe")
        self.results = {}
        
    def test_hypothesis_1_breadth(self):
        """Test if higher degree nodes cause more knowledge disruption"""
        logger.info("Testing Hypothesis 1: Node degree vs disruption")
        
        if len(self.perturbation_df) == 0:
            return {
                'hypothesis': 'Higher degree nodes cause more disruption',
                'error': 'No perturbation data available',
                'significant': False,
                'conclusion': 'Insufficient data'
            }
        
        high_degree = self.perturbation_df[self.perturbation_df['influencer_degree'] > 
                                         self.perturbation_df['influencer_degree'].median()]
        low_degree = self.perturbation_df[self.perturbation_df['influencer_degree'] <= 
                                        self.perturbation_df['influencer_degree'].median()]
        
        t_stat, p_value = stats.ttest_ind(
            high_degree['correctness_flipped'].values,
            low_degree['correctness_flipped'].values
        )
        
        result = {
            'hypothesis': 'Higher degree nodes cause more disruption',
            'high_degree_flip_rate': high_degree['correctness_flipped'].mean(),
            'low_degree_flip_rate': low_degree['correctness_flipped'].mean(),
            'high_degree_confidence_change': high_degree['confidence_change'].mean(),
            'low_degree_confidence_change': low_degree['confidence_change'].mean(),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'conclusion': 'Supported' if p_value < 0.05 and t_stat > 0 else 'Not supported'
        }
        
        self.results['hypothesis_1'] = result
        return result
    
    def test_hypothesis_2_distance(self):
        """Test if closer nodes have stronger influence"""
        logger.info("Testing Hypothesis 2: Distance vs influence strength")
        
        if len(self.perturbation_df) == 0:
            return {
                'hypothesis': 'Closer nodes have stronger influence',
                'error': 'No perturbation data available',
                'significant': False,
                'conclusion': 'Insufficient data'
            }
        
        distance_analysis = []
        for distance in [1, 2, 3]:
            subset = self.perturbation_df[self.perturbation_df['influencer_distance'] == distance]
            if len(subset) > 0:
                distance_analysis.append({
                    'distance': distance,
                    'flip_rate': subset['correctness_flipped'].mean(),
                    'confidence_change': subset['confidence_change'].mean(),
                    'sample_size': len(subset)
                })
        
        df_dist = pd.DataFrame(distance_analysis)
        if len(df_dist) > 1:
            correlation_flip = stats.spearmanr(df_dist['distance'], df_dist['flip_rate'])
            correlation_conf = stats.spearmanr(df_dist['distance'], df_dist['confidence_change'])
        else:
            correlation_flip = (0, 1)
            correlation_conf = (0, 1)
        
        result = {
            'hypothesis': 'Closer nodes have stronger influence',
            'distance_analysis': distance_analysis,
            'correlation_distance_flip': correlation_flip[0],
            'p_value_flip': correlation_flip[1],
            'correlation_distance_confidence': correlation_conf[0],
            'p_value_confidence': correlation_conf[1],
            'significant': correlation_flip[1] < 0.05 or correlation_conf[1] < 0.05,
            'conclusion': 'Supported' if correlation_flip[0] < -0.3 and correlation_flip[1] < 0.05 else 'Not supported'
        }
        
        self.results['hypothesis_2'] = result
        return result
    
    def test_hypothesis_3_centrality(self):
        """Test if high centrality nodes are more influential"""
        logger.info("Testing Hypothesis 3: Centrality vs influence")
        
        if len(self.perturbation_df) == 0:
            return {
                'hypothesis': 'High centrality nodes are more influential',
                'error': 'No perturbation data available',
                'significant': False,
                'conclusion': 'Insufficient data'
            }
        
        high_centrality = self.perturbation_df[self.perturbation_df['influencer_centrality'] > 
                                             self.perturbation_df['influencer_centrality'].median()]
        low_centrality = self.perturbation_df[self.perturbation_df['influencer_centrality'] <= 
                                            self.perturbation_df['influencer_centrality'].median()]
        
        if len(high_centrality) > 0 and len(low_centrality) > 0:
            t_stat, p_value = stats.ttest_ind(
                high_centrality['correctness_flipped'].values,
                low_centrality['correctness_flipped'].values
            )
        else:
            t_stat, p_value = 0, 1
        
        result = {
            'hypothesis': 'High centrality nodes are more influential',
            'high_centrality_flip_rate': high_centrality['correctness_flipped'].mean() if len(high_centrality) > 0 else 0,
            'low_centrality_flip_rate': low_centrality['correctness_flipped'].mean() if len(low_centrality) > 0 else 0,
            'high_centrality_confidence_change': high_centrality['confidence_change'].mean() if len(high_centrality) > 0 else 0,
            'low_centrality_confidence_change': low_centrality['confidence_change'].mean() if len(low_centrality) > 0 else 0,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'conclusion': 'Supported' if p_value < 0.05 and t_stat > 0 else 'Not supported'
        }
        
        self.results['hypothesis_3'] = result
        return result
    
    def analyze_ripple_patterns(self):
        """Analyze patterns in the ripple effect"""
        logger.info("Analyzing ripple effect patterns")
        
        if len(self.perturbation_df) == 0:
            return {
                'error': 'No perturbation data available',
                'total_samples_analyzed': 0,
                'vulnerable_samples': 0,
                'robust_samples': 0
            }
        
        sample_analysis = self.perturbation_df.groupby('sample_id').agg({
            'correctness_flipped': ['sum', 'mean'],
            'confidence_change': 'mean',
            'influencer_degree': 'mean',
            'influencer_distance': 'mean'
        }).reset_index()
        
        vulnerable_samples = sample_analysis[sample_analysis[('correctness_flipped', 'sum')] > 2]
        robust_samples = sample_analysis[sample_analysis[('correctness_flipped', 'sum')] == 0]
        
        pattern_analysis = {
            'total_samples_analyzed': len(sample_analysis),
            'vulnerable_samples': len(vulnerable_samples),
            'robust_samples': len(robust_samples),
            'avg_flips_per_sample': sample_analysis[('correctness_flipped', 'sum')].mean(),
            'vulnerability_patterns': {
                'avg_degree_vulnerable': vulnerable_samples[('influencer_degree', 'mean')].mean() if len(vulnerable_samples) > 0 else 0,
                'avg_degree_robust': robust_samples[('influencer_degree', 'mean')].mean() if len(robust_samples) > 0 else 0,
                'avg_distance_vulnerable': vulnerable_samples[('influencer_distance', 'mean')].mean() if len(vulnerable_samples) > 0 else 0,
                'avg_distance_robust': robust_samples[('influencer_distance', 'mean')].mean() if len(robust_samples) > 0 else 0
            }
        }
        
        self.results['ripple_patterns'] = pattern_analysis
        return pattern_analysis
    
    def create_visualizations(self, output_dir: Path):
        """Create comprehensive visualizations"""
        logger.info("Creating visualizations")
        
        if len(self.perturbation_df) == 0:
            logger.warning("No perturbation data available for visualization")
            # Create a simple plot showing no data
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.text(0.5, 0.5, 'No Perturbation Data Available\nAll baseline answers were incorrect', 
                   ha='center', va='center', fontsize=16)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            plt.savefig(output_dir / 'ripple_effect_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Flip rate by strategy
        strategy_data = self.perturbation_df.groupby('perturbation_strategy')['correctness_flipped'].mean()
        axes[0, 0].bar(strategy_data.index, strategy_data.values)
        axes[0, 0].set_title('Knowledge Disruption by Strategy')
        axes[0, 0].set_ylabel('Flip Rate')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Confidence change distribution
        axes[0, 1].hist(self.perturbation_df['confidence_change'], bins=30, alpha=0.7)
        axes[0, 1].axvline(0, color='red', linestyle='--')
        axes[0, 1].set_title('Distribution of Confidence Changes')
        axes[0, 1].set_xlabel('Confidence Change')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Distance vs flip rate
        distance_flip = self.perturbation_df.groupby('influencer_distance')['correctness_flipped'].mean()
        axes[0, 2].plot(distance_flip.index, distance_flip.values, marker='o')
        axes[0, 2].set_title('Distance vs Knowledge Disruption')
        axes[0, 2].set_xlabel('Distance from Core')
        axes[0, 2].set_ylabel('Flip Rate')
        
        # 4. Degree vs confidence change scatter
        axes[1, 0].scatter(self.perturbation_df['influencer_degree'], 
                          self.perturbation_df['confidence_change'], alpha=0.5)
        axes[1, 0].set_title('Node Degree vs Confidence Change')
        axes[1, 0].set_xlabel('Node Degree')
        axes[1, 0].set_ylabel('Confidence Change')
        
        # 5. Centrality vs flip rate
        centrality_bins = pd.qcut(self.perturbation_df['influencer_centrality'], q=5, duplicates='drop')
        centrality_flip = self.perturbation_df.groupby(centrality_bins)['correctness_flipped'].mean()
        axes[1, 1].bar(range(len(centrality_flip)), centrality_flip.values)
        axes[1, 1].set_title('Centrality Quintiles vs Disruption')
        axes[1, 1].set_xlabel('Centrality Quintile')
        axes[1, 1].set_ylabel('Flip Rate')
        axes[1, 1].set_xticks(range(len(centrality_flip)))
        # Dynamic labels based on actual number of bins
        axes[1, 1].set_xticklabels([f'Q{i+1}' for i in range(len(centrality_flip))])
        
        # 6. Heatmap of strategy effectiveness
        strategy_metrics = self.perturbation_df.pivot_table(
            values=['correctness_flipped', 'confidence_change'],
            index='perturbation_strategy',
            aggfunc='mean'
        )
        sns.heatmap(strategy_metrics, annot=True, fmt='.3f', cmap='coolwarm', ax=axes[1, 2])
        axes[1, 2].set_title('Strategy Effectiveness Heatmap')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'ripple_effect_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}")
    
    def generate_report(self, output_dir: Path):
        """Generate comprehensive analysis report"""
        self.test_hypothesis_1_breadth()
        self.test_hypothesis_2_distance()
        self.test_hypothesis_3_centrality()
        self.analyze_ripple_patterns()
        
        with open(output_dir / 'hypothesis_test_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, cls=NumpyJSONEncoder)
        
        self.create_visualizations(output_dir)
        
        # Generate summary report
        summary = {
            'experiment_summary': {
                'total_baseline_samples': len(self.baseline_df),
                'bright_knowledge_samples': len(self.baseline_df[self.baseline_df['knowledge_type'] == 'bright']),
                'total_perturbations': len(self.perturbation_df),
                'overall_disruption_rate': self.perturbation_df['correctness_flipped'].mean(),
                'avg_confidence_change': self.perturbation_df['confidence_change'].mean()
            },
            'key_findings': {
                'hypothesis_1_breadth': self.results['hypothesis_1']['conclusion'],
                'hypothesis_2_distance': self.results['hypothesis_2']['conclusion'],
                'hypothesis_3_centrality': self.results['hypothesis_3']['conclusion'],
                'most_vulnerable_samples': self.results['ripple_patterns']['vulnerable_samples'],
                'most_robust_samples': self.results['ripple_patterns']['robust_samples']
            }
        }
        
        with open(output_dir / 'experiment_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, cls=NumpyJSONEncoder)
        
        logger.info("Analysis complete. Results saved.")
        return summary