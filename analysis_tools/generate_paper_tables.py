import json
import pandas as pd
from pathlib import Path
import numpy as np

def generate_paper_tables(experiment_dir):
    """生成适合论文使用的数据表格"""
    
    # 加载数据
    with open(experiment_dir / 'analysis' / 'experiment_summary.json', 'r') as f:
        summary = json.load(f)
    
    with open(experiment_dir / 'perturbation/perturbation_results.json', 'r') as f:
        results = json.load(f)
    
    df = pd.DataFrame(results)
    
    # 表1：实验概览
    print("Table 1: Experiment Overview")
    print("=" * 60)
    overview_data = {
        'Metric': [
            'Total Samples',
            'Bright Knowledge Samples', 
            'Total Perturbations',
            'Unique Strategies',
            'Overall Flip Rate',
            'Avg Confidence Change'
        ],
        'Value': [
            20,
            summary['baseline_statistics']['bright_knowledge_count'],
            summary['overall_statistics']['total_perturbations'],
            7,
            f"{summary['overall_statistics']['flip_rate']:.1%}",
            f"{summary['overall_statistics']['avg_confidence_change']:.4f}"
        ]
    }
    overview_df = pd.DataFrame(overview_data)
    print(overview_df.to_string(index=False))
    
    # 表2：策略比较
    print("\n\nTable 2: Strategy Comparison")
    print("=" * 80)
    strategy_data = []
    
    for strategy, stats in summary['by_strategy'].items():
        row = {
            'Strategy': strategy.replace('_', ' ').title(),
            'Count': stats['count'],
            'Flip Rate': f"{stats['flip_rate']:.1%}",
            'Conf Change': f"{stats['avg_confidence_change']:.3f}",
            'Conf Up Rate': f"{stats['confidence_increase_rate']:.1%}",
            'Avg Degree': f"{stats['avg_influencer_degree']:.1f}",
            'Avg Distance': f"{stats['avg_influencer_distance']:.1f}"
        }
        if 'avg_semantic_similarity' in stats:
            row['Avg Similarity'] = f"{stats['avg_semantic_similarity']:.3f}"
        else:
            row['Avg Similarity'] = '-'
        strategy_data.append(row)
    
    strategy_df = pd.DataFrame(strategy_data)
    print(strategy_df.to_string(index=False))
    
    # 表3：语义相似度效应分析
    print("\n\nTable 3: Semantic Similarity Effects")
    print("=" * 60)
    
    semantic_df = df[df['influencer_semantic_similarity'] > 0]
    if len(semantic_df) > 0:
        bins = [0, 0.3, 0.5, 0.7, 1.0]
        labels = ['Low (0-0.3)', 'Med-Low (0.3-0.5)', 'Med-High (0.5-0.7)', 'High (0.7-1.0)']
        semantic_df['sim_category'] = pd.cut(semantic_df['influencer_semantic_similarity'], 
                                            bins=bins, labels=labels)
        
        sim_analysis = semantic_df.groupby('sim_category', observed=False).agg({
            'correctness_flipped': ['count', 'sum', 'mean'],
            'confidence_change': 'mean',
            'influencer_semantic_similarity': 'mean'
        }).round(3)
        
        sim_table = pd.DataFrame({
            'Similarity Range': labels,
            'Samples': sim_analysis[('correctness_flipped', 'count')].values,
            'Flips': sim_analysis[('correctness_flipped', 'sum')].values.astype(int),
            'Flip Rate': [f"{x:.1%}" for x in sim_analysis[('correctness_flipped', 'mean')].values],
            'Avg Conf Change': [f"{x:.3f}" for x in sim_analysis[('confidence_change', 'mean')].values],
            'Avg Similarity': [f"{x:.3f}" for x in sim_analysis[('influencer_semantic_similarity', 'mean')].values]
        })
        
        print(sim_table.to_string(index=False))
    
    # 表4：成功翻转案例分析
    print("\n\nTable 4: Successful Flip Cases")
    print("=" * 100)
    
    flip_cases = df[df['correctness_flipped'] == True]
    if len(flip_cases) > 0:
        flip_summary = []
        for _, case in flip_cases.iterrows():
            row = {
                'Question': case['question'][:40] + '...',
                'Strategy': case['perturbation_strategy'].replace('_', ' ').title(),
                'Node Degree': case['influencer_degree'],
                'Distance': case['influencer_distance'],
                'Conf Change': f"{case['confidence_change']:.3f}"
            }
            if case.get('influencer_semantic_similarity', -1) > 0:
                row['Similarity'] = f"{case['influencer_semantic_similarity']:.3f}"
            else:
                row['Similarity'] = '-'
            flip_summary.append(row)
        
        flip_df = pd.DataFrame(flip_summary)
        print(flip_df.to_string(index=False))
    
    # 保存LaTeX格式的表格
    print("\n\nGenerating LaTeX tables...")
    
    with open(experiment_dir / 'analysis' / 'paper_tables.tex', 'w') as f:
        f.write("% Table 1: Experiment Overview\n")
        f.write(overview_df.to_latex(index=False))
        f.write("\n\n% Table 2: Strategy Comparison\n")
        f.write(strategy_df.to_latex(index=False))
        if len(semantic_df) > 0:
            f.write("\n\n% Table 3: Semantic Similarity Effects\n")
            f.write(sim_table.to_latex(index=False))
        if len(flip_cases) > 0:
            f.write("\n\n% Table 4: Successful Flip Cases\n")
            f.write(flip_df.to_latex(index=False))
    
    print(f"\nLaTeX tables saved to: {experiment_dir / 'analysis' / 'paper_tables.tex'}")
    
    # 生成CSV文件供进一步分析
    strategy_df.to_csv(experiment_dir / 'analysis' / 'strategy_comparison.csv', index=False)
    if len(semantic_df) > 0:
        sim_table.to_csv(experiment_dir / 'analysis' / 'semantic_effects.csv', index=False)
    if len(flip_cases) > 0:
        flip_df.to_csv(experiment_dir / 'analysis' / 'successful_flips.csv', index=False)
    
    print("\nCSV files saved for further analysis")

if __name__ == "__main__":
    experiment_dir = Path("/root/graph/knowledge_ripple_effect/results/semantic_experiment_20250617_090658")
    generate_paper_tables(experiment_dir)