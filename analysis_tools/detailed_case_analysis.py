import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def load_and_analyze_cases(experiment_dir):
    """加载并分析具体案例"""
    
    # 加载扰动结果
    with open(experiment_dir / 'perturbation/perturbation_results.json', 'r') as f:
        perturbation_results = json.load(f)
    
    # 转换为DataFrame
    df = pd.DataFrame(perturbation_results)
    
    print("=== 实验数据概览 ===")
    print(f"总扰动次数: {len(df)}")
    print(f"涉及样本数: {df['sample_id'].nunique()}")
    print(f"成功翻转次数: {df['correctness_flipped'].sum()}")
    print(f"平均置信度变化: {df['confidence_change'].mean():.4f}")
    
    # 1. 找出所有成功翻转的案例
    print("\n\n=== 成功翻转答案的案例 ===")
    flipped_cases = df[df['correctness_flipped'] == True]
    
    for idx, case in flipped_cases.iterrows():
        print(f"\n案例 {idx+1}:")
        print(f"策略: {case['perturbation_strategy']}")
        print(f"问题: {case['question']}")
        print(f"基准答案: {case['baseline_answer'][:100]}...")
        print(f"扰动后答案: {case['perturbed_answer'][:100]}...")
        print(f"干扰节点: {case['influencer_node']}")
        print(f"干扰内容: {case['influencer_context'][:200]}...")
        print(f"节点度数: {case['influencer_degree']}")
        print(f"距离: {case['influencer_distance']}")
        if 'influencer_semantic_similarity' in case and case['influencer_semantic_similarity'] > 0:
            print(f"语义相似度: {case['influencer_semantic_similarity']:.4f}")
        print(f"置信度变化: {case['confidence_change']:.4f}")
        print("-" * 80)
    
    # 2. 分析高语义相似度但增强置信度的案例
    print("\n\n=== 高语义相似度增强置信度的案例 ===")
    high_sim_cases = df[(df['perturbation_strategy'] == 'high_semantic_similarity') & 
                        (df['confidence_change'] > 0)]
    
    for idx, case in high_sim_cases.head(3).iterrows():
        print(f"\n案例 {idx+1}:")
        print(f"问题: {case['question']}")
        print(f"基准答案: {case['baseline_answer'][:100]}...")
        print(f"基准置信度: {case['baseline_confidence']:.4f}")
        print(f"扰动后置信度: {case['perturbed_confidence']:.4f}")
        print(f"置信度提升: {case['confidence_change']:.4f}")
        print(f"干扰内容: {case['influencer_context'][:200]}...")
        print(f"语义相似度: {case['influencer_semantic_similarity']:.4f}")
        print("-" * 80)
    
    # 3. 分析远距离但造成虚假自信的案例
    print("\n\n=== 远距离虚假自信案例 ===")
    far_confidence_cases = df[(df['influencer_distance'] >= 3) & 
                             (df['confidence_change'] > 0)]
    
    for idx, case in far_confidence_cases.head(3).iterrows():
        print(f"\n案例 {idx+1}:")
        print(f"问题: {case['question']}")
        print(f"干扰节点: {case['influencer_node']}")
        print(f"距离: {case['influencer_distance']}")
        print(f"干扰内容: {case['influencer_context'][:200]}...")
        print(f"置信度变化: {case['confidence_change']:.4f}")
        print("-" * 80)
    
    return df

def create_detailed_visualizations(df, save_dir):
    """创建详细的可视化图表"""
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 策略效果热力图
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    strategy_metrics = df.groupby('perturbation_strategy').agg({
        'correctness_flipped': 'mean',
        'confidence_change': 'mean',
        'answer_changed': 'mean'
    }).round(3)
    
    sns.heatmap(strategy_metrics.T, annot=True, cmap='RdBu_r', center=0, 
                fmt='.3f', cbar_kws={'label': 'Effect Magnitude'})
    plt.title('Strategy Effectiveness Heatmap')
    plt.tight_layout()
    plt.savefig(save_dir / 'strategy_heatmap.png', dpi=300)
    plt.close()
    
    # 2. 语义相似度分布图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 2.1 语义相似度与翻转率的关系
    ax = axes[0, 0]
    semantic_df = df[df['influencer_semantic_similarity'] > 0]
    if len(semantic_df) > 0:
        ax.scatter(semantic_df['influencer_semantic_similarity'], 
                  semantic_df['correctness_flipped'].astype(int),
                  alpha=0.6, s=50)
        ax.set_xlabel('Semantic Similarity')
        ax.set_ylabel('Correctness Flipped (0/1)')
        ax.set_title('Semantic Similarity vs Answer Flip')
        
        # 添加趋势线
        z = np.polyfit(semantic_df['influencer_semantic_similarity'], 
                       semantic_df['correctness_flipped'].astype(int), 1)
        p = np.poly1d(z)
        ax.plot(semantic_df['influencer_semantic_similarity'].sort_values(), 
                p(semantic_df['influencer_semantic_similarity'].sort_values()), 
                "r--", alpha=0.8)
    
    # 2.2 语义相似度与置信度变化
    ax = axes[0, 1]
    if len(semantic_df) > 0:
        ax.scatter(semantic_df['influencer_semantic_similarity'], 
                  semantic_df['confidence_change'],
                  alpha=0.6, s=50)
        ax.set_xlabel('Semantic Similarity')
        ax.set_ylabel('Confidence Change')
        ax.set_title('Semantic Similarity vs Confidence Change')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # 2.3 节点度数分布
    ax = axes[1, 0]
    degree_flip = df.groupby(pd.qcut(df['influencer_degree'], q=5, duplicates='drop')).agg({
        'correctness_flipped': ['mean', 'count']
    })
    degree_flip.columns = ['flip_rate', 'count']
    degree_flip['flip_rate'].plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title('Flip Rate by Node Degree Quintiles')
    ax.set_xlabel('Node Degree Quintile')
    ax.set_ylabel('Flip Rate')
    ax.tick_params(axis='x', rotation=45)
    
    # 在柱状图上添加样本数
    for i, (idx, row) in enumerate(degree_flip.iterrows()):
        ax.text(i, row['flip_rate'] + 0.01, f"n={row['count']}", 
                ha='center', va='bottom')
    
    # 2.4 距离与置信度变化箱线图
    ax = axes[1, 1]
    distance_data = []
    for dist in sorted(df['influencer_distance'].unique()):
        if dist >= 0:
            data = df[df['influencer_distance'] == dist]['confidence_change']
            if len(data) > 0:
                distance_data.append(data)
    
    if distance_data:
        ax.boxplot(distance_data, labels=[str(int(d)) for d in sorted(df['influencer_distance'].unique()) if d >= 0])
        ax.set_xlabel('Distance to Core')
        ax.set_ylabel('Confidence Change')
        ax.set_title('Confidence Change Distribution by Distance')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'detailed_analysis.png', dpi=300)
    plt.close()
    
    # 3. 策略对比雷达图
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    strategies = df['perturbation_strategy'].unique()
    metrics = ['flip_rate', 'avg_confidence_change', 'confidence_up_rate', 'avg_degree', 'avg_distance']
    
    strategy_radar_data = []
    for strategy in strategies:
        strategy_df = df[df['perturbation_strategy'] == strategy]
        data = [
            strategy_df['correctness_flipped'].mean(),
            strategy_df['confidence_change'].mean() + 0.5,  # 归一化到0-1
            (strategy_df['confidence_change'] > 0).mean(),
            strategy_df['influencer_degree'].mean() / df['influencer_degree'].max(),
            1 - (strategy_df['influencer_distance'].mean() / df['influencer_distance'].max()) if strategy_df['influencer_distance'].mean() > 0 else 0.5
        ]
        strategy_radar_data.append(data)
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    for i, (strategy, data) in enumerate(zip(strategies, strategy_radar_data)):
        data += data[:1]
        ax.plot(angles, data, 'o-', linewidth=2, label=strategy)
        ax.fill(angles, data, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title('Strategy Comparison Radar Chart', size=20, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'strategy_radar.png', dpi=300)
    plt.close()
    
    print(f"\n图表已保存到: {save_dir}")

if __name__ == "__main__":
    # 使用最新的实验结果
    experiment_dir = Path("/root/graph/knowledge_ripple_effect/results/semantic_experiment_20250617_090658")
    
    # 分析案例
    df = load_and_analyze_cases(experiment_dir)
    
    # 创建可视化
    analysis_dir = experiment_dir / 'analysis'
    create_detailed_visualizations(df, analysis_dir)