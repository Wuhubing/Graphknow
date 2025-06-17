import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from collections import defaultdict

def analyze_semantic_patterns(experiment_dir):
    """深入分析语义相似度模式"""
    
    # 加载数据
    with open(experiment_dir / 'perturbation/perturbation_results.json', 'r') as f:
        results = json.load(f)
    
    df = pd.DataFrame(results)
    
    # 分析每个问题的详细情况
    print("=== 按问题分组的详细分析 ===\n")
    
    question_groups = df.groupby('question')
    
    for question, group in question_groups:
        print(f"\n问题: {question}")
        print(f"基准答案: {group.iloc[0]['baseline_answer'][:50]}...")
        print(f"样本数: {len(group)}")
        
        # 分析不同策略的效果
        strategy_effects = group.groupby('perturbation_strategy').agg({
            'correctness_flipped': 'sum',
            'confidence_change': 'mean',
            'influencer_semantic_similarity': lambda x: x[x > 0].mean() if any(x > 0) else -1
        })
        
        print("\n策略效果:")
        for strategy, row in strategy_effects.iterrows():
            if row['influencer_semantic_similarity'] > 0:
                print(f"  {strategy}: 翻转{row['correctness_flipped']}次, "
                      f"置信度变化{row['confidence_change']:.3f}, "
                      f"语义相似度{row['influencer_semantic_similarity']:.3f}")
            else:
                print(f"  {strategy}: 翻转{row['correctness_flipped']}次, "
                      f"置信度变化{row['confidence_change']:.3f}")
        
        # 找出最有效的干扰
        most_effective = group[group['correctness_flipped'] == True]
        if len(most_effective) > 0:
            print("\n成功干扰的案例:")
            for _, case in most_effective.iterrows():
                print(f"  - 策略: {case['perturbation_strategy']}")
                print(f"    干扰: {case['influencer_context'][:100]}...")
                print(f"    效果: 从'{case['baseline_answer'][:30]}...' 变为 '{case['perturbed_answer'][:30]}...'")
        
        print("-" * 100)
    
    # 创建语义相似度的详细分布图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 语义相似度的分布
    ax = axes[0, 0]
    semantic_data = df[df['influencer_semantic_similarity'] > 0]
    ax.hist(semantic_data['influencer_semantic_similarity'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_xlabel('Semantic Similarity')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Semantic Similarity Scores')
    ax.axvline(x=0.7, color='red', linestyle='--', label='High threshold')
    ax.axvline(x=0.4, color='orange', linestyle='--', label='Medium threshold')
    ax.legend()
    
    # 2. 按语义相似度分组的效果
    ax = axes[0, 1]
    if len(semantic_data) > 0:
        bins = [0, 0.3, 0.5, 0.7, 1.0]
        labels = ['Low\n(0-0.3)', 'Med-Low\n(0.3-0.5)', 'Med-High\n(0.5-0.7)', 'High\n(0.7-1.0)']
        semantic_data['sim_bin'] = pd.cut(semantic_data['influencer_semantic_similarity'], bins=bins, labels=labels)
        
        bin_effects = semantic_data.groupby('sim_bin').agg({
            'correctness_flipped': 'mean',
            'confidence_change': 'mean'
        })
        
        x = np.arange(len(labels))
        width = 0.35
        
        ax.bar(x - width/2, bin_effects['correctness_flipped'], width, label='Flip Rate', color='coral')
        ax.bar(x + width/2, bin_effects['confidence_change'] + 0.5, width, label='Conf Change (+0.5)', color='lightgreen')
        
        ax.set_xlabel('Semantic Similarity Range')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Effect')
        ax.set_title('Effects by Semantic Similarity Range')
        ax.legend()
        ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.3)
    
    # 3. 节点度数 vs 语义相似度
    ax = axes[0, 2]
    if len(semantic_data) > 0:
        scatter = ax.scatter(semantic_data['influencer_degree'], 
                           semantic_data['influencer_semantic_similarity'],
                           c=semantic_data['correctness_flipped'].astype(int),
                           cmap='RdYlBu', alpha=0.6, s=50)
        ax.set_xlabel('Node Degree')
        ax.set_ylabel('Semantic Similarity')
        ax.set_title('Node Degree vs Semantic Similarity')
        plt.colorbar(scatter, ax=ax, label='Flipped (0/1)')
    
    # 4. 置信度变化的分布（按策略）
    ax = axes[1, 0]
    strategies = df['perturbation_strategy'].unique()
    confidence_changes = [df[df['perturbation_strategy'] == s]['confidence_change'] for s in strategies]
    
    bp = ax.boxplot(confidence_changes, patch_artist=True)
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.set_ylabel('Confidence Change')
    ax.set_title('Confidence Change Distribution by Strategy')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # 为箱线图着色
    colors = plt.cm.Set3(np.linspace(0, 1, len(strategies)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # 5. 初始置信度 vs 扰动效果
    ax = axes[1, 1]
    if 'influencer_initial_confidence' in df.columns:
        valid_data = df[df['influencer_initial_confidence'].notna()]
        if len(valid_data) > 0:
            scatter = ax.scatter(valid_data['influencer_initial_confidence'],
                               valid_data['confidence_change'],
                               c=valid_data['correctness_flipped'].astype(int),
                               cmap='RdYlBu', alpha=0.6, s=50)
            ax.set_xlabel('Influencer Initial Confidence')
            ax.set_ylabel('Confidence Change')
            ax.set_title('Influencer Confidence vs Effect')
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            plt.colorbar(scatter, ax=ax, label='Flipped (0/1)')
    
    # 6. 样本级别的成功率
    ax = axes[1, 2]
    sample_success = df.groupby('sample_id').agg({
        'correctness_flipped': ['sum', 'count']
    })
    sample_success.columns = ['flips', 'attempts']
    sample_success['flip_rate'] = sample_success['flips'] / sample_success['attempts']
    
    ax.bar(range(len(sample_success)), sample_success['flip_rate'], color='skyblue', edgecolor='black')
    ax.set_xlabel('Sample ID (indexed)')
    ax.set_ylabel('Flip Rate')
    ax.set_title('Flip Rate by Sample')
    
    # 添加样本ID标签
    for i, (idx, row) in enumerate(sample_success.iterrows()):
        ax.text(i, row['flip_rate'] + 0.02, f"{row['flips']}/{row['attempts']}", 
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(experiment_dir / 'analysis' / 'semantic_detailed_analysis.png', dpi=300)
    plt.close()
    
    # 分析语义相似度的具体内容
    print("\n\n=== 语义相似度内容分析 ===")
    
    # 高相似度案例
    high_sim = semantic_data[semantic_data['influencer_semantic_similarity'] > 0.8]
    if len(high_sim) > 0:
        print("\n高语义相似度案例 (>0.8):")
        for _, case in high_sim.head(3).iterrows():
            print(f"\n相似度: {case['influencer_semantic_similarity']:.4f}")
            print(f"问题: {case['question']}")
            print(f"干扰内容: {case['influencer_context'][:150]}...")
            print(f"效果: 翻转={case['correctness_flipped']}, 置信度变化={case['confidence_change']:.4f}")
    
    # 中等相似度但成功翻转的案例
    medium_success = semantic_data[(semantic_data['influencer_semantic_similarity'] > 0.3) & 
                                  (semantic_data['influencer_semantic_similarity'] < 0.7) &
                                  (semantic_data['correctness_flipped'] == True)]
    if len(medium_success) > 0:
        print("\n\n中等语义相似度但成功翻转的案例:")
        for _, case in medium_success.iterrows():
            print(f"\n相似度: {case['influencer_semantic_similarity']:.4f}")
            print(f"问题: {case['question']}")
            print(f"原答案: {case['baseline_answer'][:50]}...")
            print(f"新答案: {case['perturbed_answer'][:50]}...")
            print(f"干扰内容: {case['influencer_context'][:150]}...")

if __name__ == "__main__":
    experiment_dir = Path("/root/graph/knowledge_ripple_effect/results/semantic_experiment_20250617_090658")
    analyze_semantic_patterns(experiment_dir)