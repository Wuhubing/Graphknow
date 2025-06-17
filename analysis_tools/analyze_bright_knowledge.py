import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_bright_knowledge_in_depth():
    """深入分析100个Bright Knowledge样本"""
    
    # 加载数据
    experiment_dir = Path("/root/graph/knowledge_ripple_effect/results/large_scale_experiment_20250617_093716")
    with open(experiment_dir / 'baseline/baseline_results.json', 'r') as f:
        baseline_results = json.load(f)
    
    df = pd.DataFrame(baseline_results)
    bright_df = df[df['knowledge_type'] == 'bright']
    
    print("="*100)
    print("深度分析：100个Bright Knowledge样本")
    print("="*100)
    
    # 1. 比较类问题的详细分析
    print("\n### 1. 比较类问题分析 ###")
    print("-"*80)
    
    comparison_keywords = ['more', 'less', 'bigger', 'smaller', 'older', 'younger', 
                          'higher', 'lower', 'greater', 'fewer', 'better', 'worse']
    
    comparison_samples = []
    for _, sample in bright_df.iterrows():
        question_lower = sample['question'].lower()
        if any(keyword in question_lower for keyword in comparison_keywords):
            comparison_samples.append(sample)
    
    print(f"找到 {len(comparison_samples)} 个比较类问题")
    
    # 展示所有比较类问题
    for i, sample in enumerate(comparison_samples):
        print(f"\n比较问题 {i+1}:")
        print(f"问题: {sample['question']}")
        print(f"答案: {sample['predicted_answer'][:100]}...")
        print(f"置信度: {sample['confidence']:.3f}")
        
        # 分析比较的对象
        question_words = sample['question'].split()
        comparison_word = None
        for word in comparison_keywords:
            if word in sample['question'].lower():
                comparison_word = word
                break
        print(f"比较类型: {comparison_word}")
    
    # 2. 高置信度样本分析
    print("\n\n### 2. 高置信度样本分析 (置信度 > -0.2) ###")
    print("-"*80)
    
    high_confidence = bright_df[bright_df['confidence'] > -0.2].sort_values('confidence', ascending=False)
    print(f"找到 {len(high_confidence)} 个高置信度样本")
    
    for idx, (_, sample) in enumerate(high_confidence.head(5).iterrows()):
        print(f"\n高置信样本 {idx+1}:")
        print(f"问题: {sample['question']}")
        print(f"答案: {sample['predicted_answer'][:100]}...")
        print(f"置信度: {sample['confidence']:.3f}")
        
        # 分析为什么置信度高
        if 'what' in sample['question'].lower():
            print("分析: 'What'类问题，通常有明确答案")
        if len(sample['question'].split()) < 15:
            print("分析: 问题较短，可能更直接")
    
    # 3. 问题长度与置信度的关系
    print("\n\n### 3. 问题复杂度分析 ###")
    print("-"*80)
    
    bright_df['question_length'] = bright_df['question'].str.split().str.len()
    
    # 按问题长度分组
    length_bins = [0, 10, 20, 30, 100]
    length_labels = ['Short (≤10)', 'Medium (11-20)', 'Long (21-30)', 'Very Long (>30)']
    bright_df['length_category'] = pd.cut(bright_df['question_length'], 
                                          bins=length_bins, labels=length_labels)
    
    length_stats = bright_df.groupby('length_category', observed=False).agg({
        'confidence': ['mean', 'std', 'count']
    })
    
    print("问题长度与置信度的关系:")
    print(length_stats)
    
    # 4. 特殊案例识别
    print("\n\n### 4. 特殊案例 ###")
    print("-"*80)
    
    # 找出置信度最低的bright knowledge
    lowest_confidence = bright_df.nsmallest(3, 'confidence')
    print("\n置信度最低但仍然正确的案例:")
    for _, sample in lowest_confidence.iterrows():
        print(f"\n问题: {sample['question'][:100]}...")
        print(f"置信度: {sample['confidence']:.3f}")
    
    # 5. 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 5.1 置信度分布
    ax = axes[0, 0]
    bright_df['confidence'].hist(bins=20, ax=ax, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(bright_df['confidence'].mean(), color='red', linestyle='--', 
               label=f'Mean: {bright_df["confidence"].mean():.3f}')
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Count')
    ax.set_title('Confidence Distribution of Bright Knowledge')
    ax.legend()
    
    # 5.2 问题长度分布
    ax = axes[0, 1]
    bright_df['question_length'].hist(bins=15, ax=ax, alpha=0.7, color='blue', edgecolor='black')
    ax.set_xlabel('Question Length (words)')
    ax.set_ylabel('Count')
    ax.set_title('Question Length Distribution')
    
    # 5.3 问题类型饼图
    ax = axes[1, 0]
    question_types = {
        'What': sum(1 for _, s in bright_df.iterrows() if 'what' in s['question'].lower()),
        'Which': sum(1 for _, s in bright_df.iterrows() if 'which' in s['question'].lower()),
        'Who': sum(1 for _, s in bright_df.iterrows() if 'who' in s['question'].lower()),
        'Where': sum(1 for _, s in bright_df.iterrows() if 'where' in s['question'].lower()),
        'When': sum(1 for _, s in bright_df.iterrows() if 'when' in s['question'].lower()),
        'Other': 0
    }
    question_types['Other'] = len(bright_df) - sum(list(question_types.values()))
    
    ax.pie(question_types.values(), labels=question_types.keys(), autopct='%1.1f%%')
    ax.set_title('Question Type Distribution')
    
    # 5.4 置信度 vs 问题长度散点图
    ax = axes[1, 1]
    ax.scatter(bright_df['question_length'], bright_df['confidence'], alpha=0.5)
    ax.set_xlabel('Question Length (words)')
    ax.set_ylabel('Confidence Score')
    ax.set_title('Confidence vs Question Length')
    
    # 添加趋势线
    z = np.polyfit(bright_df['question_length'], bright_df['confidence'], 1)
    p = np.poly1d(z)
    ax.plot(bright_df['question_length'].sort_values(), 
            p(bright_df['question_length'].sort_values()), 
            "r--", alpha=0.8, label=f'Trend: {z[0]:.4f}x + {z[1]:.3f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(experiment_dir / 'bright_knowledge_analysis.png', dpi=300)
    plt.close()
    
    print(f"\n\n可视化已保存到: {experiment_dir / 'bright_knowledge_analysis.png'}")
    
    # 6. 为扰动实验准备重点样本
    print("\n\n### 5. 推荐的重点扰动样本 ###")
    print("-"*80)
    
    # 选择最有价值的样本进行深度扰动实验
    priority_samples = {
        'comparison': comparison_samples[:3] if len(comparison_samples) >= 3 else comparison_samples,
        'high_confidence': high_confidence.head(3).to_dict('records'),
        'low_confidence': lowest_confidence.to_dict('records'),
        'typical': bright_df.sample(3).to_dict('records')
    }
    
    priority_ids = []
    for category, samples in priority_samples.items():
        print(f"\n{category.upper()} 类别:")
        for sample in samples:
            priority_ids.append(sample['id'])
            print(f"  - ID: {sample['id']}, 问题: {sample['question'][:60]}...")
    
    # 保存优先样本列表
    with open(experiment_dir / 'priority_samples_for_perturbation.json', 'w') as f:
        json.dump({
            'priority_sample_ids': priority_ids,
            'categories': {k: [s['id'] for s in v] for k, v in priority_samples.items()},
            'total_bright_knowledge': len(bright_df),
            'analysis_summary': {
                'comparison_questions': len(comparison_samples),
                'high_confidence_samples': len(high_confidence),
                'avg_question_length': bright_df['question_length'].mean(),
                'confidence_length_correlation': np.corrcoef(bright_df['question_length'], 
                                                            bright_df['confidence'])[0, 1]
            }
        }, f, indent=2)
    
    return bright_df, comparison_samples

if __name__ == "__main__":
    import numpy as np
    bright_df, comparison_samples = analyze_bright_knowledge_in_depth()