import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import textwrap

def create_case_comparison_figure(experiment_dir):
    """创建案例对比可视化图"""
    
    # 加载数据
    with open(experiment_dir / 'perturbation/perturbation_results.json', 'r') as f:
        results = json.load(f)
    
    df = pd.DataFrame(results)
    
    # 选择有代表性的案例
    cases_to_show = []
    
    # 1. 高语义相似度增强置信度的案例
    high_sim_positive = df[(df['perturbation_strategy'] == 'high_semantic_similarity') & 
                          (df['confidence_change'] > 0)].iloc[0] if len(df[(df['perturbation_strategy'] == 'high_semantic_similarity') & (df['confidence_change'] > 0)]) > 0 else None
    if high_sim_positive is not None:
        cases_to_show.append(('High Similarity\n(Reinforcement)', high_sim_positive))
    
    # 2. 中等语义相似度翻转答案的案例
    medium_sim_flip = df[(df['perturbation_strategy'] == 'medium_semantic_similarity') & 
                        (df['correctness_flipped'] == True)].iloc[0] if len(df[(df['perturbation_strategy'] == 'medium_semantic_similarity') & (df['correctness_flipped'] == True)]) > 0 else None
    if medium_sim_flip is not None:
        cases_to_show.append(('Medium Similarity\n(Confusion)', medium_sim_flip))
    
    # 3. 高节点度翻转答案的案例
    high_degree_flip = df[(df['perturbation_strategy'] == 'high_degree') & 
                         (df['correctness_flipped'] == True)].iloc[0] if len(df[(df['perturbation_strategy'] == 'high_degree') & (df['correctness_flipped'] == True)]) > 0 else None
    if high_degree_flip is not None:
        cases_to_show.append(('High Degree\n(Broadcast)', high_degree_flip))
    
    # 4. 远距离虚假自信的案例
    far_distance_confident = df[(df['influencer_distance'] >= 3) & 
                               (df['confidence_change'] > 0.2)].iloc[0] if len(df[(df['influencer_distance'] >= 3) & (df['confidence_change'] > 0.2)]) > 0 else None
    if far_distance_confident is not None:
        cases_to_show.append(('Far Distance\n(False Confidence)', far_distance_confident))
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    for idx, (title, case) in enumerate(cases_to_show):
        if idx >= 4:
            break
            
        ax = axes[idx]
        
        # 设置子图标题
        ax.text(0.5, 0.95, title, transform=ax.transAxes, 
                fontsize=16, fontweight='bold', ha='center')
        
        # 问题
        question_wrapped = textwrap.fill(case['question'], width=60)
        ax.text(0.05, 0.85, f"Question:\n{question_wrapped}", 
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.3))
        
        # 基准答案
        baseline_wrapped = textwrap.fill(case['baseline_answer'][:100] + "...", width=60)
        ax.text(0.05, 0.65, f"Original Answer:\n{baseline_wrapped}", 
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.3))
        
        # 干扰信息
        influencer_wrapped = textwrap.fill(case['influencer_context'][:150] + "...", width=60)
        ax.text(0.05, 0.45, f"Influencer ({case['influencer_node']}):\n{influencer_wrapped}", 
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))
        
        # 扰动后答案
        perturbed_wrapped = textwrap.fill(case['perturbed_answer'][:100] + "...", width=60)
        color = "lightcoral" if case['correctness_flipped'] else "lightgreen"
        ax.text(0.05, 0.25, f"Perturbed Answer:\n{perturbed_wrapped}", 
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
        
        # 统计信息
        stats_text = f"Metrics:\n"
        stats_text += f"• Node Degree: {case['influencer_degree']}\n"
        stats_text += f"• Distance: {case['influencer_distance']}\n"
        if 'influencer_semantic_similarity' in case and case['influencer_semantic_similarity'] > 0:
            stats_text += f"• Semantic Similarity: {case['influencer_semantic_similarity']:.3f}\n"
        stats_text += f"• Confidence Change: {case['confidence_change']:.3f}\n"
        stats_text += f"• Answer Flipped: {'Yes' if case['correctness_flipped'] else 'No'}"
        
        ax.text(0.05, 0.05, stats_text, transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.3))
        
        # 移除坐标轴
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(experiment_dir / 'analysis' / 'case_comparisons.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建第二个图：展示语义相似度的影响机制
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # 绘制概念图
    ax.text(0.5, 0.9, 'Knowledge Ripple Effect: Semantic Similarity Mechanism', 
            transform=ax.transAxes, fontsize=18, fontweight='bold', ha='center')
    
    # 核心知识
    core = plt.Circle((0.5, 0.5), 0.08, color='blue', alpha=0.8)
    ax.add_patch(core)
    ax.text(0.5, 0.5, 'Core\nKnowledge', ha='center', va='center', 
            fontsize=10, color='white', fontweight='bold')
    
    # 不同相似度的干扰知识
    # 高相似度
    high_sim = plt.Circle((0.2, 0.5), 0.06, color='green', alpha=0.6)
    ax.add_patch(high_sim)
    ax.text(0.2, 0.5, 'High\nSim', ha='center', va='center', fontsize=9)
    ax.annotate('', xy=(0.42, 0.5), xytext=(0.26, 0.5),
                arrowprops=dict(arrowstyle='->', lw=3, color='green'))
    ax.text(0.34, 0.52, '+3.2%', ha='center', fontsize=8, color='green')
    
    # 中等相似度
    medium_sim = plt.Circle((0.35, 0.25), 0.06, color='orange', alpha=0.6)
    ax.add_patch(medium_sim)
    ax.text(0.35, 0.25, 'Med\nSim', ha='center', va='center', fontsize=9)
    ax.annotate('', xy=(0.47, 0.43), xytext=(0.38, 0.31),
                arrowprops=dict(arrowstyle='->', lw=2, color='orange', linestyle='dashed'))
    ax.text(0.42, 0.35, '13.3% flip', ha='center', fontsize=8, color='orange')
    
    # 低相似度
    low_sim = plt.Circle((0.65, 0.25), 0.06, color='red', alpha=0.6)
    ax.add_patch(low_sim)
    ax.text(0.65, 0.25, 'Low\nSim', ha='center', va='center', fontsize=9)
    ax.annotate('', xy=(0.53, 0.43), xytext=(0.62, 0.31),
                arrowprops=dict(arrowstyle='->', lw=1, color='red', linestyle='dotted'))
    ax.text(0.58, 0.35, '-17% conf', ha='center', fontsize=8, color='red')
    
    # 高节点度
    high_degree = plt.Circle((0.8, 0.5), 0.06, color='purple', alpha=0.6)
    ax.add_patch(high_degree)
    ax.text(0.8, 0.5, 'High\nDegree', ha='center', va='center', fontsize=9)
    ax.annotate('', xy=(0.58, 0.5), xytext=(0.74, 0.5),
                arrowprops=dict(arrowstyle='->', lw=4, color='purple'))
    ax.text(0.66, 0.52, '40% flip', ha='center', fontsize=8, color='purple')
    
    # 添加图例
    legend_elements = [
        mpatches.Patch(color='green', label='Reinforcement Effect'),
        mpatches.Patch(color='orange', label='Confusion Effect'),
        mpatches.Patch(color='red', label='Burden Effect'),
        mpatches.Patch(color='purple', label='Broadcast Effect')
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=4, 
              bbox_to_anchor=(0.5, -0.05))
    
    # 添加说明文字
    ax.text(0.5, 0.08, 'Arrow thickness indicates effect strength', 
            transform=ax.transAxes, ha='center', fontsize=10, style='italic')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(experiment_dir / 'analysis' / 'semantic_mechanism.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"案例对比图已保存到: {experiment_dir / 'analysis'}")

if __name__ == "__main__":
    experiment_dir = Path("/root/graph/knowledge_ripple_effect/results/semantic_experiment_20250617_090658")
    create_case_comparison_figure(experiment_dir)