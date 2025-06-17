import json
import pandas as pd
from collections import defaultdict
import numpy as np

def analyze_perturbation_patterns(results_file):
    """深入分析扰动模式，特别关注距离=1的成功案例和远距离的置信度上升案例"""
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # 分类不同的扰动效果
    distance_1_flips = []  # 距离=1且翻转答案的案例
    distance_1_confidence_up = []  # 距离=1且置信度上升的案例
    far_confidence_up = []  # 距离>=2且置信度上升的案例
    
    for result in results:
        distance = result['influencer_distance']
        confidence_change = result['confidence_change']
        answer_changed = result['answer_changed']
        correctness_flipped = result['correctness_flipped']
        
        if distance == 1:
            if correctness_flipped:
                distance_1_flips.append(result)
            elif confidence_change > 0:
                distance_1_confidence_up.append(result)
        elif distance >= 2 and confidence_change > 0:
            far_confidence_up.append(result)
    
    print(f"总实验数: {len(results)}")
    print(f"\n距离=1且成功翻转正确性的案例: {len(distance_1_flips)}")
    print(f"距离=1且置信度上升的案例: {len(distance_1_confidence_up)}")
    print(f"距离>=2且置信度上升的案例: {len(far_confidence_up)}")
    
    # 分析距离=1的成功翻转案例
    print("\n=== 距离=1的成功翻转案例分析 ===")
    for i, case in enumerate(distance_1_flips[:5]):  # 显示前5个
        print(f"\n案例 {i+1}:")
        print(f"问题: {case['question']}")
        print(f"基准答案: {case['baseline_answer'][:100]}...")
        print(f"扰动后答案: {case['perturbed_answer'][:100]}...")
        print(f"干扰节点: {case['influencer_node']}")
        print(f"干扰内容: {case['influencer_context'][:200]}...")
        print(f"节点度数: {case['influencer_degree']}")
        print(f"置信度变化: {case['confidence_change']:.4f}")
    
    # 分析远距离置信度上升案例
    print("\n\n=== 距离>=2但置信度上升的案例分析 ===")
    # 按置信度上升幅度排序
    far_confidence_up.sort(key=lambda x: x['confidence_change'], reverse=True)
    for i, case in enumerate(far_confidence_up[:5]):  # 显示前5个
        print(f"\n案例 {i+1}:")
        print(f"问题: {case['question']}")
        print(f"基准答案: {case['baseline_answer'][:100]}...")
        print(f"扰动后答案: {case['perturbed_answer'][:100]}...")
        print(f"干扰节点: {case['influencer_node']}")
        print(f"干扰内容: {case['influencer_context'][:200]}...")
        print(f"节点距离: {case['influencer_distance']}")
        print(f"节点度数: {case['influencer_degree']}")
        print(f"置信度变化: {case['confidence_change']:.4f}")
    
    # 统计不同策略的效果
    print("\n\n=== 不同扰动策略的效果统计 ===")
    strategy_stats = defaultdict(lambda: {'total': 0, 'flips': 0, 'confidence_up': 0, 'avg_confidence_change': []})
    
    for result in results:
        strategy = result['perturbation_strategy']
        strategy_stats[strategy]['total'] += 1
        if result['correctness_flipped']:
            strategy_stats[strategy]['flips'] += 1
        if result['confidence_change'] > 0:
            strategy_stats[strategy]['confidence_up'] += 1
        strategy_stats[strategy]['avg_confidence_change'].append(result['confidence_change'])
    
    for strategy, stats in strategy_stats.items():
        avg_change = np.mean(stats['avg_confidence_change'])
        print(f"\n策略: {strategy}")
        print(f"  总数: {stats['total']}")
        print(f"  翻转率: {stats['flips']/stats['total']*100:.2f}%")
        print(f"  置信度上升率: {stats['confidence_up']/stats['total']*100:.2f}%")
        print(f"  平均置信度变化: {avg_change:.4f}")
    
    return distance_1_flips, far_confidence_up

if __name__ == "__main__":
    results_file = "/root/graph/knowledge_ripple_effect/results/experiment_20250617_074359/perturbation/perturbation_results.json"
    analyze_perturbation_patterns(results_file)