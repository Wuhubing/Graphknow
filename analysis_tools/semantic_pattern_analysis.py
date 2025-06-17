import json
import re
from collections import defaultdict

def extract_semantic_patterns(results_file):
    """分析成功翻转案例的语义模式"""
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # 收集距离=1的成功翻转案例
    successful_flips = []
    for result in results:
        if result['influencer_distance'] == 1 and result['correctness_flipped']:
            successful_flips.append(result)
    
    print(f"找到 {len(successful_flips)} 个距离=1的成功翻转案例\n")
    
    # 分析语义模式
    patterns = {
        'entity_confusion': [],  # 实体混淆
        'temporal_confusion': [],  # 时间混淆
        'attribute_conflict': [],  # 属性冲突
        'context_shift': []  # 上下文转移
    }
    
    for case in successful_flips:
        question = case['question'].lower()
        baseline_answer = case['baseline_answer'].lower()
        perturbed_answer = case['perturbed_answer'].lower()
        influencer_context = case['influencer_context'].lower()
        
        # 检测实体混淆：干扰内容包含与问题相关的其他实体
        question_entities = set(re.findall(r'\b[A-Z][a-z]+\b', case['question']))
        influencer_entities = set(re.findall(r'\b[A-Z][a-z]+\b', case['influencer_context']))
        if len(question_entities & influencer_entities) > 0:
            patterns['entity_confusion'].append(case)
        
        # 检测时间混淆：包含年份或时间相关词汇
        if any(year in influencer_context for year in ['19', '20']) or \
           any(word in influencer_context for word in ['year', 'date', 'time', 'when']):
            patterns['temporal_confusion'].append(case)
        
        # 检测属性冲突：问题询问某个属性，干扰内容提供了不同的属性值
        if 'who' in question or 'what' in question or 'which' in question:
            if baseline_answer != perturbed_answer:
                patterns['attribute_conflict'].append(case)
        
        # 检测上下文转移：答案完全改变了主题
        baseline_words = set(baseline_answer.split()[:20])
        perturbed_words = set(perturbed_answer.split()[:20])
        overlap_ratio = len(baseline_words & perturbed_words) / max(len(baseline_words), 1)
        if overlap_ratio < 0.3:
            patterns['context_shift'].append(case)
    
    # 输出分析结果
    for pattern_type, cases in patterns.items():
        print(f"\n=== {pattern_type.upper()} 模式 ({len(cases)} 个案例) ===")
        for i, case in enumerate(cases[:2]):  # 每种模式显示2个例子
            print(f"\n例子 {i+1}:")
            print(f"问题: {case['question']}")
            print(f"基准答案开头: {case['baseline_answer'][:100]}...")
            print(f"扰动后答案开头: {case['perturbed_answer'][:100]}...")
            print(f"干扰内容: {case['influencer_context'][:150]}...")
    
    # 分析远距离置信度上升的模式
    print("\n\n=== 远距离置信度上升模式分析 ===")
    far_confidence_cases = []
    for result in results:
        if result['influencer_distance'] >= 2 and result['confidence_change'] > 0.3:
            far_confidence_cases.append(result)
    
    print(f"找到 {len(far_confidence_cases)} 个远距离(>=2)且置信度大幅上升(>0.3)的案例")
    
    # 分析这些案例的共同特征
    topic_overlap = []
    reinforcement = []
    
    for case in far_confidence_cases[:5]:
        question_words = set(case['question'].lower().split())
        influencer_words = set(case['influencer_context'].lower().split())
        
        # 主题重叠
        overlap = len(question_words & influencer_words)
        if overlap > 3:
            topic_overlap.append(case)
        
        # 答案强化
        if case['baseline_correct'] and not case['correctness_flipped']:
            reinforcement.append(case)
    
    print(f"\n主题重叠案例: {len(topic_overlap)}")
    print(f"答案强化案例: {len(reinforcement)}")
    
    return patterns, far_confidence_cases

if __name__ == "__main__":
    results_file = "/root/graph/knowledge_ripple_effect/results/experiment_20250617_074359/perturbation/perturbation_results.json"
    extract_semantic_patterns(results_file)