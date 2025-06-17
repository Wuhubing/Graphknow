import json
from pathlib import Path
import pandas as pd

def perform_qualitative_analysis():
    """对现有实验结果进行深入的定性分析"""
    
    # 加载数据
    experiment_dir = Path("/root/graph/knowledge_ripple_effect/results/semantic_experiment_20250617_090658")
    with open(experiment_dir / 'perturbation/perturbation_results.json', 'r') as f:
        results = json.load(f)
    
    df = pd.DataFrame(results)
    
    print("="*100)
    print("深度案例分析：理解知识涟漪效应的机制")
    print("="*100)
    
    # 1. 分析high_degree策略的成功案例
    print("\n\n### 1. HIGH_DEGREE策略成功案例分析 ###")
    print("-"*80)
    
    high_degree_flips = df[(df['perturbation_strategy'] == 'high_degree') & 
                           (df['correctness_flipped'] == True)]
    
    for idx, case in high_degree_flips.iterrows():
        print(f"\n案例 {idx}:")
        print(f"问题: {case['question']}")
        print(f"金标准答案: {case['gold_answer'] if 'gold_answer' in case else 'N/A'}")
        print(f"基准答案: {case['baseline_answer']}")
        print(f"扰动后答案: {case['perturbed_answer']}")
        print(f"\n干扰节点: '{case['influencer_node']}' (度数={case['influencer_degree']})")
        print(f"干扰内容: {case['influencer_context']}")
        print(f"\n分析:")
        
        # 案例具体分析
        if "Arthur's Magazine" in case['question']:
            print("  - 问题询问两个杂志的创刊时间先后")
            print("  - 干扰节点'91.1'是一个广播频率，看似完全无关")
            print("  - 但该节点连接了多个时间相关的实体（2001, 2003, 2004）")
            print("  - 这些时间信息可能干扰了模型对'first'这个时间概念的判断")
            print("  - 机制：高度数节点通过引入大量时间噪音，破坏了原有的时序推理")
        
        if "tennis player" in case['question']:
            print("  - 问题询问谁赢得更多Grand Slam冠军")
            print("  - 干扰内容提到了另一位网球选手Pam Teeguarden")
            print("  - 虽然没有直接提供Grand Slam数量，但引入了'professional tennis player'概念")
            print("  - 机制：通过引入相关领域的其他实体，稀释了原有答案的确定性")
    
    # 2. 分析medium_semantic_similarity的成功案例
    print("\n\n### 2. MEDIUM_SEMANTIC_SIMILARITY策略成功案例分析 ###")
    print("-"*80)
    
    medium_sem_flips = df[(df['perturbation_strategy'] == 'medium_semantic_similarity') & 
                          (df['correctness_flipped'] == True)]
    
    for idx, case in medium_sem_flips.iterrows():
        print(f"\n案例 {idx}:")
        print(f"问题: {case['question']}")
        print(f"基准答案: {case['baseline_answer'][:100]}...")
        print(f"扰动后答案: {case['perturbed_answer'][:100]}...")
        print(f"\n干扰节点: '{case['influencer_node']}'")
        print(f"语义相似度: {case['influencer_semantic_similarity']:.3f}")
        print(f"干扰内容: {case['influencer_context']}")
        print(f"\n分析:")
        
        # 具体分析
        if "Grand Slam" in case['influencer_context']:
            print("  - 干扰内容提到'Li won...two Grand Slam singles titles'")
            print("  - 语义相似度0.402表明内容相关但不完全一致")
            print("  - 关键冲突点：问题问的是Henri Leconte vs Jonathan Stark")
            print("  - 但干扰引入了第三个人Li Na的Grand Slam信息")
            print("  - 机制：中等相似度让模型'认为'这是相关信息，但实际上引入了混淆")
            print("  - 这种'似是而非'的信息最容易破坏推理链")
    
    # 3. 分析high_semantic_similarity的增强案例
    print("\n\n### 3. HIGH_SEMANTIC_SIMILARITY增强置信度案例分析 ###")
    print("-"*80)
    
    high_sem_boost = df[(df['perturbation_strategy'] == 'high_semantic_similarity') & 
                        (df['confidence_change'] > 0)]
    
    for idx, case in high_sem_boost.head(2).iterrows():
        print(f"\n案例 {idx}:")
        print(f"问题: {case['question']}")
        print(f"答案: {case['baseline_answer'][:100]}...")
        print(f"\n干扰节点: '{case['influencer_node']}'")
        print(f"语义相似度: {case['influencer_semantic_similarity']:.3f}")
        print(f"干扰内容: {case['influencer_context']}")
        print(f"置信度变化: {case['baseline_confidence']:.3f} → {case['perturbed_confidence']:.3f} "
              f"(+{case['confidence_change']:.3f})")
        print(f"\n分析:")
        
        # 具体分析
        if "Malcolm Smith" in case['question'] and "Super Bowl" in case['influencer_context']:
            print("  - 问题询问Malcolm Smith在哪场比赛获得MVP")
            print("  - 干扰内容直接说明'Smith was named the Most Valuable Player of Super Bowl XLVIII'")
            print("  - 语义相似度0.929表明这几乎是对问题的直接回答")
            print("  - 机制：高度相关的信息起到了'确认'和'强化'的作用")
            print("  - 就像考试时看到参考书上的原题，增强了答题信心")
    
    # 4. 总结发现的机制
    print("\n\n### 4. 机制总结 ###")
    print("-"*80)
    
    print("\n发现的知识影响机制:")
    print("\n1. 广播干扰机制（High Degree）:")
    print("   - 高度数节点像'广播站'，向多个方向发送信号")
    print("   - 即使内容看似无关，但通过大量连接产生'认知噪音'")
    print("   - 特别有效于时间、数量等需要精确判断的问题")
    
    print("\n2. 认知迷雾机制（Medium Similarity）:")
    print("   - 中等相似度的信息处于'似是而非'的危险区间")
    print("   - 模型认为信息相关，但实际包含误导性细节")
    print("   - 最容易在比较类问题中造成混淆")
    
    print("\n3. 回声室机制（High Similarity）:")
    print("   - 高度相似的信息形成'回声室'效应")
    print("   - 相互印证和强化，提升模型置信度")
    print("   - 但如果是错误信息的回声，可能加深错误")
    
    print("\n4. 距离悖论机制（Far Distance）:")
    print("   - 远距离节点通过间接路径影响")
    print("   - 不改变答案但影响置信度")
    print("   - 可能通过激活更广泛的知识网络造成'虚假自信'")

if __name__ == "__main__":
    perform_qualitative_analysis()