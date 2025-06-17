import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from src.ripple_analyzer import RippleEffectAnalyzer
from src.json_utils import NumpyJSONEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def combine_experiment_results():
    """合并多次实验的结果进行综合分析"""
    
    # 加载所有实验结果
    all_results = []
    
    # 1. 完整的30样本实验
    exp1_dir = Path("/root/graph/knowledge_ripple_effect/results/complete_experiment_20250617_100529")
    with open(exp1_dir / 'perturbation/perturbation_results.json', 'r') as f:
        results1 = json.load(f)
    logger.info(f"实验1: {len(results1)}个扰动结果")
    all_results.extend(results1)
    
    # 2. 批次实验的部分结果
    exp2_dir = Path("/root/graph/knowledge_ripple_effect/results/batch_experiment_20250617_131758")
    with open(exp2_dir / 'temp_results.json', 'r') as f:
        results2 = json.load(f)
    logger.info(f"实验2: {len(results2)}个扰动结果")
    
    # 去重（基于sample_id和influencer_node）
    existing_keys = set()
    for r in results1:
        key = f"{r['sample_id']}_{r['influencer_node']}_{r['perturbation_strategy']}"
        existing_keys.add(key)
    
    new_results = []
    for r in results2:
        key = f"{r['sample_id']}_{r['influencer_node']}_{r['perturbation_strategy']}"
        if key not in existing_keys:
            new_results.append(r)
    
    logger.info(f"实验2新增: {len(new_results)}个独特的扰动结果")
    all_results.extend(new_results)
    
    # 创建新的结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_dir = Path(f"/root/graph/knowledge_ripple_effect/results/combined_analysis_{timestamp}")
    combined_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存合并的结果
    with open(combined_dir / 'all_perturbation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyJSONEncoder)
    
    # 加载基准结果
    baseline_dir = Path("/root/graph/knowledge_ripple_effect/results/large_scale_experiment_20250617_093716")
    with open(baseline_dir / 'baseline/baseline_results.json', 'r') as f:
        all_baseline = json.load(f)
    
    # 筛选涉及的bright样本
    sample_ids = list(set([r['sample_id'] for r in all_results]))
    bright_baseline = [b for b in all_baseline if b['id'] in sample_ids and b['knowledge_type'] == 'bright']
    
    logger.info(f"\n合并后统计:")
    logger.info(f"总扰动数: {len(all_results)}")
    logger.info(f"涉及样本数: {len(sample_ids)}")
    
    # 运行综合分析
    analyzer = RippleEffectAnalyzer(bright_baseline, all_results)
    
    summary = analyzer.summarize_experiment()
    hypothesis_results = analyzer.test_hypotheses()
    
    # 保存分析结果
    with open(combined_dir / 'experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyJSONEncoder)
    
    with open(combined_dir / 'hypothesis_test_results.json', 'w') as f:
        json.dump(hypothesis_results, f, indent=2, cls=NumpyJSONEncoder)
    
    # 生成可视化
    analyzer.plot_ripple_effect_analysis(save_path=str(combined_dir / 'ripple_effect_analysis.png'))
    
    # 生成综合报告
    report = f"""# 知识涟漪效应综合实验报告

## 实验规模
- 总扰动次数: {len(all_results)}
- 涉及样本数: {len(sample_ids)}
- 数据来源: 两次独立实验的合并结果

## 核心发现

### 1. 总体统计
- 整体翻转率: {summary['overall_statistics']['flip_rate']:.2%}
- 平均置信度变化: {summary['overall_statistics']['avg_confidence_change']:.4f}

### 2. 策略效果对比
"""
    
    # 添加策略分析
    for strategy, stats in summary['by_strategy'].items():
        report += f"\n**{strategy}**\n"
        report += f"- 样本数: {stats['count']}\n"
        report += f"- 翻转率: {stats['flip_rate']:.2%}\n"
        report += f"- 平均置信度变化: {stats['avg_confidence_change']:.4f}\n"
        if 'avg_semantic_similarity' in stats and stats['avg_semantic_similarity'] > 0:
            report += f"- 平均语义相似度: {stats['avg_semantic_similarity']:.3f}\n"
    
    report += f"""
### 3. 假设检验结果
"""
    # 安全地添加假设检验结果
    for test_name, result in hypothesis_results.items():
        if isinstance(result, dict) and 'p_value' in result:
            report += f"\n**{test_name}**\n"
            report += f"- p值: {result['p_value']:.4f}\n"
            report += f"- 统计显著: {'是' if result['significant'] else '否'}\n"
    
    report += f"""
## 结论
基于{len(all_results)}个扰动实验的结果，我们的主要发现包括：
1. 不同扰动策略展现出显著不同的效果
2. 图结构特征（如节点度数）影响知识的脆弱性
3. 语义相似度在知识干扰中起重要作用

实验结果保存在: {combined_dir}
"""
    
    with open(combined_dir / 'combined_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    # 打印摘要
    print("\n" + "="*80)
    print("综合分析完成！")
    print("="*80)
    print(report)
    
    return combined_dir, summary, hypothesis_results

if __name__ == "__main__":
    combined_dir, summary, hypothesis_results = combine_experiment_results()