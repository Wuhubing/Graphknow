import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from tqdm import tqdm

from src.config import MODEL_CONFIG, EXPERIMENT_CONFIG
from src.model_utils import LlamaInference
from src.knowledge_graph import LocalKnowledgeGraphBuilder
from src.data_processing import HotpotQAProcessor
from src.perturbation_experiment import PerturbationExperiment
from src.ripple_analyzer import RippleEffectAnalyzer
from src.json_utils import NumpyJSONEncoder

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_complete_experiment(sample_size=100, use_existing_baseline=True):
    """运行完整的实验，包括基准评估和扰动实验"""
    
    # 创建实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(f"/root/graph/knowledge_ripple_effect/results/complete_experiment_{timestamp}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 初始化模型
    logger.info("初始化模型...")
    model = LlamaInference(MODEL_CONFIG['model_name'])
    graph_builder = LocalKnowledgeGraphBuilder()
    
    # 如果使用现有的基准结果
    if use_existing_baseline:
        logger.info("使用现有的基准评估结果...")
        existing_exp_dir = Path("/root/graph/knowledge_ripple_effect/results/large_scale_experiment_20250617_093716")
        
        # 复制基准结果
        import shutil
        shutil.copytree(existing_exp_dir / 'baseline', experiment_dir / 'baseline')
        
        # 加载基准结果
        with open(existing_exp_dir / 'baseline/baseline_results.json', 'r') as f:
            baseline_results = json.load(f)
        
        # 筛选bright knowledge样本
        baseline_df = pd.DataFrame(baseline_results)
        bright_samples = baseline_df[baseline_df['knowledge_type'] == 'bright']
        
        # 限制样本数量
        if sample_size < len(bright_samples):
            # 优先选择已识别的优先样本
            try:
                with open(existing_exp_dir / 'priority_samples_for_perturbation.json', 'r') as f:
                    priority_info = json.load(f)
                priority_ids = priority_info['priority_sample_ids']
                
                # 先选择优先样本
                selected_bright = bright_samples[bright_samples['id'].isin(priority_ids)]
                
                # 如果不够，再随机补充
                if len(selected_bright) < sample_size:
                    remaining = bright_samples[~bright_samples['id'].isin(priority_ids)]
                    additional = remaining.sample(n=min(sample_size - len(selected_bright), len(remaining)))
                    selected_bright = pd.concat([selected_bright, additional])
                
                bright_baseline_results = selected_bright.to_dict('records')
            except:
                # 如果没有优先样本文件，就随机选择
                bright_baseline_results = bright_samples.sample(n=min(sample_size, len(bright_samples))).to_dict('records')
        else:
            bright_baseline_results = bright_samples.to_dict('records')
        
        logger.info(f"选择了{len(bright_baseline_results)}个Bright Knowledge样本进行扰动实验")
        
    else:
        # 运行新的基准评估
        logger.info("运行新的基准评估...")
        processor = HotpotQAProcessor(split='train', sample_size=sample_size * 3)  # 多加载一些以确保有足够的bright samples
        processor.load_data()
        samples = processor.prepare_baseline_prompts()
        
        from src.baseline_evaluation import BaselineEvaluator
        baseline_evaluator = BaselineEvaluator(model, processor)
        baseline_results = baseline_evaluator.run_baseline_evaluation()
        
        # 保存基准结果
        baseline_dir = experiment_dir / 'baseline'
        baseline_dir.mkdir(exist_ok=True)
        baseline_evaluator.save_results(baseline_dir)
        
        # 筛选bright knowledge
        baseline_df = pd.DataFrame(baseline_results)
        bright_baseline_results = baseline_df[baseline_df['knowledge_type'] == 'bright'].to_dict('records')
    
    # 加载样本数据
    logger.info("加载样本数据...")
    processor = HotpotQAProcessor(split='train', sample_size=len(baseline_results))
    processor.load_data()
    all_samples = processor.prepare_baseline_prompts()
    sample_dict = {s['id']: s for s in all_samples}
    
    # 准备扰动实验的样本
    bright_samples_for_perturbation = []
    for br in bright_baseline_results:
        if br['id'] in sample_dict:
            bright_samples_for_perturbation.append(sample_dict[br['id']])
    
    logger.info(f"准备对{len(bright_samples_for_perturbation)}个样本进行扰动实验")
    
    # 保存实验元数据
    metadata = {
        'timestamp': timestamp,
        'model_name': MODEL_CONFIG['model_name'],
        'device': str(device),
        'total_baseline_samples': len(baseline_results),
        'bright_knowledge_samples': len(bright_baseline_results),
        'samples_for_perturbation': len(bright_samples_for_perturbation),
        'use_existing_baseline': use_existing_baseline
    }
    
    with open(experiment_dir / 'experiment_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # 运行扰动实验
    logger.info("\n开始扰动实验...")
    perturbation_exp = PerturbationExperiment(model, graph_builder)
    
    # 使用完整的baseline_results列表（包含所有样本）
    perturbation_results = perturbation_exp.run_full_experiment(
        bright_samples_for_perturbation, 
        bright_baseline_results
    )
    
    # 保存扰动结果
    perturbation_dir = experiment_dir / 'perturbation'
    perturbation_dir.mkdir(exist_ok=True)
    analysis_stats = perturbation_exp.save_results(perturbation_dir)
    
    # 运行分析
    logger.info("\n运行分析和可视化...")
    analyzer = RippleEffectAnalyzer(bright_baseline_results, perturbation_results)
    
    summary = analyzer.summarize_experiment()
    hypothesis_results = analyzer.test_hypotheses()
    
    # 保存分析结果
    analysis_dir = experiment_dir / 'analysis'
    analysis_dir.mkdir(exist_ok=True)
    
    with open(analysis_dir / 'experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyJSONEncoder)
    
    with open(analysis_dir / 'hypothesis_test_results.json', 'w') as f:
        json.dump(hypothesis_results, f, indent=2, cls=NumpyJSONEncoder)
    
    # 生成可视化
    analyzer.plot_ripple_effect_analysis(save_path=str(analysis_dir / 'ripple_effect_analysis.png'))
    
    # 打印结果摘要
    print("\n" + "="*80)
    print("实验完成！")
    print("="*80)
    
    print(f"\n基准评估:")
    print(f"  总样本数: {len(baseline_results)}")
    print(f"  Bright Knowledge样本: {len(bright_baseline_results)}")
    
    if perturbation_results:
        print(f"\n扰动实验:")
        print(f"  扰动样本数: {len(bright_samples_for_perturbation)}")
        print(f"  总扰动次数: {len(perturbation_results)}")
        print(f"  翻转率: {summary['overall_statistics']['flip_rate']:.2%}")
        print(f"  平均置信度变化: {summary['overall_statistics']['avg_confidence_change']:.4f}")
        
        print(f"\n策略效果:")
        for strategy, stats in summary['by_strategy'].items():
            print(f"\n  {strategy}:")
            print(f"    样本数: {stats['count']}")
            print(f"    翻转率: {stats['flip_rate']:.2%}")
            print(f"    平均置信度变化: {stats['avg_confidence_change']:.4f}")
            if 'avg_semantic_similarity' in stats and stats['avg_semantic_similarity'] > 0:
                print(f"    平均语义相似度: {stats['avg_semantic_similarity']:.3f}")
        
        print(f"\n假设检验:")
        for test_name, result in hypothesis_results.items():
            if isinstance(result, dict) and 'p_value' in result:
                print(f"\n  {test_name}:")
                print(f"    p值: {result['p_value']:.4f}")
                print(f"    统计显著: {'是' if result['significant'] else '否'}")
    
    print(f"\n结果保存在: {experiment_dir}")
    
    return experiment_dir, summary, hypothesis_results

if __name__ == "__main__":
    # 运行实验，使用现有的基准结果，对全部100个bright knowledge样本进行扰动
    experiment_dir, summary, hypothesis_results = run_complete_experiment(
        sample_size=100, 
        use_existing_baseline=True
    )