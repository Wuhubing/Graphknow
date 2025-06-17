import os
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

from src.config import DATA_DIR, RESULTS_DIR, MODEL_CONFIG
from src.data_processing import HotpotQAProcessor
from src.model_utils import LlamaInference
from src.knowledge_graph import LocalKnowledgeGraphBuilder
from src.baseline_evaluation import BaselineEvaluator
from src.perturbation_experiment import PerturbationExperiment
from src.ripple_analyzer import RippleEffectAnalyzer
from src.json_utils import NumpyJSONEncoder

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # 创建实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = RESULTS_DIR / f"semantic_experiment_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 初始化组件
    logger.info("初始化模型和数据处理器...")
    model = LlamaInference(MODEL_CONFIG['model_name'])
    processor = HotpotQAProcessor(split='train', sample_size=20)
    graph_builder = LocalKnowledgeGraphBuilder()
    
    # 加载数据 - 使用较小的样本量进行测试
    logger.info("加载数据...")
    processor.load_data()
    samples = processor.prepare_baseline_prompts()
    
    # 保存实验元数据
    metadata = {
        'timestamp': timestamp,
        'model_name': MODEL_CONFIG['model_name'],
        'device': str(device),
        'num_samples': len(samples),
        'experiment_type': 'semantic_similarity_based_perturbation',
        'features': [
            'semantic_similarity_strategies',
            'influencer_initial_confidence',
            'extended_perturbation_strategies'
        ]
    }
    
    with open(experiment_dir / 'experiment_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # 运行基准评估
    logger.info("运行基准评估...")
    baseline_evaluator = BaselineEvaluator(model, processor)
    baseline_results = baseline_evaluator.run_baseline_evaluation()
    
    # 计算统计信息
    df = pd.DataFrame(baseline_results)
    baseline_stats = {
        'total_samples': len(df),
        'correct_predictions': df['is_correct'].sum(),
        'accuracy': df['is_correct'].mean(),
        'bright_knowledge_count': (df['knowledge_type'] == 'bright').sum(),
        'dark_knowledge_count': (df['knowledge_type'] == 'dark').sum(),
        'avg_confidence': df['confidence'].mean()
    }
    
    # 保存基准结果
    baseline_dir = experiment_dir / 'baseline'
    baseline_dir.mkdir(exist_ok=True)
    baseline_evaluator.save_results(baseline_dir)
    
    # 运行扰动实验
    logger.info("运行语义相似度扰动实验...")
    perturbation_exp = PerturbationExperiment(model, graph_builder)
    perturbation_results = perturbation_exp.run_full_experiment(samples, baseline_results)
    
    # 保存扰动结果
    perturbation_dir = experiment_dir / 'perturbation'
    perturbation_dir.mkdir(exist_ok=True)
    analysis_stats = perturbation_exp.save_results(perturbation_dir)
    
    # 运行分析和可视化
    logger.info("运行分析和可视化...")
    analyzer = RippleEffectAnalyzer(baseline_results, perturbation_results)
    
    # 执行分析
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
    fig_path = analysis_dir / 'ripple_effect_analysis.png'
    analyzer.plot_ripple_effect_analysis(save_path=str(fig_path))
    
    logger.info(f"实验完成！结果保存在: {experiment_dir}")
    
    # 打印关键发现
    print("\n=== 实验关键发现 ===")
    print(f"总扰动次数: {len(perturbation_results)}")
    print(f"成功翻转率: {summary['overall_statistics']['flip_rate']:.2%}")
    print(f"平均置信度变化: {summary['overall_statistics']['avg_confidence_change']:.4f}")
    
    # 按策略打印结果
    print("\n=== 各策略效果 ===")
    for strategy, stats in summary['by_strategy'].items():
        print(f"\n策略: {strategy}")
        print(f"  翻转率: {stats['flip_rate']:.2%}")
        print(f"  置信度上升率: {stats['confidence_increase_rate']:.2%}")
        print(f"  平均置信度变化: {stats['avg_confidence_change']:.4f}")

if __name__ == "__main__":
    main()