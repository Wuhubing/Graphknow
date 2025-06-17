import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class SemanticAnalyzer:
    """计算句子之间的语义相似度"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """初始化语义分析器
        
        Args:
            model_name: sentence-transformers模型名称
        """
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"成功加载语义模型: {model_name}")
        except Exception as e:
            logger.error(f"加载语义模型失败: {str(e)}")
            self.model = None
    
    def encode_sentences(self, sentences: List[str]) -> np.ndarray:
        """将句子编码为向量
        
        Args:
            sentences: 句子列表
            
        Returns:
            句子向量矩阵
        """
        if not self.model:
            return np.array([])
        
        try:
            embeddings = self.model.encode(sentences, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            logger.error(f"编码句子失败: {str(e)}")
            return np.array([])
    
    def compute_similarity(self, sentence1: str, sentence2: str) -> float:
        """计算两个句子的余弦相似度
        
        Args:
            sentence1: 第一个句子
            sentence2: 第二个句子
            
        Returns:
            相似度分数 (0-1)
        """
        if not self.model:
            return 0.0
        
        try:
            embeddings = self.encode_sentences([sentence1, sentence2])
            if len(embeddings) < 2:
                return 0.0
            
            # 计算余弦相似度
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            logger.error(f"计算相似度失败: {str(e)}")
            return 0.0
    
    def find_most_similar(self, 
                         target_sentence: str, 
                         candidate_sentences: List[str], 
                         top_k: int = 5) -> List[Tuple[int, str, float]]:
        """找出与目标句子最相似的候选句子
        
        Args:
            target_sentence: 目标句子
            candidate_sentences: 候选句子列表
            top_k: 返回最相似的k个句子
            
        Returns:
            [(索引, 句子, 相似度分数)]
        """
        if not self.model or not candidate_sentences:
            return []
        
        try:
            # 编码所有句子
            all_sentences = [target_sentence] + candidate_sentences
            embeddings = self.encode_sentences(all_sentences)
            
            if len(embeddings) < 2:
                return []
            
            target_embedding = embeddings[0]
            candidate_embeddings = embeddings[1:]
            
            # 计算相似度
            similarities = []
            for i, candidate_embedding in enumerate(candidate_embeddings):
                similarity = np.dot(target_embedding, candidate_embedding) / (
                    np.linalg.norm(target_embedding) * np.linalg.norm(candidate_embedding)
                )
                similarities.append((i, candidate_sentences[i], float(similarity)))
            
            # 按相似度排序
            similarities.sort(key=lambda x: x[2], reverse=True)
            
            return similarities[:top_k]
        except Exception as e:
            logger.error(f"查找相似句子失败: {str(e)}")
            return []
    
    def categorize_by_similarity(self,
                               target_sentence: str,
                               candidate_sentences: List[str],
                               thresholds: Dict[str, Tuple[float, float]] = None) -> Dict[str, List[Tuple[int, str, float]]]:
        """根据相似度将候选句子分类
        
        Args:
            target_sentence: 目标句子
            candidate_sentences: 候选句子列表
            thresholds: 相似度阈值字典，默认为 {
                'high': (0.7, 1.0),
                'medium': (0.4, 0.7),
                'low': (0.0, 0.4)
            }
            
        Returns:
            分类后的句子字典
        """
        if thresholds is None:
            thresholds = {
                'high': (0.7, 1.0),
                'medium': (0.4, 0.7),
                'low': (0.0, 0.4)
            }
        
        categorized = {category: [] for category in thresholds}
        
        if not self.model or not candidate_sentences:
            return categorized
        
        try:
            # 计算所有相似度
            all_sentences = [target_sentence] + candidate_sentences
            embeddings = self.encode_sentences(all_sentences)
            
            if len(embeddings) < 2:
                return categorized
            
            target_embedding = embeddings[0]
            candidate_embeddings = embeddings[1:]
            
            # 分类
            for i, candidate_embedding in enumerate(candidate_embeddings):
                similarity = np.dot(target_embedding, candidate_embedding) / (
                    np.linalg.norm(target_embedding) * np.linalg.norm(candidate_embedding)
                )
                similarity = float(similarity)
                
                # 找到对应的类别
                for category, (min_threshold, max_threshold) in thresholds.items():
                    if min_threshold <= similarity < max_threshold:
                        categorized[category].append((i, candidate_sentences[i], similarity))
                        break
            
            return categorized
        except Exception as e:
            logger.error(f"分类句子失败: {str(e)}")
            return categorized