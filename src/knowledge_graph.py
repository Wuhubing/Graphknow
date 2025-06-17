import spacy
import networkx as nx
from typing import List, Dict, Tuple, Set
import pandas as pd
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class LocalKnowledgeGraphBuilder:
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        logger.info(f"Loading spaCy model: {spacy_model}")
        self.nlp = spacy.load(spacy_model)
        self.graphs = {}
        
    def extract_entities(self, text: str) -> List[Dict]:
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        for token in doc:
            if token.pos_ in ['PROPN', 'NOUN'] and not any(token.text in ent['text'] for ent in entities):
                if token.text.istitle() and len(token.text) > 2:
                    entities.append({
                        'text': token.text,
                        'label': 'NOUN',
                        'start': token.idx,
                        'end': token.idx + len(token.text)
                    })
        
        return entities
    
    def build_local_graph(self, sample_data: Dict) -> nx.Graph:
        G = nx.Graph()
        
        question_text = sample_data['question']
        supporting_facts = sample_data['supporting_facts']
        all_context = sample_data['all_context']
        
        core_entities = set()
        for fact in supporting_facts:
            entities = self.extract_entities(fact['sentence'])
            for ent in entities:
                core_entities.add(ent['text'])
                G.add_node(ent['text'], node_type='core', label=ent['label'])
        
        question_entities = self.extract_entities(question_text)
        for ent in question_entities:
            if ent['text'] not in G:
                G.add_node(ent['text'], node_type='question', label=ent['label'])
            core_entities.add(ent['text'])
        
        for title, sentences in zip(all_context['title'], all_context['sentences']):
            for sentence in sentences[:5]:
                entities = self.extract_entities(sentence)
                
                sentence_entities = []
                for ent in entities:
                    if ent['text'] not in G:
                        node_type = 'peripheral'
                        G.add_node(ent['text'], node_type=node_type, label=ent['label'])
                    sentence_entities.append(ent['text'])
                
                for i, ent1 in enumerate(sentence_entities):
                    for ent2 in sentence_entities[i+1:]:
                        if ent1 != ent2:
                            G.add_edge(ent1, ent2)
        
        for node in G.nodes():
            if node in core_entities:
                G.nodes[node]['is_core'] = True
            else:
                G.nodes[node]['is_core'] = False
        
        return G
    
    def compute_graph_metrics(self, G: nx.Graph, core_entities: Set[str]) -> Dict:
        metrics = {}
        
        try:
            core_subgraph = G.subgraph(core_entities)
            if len(core_subgraph) > 1 and nx.is_connected(core_subgraph):
                core_path_nodes = set(core_entities)
            else:
                core_path_nodes = core_entities
        except:
            core_path_nodes = core_entities
        
        for node in G.nodes():
            node_metrics = {
                'degree': G.degree(node),
                'is_core': node in core_entities,
                'distance_to_core': float('inf'),
                'betweenness_centrality': 0,
                'closeness_centrality': 0,
                'clustering_coefficient': 0
            }
            
            if node not in core_entities:
                min_distance = float('inf')
                for core_node in core_entities:
                    try:
                        distance = nx.shortest_path_length(G, node, core_node)
                        min_distance = min(min_distance, distance)
                    except nx.NetworkXNoPath:
                        continue
                node_metrics['distance_to_core'] = min_distance
            else:
                node_metrics['distance_to_core'] = 0
            
            metrics[node] = node_metrics
        
        if len(G) > 2:
            betweenness = nx.betweenness_centrality(G)
            for node, centrality in betweenness.items():
                metrics[node]['betweenness_centrality'] = centrality
            
            if nx.is_connected(G):
                closeness = nx.closeness_centrality(G)
                for node, centrality in closeness.items():
                    metrics[node]['closeness_centrality'] = centrality
            
            clustering = nx.clustering(G)
            for node, coeff in clustering.items():
                metrics[node]['clustering_coefficient'] = coeff
        
        return metrics
    
    def select_influencer_nodes(self, G: nx.Graph, metrics: Dict, strategy: str) -> List[Tuple[str, Dict]]:
        non_core_nodes = [(node, data) for node, data in metrics.items() 
                         if not data['is_core'] and data['distance_to_core'] < float('inf')]
        
        if not non_core_nodes:
            return []
        
        if strategy == 'high_degree':
            sorted_nodes = sorted(non_core_nodes, key=lambda x: x[1]['degree'], reverse=True)
        elif strategy == 'close_distance':
            sorted_nodes = [n for n in non_core_nodes if n[1]['distance_to_core'] == 1]
        elif strategy == 'far_distance':
            sorted_nodes = [n for n in non_core_nodes if n[1]['distance_to_core'] >= 3]
        elif strategy == 'high_centrality':
            sorted_nodes = sorted(non_core_nodes, key=lambda x: x[1]['betweenness_centrality'], reverse=True)
        else:
            sorted_nodes = non_core_nodes
        
        return sorted_nodes[:5]
    
    def visualize_graph_stats(self, G: nx.Graph, metrics: Dict) -> Dict:
        stats = {
            'num_nodes': len(G),
            'num_edges': len(G.edges()),
            'num_core_nodes': sum(1 for n, d in metrics.items() if d['is_core']),
            'avg_degree': np.mean([d['degree'] for d in metrics.values()]),
            'max_degree': max([d['degree'] for d in metrics.values()]),
            'avg_distance_to_core': np.mean([d['distance_to_core'] for d in metrics.values() 
                                            if not d['is_core'] and d['distance_to_core'] < float('inf')]),
            'connected_components': nx.number_connected_components(G)
        }
        return stats