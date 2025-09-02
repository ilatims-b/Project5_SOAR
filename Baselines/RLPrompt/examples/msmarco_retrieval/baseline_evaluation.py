#!/usr/bin/env python3
"""
MS MARCO V2.1 Baseline Evaluation Script

This script evaluates the baseline performance of original prompts
on the MS MARCO V2.1 dataset to establish performance benchmarks
for compression comparison.
"""

import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Any
import numpy as np
from tqdm import tqdm
import os

class MSMarcoBaselineEvaluator:
    """Evaluates baseline performance on MS MARCO V2.1 dataset."""
    
    def __init__(self, model_name: str = "roberta-base"):
        """Initialize with specified model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # MS MARCO specific settings
        self.max_query_length = 128
        self.max_passage_length = 512
        
    def load_msmarco_data(self, data_file: str) -> List[Dict[str, Any]]:
        """Load MS MARCO V2.1 data from JSONL file."""
        data = []
        with open(data_file, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def encode_query_passage_pair(self, query: str, passage: str) -> Dict[str, torch.Tensor]:
        """Encode query-passage pair for the model."""
        # Combine query and passage with separator
        combined_text = f"{query} [SEP] {passage}"
        
        # Tokenize
        encoding = self.tokenizer(
            combined_text,
            max_length=self.max_query_length + self.max_passage_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device)
        }
    
    def compute_similarity_score(self, query: str, passage: str) -> float:
        """Compute similarity score between query and passage."""
        with torch.no_grad():
            encoding = self.encode_query_passage_pair(query, passage)
            
            # Get model outputs
            outputs = self.model(**encoding)
            
            # Use [CLS] token representation for similarity
            cls_representation = outputs.last_hidden_state[:, 0, :]
            
            # Compute similarity score (cosine similarity)
            similarity = torch.nn.functional.cosine_similarity(
                cls_representation, cls_representation, dim=1
            )
            
            return similarity.item()
    
    def evaluate_retrieval_accuracy(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate retrieval accuracy on MS MARCO data."""
        total_queries = len(data)
        correct_retrievals = 0
        query_metrics = {}
        
        print(f"Evaluating {total_queries} queries...")
        
        for i, sample in enumerate(tqdm(data, desc="Evaluating queries")):
            query = sample['query']
            passages = sample['passages']
            relevant_ids = set(sample['relevant_passage_ids'])
            
            # Compute similarity scores for all passages
            passage_scores = []
            for passage in passages:
                score = self.compute_similarity_score(query, passage['passage_text'])
                passage_scores.append({
                    'passage_id': passage['passage_id'],
                    'score': score,
                    'is_relevant': passage['passage_id'] in relevant_ids
                })
            
            # Sort by similarity score (descending)
            passage_scores.sort(key=lambda x: x['score'], reverse=True)
            
            # Check if top-ranked passage is relevant
            top_passage = passage_scores[0]
            if top_passage['is_relevant']:
                correct_retrievals += 1
            
            # Store metrics for this query
            query_metrics[f"query_{i}"] = {
                'query_text': query,
                'num_passages': len(passages),
                'num_relevant': len(relevant_ids),
                'top_passage_id': top_passage['passage_id'],
                'top_score': top_passage['score'],
                'correct_retrieval': top_passage['is_relevant'],
                'accuracy': 1.0 if top_passage['is_relevant'] else 0.0,
                'tokens': len(self.tokenizer.encode(query)),
                'passage_scores': passage_scores[:5]  # Top 5 scores
            }
        
        # Calculate overall metrics
        overall_accuracy = correct_retrievals / total_queries if total_queries > 0 else 0
        
        # Calculate F1 score
        true_positives = correct_retrievals
        false_positives = total_queries - correct_retrievals
        false_negatives = sum(1 for sample in data if len(sample['relevant_passage_ids']) > 0)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'overall_metrics': {
                'total_queries': total_queries,
                'accuracy': round(overall_accuracy, 4),
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1_score': round(f1_score, 4),
                'correct_retrievals': correct_retrievals
            },
            'query_metrics': query_metrics,
            'model_info': {
                'model_name': self.model.config._name_or_path,
                'device': str(self.device),
                'max_query_length': self.max_query_length,
                'max_passage_length': self.max_passage_length
            }
        }
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save evaluation results to JSON file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Baseline evaluation results saved to: {output_file}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary."""
        metrics = results['overall_metrics']
        model_info = results['model_info']
        
        print("\n" + "="*50)
        print("MS MARCO V2.1 BASELINE EVALUATION SUMMARY")
        print("="*50)
        print(f"Model: {model_info['model_name']}")
        print(f"Device: {model_info['device']}")
        print(f"Total Queries: {metrics['total_queries']}")
        print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"Correct Retrievals: {metrics['correct_retrievals']}")
        print("="*50)

def main():
    parser = argparse.ArgumentParser(description='Evaluate baseline performance on MS MARCO V2.1')
    parser.add_argument('--data', required=True, help='Path to MS MARCO V2.1 data file (JSONL)')
    parser.add_argument('--model', default='roberta-base', help='Model name for evaluation')
    parser.add_argument('--output', default='baseline_results.json', help='Output results file')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to evaluate')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = MSMarcoBaselineEvaluator(args.model)
    
    # Load data
    print(f"Loading MS MARCO V2.1 data from: {args.data}")
    data = evaluator.load_msmarco_data(args.data)
    
    if args.max_samples:
        data = data[:args.max_samples]
        print(f"Limited to {len(data)} samples for evaluation")
    
    # Run evaluation
    print("Starting baseline evaluation...")
    results = evaluator.evaluate_retrieval_accuracy(data)
    
    # Save results
    evaluator.save_results(results, args.output)
    
    # Print summary
    evaluator.print_summary(results)

if __name__ == "__main__":
    main() 