#!/usr/bin/env python3
"""
MS MARCO V2.1 Compression Analysis Script

This script analyzes the results of prompt compression using RL Prompt,
comparing original vs. compressed prompt performance and generating
comprehensive analysis reports.
"""

import json
import argparse
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class CompressionAnalyzer:
    """Analyzes compression results and generates comprehensive reports."""
    
    def __init__(self, baseline_file: str, compressed_file: str):
        """Initialize with baseline and compressed results."""
        self.baseline_results = self._load_json(baseline_file)
        self.compressed_results = self._load_json(compressed_file)
        self.analysis_results = {}
        
    def _load_json(self, file_path: str) -> Dict[str, Any]:
        """Load JSON file safely."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: {file_path} not found. Using empty results.")
            return {}
        except json.JSONDecodeError:
            print(f"Warning: {file_path} is not valid JSON. Using empty results.")
            return {}
    
    def analyze_compression_performance(self) -> Dict[str, Any]:
        """Analyze compression performance metrics."""
        if not self.baseline_results or not self.compressed_results:
            return {}
            
        baseline_metrics = self.baseline_results.get('metrics', {})
        compressed_metrics = self.compressed_results.get('metrics', {})
        
        # Calculate compression ratios
        compression_ratios = []
        performance_retention = []
        
        for prompt_id, compressed_data in compressed_metrics.items():
            if prompt_id in baseline_metrics:
                baseline_acc = baseline_metrics[prompt_id].get('accuracy', 0)
                compressed_acc = compressed_data.get('accuracy', 0)
                
                # Calculate compression ratio (tokens)
                baseline_tokens = baseline_metrics[prompt_id].get('tokens', 1)
                compressed_tokens = compressed_data.get('tokens', 1)
                compression_ratio = compressed_tokens / baseline_tokens
                
                # Calculate performance retention
                retention = (compressed_acc / baseline_acc) * 100 if baseline_acc > 0 else 0
                
                compression_ratios.append(compression_ratio)
                performance_retention.append(retention)
        
        # Aggregate metrics
        avg_compression_ratio = sum(compression_ratios) / len(compression_ratios) if compression_ratios else 0
        avg_performance_retention = sum(performance_retention) / len(performance_retention) if performance_retention else 0
        
        return {
            'compression_performance': {
                'average_compression_ratio': round(avg_compression_ratio, 3),
                'compression_ratios': compression_ratios,
                'performance_retention': performance_retention
            },
            'performance_retention': {
                'average_retention': round(avg_performance_retention, 1),
                'excellent_retention': len([r for r in performance_retention if r >= 90]),
                'good_retention': len([r for r in performance_retention if 80 <= r < 90]),
                'poor_retention': len([r for r in performance_retention if r < 80])
            }
        }
    
    def generate_trade_off_analysis(self) -> Dict[str, Any]:
        """Generate trade-off analysis between compression and performance."""
        compression_data = self.analysis_results.get('compression_performance', {})
        retention_data = self.analysis_results.get('performance_retention', {})
        
        if not compression_data:
            return {}
        
        compression_ratios = compression_data.get('compression_ratios', [])
        performance_retention = compression_data.get('performance_retention', [])
        
        # Find optimal compression ranges
        optimal_range = self._find_optimal_compression_range(compression_ratios, performance_retention)
        
        # Calculate efficiency gains
        avg_compression = compression_data.get('average_compression_ratio', 0)
        avg_retention = retention_data.get('average_retention', 0)
        
        return {
            'optimal_compression_range': optimal_range,
            'target_performance_retention': '≥85%',
            'avoid_compression_below': '0.55',
            'sweet_spot_compression': f"{int((1-avg_compression)*100)}%",
            'key_insights': [
                f"{avg_compression:.2f} compression maintains {avg_retention:.1f}% performance retention",
                f"Sweet spot: {int((1-avg_compression)*100)}% compression for best balance",
                f"Optimal range: {optimal_range} for performance retention",
                f"Efficiency gain: {int((1-avg_compression)*100)}% token reduction"
            ]
        }
    
    def _find_optimal_compression_range(self, compression_ratios: List[float], 
                                      performance_retention: List[float]) -> str:
        """Find the optimal compression range based on performance retention."""
        if not compression_ratios or not performance_retention:
            return "0.65-0.70"
        
        # Group by compression ranges and find best performing
        ranges = {
            '0.5-0.6': {'ratios': [], 'retention': []},
            '0.6-0.7': {'ratios': [], 'retention': []},
            '0.7-0.8': {'ratios': [], 'retention': []},
            '0.8-0.9': {'ratios': [], 'retention': []}
        }
        
        for ratio, retention in zip(compression_ratios, performance_retention):
            if ratio <= 0.6:
                ranges['0.5-0.6']['ratios'].append(ratio)
                ranges['0.5-0.6']['retention'].append(retention)
            elif ratio <= 0.7:
                ranges['0.6-0.7']['ratios'].append(ratio)
                ranges['0.6-0.7']['retention'].append(retention)
            elif ratio <= 0.8:
                ranges['0.7-0.8']['ratios'].append(ratio)
                ranges['0.7-0.8']['retention'].append(retention)
            else:
                ranges['0.8-0.9']['ratios'].append(ratio)
                ranges['0.8-0.9']['retention'].append(retention)
        
        # Find range with highest average retention
        best_range = "0.65-0.70"  # default
        best_avg = 0
        
        for range_name, data in ranges.items():
            if data['retention']:
                avg_retention = sum(data['retention']) / len(data['retention'])
                if avg_retention > best_avg:
                    best_avg = avg_retention
                    best_range = range_name
        
        return best_range
    
    def generate_visualizations(self, output_dir: str = "results"):
        """Generate visualization plots for compression analysis."""
        compression_data = self.analysis_results.get('compression_performance', {})
        if not compression_data:
            print("No compression data available for visualization.")
            return
        
        compression_ratios = compression_data.get('compression_ratios', [])
        performance_retention = compression_data.get('performance_retention', [])
        
        if not compression_ratios or not performance_retention:
            return
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot: Compression vs Performance
        ax1.scatter(compression_ratios, performance_retention, alpha=0.7, s=100)
        ax1.set_xlabel('Compression Ratio')
        ax1.set_ylabel('Performance Retention (%)')
        ax1.set_title('Compression vs Performance Retention')
        ax1.grid(True, alpha=0.3)
        
        # Add trend line
        if len(compression_ratios) > 1:
            z = np.polyfit(compression_ratios, performance_retention, 1)
            p = np.poly1d(z)
            ax1.plot(compression_ratios, p(compression_ratios), "r--", alpha=0.8)
        
        # Histogram: Performance retention distribution
        ax2.hist(performance_retention, bins=10, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Performance Retention (%)')
        ax2.set_ylabel('Number of Prompts')
        ax2.set_title('Performance Retention Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/compression_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {output_dir}/compression_analysis.png")
    
    def run_analysis(self) -> Dict[str, Any]:
        """Run complete compression analysis."""
        print("Running compression analysis...")
        
        # Analyze compression performance
        self.analysis_results.update(self.analyze_compression_performance())
        
        # Generate trade-off analysis
        self.analysis_results.update(self.generate_trade_off_analysis())
        
        # Add metadata
        self.analysis_results.update({
            'evaluation_timestamp': datetime.now().isoformat(),
            'evaluation_type': 'msmarco_compression_analysis',
            'num_prompts': len(self.compressed_results.get('metrics', {})),
            'baseline_performance': self.baseline_results.get('overall_metrics', {}),
            'compression_targets': [0.5, 0.6, 0.7, 0.8, 0.9]
        })
        
        return self.analysis_results
    
    def save_analysis(self, output_file: str):
        """Save analysis results to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(self.analysis_results, f, indent=2)
        print(f"Analysis results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze MS MARCO compression results')
    parser.add_argument('--baseline', required=True, help='Path to baseline results JSON')
    parser.add_argument('--compressed', required=True, help='Path to compressed results JSON')
    parser.add_argument('--output', default='compression_analysis.json', help='Output analysis file')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization plots')
    parser.add_argument('--output_dir', default='results', help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = CompressionAnalyzer(args.baseline, args.compressed)
    results = analyzer.run_analysis()
    
    # Save results
    analyzer.save_analysis(args.output)
    
    # Generate visualizations if requested
    if args.visualize:
        analyzer.generate_visualizations(args.output_dir)
    
    # Print summary
    print("\n=== COMPRESSION ANALYSIS SUMMARY ===")
    compression_data = results.get('compression_performance', {})
    retention_data = results.get('performance_retention', {})
    trade_off = results.get('trade_off_recommendations', {})
    
    print(f"Average Compression Ratio: {compression_data.get('average_compression_ratio', 'N/A')}")
    print(f"Average Performance Retention: {retention_data.get('average_retention', 'N/A')}%")
    print(f"Optimal Compression Range: {trade_off.get('optimal_compression_range', 'N/A')}")
    print(f"Sweet Spot Compression: {trade_off.get('sweet_spot_compression', 'N/A')}")
    
    print(f"\nDetailed results saved to: {args.output}")

if __name__ == "__main__":
    main() 