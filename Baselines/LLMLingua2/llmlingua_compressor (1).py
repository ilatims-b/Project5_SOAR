#!/usr/bin/env python3
"""
LLMLingua2 Compression Module
Handles compression using the three methods and context analysis
"""

import json
import requests
import time
from typing import List, Dict, Any, Tuple
from datasets import load_dataset
from llmlingua import PromptCompressor


class ContextTracker:
    """Track context retention with separators"""
    
    def __init__(self, separator: str):
        self.separator = separator
    
    def prepare_contexts_with_separators(self, contexts: List[str]) -> Tuple[str, Dict[int, Tuple[int, int]]]:
        """Combine contexts with separators and track positions"""
        combined_text = ""
        context_positions = {}
        current_pos = 0
        
        for i, context in enumerate(contexts):
            if i > 0:
                combined_text += f" {self.separator} "
                current_pos += len(f" {self.separator} ")
            
            start_pos = current_pos
            combined_text += context
            current_pos += len(context)
            end_pos = current_pos
            
            context_positions[i] = (start_pos, end_pos)
            
        return combined_text, context_positions
    
    def analyze_context_retention(self, original_text: str, compressed_text: str, 
                                context_positions: Dict[int, Tuple[int, int]]) -> Dict[int, Dict[str, Any]]:
        """Analyze which tokens from each context are retained"""
        cleaned_compressed = compressed_text.replace(self.separator, " ").strip()
        compressed_words = cleaned_compressed.split()
        
        context_stats = {}
        
        for context_id, (start, end) in context_positions.items():
            original_context_text = original_text[start:end]
            original_words = original_context_text.split()
            
            # Count retained words (including repetitions)
            retained_count = 0
            for word in original_words:
                if word.lower() in [w.lower() for w in compressed_words]:
                    retained_count += 1
            
            context_stats[context_id] = {
                'original_length': len(original_words),
                'retained_count': retained_count,
                'retention_ratio': retained_count / max(len(original_words), 1)
            }
        
        return context_stats


class ScaleDownAPI:
    """Simple API client for ScaleDown"""
    
    def __init__(self, api_key: str, base_url: str, model: str = "gemini/gemini-2.0-flash"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
    
    def get_response(self, context: str, prompt: str) -> str:
        """Get response from API"""
        payload = {
            "context": context,
            "model": self.model,
            "scaledown": {"rate": 0},
            "prompt": prompt
        }
        
        headers = {
            'x-api-key': self.api_key,
            'Content-Type': 'application/json'
        }
        #print(self.api_key)
        #print(self.base_url)
        #print(self.model)
        #print(context)
        print(prompt)
        try:
            response = requests.post(self.base_url, headers=headers, data=json.dumps(payload))
            #print(response.json().get('full_response'))
            # Debug response details
            #print(f"Response Status Code: {response.status_code}")
            #print(f"Response Headers: {dict(response.headers)}")
            #print(response.json().get('full_response'))
            #if response.status_code == 200:

            return response.json().get('full_response')
        except Exception as e:
            print(f"API Error: {e}")
        
        #return ""


class LLMLingua2Compressor:
    """Main compression class"""
    
    def __init__(self, config_path: str = "config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize components
        self.context_tracker = ContextTracker(self.config['context_separator'])
        self.api_client = ScaleDownAPI(
            self.config['api_config']['api_key'],
            self.config['api_config']['base_url'],
            self.config['api_config']['model']
        )
        
        # Initialize LLMLingua2 compressor
        print("Initializing LLMLingua2 compressor...")
        self.compressor = PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            use_llmlingua2=True,
            device_map="auto"  # Will use GPU if available
        )
    
    def load_dataset(self) -> List[Dict]:
        """Load and filter MS MARCO dataset"""
        print("Loading MS MARCO dataset...")
        dataset = load_dataset('microsoft/ms_marco', self.config['dataset_config']['version'])['validation']
        
        filtered_examples = []
        for example in dataset:
            if len(filtered_examples) >= self.config['dataset_config']['max_examples']:
                break
                
            if example['query_type'] != self.config['dataset_config']['query_type']:
                continue
                
            if not example['answers'] or not example['answers'][0]:
                continue
                
            if example['answers'][0].lower().strip() in ['no answer', 'no answer present', 'no answer present.']:
                continue
            print(example['query_type'])    
            filtered_examples.append(example)
        
        print(f"Loaded {len(filtered_examples)} examples")
        return filtered_examples
    
    def compress_with_method(self, contexts: List[str], query: str, method_config: Dict) -> Dict:
        """Apply compression method"""
        # Prepare contexts with separators
        combined_context, context_positions = self.context_tracker.prepare_contexts_with_separators(contexts)
        
        # Configure force tokens to preserve separators
        force_tokens = [self.config['context_separator'], '\n', '.', '?']
        
        try:
            if 'rate' in method_config:
                result = self.compressor.compress_prompt(
                    context=[combined_context],
                    question=query,
                    rate=method_config['rate'],
                    force_tokens=force_tokens,
                    use_token_level_filter=True
                )
            elif 'target_token' in method_config:
                result = self.compressor.compress_prompt(
                    context=[combined_context],
                    question=query,
                    target_token=method_config['target_token'],
                    force_tokens=force_tokens,
                    use_token_level_filter=True
                )
            elif 'target_context' in method_config:
                result = self.compressor.compress_prompt(
                    context=contexts,  # Use separate contexts for context-level filtering
                    question=query,
                    target_context=method_config['target_context'],
                    force_tokens=force_tokens,
                    use_context_level_filter=True,
                    use_token_level_filter=True
                )
            
            # Analyze context retention
            context_analysis = self.context_tracker.analyze_context_retention(
                combined_context, result['compressed_prompt'], context_positions
            )
            
            return {
                'compressed_prompt': result['compressed_prompt'],
                'compression_rate': result['rate'],
                'compression_ratio': result['ratio'],
                'original_tokens': result['origin_tokens'],
                'compressed_tokens': result['compressed_tokens'],
                'context_analysis': context_analysis
            }
            
        except Exception as e:
            print(f"Compression failed: {e}")
            return {
                'compressed_prompt': combined_context,
                'compression_rate': "100%",
                'compression_ratio': "1.0x",
                'original_tokens': len(combined_context.split()),  
                'compressed_tokens': len(combined_context.split()),
                'context_analysis': {},
                'error': str(e)
            }
    
    def process_example(self, example: Dict) -> Dict:
        """Process a single example through all compression methods"""
        query = example['query']
        contexts = example['passages']['passage_text']
        ground_truth = example['answers'][0]
        
        result = {
            'query_id': example['query_id'],
            'query': query,
            'ground_truth': ground_truth,
            'contexts': contexts,
            'is_selected': example['passages']['is_selected']
        }
        
        # Original (uncompressed) prompt
        original_context = "\n\n".join(contexts)
        original_response = self.api_client.get_response(original_context, query)
        
        result['original'] = {
            'context': original_context,
            'response': original_response,
            'token_count': len(original_context.split())
        }
        
        # Apply each compression method
        for method_name, method_config in self.config['compression_methods'].items():
            print(f"  Processing {method_name}...")
            
            compression_result = self.compress_with_method(contexts, query, method_config)
            compressed_context = compression_result['compressed_prompt']
            
            # Get API response for compressed context
            compressed_response = self.api_client.get_response(compressed_context, query)
            
            result[method_name] = {
                'compression_result': compression_result,
                'response': compressed_response,
                'context': compressed_context
            }
        
        return result
    
    def run_compression(self, num_examples: int = None) -> List[Dict]:
        """Run compression on dataset"""
        examples = self.load_dataset()
        
        if num_examples:
            examples = examples[:num_examples]
        
        print(f"Processing {len(examples)} examples...")
        
        results = []
        for i, example in enumerate(examples):
            print(f"Processing example {i+1}/{len(examples)}: {example['query'][:50]}...")
            
            try:
                result = self.process_example(example)
                results.append(result)
            except Exception as e:
                print(f"Error processing example {i+1}: {e}")
                continue
        
        return results
    
    def save_results(self, results: List[Dict], output_file: str):
        """Save compression results"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='LLMLingua2 Compression')
    parser.add_argument('--num_examples', type=int, help='Number of examples to process')
    parser.add_argument('--output', default='compression_results.json', help='Output file')
    parser.add_argument('--config', default='config.json', help='Config file')
    
    args = parser.parse_args()
    
    compressor = LLMLingua2Compressor(args.config)
    results = compressor.run_compression(args.num_examples)
    compressor.save_results(results, args.output)


if __name__ == '__main__':
    main()
