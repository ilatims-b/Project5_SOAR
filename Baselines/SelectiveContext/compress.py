import os
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from compressor.selective_context import SelectiveContext

def compress_dataset(args):
    """Compress the LongBench-v2 dataset using SelectiveContext"""
    print("Loading LongBench-v2 dataset...")
    dataset = load_dataset('THUDM/LongBench-v2', split='train')
    
    # Apply dryrun limit if specified
    if args.dryrun > 0:
        dataset = dataset.select(range(min(args.dryrun, len(dataset))))
        print(f"DRYRUN MODE: Processing only {len(dataset)} samples for testing")
    
    # Initialize the compressor
    compressor = SelectiveContext(
        model_type=args.compressor_model_type,
        lang=args.compressor_lang,
        reduce_ratio=args.compressor_reduce_ratio,
        reduce_level=args.compressor_reduce_level
    )
    
    compressed_data = []
    
    print(f"Compressing {len(dataset)} samples...")
    for item in tqdm(dataset):
        try:
            # Extract the data
            question = item['question']
            context = item['context']
            choice_A = item['choice_A']
            choice_B = item['choice_B']
            choice_C = item['choice_C']
            choice_D = item['choice_D']
            
            # Compress using SelectiveContext and get the token mask directly
            (
                compressed_question, compressed_context, 
                compressed_A, compressed_B, compressed_C, compressed_D,
                unit_mask, original_units
            ) = compressor.run(
                question, context, choice_A, choice_B, choice_C, choice_D
            )
            
            # Create compressed item
            compressed_item = {
                "_id": item["_id"],
                "domain": item["domain"],
                "sub_domain": item["sub_domain"],
                "difficulty": item["difficulty"],
                "length": item["length"],
                "question": compressed_question,
                "choice_A": compressed_A,
                "choice_B": compressed_B,
                "choice_C": compressed_C,
                "choice_D": compressed_D,
                "answer": item["answer"],
                "context": compressed_context,
                "original_context": context,  # Keep original for comparison
                "context_unit": original_units,
                "kept_context_unit_mask": unit_mask,
                "compression_words_ratio": len(compressed_context.split()) / len(context.split()) if context.split() else 1.0,
                "compression_error": ""
            }
            
            compressed_data.append(compressed_item)
            
        except Exception as e:
            print(f"Error processing item {item['_id']}: {e}")
            # If compression fails, keep original data
            compressed_item = {
                "_id": item["_id"],
                "domain": item["domain"],
                "sub_domain": item["sub_domain"],
                "difficulty": item["difficulty"],
                "length": item["length"],
                "question": item["question"],
                "choice_A": item["choice_A"],
                "choice_B": item["choice_B"],
                "choice_C": item["choice_C"],
                "choice_D": item["choice_D"],
                "answer": item["answer"],
                "context": item["context"],
                "original_context": item["context"],
                "context_unit": [],
                "kept_context_unit_mask": [],
                "compression_words_ratio": 1.0,
                "compression_error": str(e)
            }
            compressed_data.append(compressed_item)
    
    # Create and save compressed dataset
    compressed_dataset = Dataset.from_list(compressed_data)
    output_path = os.path.join(args.save_dir, "LongBench-v2-compressed")
    compressed_dataset.save_to_disk(output_path)
    
    # Save metadata
    metadata = {
        "original_dataset": "THUDM/LongBench-v2",
        "compressed_dataset": "LongBench-v2-compressed",
        "compressor": "SelectiveContext",
        "compressor_params": {
            "model_type": args.compressor_model_type,
            "lang": args.compressor_lang,
            "reduce_ratio": args.compressor_reduce_ratio,
            "reduce_level": args.compressor_reduce_level
        },
        "total_samples": len(compressed_data),
        "average_compression_words_ratio": sum(item.get("compression_words_ratio", 1.0) for item in compressed_data) / len(compressed_data),
        "save_dir": args.save_dir
    }
    
    metadata_path = os.path.join(args.save_dir, "compression_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Compressed dataset saved to: {output_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Average compression ratio: {metadata['average_compression_words_ratio']:.3f}")

def main():
    parser = argparse.ArgumentParser(description="Compress LongBench-v2 dataset using SelectiveContext")
    parser.add_argument("--save_dir", "-s", type=str, default="results", 
                       help="Directory to save compressed dataset")
    parser.add_argument("--compressor_model_type", "-cmt", type=str, default="microsoft/Phi-3-mini-128k-instruct",
                       help="Model type for SelectiveContext")
    parser.add_argument("--compressor_lang", "-cl", type=str, default="en",
                       help="Language for SelectiveContext")
    parser.add_argument("--compressor_reduce_ratio", "-crr", type=float, default=0.35,
                       help="Reduction ratio for compression")
    parser.add_argument("--compressor_reduce_level", "-crl", type=str, default="phrase",
                       help="Reduction level for compression")
    parser.add_argument("--dryrun", "-d", type=int, default=0,
                       help="Dryrun mode: process only N samples for testing (0 = process all)")
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("Starting dataset compression...")
    compress_dataset(args)
    print("Dataset compression completed!")

if __name__ == "__main__":
    main()
