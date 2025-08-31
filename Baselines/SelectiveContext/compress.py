import os
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from compressor.selective_context import SelectiveContext

def compress_dataset(args):
    """Compress the MS-MARCO dataset using SelectiveContext"""
    print("Loading MS-MARCO dataset...")
    dataset = load_dataset('microsoft/ms_marco', 'v2.1', split='test')
    
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
            question = item['query']
            passages = item['passages']
            passage_texts = passages['passage_text']
            full_context = " ".join(passage_texts)

            # Compress each passage individually
            compressed_passage_texts = []
            context_units_per_passage = []
            kept_masks_per_passage = []

            for passage_text in passage_texts:
                (
                    _, # We dont compress the question
                    compressed_passage, 
                    _, _, _, _, # We don't have answer choices (legacy code from LongBench)
                    unit_mask, 
                    original_units
                ) = compressor.run(
                    question, passage_text
                )
                compressed_passage_texts.append(compressed_passage)
                context_units_per_passage.append(original_units)
                kept_masks_per_passage.append(unit_mask)

            # Reconstruct compressed context for ratio calculation
            compressed_context_full = " ".join(compressed_passage_texts)
            
            # Create compressed item
            compressed_item = {
                "query_id": item["query_id"],
                "query_type": item["query_type"],
                "query": item["query"],
                "answers": item["answers"],
                "wellFormedAnswers": item.get("wellFormedAnswers", []),
                "passages": {
                    "passage_text": compressed_passage_texts,
                    "is_selected": passages["is_selected"],
                    "url": passages["url"],
                    "context_unit": context_units_per_passage,
                    "kept_context_unit_mask": kept_masks_per_passage,
                },
                "original_passages": item["passages"],
                "compression_words_ratio": len(compressed_context_full.split()) / len(full_context.split()) if full_context.split() else 1.0,
                "compression_characters_ratio": len(compressed_context_full) / len(full_context) if full_context else 1.0,
                "compression_rate": args.compressor_reduce_ratio,
                "compression_level": args.compressor_reduce_level,
                "compression_error": ""
            }
            
            compressed_data.append(compressed_item)
            
        except Exception as e:
            print(f"Error processing item {item['query_id']}: {e}")
            # If compression fails, keep original data
            context = " ".join(item['passages']['passage_text'])
            compressed_item = {
                "query_id": item["query_id"],
                "query_type": item["query_type"],
                "query": item["query"],
                "answers": item["answers"],
                "wellFormedAnswers": item.get("wellFormedAnswers", []),
                "passages": item["passages"],
                "original_passages": item["passages"],
                "context_unit": [],
                "kept_context_unit_mask": [],
                "compression_words_ratio": 1.0,
                "compression_characters_ratio": 1.0,
                "compression_rate": args.compressor_reduce_ratio,
                "compression_level": args.compressor_reduce_level,
                "compression_error": str(e)
            }
            compressed_data.append(compressed_item)
    
    # Create and save compressed dataset
    compressed_dataset = Dataset.from_list(compressed_data)
    output_path = os.path.join(args.save_dir, "MS-MARCO-compressed")
    compressed_dataset.save_to_disk(output_path)
    
    # Save metadata
    metadata = {
        "original_dataset": "MS-MARCO",
        "compressed_dataset": "MS-MARCO-compressed",
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
    parser = argparse.ArgumentParser(description="Compress MS-MARCO dataset using SelectiveContext")
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
