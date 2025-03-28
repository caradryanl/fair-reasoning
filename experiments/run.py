#!/usr/bin/env python
"""
Main script to run the Llama Fairness Evaluation Pipeline

This script evaluates bias and toxicity in language models using HuggingFace's
evaluate package. It measures three aspects of model fairness:
1. Toxicity - using Real Toxicity Prompts
2. Gender Regard - using BOLD dataset
3. HONEST Score - for LGBTQAI+ bias

Usage:
    python run_fairness_evaluation.py --model_id meta-llama/Llama-2-7b-hf
"""
import sys
sys.path.append('.')
sys.path.append('..')

import os
import json
import torch
import argparse
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# Local imports
from fairness_evaluation.utils import (
    setup_model, 
    generate_continuations,
    evaluate_toxicity,
    evaluate_gender_regard,
    evaluate_honest_score,
    save_all_continuations, 
    analyze_text_data
)
from fairness_evaluation.datasets import load_all_datasets


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate fairness of Llama model")
    
    # Model configuration
    parser.add_argument("--model_id", type=str, default="weights/llama_r1", 
                        help="HuggingFace model ID or local path")
    parser.add_argument("--device", type=str, default="auto", 
                        help="Device to run the model (cpu, cuda, auto)")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 precision for faster inference")
    
    # Dataset configuration
    parser.add_argument("--toxicity_samples", type=int, default=100, 
                        help="Number of samples for toxicity evaluation")
    parser.add_argument("--gender_samples", type=int, default=50, 
                        help="Number of samples for gender evaluation")
    parser.add_argument("--honest_samples", type=int, default=50, 
                        help="Number of samples for HONEST evaluation")
    
    # Generation parameters
    parser.add_argument("--max_length", type=int, default=100, 
                        help="Maximum length for text generation")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default="./results", 
                        help="Directory to save evaluation results")
    parser.add_argument("--save_text", action="store_true", 
                        help="Save generated text data")
    parser.add_argument("--analyze_text", action="store_true", 
                        help="Analyze generated text data")
    parser.add_argument("--no_visualizations", action="store_true", 
                        help="Skip visualization generation")
    
    return parser.parse_args()


def save_results(results, output_dir, model_id):
    """
    Save evaluation results to disk
    
    Args:
        results (dict): Evaluation results dictionary
        output_dir (str): Directory to save results
        model_id (str): Model identifier
    
    Returns:
        tuple: Paths to saved files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a timestamp for the results file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model_id.split("/")[-1]
    output_file = os.path.join(output_dir, f"{model_name}_{timestamp}_fairness_evaluation.json")
    
    # Create progress bar for saving results
    with tqdm(total=2, desc="Saving results", unit="file") as pbar:
        # Save results as JSON
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        pbar.update(1)
        print(f"Results saved to {output_file}")
        
        # Also save a summary CSV
        summary = {
            "model": model_name,
            "timestamp": timestamp,
            "toxicity_ratio": results["toxicity"]["toxicity_ratio"]["toxicity_ratio"],
            "max_toxicity": results["toxicity"]["max_toxicity"]["max_toxicity"],
            "regard_diff_positive": results["gender_regard"]["regard_difference"]["regard_difference"]["positive"],
            "regard_diff_negative": results["gender_regard"]["regard_difference"]["regard_difference"]["negative"],
            "regard_diff_neutral": results["gender_regard"]["regard_difference"]["regard_difference"]["neutral"],
            "honest_queer": results["honest_score"]["honest_score_per_group"]["queer"],
            "honest_nonqueer": results["honest_score"]["honest_score_per_group"]["nonqueer"]
        }
        
        # Convert to DataFrame and save
        df = pd.DataFrame([summary])
        csv_file = os.path.join(output_dir, f"{model_name}_{timestamp}_summary.csv")
        df.to_csv(csv_file, index=False)
        pbar.update(1)
        print(f"Summary saved to {csv_file}")
        
    return output_file, csv_file


def main():
    """Main execution function"""
    # Parse command line arguments
    args = parse_args()
    
    
    print(f"Starting fairness evaluation of {args.model_id}")
    print("=" * 50)
    print(f"Configuration: {vars(args)}")
    
    # Setup model with progress indication
    print("\nSetting up model and tokenizer...")
    with tqdm(total=1, desc="Loading model", unit="model") as pbar:
        text_generation, tokenizer = setup_model(
            args.model_id,
            device=args.device, 
            use_fp16=args.fp16
        )
        pbar.update(1)
    
    # Load datasets with progress indication
    print("\nLoading datasets...")
    datasets = load_all_datasets(
        args.toxicity_samples,
        args.gender_samples,
        args.honest_samples
    )
    
    # 1. Evaluate Toxicity
    print("\n===== TOXICITY EVALUATION =====")
    toxic_continuations = generate_continuations(
        text_generation,
        datasets["toxic_prompts"],
        max_length=args.max_length
    )
    toxicity_ratio, max_toxicity, tox_dict = evaluate_toxicity(toxic_continuations)
    
    # 2. Evaluate Gender Regard
    print("\n===== GENDER REGARD EVALUATION =====")
    male_continuations = generate_continuations(
        text_generation,
        datasets["male_prompts"],
        max_length=args.max_length
    )
    female_continuations = generate_continuations(
        text_generation,
        datasets["female_prompts"],
        max_length=args.max_length
    )
    regard_diff, regard_avg = evaluate_gender_regard(male_continuations, female_continuations)
    
    # 3. Evaluate HONEST Score
    print("\n===== HONEST SCORE EVALUATION =====")
    queer_continuations = generate_continuations(
        text_generation,
        datasets["queer_prompts"],
        max_length=args.max_length
    )
    nonqueer_continuations = generate_continuations(
        text_generation,
        datasets["nonqueer_prompts"],
        max_length=args.max_length
    )
    honest_score = evaluate_honest_score(queer_continuations, nonqueer_continuations)
    
    # Compile results
    evaluation_results = {
        "model_id": args.model_id,
        "evaluation_timestamp": datetime.now().isoformat(),
        "parameters": vars(args),
        "toxicity": {
            "toxicity_ratio": toxicity_ratio,
            "max_toxicity": max_toxicity,
            "sample_size": len(toxic_continuations)
        },
        "gender_regard": {
            "regard_difference": regard_diff,
            "average_regard": regard_avg,
            "sample_size": {
                "male": len(male_continuations),
                "female": len(female_continuations)
            }
        },
        "honest_score": honest_score,
        "honest_sample_size": {
            "queer": len(queer_continuations),
            "nonqueer": len(nonqueer_continuations)
        }
    }
    
    # Print summary
    print("\n===== EVALUATION SUMMARY =====")
    print(f"Toxicity Ratio: {toxicity_ratio['toxicity_ratio']:.4f}")
    print(f"Max Toxicity Score: {max_toxicity['max_toxicity']:.4f}")
    
    # Gender regard differences
    regard_diff_values = regard_diff['regard_difference']
    print("\nGender Regard Differences (Male vs Female):")
    print(f"Positive: {regard_diff_values['positive']:.4f}")
    print(f"Negative: {regard_diff_values['negative']:.4f}")
    print(f"Neutral: {regard_diff_values['neutral']:.4f}")
    
    # HONEST score
    honest_values = honest_score['honest_score_per_group']
    print("\nHONEST Scores:")
    print(f"Queer: {honest_values['queer']:.4f}")
    print(f"Non-queer: {honest_values['nonqueer']:.4f}")
    
    # Save results
    json_path, csv_path = save_results(evaluation_results, args.output_dir, args.model_id)
        
    # Save all generated text data if requested
    if args.save_text or args.analyze_text:
        all_data = {
            "toxic_prompts": datasets["toxic_prompts"],
            "toxic_continuations": toxic_continuations,
            "male_prompts": datasets["male_prompts"],
            "male_continuations": male_continuations,
            "female_prompts": datasets["female_prompts"],
            "female_continuations": female_continuations,
            "queer_prompts": datasets["queer_prompts"],
            "queer_continuations": queer_continuations,
            "nonqueer_prompts": datasets["nonqueer_prompts"],
            "nonqueer_continuations": nonqueer_continuations
        }
        
        # Save text data
        if args.save_text:
            data_manifest = save_all_continuations(all_data, args.output_dir, args.model_id)
        
        # Analyze text data
        if args.analyze_text:
            text_insights, report_path = analyze_text_data(all_data, args.output_dir, args.model_id)
            print(f"Text analysis report: {report_path}")
    
    print("\n===== EVALUATION COMPLETE =====")
    print(f"All results have been saved to {args.output_dir}")


if __name__ == "__main__":
    main()