import sys
sys.path.append('.')
sys.path.append('..')


import os
import json
import torch
import argparse
import pandas as pd
from datetime import datetime

# Import from our modules
from fairness_evaluation import setup_model, generate_continuations, evaluate_toxicity
from fairness_evaluation import evaluate_gender_regard, evaluate_honest_score
from fairness_evaluation_datasets import load_all_datasets

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fairness of Llama model")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-hf", 
                        help="HuggingFace model ID or local path")
    parser.add_argument("--toxicity_samples", type=int, default=100, 
                        help="Number of samples for toxicity evaluation")
    parser.add_argument("--gender_samples", type=int, default=50, 
                        help="Number of samples for gender evaluation")
    parser.add_argument("--honest_samples", type=int, default=50, 
                        help="Number of samples for HONEST evaluation")
    parser.add_argument("--output_dir", type=str, default="./results", 
                        help="Directory to save evaluation results")
    parser.add_argument("--max_length", type=int, default=50, 
                        help="Maximum length for text generation")
    return parser.parse_args()

def save_results(results, output_dir, model_id):
    """
    Save evaluation results to disk
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a timestamp for the results file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model_id.split("/")[-1]
    output_file = os.path.join(output_dir, f"{model_name}_{timestamp}_fairness_evaluation.json")
    
    # Save results as JSON
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
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
    print(f"Summary saved to {csv_file}")

def main():
    # Parse command line arguments
    args = parse_args()
    
    print(f"Starting fairness evaluation of {args.model_id}")
    print("=" * 50)
    
    # Setup model
    text_generation, tokenizer = setup_model(args.model_id)
    
    # Load datasets
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
    save_results(evaluation_results, args.output_dir, args.model_id)

if __name__ == "__main__":
    main()