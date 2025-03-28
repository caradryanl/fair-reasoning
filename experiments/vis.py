import sys
sys.path.append('.')
sys.path.append('..')


import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize fairness evaluation results")
    parser.add_argument("--results_file", type=str, required=True, 
                        help="Path to the JSON results file")
    parser.add_argument("--output_dir", type=str, default="./visualizations", 
                        help="Directory to save visualizations")
    return parser.parse_args()

def load_results(results_file):
    """
    Load results from JSON file
    """
    with open(results_file, 'r') as f:
        results = json.load(f)
    return results

def plot_toxicity(results, output_dir):
    """
    Create visualizations for toxicity results
    """
    toxicity_ratio = results["toxicity"]["toxicity_ratio"]["toxicity_ratio"]
    max_toxicity = results["toxicity"]["max_toxicity"]["max_toxicity"]
    
    # Create bar chart for toxicity ratio
    plt.figure(figsize=(10, 6))
    plt.bar(['Toxicity Ratio'], [toxicity_ratio], color='darkred')
    plt.ylim(0, 1)
    plt.title(f'Toxicity Ratio: {toxicity_ratio:.4f}')
    plt.ylabel('Ratio of Toxic Continuations')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add horizontal line for reference
    plt.axhline(y=0.1, color='black', linestyle='--', alpha=0.5, label='10% Reference')
    plt.legend()
    
    # Save figure
    toxicity_ratio_file = os.path.join(output_dir, 'toxicity_ratio.png')
    plt.savefig(toxicity_ratio_file, bbox_inches='tight')
    plt.close()
    
    # Create bar chart for max toxicity
    plt.figure(figsize=(10, 6))
    plt.bar(['Max Toxicity'], [max_toxicity], color='darkred')
    plt.ylim(0, 1)
    plt.title(f'Maximum Toxicity Score: {max_toxicity:.4f}')
    plt.ylabel('Toxicity Score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save figure
    max_toxicity_file = os.path.join(output_dir, 'max_toxicity.png')
    plt.savefig(max_toxicity_file, bbox_inches='tight')
    plt.close()
    
    print(f"Toxicity visualizations saved to {output_dir}")

def plot_gender_regard(results, output_dir):
    """
    Create visualizations for gender regard results
    """
    regard_diff = results["gender_regard"]["regard_difference"]["regard_difference"]
    average_regard = results["gender_regard"]["average_regard"]["average_data_regard"]
    average_references = results["gender_regard"]["average_regard"]["average_references_regard"]
    
    # Prepare data for regard difference
    categories = list(regard_diff.keys())
    values = list(regard_diff.values())
    
    # Create bar chart for regard difference
    plt.figure(figsize=(12, 7))
    colors = ['darkblue' if v < 0 else 'darkred' for v in values]
    bars = plt.bar(categories, values, color=colors)
    
    # Add data labels
    for bar in bars:
        height = bar.get_height()
        if height < 0:
            plt.text(bar.get_x() + bar.get_width()/2., height - 0.02,
                    f'{height:.4f}', ha='center', va='top', color='white', fontweight='bold')
        else:
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.title('Gender Regard Differences (Male vs Female)')
    plt.ylabel('Difference in Regard Score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save figure
    regard_diff_file = os.path.join(output_dir, 'gender_regard_difference.png')
    plt.savefig(regard_diff_file, bbox_inches='tight')
    plt.close()
    
    # Create bar chart comparing average regard for male vs female
    categories = list(average_regard.keys())
    male_values = list(average_regard.values())
    female_values = list(average_references.values())
    
    # Set up the figure
    plt.figure(figsize=(12, 7))
    x = np.arange(len(categories))
    width = 0.35
    
    # Plot bars
    plt.bar(x - width/2, male_values, width, label='Male', color='royalblue')
    plt.bar(x + width/2, female_values, width, label='Female', color='lightcoral')
    
    # Add labels and formatting
    plt.ylabel('Average Regard Score')
    plt.title('Average Regard Scores by Gender')
    plt.xticks(x, categories)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save figure
    avg_regard_file = os.path.join(output_dir, 'average_gender_regard.png')
    plt.savefig(avg_regard_file, bbox_inches='tight')
    plt.close()