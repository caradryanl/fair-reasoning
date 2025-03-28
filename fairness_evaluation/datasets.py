import random
from datasets import load_dataset
from collections import Counter

def load_toxicity_prompts(num_samples=100):
    """
    Load Real Toxicity Prompts dataset
    """
    print("Loading Real Toxicity Prompts dataset...")
    toxicity_prompts = load_dataset("allenai/real-toxicity-prompts", split="train")
    
    # Sample random prompts
    toxic_sample = get_random_prompts(toxicity_prompts, num_samples)
    
    # Extract text from prompts
    text_prompts = [p['text'] for p in toxic_sample['prompt']]
    
    # Print a few examples
    print(f"Loaded {len(text_prompts)} prompts from Real Toxicity Prompts.")
    print("Sample prompts:")
    for prompt in text_prompts[:3]:
        print(f"- \"{prompt}\"")
    
    return text_prompts

def load_bold_prompts(num_samples=50):
    """
    Load BOLD dataset for gender evaluation
    """
    print("Loading BOLD dataset...")
    bold = load_dataset("AlexaAI/bold", split="train")
    
    # Filter for American actresses and actors
    female_bold = random.sample([p for p in bold if p.get('category') == 'American_actresses'], num_samples)
    male_bold = random.sample([p for p in bold if p.get('category') == 'American_actors'], num_samples)
    
    # Extract prompts
    female_prompts = [p['prompts'][0] for p in female_bold]
    male_prompts = [p['prompts'][0] for p in male_bold]
    
    # Print examples
    print(f"Loaded {len(female_prompts)} female and {len(male_prompts)} male prompts from BOLD.")
    print("Sample female prompts:")
    for prompt in female_prompts[:3]:
        print(f"- \"{prompt}\"")
    print("Sample male prompts:")
    for prompt in male_prompts[:3]:
        print(f"- \"{prompt}\"")
    
    return female_prompts, male_prompts

def load_honest_prompts(num_samples=50):
    """
    Load HONEST dataset for LGBTQAI+ evaluation
    """
    print("Loading HONEST dataset...")
    honest_dataset = load_dataset("MilaNLProc/honest", 'en_queer_nonqueer', split='honest')
    
    # Count categories
    categories = [h['category'] for h in honest_dataset]
    category_counts = Counter(categories)
    print("Categories in HONEST dataset:")
    for category, count in category_counts.items():
        print(f"- {category}: {count} prompts")
    
    # Sample from queer and nonqueer categories
    queer_prompts = random.sample(
        [h['template_masked'].replace(' [M].', '') for h in honest_dataset if h['category'].startswith('queer')], 
        num_samples
    )
    
    nonqueer_prompts = random.sample(
        [h['template_masked'].replace(' [M].', '') for h in honest_dataset if h['category'].startswith('nonqueer')], 
        num_samples
    )
    
    # Print examples
    print(f"Loaded {len(queer_prompts)} queer and {len(nonqueer_prompts)} non-queer prompts from HONEST.")
    print("Sample queer prompts:")
    for prompt in queer_prompts[:3]:
        print(f"- \"{prompt}\"")
    print("Sample non-queer prompts:")
    for prompt in nonqueer_prompts[:3]:
        print(f"- \"{prompt}\"")
    
    return queer_prompts, nonqueer_prompts

def get_random_prompts(dataset, num_examples=100):
    """
    Random sample a subset of prompts from a dataset
    """
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    return dataset[picks]

def load_all_datasets(toxicity_samples=100, gender_samples=50, honest_samples=50):
    """
    Load all datasets for fairness evaluation
    """
    # Set random seed for reproducibility
    random.seed(42)
    
    # Load toxicity prompts
    toxic_prompts = load_toxicity_prompts(toxicity_samples)
    
    # Load gender prompts
    female_prompts, male_prompts = load_bold_prompts(gender_samples)
    
    # Load HONEST prompts
    queer_prompts, nonqueer_prompts = load_honest_prompts(honest_samples)
    
    return {
        "toxic_prompts": toxic_prompts,
        "female_prompts": female_prompts,
        "male_prompts": male_prompts,
        "queer_prompts": queer_prompts,
        "nonqueer_prompts": nonqueer_prompts
    }

if __name__ == "__main__":
    # Test dataset loading
    datasets = load_all_datasets(10, 5, 5)  # Small sample sizes for testing
    print("\nDataset loading complete!")