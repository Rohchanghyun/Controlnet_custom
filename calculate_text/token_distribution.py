import json
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import os
from transformers import CLIPTokenizer

def analyze_caption_tokens(json_path, output_filename):
    # Load CLIP tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    # Load json file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Calculate token counts for each caption
    token_counts = []
    for item in data:
        caption = item['caption']
        tokens = tokenizer(caption)['input_ids']
        token_counts.append(len(tokens))
    
    # Calculate token frequency
    counter = Counter(token_counts)
    
    # Create bar graph
    plt.figure(figsize=(12, 6))
    
    # Extract token counts and frequencies
    tokens = sorted(list(counter.keys()))
    frequencies = [counter[token] for token in tokens]
    
    # Draw bar graph
    bars = plt.bar(tokens, frequencies)
    
    # Style the graph
    plt.title(f'CLIP Token Distribution - {os.path.basename(json_path)}', fontsize=14)
    plt.xlabel('Number of Tokens', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    # Set integer x-axis
    plt.xticks(tokens)
    
    # Add grid
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add frequency numbers on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    # Print basic statistics
    print(f'\nStatistics for {os.path.basename(json_path)}:')
    print(f'Average token count: {np.mean(token_counts):.2f}')
    print(f'Maximum token count: {max(token_counts)}')
    print(f'Minimum token count: {min(token_counts)}')
    print(f'Total number of captions: {len(token_counts)}')
    
    # Save graph
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    base_path = '/workspace/data/changhyun/dataset/emoji_data/captions_llava'
    output_dir = 'token_distributions'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze each json file
    json_files = [
        'query_captions.json',
        'test_captions.json',
        'train_captions.json'
    ]
    
    for json_file in json_files:
        json_path = os.path.join(base_path, json_file)
        output_filename = os.path.join(output_dir, f'token_distribution_llava_{json_file.replace(".json", ".png")}')
        analyze_caption_tokens(json_path, output_filename)

if __name__ == "__main__":
    main()