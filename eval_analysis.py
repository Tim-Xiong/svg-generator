import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Set style
plt.style.use('ggplot')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)

# Load the data
results_csv = "results/summary_20250421_230054.csv"
results_json = "results/results_20250421_230054.json"

df = pd.read_csv(results_csv)

# Extract category from description if not already available
def extract_category(row):
    """
    Determines the category of an image based on its description or existing category.
    
    Args:
        row: A pandas DataFrame row containing 'category' and 'description' fields
        
    Returns:
        str: The determined category ('fashion', 'landscape', 'abstract', or 'unknown')
    """
    if pd.notna(row['category']) and row['category'] != 'unknown':
        return row['category']
    
    # Try to extract from description
    desc = row['description'].lower()
    if any(keyword in desc for keyword in ['coat', 'pants', 'shirt', 'dress', 'scarf', 'shoes']):
        return 'fashion'
    elif any(keyword in desc for keyword in ['forest', 'beach', 'mountain', 'ocean', 'lake', 'sky']):
        return 'landscape'
    elif any(keyword in desc for keyword in ['rectangle', 'circle', 'triangle', 'shape', 'spiral']):
        return 'abstract'
    else:
        return 'unknown'

# Clean the data
df['category'] = df.apply(extract_category, axis=1)
df['generation_time'] = pd.to_numeric(df['generation_time'], errors='coerce')

# 1. Model Performance Comparison
def plot_model_comparison():
    """
    Creates boxplots comparing model performance across three metrics:
    VQA score, aesthetic score, and fidelity score.
    
    Saves the resulting plot to 'results/model_comparison.png'.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['vqa_score', 'aesthetic_score', 'fidelity_score']
    titles = ['VQA Score', 'Aesthetic Score', 'Fidelity Score']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        sns.boxplot(x='model', y=metric, data=df, ax=axes[i])
        axes[i].set_title(f'{title} by Model')
        axes[i].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png')
    plt.close()

# 2. Category Performance Analysis
def plot_category_performance():
    """
    Creates boxplots showing performance by category and model for three metrics:
    VQA score, aesthetic score, and fidelity score.
    
    Saves the resulting plot to 'results/category_performance.png'.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['vqa_score', 'aesthetic_score', 'fidelity_score']
    titles = ['VQA Score', 'Aesthetic Score', 'Fidelity Score']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        sns.boxplot(x='category', y=metric, hue='model', data=df, ax=axes[i])
        axes[i].set_title(f'{title} by Category and Model')
        axes[i].set_ylim([0, 1])
        if i > 0:
            axes[i].get_legend().remove()
    
    axes[0].legend(title='Model')
    plt.tight_layout()
    plt.savefig('results/category_performance.png')
    plt.close()

# 3. Generation Time Analysis
def plot_generation_time():
    """
    Creates visualizations of generation time analysis:
    1. A boxplot showing generation time by model
    2. Scatter plots showing the relationship between generation time and quality metrics
    
    Saves the resulting plots to 'results/generation_time.png' and 'results/quality_vs_time.png'.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='model', y='generation_time', data=df)
    plt.title('Generation Time by Model')
    plt.ylabel('Time (seconds)')
    plt.tight_layout()
    plt.savefig('results/generation_time.png')
    plt.close()
    
    # Generation time vs quality scatter plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['vqa_score', 'aesthetic_score', 'fidelity_score']
    titles = ['VQA Score', 'Aesthetic Score', 'Fidelity Score']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        for model, color in zip(df['model'].unique(), ['#1f77b4', '#ff7f0e']):
            model_data = df[df['model'] == model]
            axes[i].scatter(model_data['generation_time'], model_data[metric], 
                          alpha=0.6, label=model, c=color)
            
        axes[i].set_title(f'{title} vs. Generation Time')
        axes[i].set_xlabel('Generation Time (seconds)')
        axes[i].set_ylabel(title)
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('results/quality_vs_time.png')
    plt.close()

# 4. Description complexity vs performance
def plot_complexity_performance():
    """
    Analyzes the relationship between description complexity (word count) and 
    performance metrics, creating scatter plots with trend lines.
    
    Saves the resulting plot to 'results/complexity_performance.png'.
    """
    df['description_length'] = df['description'].str.len()
    df['word_count'] = df['description'].str.split().str.len()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['vqa_score', 'aesthetic_score', 'fidelity_score']
    titles = ['VQA Score', 'Aesthetic Score', 'Fidelity Score']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        for model, color in zip(df['model'].unique(), ['#1f77b4', '#ff7f0e']):
            model_data = df[df['model'] == model]
            axes[i].scatter(model_data['word_count'], model_data[metric], 
                          alpha=0.6, label=model, c=color)
            
            # Add trendline
            z = np.polyfit(model_data['word_count'], model_data[metric], 1)
            p = np.poly1d(z)
            axes[i].plot(sorted(model_data['word_count']), p(sorted(model_data['word_count'])), 
                       c=color, linestyle='--')
            
        axes[i].set_title(f'{title} vs. Description Complexity')
        axes[i].set_xlabel('Word Count')
        axes[i].set_ylabel(title)
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('results/complexity_performance.png')
    plt.close()

# 5. Success and failure examples
def analyze_best_worst_examples():
    """
    Identifies and prints the top 10 most successful and least successful generations
    based on fidelity score.
    
    Creates directories for sample SVG and PNG files if they don't exist.
    
    Returns:
        tuple: (success_df, failure_df) DataFrames containing the best and worst examples
    """
    # Create directory for result samples
    Path("results/sample_svg").mkdir(exist_ok=True)
    Path("results/sample_png").mkdir(exist_ok=True)
    
    # Load detailed results
    with open(results_json, 'r') as f:
        results_data = json.load(f)
    
    # Create success/failure dataframes
    success_df = df.nlargest(10, 'fidelity_score')
    failure_df = df.nsmallest(10, 'fidelity_score')
    
    # Print success examples
    print("Top 10 Successful Generations:")
    print(success_df[['model', 'description', 'vqa_score', 'aesthetic_score', 'fidelity_score']].to_string(index=False))
    
    # Print failure examples
    print("\nTop 10 Failed Generations:")
    print(failure_df[['model', 'description', 'vqa_score', 'aesthetic_score', 'fidelity_score']].to_string(index=False))
    
    return success_df, failure_df

# 6. Summary statistics
def print_summary_stats():
    """
    Calculates and prints summary statistics for model performance:
    1. Overall stats by model (mean, std, min, max for each metric)
    2. Performance by category and model
    
    Also creates a radar chart visualizing fidelity scores by category and model,
    saved to 'results/category_radar.png'.
    """
    # Overall stats by model
    model_stats = df.groupby('model').agg({
        'vqa_score': ['mean', 'std', 'min', 'max'],
        'aesthetic_score': ['mean', 'std', 'min', 'max'],
        'fidelity_score': ['mean', 'std', 'min', 'max'],
        'generation_time': ['mean', 'std', 'min', 'max']
    })
    
    print("Overall Model Performance:")
    print(model_stats)
    
    # Stats by category and model
    category_stats = df.groupby(['model', 'category']).agg({
        'vqa_score': 'mean',
        'aesthetic_score': 'mean',
        'fidelity_score': 'mean',
        'generation_time': 'mean'
    }).reset_index()
    
    print("\nPerformance by Category and Model:")
    print(category_stats.to_string())
    
    # Create a radar chart for category performance
    categories = category_stats['category'].unique()
    models = category_stats['model'].unique()
    
    plt.figure(figsize=(10, 8))
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    ax = plt.subplot(111, polar=True)
    
    for model in models:
        model_data = category_stats[category_stats['model'] == model]
        values = []
        for category in categories:
            cat_data = model_data[model_data['category'] == category]
            if not cat_data.empty:
                values.append(cat_data['fidelity_score'].values[0])
            else:
                values.append(0)
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title('Fidelity Score by Category and Model')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('results/category_radar.png')
    plt.close()

# Main analysis function
def run_analysis():
    """
    Main function that runs the complete analysis pipeline:
    1. Creates necessary directories
    2. Generates all visualization plots
    3. Prints summary statistics
    4. Analyzes best and worst examples
    
    All results are saved to the 'results/' directory.
    """
    print("Starting analysis of evaluation results...")
    
    # Create plots directory if it doesn't exist
    Path("results").mkdir(exist_ok=True)
    
    # Generate all plots
    plot_model_comparison()
    plot_category_performance()
    plot_generation_time()
    plot_complexity_performance()
    
    # Print summary statistics
    print_summary_stats()
    
    # Analyze best and worst examples
    success_df, failure_df = analyze_best_worst_examples()
    
    print("\nAnalysis complete. Visualizations saved to 'results/' directory.")

if __name__ == "__main__":
    run_analysis()