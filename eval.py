import ast
import argparse
import logging
import numpy as np
import pandas as pd
import json
from datetime import datetime
import os
from PIL import Image
from ml import MLModel
from dl import DLModel
from naive import NaiveModel
import cairosvg
import io
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
from metric import harmonic_mean, VQAEvaluator, AestheticEvaluator
import gc
import torch

# Setup logging
os.makedirs("logs", exist_ok=True)
log_file = f"logs/eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def svg_to_png(svg_code: str, size: tuple = (384, 384)) -> Image.Image:
    """Converts SVG code to a PNG image.
    
    Args:
        svg_code (str): SVG code to convert
        size (tuple, optional): Output image size. Defaults to (384, 384).
        
    Returns:
        PIL.Image.Image: The converted PNG image
    """
    try:
        png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'), output_width=size[0], output_height=size[1])
        return Image.open(io.BytesIO(png_data))
    except Exception as e:
        logger.error(f"Error converting SVG to PNG: {e}")
        # Return a default red circle if conversion fails
        default_svg = """<svg width="384" height="384" viewBox="0 0 256 256"><circle cx="128" cy="128" r="64" fill="red" /></svg>"""
        png_data = cairosvg.svg2png(bytestring=default_svg.encode('utf-8'), output_width=size[0], output_height=size[1])
        return Image.open(io.BytesIO(png_data))

def load_evaluation_data(eval_csv_path: str, descriptions_csv_path: str, index: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load evaluation data from CSV files.
    
    Args:
        eval_csv_path (str): Path to the evaluation CSV
        descriptions_csv_path (str): Path to the descriptions CSV
        index (int, optional): Specific index to load. Defaults to None (load all).
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Loaded evaluation and descriptions dataframes
    """
    logger.info(f"Loading evaluation data from {eval_csv_path} and {descriptions_csv_path}")
    
    with tqdm(total=2, desc="Loading data files") as pbar:
        eval_df = pd.read_csv(eval_csv_path)
        pbar.update(1)
        
        descriptions_df = pd.read_csv(descriptions_csv_path)
        pbar.update(1)
    
    if index is not None:
        eval_df = eval_df.iloc[[index]]
        descriptions_df = descriptions_df.iloc[[index]]
        logger.info(f"Selected description at index {index}: {descriptions_df.iloc[0]['description']}")
    
    return eval_df, descriptions_df

def generate_svg(model: Any, description: str, eval_data: pd.Series, 
                results_dir: str = "results") -> Dict[str, Any]:
    """Generate SVG using the model and save it.
    
    Args:
        model (Any): The model to evaluate (MLModel, DLModel, or NaiveModel)
        description (str): Text description to generate SVG from
        eval_data (pd.Series): Evaluation data with questions, choices, and answers
        results_dir (str): Directory to save results to
        
    Returns:
        Dict[str, Any]: Generation results
    """
    # Create output directories
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f"{results_dir}/svg", exist_ok=True)
    os.makedirs(f"{results_dir}/png", exist_ok=True)
    
    model_name = model.__class__.__name__
    results = {
        "description": description,
        "model_type": model_name,
        "id": eval_data.get('id', '0'),
        "category": description.split(',')[-1] if ',' in description else "unknown",
        "timestamp": datetime.now().isoformat(),
    }
    
    # Generate SVG
    logger.info(f"Generating SVG for description: {description}")
    start_time = datetime.now()
    svg = model.predict(description)
    generation_time = (datetime.now() - start_time).total_seconds()
    results["svg"] = svg
    results["generation_time_seconds"] = generation_time
    
    # Convert SVG to PNG for visual evaluation
    image = svg_to_png(svg)
    results["image_width"] = image.width
    results["image_height"] = image.height
    
    # Save the SVG and PNG for inspection
    output_filename = f"{results['id']}_{model_name}"
    with open(f"{results_dir}/svg/{output_filename}.svg", "w") as f:
        f.write(svg)
    image.save(f"{results_dir}/png/{output_filename}.png")
    
    logger.info(f"Generated SVG for model {model_name} in {generation_time:.2f} seconds")
    
    return results

def evaluate_results(results_list: List[Dict[str, Any]], 
                    vqa_evaluator, aesthetic_evaluator,
                    results_dir: str = "results") -> List[Dict[str, Any]]:
    """Evaluate generated SVGs.
    
    Args:
        results_list (List[Dict[str, Any]]): List of generation results
        vqa_evaluator: VQA evaluation model
        aesthetic_evaluator: Aesthetic evaluation model
        results_dir (str): Directory with saved results
        
    Returns:
        List[Dict[str, Any]]: Evaluation results
    """
    evaluated_results = []
    
    for result in tqdm(results_list, desc="Evaluating results"):
        model_name = result["model_type"]
        output_filename = f"{result['id']}_{model_name}"
        
        # Load the PNG image
        image = Image.open(f"{results_dir}/png/{output_filename}.png").convert('RGB')
        
        try:
            # Parse evaluation data
            questions = result.get("questions")
            choices = result.get("choices")
            answers = result.get("answers")
            
            if not all([questions, choices, answers]):
                logger.warning(f"Missing evaluation data for {output_filename}")
                continue
                
            # Calculate scores
            logger.info(f"Calculating VQA score for model: {model_name}")
            vqa_score = vqa_evaluator.score(questions, choices, answers, image)
            
            logger.info(f"Calculating aesthetic score for model: {model_name}")
            aesthetic_score = aesthetic_evaluator.score(image)
            
            # Calculate final fidelity score using harmonic mean
            instance_score = harmonic_mean(vqa_score, aesthetic_score, beta=0.5)
            
            # Add scores to results
            result["vqa_score"] = vqa_score
            result["aesthetic_score"] = aesthetic_score
            result["fidelity_score"] = instance_score
            
            logger.info(f"VQA Score: {vqa_score:.4f}")
            logger.info(f"Aesthetic Score: {aesthetic_score:.4f}")
            logger.info(f"Final Fidelity Score: {instance_score:.4f}")
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            result["error"] = str(e)
        
        evaluated_results.append(result)
    
    return evaluated_results

def create_model(model_type: str, device: str = "cuda") -> Any:
    """Create a model instance based on model type.
    
    Args:
        model_type (str): Type of model ('ml', 'dl', or 'naive')
        device (str, optional): Device to run model on. Defaults to "cuda".
        
    Returns:
        Any: Model instance
    """
    logger.info(f"Creating {model_type.upper()} model on {device}")
    with tqdm(total=1, desc=f"Loading {model_type.upper()} model") as pbar:
        if model_type.lower() == 'ml':
            model = MLModel(device=device)
        elif model_type.lower() == 'dl':
            model = DLModel(device=device)
        elif model_type.lower() == 'naive':
            model = NaiveModel(device=device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        pbar.update(1)
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Evaluate SVG generation models')
    # dl is not working and takes too long, so we don't evaluate it by default
    parser.add_argument('--models', nargs='+', choices=['ml', 'dl', 'naive'], default=['ml', 'naive'], 
                        help='Models to evaluate (ml, dl, naive)')
    parser.add_argument('--index', type=int, default=None,
                        help='Index of the description to evaluate (default: None, evaluate all)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run models on (default: cuda)')
    parser.add_argument('--eval-csv', type=str, default='data/eval.csv',
                        help='Path to evaluation CSV (default: data/eval.csv)')
    parser.add_argument('--descriptions-csv', type=str, default='data/descriptions.csv',
                        help='Path to descriptions CSV (default: data/descriptions.csv)')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directory to save results (default: results)')
    parser.add_argument('--generate-only', action='store_true',
                        help='Only generate SVGs without evaluation')
    parser.add_argument('--evaluate-only', action='store_true',
                        help='Only evaluate previously generated SVGs')
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Load evaluation data
    eval_df, descriptions_df = load_evaluation_data(args.eval_csv, args.descriptions_csv, args.index)
    
    # Load cached results or initialize new results
    cached_results_file = f"{args.results_dir}/cached_results.json"
    if os.path.exists(cached_results_file) and args.evaluate_only:
        with open(cached_results_file, 'r') as f:
            results = json.load(f)
        logger.info(f"Loaded {len(results)} cached results from {cached_results_file}")
    else:
        results = []
        
    # Step 1: Generate SVGs if not in evaluate-only mode
    if not args.evaluate_only:
        # Process one model at a time to avoid loading/unloading models repeatedly
        for model_type in args.models:
            logger.info(f"Processing all descriptions with model: {model_type}")
            model = create_model(model_type, args.device)
            
            # Process all descriptions with the current model
            for idx, (_, desc_row) in enumerate(descriptions_df.iterrows()):
                description = desc_row['description']
                eval_data = eval_df.iloc[idx]
                
                logger.info(f"Processing description {idx}: {description}")
                
                # Generate SVG and save
                result = generate_svg(model, description, eval_data, args.results_dir)
                
                # Add questions, choices and answers to the result
                try:
                    result["questions"] = ast.literal_eval(eval_data['question'])
                    result["choices"] = ast.literal_eval(eval_data['choices'])
                    result["answers"] = ast.literal_eval(eval_data['answer'])
                except Exception as e:
                    logger.error(f"Error parsing evaluation data: {e}")
                
                results.append(result)
                logger.info(f"Completed SVG generation for description {idx}")
            
            # Free up memory after processing all descriptions with this model
            logger.info(f"Completed all SVG generations for model: {model_type}")
            del model
            if args.device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
        
        # Save the results for later evaluation
        with open(cached_results_file, 'w') as f:
            # Remove image data from results to avoid large JSON files
            clean_results = []
            for result in results:
                clean_result = {k: v for k, v in result.items() if k not in ['image', 'svg']}
                clean_results.append(clean_result)
            json.dump(clean_results, f, indent=2, cls=NumpyEncoder)
        logger.info(f"Saved {len(results)} results to {cached_results_file}")
    
    # Exit if only generating
    if args.generate_only:
        logger.info("Generation completed. Skipping evaluation as requested.")
        return
    
    # Step 2: Evaluate the generated SVGs
    logger.info("Starting evaluation phase")
    
    # Initialize evaluators
    logger.info("Initializing VQA evaluator...")
    vqa_evaluator = VQAEvaluator()
    
    logger.info("Initializing Aesthetic evaluator...")
    aesthetic_evaluator = AestheticEvaluator()
    
    # Evaluate all results
    evaluated_results = evaluate_results(results, vqa_evaluator, aesthetic_evaluator, args.results_dir)
    
    # Save final results
    results_file = f"{args.results_dir}/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        # Remove image data from results to avoid large JSON files
        clean_results = []
        for result in evaluated_results:
            clean_result = {k: v for k, v in result.items() if k not in ['image', 'svg']}
            clean_results.append(clean_result)
        json.dump(clean_results, f, indent=2, cls=NumpyEncoder)
    
    # Create a summary CSV
    summary_data = []
    for result in evaluated_results:
        summary_data.append({
            'model': result['model_type'],
            'description': result['description'],
            'id': result['id'],
            'category': result['category'],
            'vqa_score': result.get('vqa_score', float('nan')),
            'aesthetic_score': result.get('aesthetic_score', float('nan')),
            'fidelity_score': result.get('fidelity_score', float('nan')),
            'generation_time': result.get('generation_time_seconds', float('nan')),
            'timestamp': result['timestamp']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = f"{args.results_dir}/summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    summary_df.to_csv(summary_file, index=False)
    
    # Print summary
    logger.info("\nEvaluation Summary:")
    for result in evaluated_results:
        logger.info(f"Model: {result['model_type']}")
        logger.info(f"Description: {result['description']}")
        logger.info(f"VQA Score: {result.get('vqa_score', 'N/A')}")
        logger.info(f"Aesthetic Score: {result.get('aesthetic_score', 'N/A')}")
        logger.info(f"Fidelity Score: {result.get('fidelity_score', 'N/A')}")
        logger.info(f"Generation Time: {result.get('generation_time_seconds', 'N/A')} seconds")
        logger.info("---")
    
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"Summary saved to: {summary_file}")
    logger.info(f"Log file: {log_file}")

if __name__ == "__main__":
    main()
