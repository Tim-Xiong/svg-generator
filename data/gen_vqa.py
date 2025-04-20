import pandas as pd
import json
import time
import os
from openai import OpenAI
from tqdm import tqdm  # for progress bar
import dotenv

dotenv.load_dotenv()

# Initialize OpenAI client
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    api_key = input("Enter your OpenAI API key: ")
client = OpenAI(api_key=api_key)

def generate_evaluation_data(description):
    """
    Use GPT-4o mini to generate evaluation questions, choices, and answers for an SVG image description
    """
    prompt = f"""
    Based on the following description of an SVG image:
    "{description}"
    
    Generate 3-5 questions about visual elements that would be in this image, along with multiple-choice options and the correct answers.
    
    For each question:
    1. The question should be answerable by looking at the image that matches the description
    2. Provide 2-4 possible answer choices for each question
    3. Indicate the correct answer that matches the description
    
    Format your response as a JSON object with exactly these three keys:
    - "question": a list of question strings
    - "choices": a list of lists, where each inner list contains the possible choices for the corresponding question
    - "answer": a list of strings, where each string is the correct answer for the corresponding question
    
    Example format:
    {{
        "question": ["Is there a red circle?", "What shape is present?"],
        "choices": [["yes", "no"], ["square", "circle", "triangle", "hexagon"]],
        "answer": ["yes", "circle"]
    }}
    
    Make sure your response is strictly in this JSON format with no additional text.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        # Parse the JSON response
        result = json.loads(response.choices[0].message.content)
        
        # Validate the response structure
        if not all(key in result for key in ["question", "choices", "answer"]):
            print(f"Warning: Response missing required keys for '{description}'")
            return None
        
        # Check that all lists are the same length
        if not (len(result["question"]) == len(result["choices"]) == len(result["answer"])):
            print(f"Warning: Lists in response have inconsistent lengths for '{description}'")
            return None
            
        return result
    
    except Exception as e:
        print(f"Error generating evaluation data for '{description}': {e}")
        return None

def create_evaluation_dataset(csv_path, output_path):
    """
    Process a CSV file with descriptions and create an evaluation dataset
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} descriptions from {csv_path}")
    
    # Initialize lists to store the evaluation data
    ids = []
    questions = []
    choices = []
    answers = []
    
    # Process each row in the CSV
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing descriptions"):
        item_id = row["id"]
        description = row["description"]
        
        # Generate evaluation data
        eval_data = generate_evaluation_data(description)
        
        if eval_data:
            ids.append(item_id)
            questions.append(json.dumps(eval_data["question"]))
            choices.append(json.dumps(eval_data["choices"]))
            answers.append(json.dumps(eval_data["answer"]))
        
        # Sleep briefly to avoid hitting API rate limits
        time.sleep(0.5)
    
    # Create a DataFrame with the evaluation data
    eval_df = pd.DataFrame({
        "id": ids,
        "question": questions,
        "choices": choices,
        "answer": answers
    })
    
    # Save as CSV
    eval_df.to_csv(output_path, index=False)
    print(f"CSV version saved to {output_path}")
    
    return eval_df

def main():
    # Get input/output paths
    input_path = "data/descriptions.csv"
    output_path = "data/eval.csv"
    
    # Create the evaluation dataset
    eval_df = create_evaluation_dataset(input_path, output_path)
    
    # Display sample of the generated dataset
    print("\nSample of generated evaluation data:")
    print(eval_df.head())
    
    # Show stats
    print(f"\nGenerated evaluation data for {len(eval_df)} out of {pd.read_csv(input_path).shape[0]} descriptions")

if __name__ == "__main__":
    main()