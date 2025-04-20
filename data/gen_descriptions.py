import openai
import pandas as pd
import random
import string
import os
from typing import List, Dict
import argparse
from dotenv import load_dotenv

starting_id = 0
def new_id() -> int:
    """Return a new ID that is the next integer in the sequence"""
    global starting_id
    temp = starting_id
    starting_id += 1
    return temp

def setup_openai_client(api_key: str):
    """Set up and return an OpenAI client."""
    client = openai.OpenAI(api_key=api_key)
    return client

def read_existing_descriptions(file_path: str) -> set:
    """Read and return existing descriptions from a CSV file using pandas."""
    if not os.path.exists(file_path):
        return set()
    
    df = pd.read_csv(file_path)
    if 'description' in df.columns:
        return set(df['description'].str.lower())
    return set()

def get_prompt_for_category(category: str, count: int, max_length: int, target_avg_length: int) -> str:
    """Return a specific prompt based on the category."""
    
    if category == "landscapes":
        return f"""
        Generate {count} short, generic descriptions of landscapes.
        
        Requirements:
        Each landscape description should be concise, around {target_avg_length} characters on average
        No description should exceed {max_length} characters
        Do NOT include any brand names, trademarks, or personal names
        Do NOT include any people, even generically
        Descriptions should be varied and creative
        Only provide the descriptions, one per line, with no numbering or additional text
        Focus on natural scenes, vistas, and environments
        
        Examples:
        a purple forest at dusk
        a lighthouse overlooking the ocean
        a green lagoon under a cloudy sky
        a snowy plain
        a starlit night over snow-covered peaks
        """
    
    elif category == "abstract":
        return f"""
        Generate {count} short, generic descriptions of abstract art or geometric compositions.
        
        Requirements:
        Each abstract description should be concise, around {target_avg_length} characters on average
        No description should exceed {max_length} characters
        Do NOT include any brand names, trademarks, or personal names
        Focus on geometric shapes, patterns, and colors
        Be creative with color combinations and spatial arrangements
        Only provide the descriptions, one per line, with no numbering or additional text
        
        Examples:
        crimson rectangles forming a chaotic grid
        purple pyramids spiraling around a bronze cone
        magenta trapezoids layered on a transluscent silver sheet
        khaki triangles and azure crescents
        a maroon dodecahedron interwoven with teal threads
        """
    
    elif category == "fashion":
        return f"""
        Generate {count} short, generic descriptions of fashion items and clothing.
        
        Requirements:
        Each fashion description should be concise, around {target_avg_length} characters on average
        No description should exceed {max_length} characters
        Do NOT include any brand names, trademarks, or personal names
        Do NOT include any people, even generically
        Focus on clothing items, accessories, fabrics, patterns, and colors
        Be specific about materials, cuts, and design features
        Only provide the descriptions, one per line, with no numbering or additional text
        
        Examples:
        gray wool coat with a faux fur collar
        burgundy corduroy pants with patch pockets and silver buttons
        orange corduroy overalls
        a purple silk scarf with tassel trim
        black and white checkered pants
        """
    
    else:
        # Generic prompt for additional categories
        return f"""
        Generate {count} short, generic descriptions of {category}.
        
        Requirements:
        - Each description should be concise, around {target_avg_length} characters on average
        - No description should exceed {max_length} characters
        - Do NOT include any brand names, trademarks, or personal names
        - Do NOT include any people, even generically
        - Descriptions should be varied and creative
        - Only provide the descriptions, one per line, with no numbering or additional text
        """

def generate_descriptions(
    client,
    categories: List[str],
    count_per_category: int,
    max_length: int = 200,
    target_avg_length: int = 50,
    existing_descriptions: set = set()
) -> Dict[str, List[str]]:
    """Generate descriptions for each category using GPT-4o mini with separate prompts."""
    
    results = {category: [] for category in categories}
    
    for category in categories:
        print(f"Generating {count_per_category} descriptions for category: {category}")
        
        # Get the specific prompt for this category
        system_prompt = get_prompt_for_category(
            category=category,
            count=count_per_category,
            max_length=max_length,
            target_avg_length=target_avg_length
        )
        
        unique_descriptions = set()
        
        try:
            while len(unique_descriptions) < count_per_category:
                remaining = count_per_category - len(unique_descriptions)
                print(f"Generating {remaining} more unique descriptions for {category}...")
                
                # Make the API call
                response = client.chat.completions.create(
                    model="gpt-4o-mini",  # Using GPT-4o mini as requested
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Generate {remaining} {category} descriptions."}
                    ],
                    temperature=0.7,
                    max_tokens=8000
                )
                
                # Process the response
                content = response.choices[0].message.content
                descriptions = [line.strip() for line in content.split('\n') if line.strip()]
                
                # Filter out any descriptions that are too long, already existing, or duplicates
                for desc in descriptions:
                    desc_lower = desc.lower()
                    if (len(desc) <= max_length and 
                        desc_lower not in {d.lower() for d in unique_descriptions} and
                        desc_lower not in existing_descriptions):
                        unique_descriptions.add(desc)
                    
                    if len(unique_descriptions) >= count_per_category:
                        break
            
            # Convert to list
            results[category] = list(unique_descriptions)
            
        except Exception as e:
            print(f"Error generating descriptions for {category}: {e}")
    
    return results

def write_to_csv_pandas(
    descriptions_dict: Dict[str, List[str]],
    output_file: str,
    append: bool = False
) -> None:
    """Write or append the generated descriptions to a CSV file using pandas."""
    
    # Create a list of dictionaries for our new data
    data = []
    for category, descriptions in descriptions_dict.items():
        for description in descriptions:
            data.append({
                "id": new_id(),
                "description": description,
                "category": category
            })
    
    # Create a DataFrame from our data
    new_df = pd.DataFrame(data)
    
    # Append or create new file
    if append and os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_csv(output_file, index=False)
        print(f"Appended {len(new_df)} descriptions to {output_file}")
    else:
        new_df.to_csv(output_file, index=False)
        print(f"Wrote {len(new_df)} descriptions to {output_file}")

def main():
    csv_path = "data/descriptions.csv"
    output_file = "data/descriptions.csv"
    append = True
    count = 50
    categories = ["landscapes", "abstract", "fashion"]
    
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("OPENAI_API_KEY is not set")
    
    # Set up the OpenAI client
    client = setup_openai_client(api_key)
    
    # If appending, read existing descriptions to avoid duplicates
    existing_descriptions = set()
    if append and os.path.exists(csv_path):
        global starting_id
        starting_id = pd.read_csv(csv_path)["id"].max() + 1
        existing_descriptions = read_existing_descriptions(csv_path)
        print(f"Found {len(existing_descriptions)} existing descriptions")
    
    # Generate the descriptions
    descriptions_dict = generate_descriptions(
        client=client,
        categories=categories,
        count_per_category=count,
        existing_descriptions=existing_descriptions
    )
    
    # Write to CSV using pandas
    write_to_csv_pandas(
        descriptions_dict=descriptions_dict,
        output_file=output_file,
        append=append
    )

if __name__ == "__main__":
    main()