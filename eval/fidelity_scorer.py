import ollama
import json

def score_fidelity(image_path, demographic_profile: dict) -> dict:
    """
    Runs LLaVA offline to check if generated image
    matches the intended demographic profile.
    Returns a score dict.
    """
    prompt = f"""
    Look at this advertisement image carefully.
    The intended demographic profile is: {json.dumps(demographic_profile)}
    
    Score on each dimension from 0-10:
    1. Age match
    2. Gender match  
    3. Ethnicity/skin tone match
    4. Overall demographic fidelity
    
    Respond in JSON only:
    {{"age_score": X, "gender_score": X, "ethnicity_score": X, "overall": X, "notes": "..."}}
    """
    
    response = ollama.chat(
        model="llava",
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [image_path]
        }]
    )
    
    return json.loads(response['message']['content'])