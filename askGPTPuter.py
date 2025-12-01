import time
import fire
import os
import json
import numpy as np
import tqdm
import requests
from utils import generatePrompt, generateShot
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.colab import userdata

puterUsername = userdata.get('puter_username')
puterPassword = userdata.get('puter_password')

def _call_puter_api(
    model_name,
    prompt,
    auth_token,
    qid,
    temperature,
    top_p,
):
    response = None
    try_idx = 0
    while True:
        try:
            url = "https://api.puter.com/drivers/call"
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {auth_token}"
            }
            
            payload = {
                "interface": "puter-chat-completion",
                "driver": model_name,
                "method": "complete",
                "args": {
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": temperature,
                    "top_p": top_p
                }
            }
            
            api_response = requests.post(url, headers=headers, json=payload)
            api_response.raise_for_status()
            response = api_response.json()
            break
        except Exception as e:
            print('Retry:', qid, 'Error:', str(e))
            try_idx += 1
            time.sleep(10 + np.random.rand()*10)
            if try_idx > 5:
                raise
    return response

def _get_puter_auth_token(username, password):
    """Authenticate with Puter and get auth token"""
    url = "https://api.puter.com/auth/sign-in"
    
    payload = {
        "username": username,
        "password": password
    }
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    
    auth_data = response.json()
    return auth_data.get('token')

def _process_single_sample(sample, model_name, auth_token, temperature, top_p, n_generations, result_dir):
    """Process a single sample - used for parallel execution"""
    prompt, qid, api = sample
    
    response = _call_puter_api(
        model_name,
        prompt,
        auth_token,
        qid,
        temperature,
        top_p,
        n_generations
    )
    
    response_content = response.get('message', {}).get('content', '')
    
    generation_dict = {
        'api': api, 
        'prompt': prompt, 
        'response': response_content,
        'puter_response': response
    }
    
    with open(os.path.join(result_dir, str(qid) + ".json"), 'w') as fout:
        fout.write(json.dumps(generation_dict))
    
    return qid

def main(
    model_name: str,
    result_dir: str,
    shot_number: int,
    username: str = puterUsername,
    password: str = puterPassword,
    temperature: float = 0.2,
    top_p: float = 0.9,
    n_generations: int = 1,
    question: str = "./dataset/question.jsonl",
    shot_type: str = "example",
    max_workers: int = 5  # New parameter for parallel execution
):
    start_time = time.time()
    os.makedirs(result_dir, exist_ok=True)

    username = puterUsername,
    password = puterPassword,

    print("Authenticating with Puter...")
    auth_token = _get_puter_auth_token(username, password)
    print("Authentication successful!")
    
    print("Generating prompt samples...")
    samples = []
    with open(question, "r") as f:
        lines = f.readlines()
        prompts = [line.strip() for line in lines]
        for i, p in enumerate(prompts):
            check_path = os.path.join(result_dir, str(i) + ".json")
            if os.path.isfile(check_path):
                continue
            p = json.loads(p)
            shots = []
            try:
                shots = generateShot(p['api'], number = shot_number, type = shot_type)
            except:
                print("ERROR: No shots for api", p['api'])
                continue
            prompt = generatePrompt(p['api'], p['question'], shots, type = shot_type)
            samples.append((prompt, i, p['api']))
    print("Total samples:", len(samples))
       
    print(f"Generating responses with {max_workers} parallel workers...")
    
    # Parallel execution
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_single_sample,
                sample,
                model_name,
                auth_token,
                temperature,
                top_p,
                n_generations,
                result_dir
            ): sample for sample in samples
        }
        
        for future in tqdm.tqdm(as_completed(futures), total=len(samples)):
            try:
                qid = future.result()
            except Exception as e:
                sample = futures[future]
                print(f"Error processing sample {sample[1]}: {e}")
    
    print(f"Elapsed time: {time.time() - start_time}")

if __name__ == "__main__":
    fire.Fire(main)