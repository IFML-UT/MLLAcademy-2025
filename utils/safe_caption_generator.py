# ***** Do not modify this file *****
# 
# 1. This function helps set guardrails for the caption generation process.
# 2. ML pipelines can and do change; this function performs a basic environment detection check
#    with error handling to reduce the risk of environment-specific issues between local and cloud runs.
# 3. It ensures that the generated captions are safe and appropriate.
# 4. It uses a pre-trained model to generate captions and checks for profanity.
# ************************************

import os
import sys
import re
import json
from pathlib import Path
from getpass import getpass
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from better_profanity import profanity
from huggingface_hub import InferenceClient
import random

# --- Runtime Environment Detection ---
def get_runtime_env():
    try:
        import google.colab
        return "colab"
    except ImportError:
        return "local"

env = get_runtime_env()
print(f"🧭 Detected environment: {env}")

# --- Hugging Face Token Handling ---
token_path = Path("/content/hf_token.txt") if env == "colab" else Path("../hf_token.txt")

if not token_path.exists():
    print("Please enter your Hugging Face API token:")
    token = getpass("Hugging Face Token: ")
    with open(token_path, "w") as f:
        f.write(token.strip())
    print(f"✅ Hugging Face token saved to {token_path}")
else:
    with open(token_path) as f:
        HF_TOKEN = f.read().strip()
    print(f"✅ Hugging Face token found at {token_path}")

HF_TOKEN = token.strip() if 'token' in locals() else HF_TOKEN

# --- Model Selection Based on Environment ---
if env == "colab":
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    print(f"🧠 We'll be using {model_name} for our LLM")
else:
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    print(f"🧠 We'll be using {model_name} for our LLM")

# --- Initialize HF InferenceClient ---
llm_client = InferenceClient(model=model_name, token=HF_TOKEN)
LLM_MODE = "conversational" if env == "colab" else "text-generation"

# --- Load Embedding Model for Topic Matching ---
print("Loading embedding model for semantic topic matching...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
profanity.load_censor_words()

APPROVED_PROMPT_TOPICS = [
    "final exams", "group projects", "studying late", "Monday mornings",
    "school cafeteria food", "summer break", "forgetting your homework",
    "getting a pop quiz", "trying to stay awake in class", "sports", "coding projects",
    "hackathons", "hanging out with friends", "summer weather", "family vacations",
    "college applications", "video games"
]
topic_embeddings = embedding_model.encode(APPROVED_PROMPT_TOPICS)

# --- Classify Topic Function ---
def classify_topic_from_input(user_input):
    input_embedding = embedding_model.encode(user_input)
    sims = util.cos_sim(input_embedding, topic_embeddings)[0]
    best_match_idx = int(sims.argmax())
    best_score = float(sims[best_match_idx])
    return APPROVED_PROMPT_TOPICS[best_match_idx] if best_score > 0.4 else None

# --- LLM Caption Generator ---
PROMPT_TEMPLATE = (
    "Write a short, funny meme caption about this topic: {user_input}.\n"
    "Only return a single caption, in quotes, with no explanation or extra text."
)

def llm_caption_generator(user_input, num_captions=3):
    prompt = PROMPT_TEMPLATE.format(user_input=user_input.strip())
    captions = []

    for _ in range(num_captions):
        seed = random.randint(0, 100000)

        if LLM_MODE == "conversational":
            response = llm_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=40,
            temperature=1.0,
            top_p=0.95,
            seed=seed  # Add seed for reproducibility/randomness if supported
            )

            output = response.choices[0].message["content"].strip()
        else:
            response = llm_client.text_generation(
                prompt=prompt,
                max_new_tokens=40,
                temperature=1.0,
                top_p=0.95,
                seed=seed
            )
            output = response.strip()

        quoted = re.findall(r'"(.*?)"', output)
        if quoted:
            cleaned = quoted[0]
        else:
            for end in [".", "—", "\n"]:
                if end in output:
                    cleaned = output.split(end)[0] + end
                    break
            else:
                cleaned = output

        caption = re.sub(r"\s+", " ", cleaned).strip()
        captions.append(caption)

    return captions


# --- Utility function to test profanity detection across generations ---
"""
def test_caption_profanity(user_input, num_captions=3):
    print(f"Testing '{user_input}' with {num_captions} caption(s):")
    captions = llama_caption_generator(user_input, num_captions=num_captions)
    for i, c in enumerate(captions, 1):
        clean_check = re.sub(r"[^a-zA-Z\s]", "", c)
        is_profane = profanity.contains_profanity(clean_check.lower())
        status = "🚫 PROFANE" if is_profane else "✅ CLEAN"
        print(f"Caption {i}: {c} Status: {status}")

    topic = classify_topic_from_input(user_input)
    if topic is None:
        raise ValueError("Your input didn't match any approved topics. Try again or rephrase.")

    captions = llama_caption_generator(user_input, num_captions=num_captions)
    filtered = [c for c in captions if len(c.split()) >= 4 and not profanity.contains_profanity(c)]

    if not filtered:
        raise ValueError("All generated captions failed our safety check!\nPlease try again or pick a new topic.")

    return filtered
"""

BLOCKED_PHRASES = ["erection", "blow me", "nudes", "bang", "morning wood", "bih", "FTW",
                  "boner", "gay", "thicc af", "porn", "drugs", "hangover", 
                  "hungover", "drunk", "get high", "weed", "smoke", "stoned", "stoner",
                  "stoned af", "drunk af", "beers", "beer", "alcohol", "drinking", "gooning", "fcuk"]

def safe_caption_generator(user_input, num_captions=3):
    for bad in BLOCKED_PHRASES:
        if bad in user_input.lower():
            raise ValueError("Your input may include inappropriate language. Please change your language and try again.")

    topic = classify_topic_from_input(user_input)
    if topic is None:
        raise ValueError("Your input didn't match any approved topics. Try again or rephrase.")

    captions = llm_caption_generator(user_input, num_captions=num_captions)

    filtered = []
    for c in captions:
        clean_check = re.sub(r"[^a-zA-Z\s]", "", c)
        if len(c.split()) >= 4 and not profanity.contains_profanity(clean_check.lower()):
            filtered.append(c)

    if not filtered:
        raise ValueError("All generated captions failed our safety check!\nPlease try again or pick a new topic.")

    return filtered