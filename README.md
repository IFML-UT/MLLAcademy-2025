
# 🤖 AI Meme Generator Project

Welcome to the **AI Meme Generator project**, built as part of the UT Austin Machine Learning Summer Academy! 

This project guides students through building a fun, safe, and AI-powered meme generator using open-source models and free-tier tools.

This multi-modal machine learning project allows students to generate meme captions from safe prompts, and match them to pre-approved meme templates using OpenCLIP.

Students will:
- Generate meme-worthy captions from text prompts.
- Select the best caption and match it to a semantically relevant image.
- Create and download a final meme combining the AI-generated text and image.

> _This lab is built and supported by the Insitute for Foundations of Machine Learning (IFML)_ | https://ifml.institute/

---

## Project Structure

```
MLLAcademy-2025/
├─Lab1_MEMEGEN
   ├── fonts/
       └── impact.ttf
   ├── images/
       └── (our image pool for meme creation [curated])
   ├── notebook/
       └── 1_meme_generator_inst.ipynb
       └── 2_meme_generator_A.ipynb
       └── 3_final_meme_assembly.ipynb
    ├── utils/
   │   └── safe_caption_generator.py
   ├── requirements.txt
   ├── README.md
   ├── selected_caption.json (generated)
   ├── selected_image.json (generated)
   ├── top_images.json (generated)
   └── captions.json (generated)
```

---

## 🧰 Tools & Libraries Used

| Tool | Purpose |
|------|---------|
| [Meta LLaMA 3.1–8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | Instruction-tuned text generation via Hugging Face Inference API |
| [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | Lightweight, chat-friendly model for Colab environments |
| [huggingface_hub](https://github.com/huggingface/huggingface_hub) | Model hosting, token management, and API interface |
| [sentence-transformers](https://www.sbert.net/) | Embedding + semantic similarity for topic classification - using "all-MiniLM-L6-v2" |
| [better-profanity](https://github.com/surge-ai/better-profanity) | Lightweight profanity detection for content safety |
| [re / regex](https://docs.python.org/3/library/re.html) | Post-processing: caption cleanup and quoted text extraction |
| [Torch](https://pytorch.org/) | Backend for sentence-transformers and HF models |
| [OpenCLIP](https://github.com/mlfoundations/open_clip) | Caption-to-image similarity for meme image selection |
| [Pillow (PIL)](https://python-pillow.org/) | Image rendering and meme creation |

---
## Lab Components

### 1️⃣ Behind the Scenes - Inference and Safe Caption Generation
#### `safe_caption_generator.py`
- Python module that:
  - Detects runtime environment (local or Colab).
  - Dynamically selects the appropriate AI model for inference:
    - **Local (Jupyter)**: `meta-llama/Llama-3.1-8B-Instruct`.
    - **Colab**: `mistralai/Mistral-7B-Instruct-v0.3`.
  - Implements a safe caption generation pipeline:
    - Semantic topic matching against a list of **approved topics**.
    - Profanity filtering and blocked phrase detection.
    - AI caption generation via Hugging Face Inference API.

### 2️⃣ Instructor Notebook
#### `1_meme_generator_inst.ipynb`
- Step-by-step Jupyter Notebook for getting started with text generation:
  - Guides students through generating captions from prompts.
  - Handles Hugging Face token input for API based inference
  - Saves captions to a JSON file for downstream use.

### 3️⃣ Student Notebook
#### `0_StudentLab-ColabNotebook.ipynb`
This is your "all in one notebook, ideal for running in a single Colab notebook.
This notebook generates captions, tests captions against image search using OpenCLIP, 
and allows you to easily combine the text and image selections using `ipywidget` within
the Google Colab UI.

### Reference Notebooks:
The same content as the **Student Notebook** above, but split across 2 notebooks, 
one for caption generation and image search, and the other for putting it all together.
- **Notebook A**: Caption Generation + Image Search
  - Loads pre-generated captions.
  - Lets students select the best caption.
  - Searches a local image set for semantically matching images using **OpenCLIP**.
  - Lets students pick the best image.

- **Notebook B**: Final Meme Assembly
  - Loads the selected caption and image.
  - Combines them into a final meme.
  - Outputs the meme as a downloadable file.

---

## 🧪 Setup Instructions:

### Prerequisites
- Python 3.11 or later.
- Free or paid Hugging Face account with API token:
  - [Sign up for Hugging Face](https://huggingface.co/join)
  - [Get your token](https://huggingface.co/settings/tokens)

  ### Running Locally (Jupyter Notebook)
1. Clone this repository:
   ```bash
   git clone https://github.com/IFML-UT/MLLAcademy-2025.git
   ```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run the first, instructor notebook:
```
jupyter `[your directory]reference_notebooks/1_meme_generator_inst.ipynb
```

4. Follow the prompts to input your Hugging Face token.

### If Running in Google Colab:
1. Open the instructor notebook in Colab using the "open with GitHub" option.
2. Follow on screen prompts and enter the Hugging Face token shared with you during lab.
3. Proceed through the lab as directed.

---

## Environment Detection Logic
This project automatically adapts based on your runtime:
- `local`: (Jupyter or VS Code): Uses LLaMA 3.1 8B model (`text-generation`).
- `colab`: (Google Colab free): Uses Mistral 7B Instruct v0.3 model (`conversational`).

---

## 🚫 Safety Features
✅ Profanity filtering via `better_profanity`
✅ Blocked phrase list (e.g., NSFW terms, slurs, inappropriate language)
✅ Topic matching to a whitelisted list of approved topics

---

## requirements.txt


```
transformers==4.50.0
sentence-transformers==2.6.1
better_profanity==0.7.0
torch>=2.7.0
open_clip_torch==2.23.0
Pillow>=11.0.0
accelerate>=0.28.0
huggingface_hub>=0.22.2
ipywidgets==7.6.5
tqdm>=4.66.1
```

---

## ✏️ Student Reflection (Optional for Portfolio)

We recommend students write a short reflection on:
- What they learned about multi-modal ML
- How prompt engineering impacted output
- What tradeoffs they made between automation and control
- What their favorite generated meme was!

---

Have fun and build responsibly! 🤘
