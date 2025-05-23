
# 🤖 AI Meme Generator Project

Welcome to the AI Meme Generator project, built as part of the UT Austin Machine Learning Summer Academy! 
This multi-modal machine learning project allows students to generate meme captions from safe prompts, match them to pre-approved meme templates using OpenCLIP, and deploy their own working meme generator app using Streamlit Cloud.
> _This lab is built and supported by the Insitute for Foundations of Machine Learning (IFML)_

---

## 📚 Project Structure

```
meme-generator/
├── app.py                       # Streamlit app for deployment (Lab D)
├── images/                     # Pre-approved meme template images
├── notebooks/
│   └── meme_generator.ipynb    # Google Colab notebook (Labs A–C)
├── utils/
│   └── safe_caption_generator.py  # Handles safe prompt and caption generation and filtering
├── requirements.txt            # Libraries needed for local or Streamlit use
└── README.md
```

---

## 🧰 Tools & Libraries Used

| Tool | Purpose |
|------|---------|
| [Meta LLaMA 3.1–8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | Instruction-tuned text generation via Hugging Face Inference API |
| [huggingface_hub](https://github.com/huggingface/huggingface_hub) | Model hosting, token management, and API interface |
| [sentence-transformers](https://www.sbert.net/) | Embedding + semantic similarity for topic classification |
| [better-profanity](https://github.com/surge-ai/better-profanity) | Lightweight profanity detection for content safety |
| [re / regex](https://docs.python.org/3/library/re.html) | Post-processing: caption cleanup and quoted text extraction |
| [Torch](https://pytorch.org/) | Required backend for sentence-transformers and HF models |
| [OpenCLIP](https://github.com/mlfoundations/open_clip) | Caption-to-image similarity for meme image selection |
| [Pillow (PIL)](https://python-pillow.org/) | Image rendering and meme creation |
| [Streamlit](https://streamlit.io/) | (Optional - Lab D) App UI + student portfolio deployment |

---

## 🚀 How to Run (Streamlit Deployment)

1. **Clone this repo** or create a GitHub copy under your own account.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud).
3. Click **"New App"** and connect your GitHub repo.
4. Set the entry point to `app.py`.
5. Click **Deploy**.
6. Your live app will allow users to:
   - Choose a safe meme topic
   - Generate a caption using AI
   - Match it to a meme image
   - Display the final meme

---

## 🧪 How to Use in Google Colab (Labs A–C)

1. Open [Google Colab](https://colab.research.google.com/)
2. Navigate to the GitHub tab
3. Paste your GitHub repo URL and open `notebooks/meme_generator.ipynb`
4. Run the notebook step-by-step:
   - Generate a caption
   - Optionally stylize it
   - Match it with OpenCLIP
   - Output a meme

---

## 📦 requirements.txt

This is used for local testing or Streamlit app builds.

```
transformers==4.40.0
sentence-transformers==2.6.1
better-profanity==0.7.0
torch>=2.7.0
open_clip_torch==2.23.0
Pillow>=11.0.0
streamlit>=1.31.0
accelerate>=0.28.0
huggingface_hub>=0.22.2
```

---

## ✏️ Student Reflection (Optional for Portfolio)

We recommend students write a short reflection on:
- What they learned about multi-modal ML
- How prompt engineering impacted output
- What tradeoffs they made between automation and control
- What their favorite generated meme was!

---

Have fun and build responsibly! 🧠💡
