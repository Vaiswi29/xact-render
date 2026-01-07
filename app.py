import os, json, zipfile
import numpy as np
from flask import Flask, request, jsonify, render_template

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ----------------------------
# Light RAG: TF-IDF retrieval (no sentence-transformers)
# ----------------------------
class xActRAG:
    def __init__(self, jsonl_file: str):
        self.questions = []
        self.answers = []

        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    msgs = data.get("messages", [])
                    user_q, assistant_a = "", ""
                    for m in msgs:
                        if m.get("role") == "user":
                            user_q = m.get("content", "")
                        elif m.get("role") == "assistant":
                            assistant_a = m.get("content", "")
                    if user_q and assistant_a:
                        self.questions.append(user_q)
                        self.answers.append(assistant_a)
                except Exception:
                    continue

        # TF-IDF is cheap and fits in free tier
        self.vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            stop_words="english"
        )
        self.q_matrix = self.vectorizer.fit_transform(self.questions)

        print(f"[RAG] Loaded {len(self.questions)} Q/A pairs and built TF-IDF matrix ✅")

    def retrieve(self, query: str, top_k: int = 3):
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.q_matrix)[0]
        idx = np.argsort(sims)[-top_k:][::-1]
        return [(self.questions[i], self.answers[i], float(sims[i])) for i in idx]


# ----------------------------
# Assistant: lazy-load the model to avoid startup OOM
# ----------------------------
class xActAssistant:
    def __init__(self, rag: xActRAG, model_path="xact_finetuned_model", zip_name="xact_finetuned_model.zip"):
        self.rag = rag
        self.model_path = model_path
        self.zip_name = zip_name

        self.tokenizer = None
        self.model = None

    def _ensure_model_loaded(self):
        if self.model is not None:
            return

        # unzip model if needed
        if os.path.exists(self.zip_name) and not os.path.exists(self.model_path):
            print("[Model] Unzipping model...")
            with zipfile.ZipFile(self.zip_name, "r") as z:
                z.extractall(".")
            print("[Model] Unzip complete ✅")

        print(f"[Model] Loading fine-tuned model from: {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)

        # Try a lower-memory dtype first (works on many CPU setups)
        # If it errors, fall back to fp32.
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                local_files_only=True,
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,  # try smaller memory
            )
            print("[Model] Loaded with bfloat16 ✅")
        except Exception as e:
            print(f"[Model] bfloat16 load failed ({e}); falling back to fp32...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
            print("[Model] Loaded with fp32 ✅")

        self.model.eval()

    def answer(self, question: str, top_k: int = 3):
        self._ensure_model_loaded()

        refs = self.rag.retrieve(question, top_k=top_k)
        context = "\n\n---\n\n".join([f"Q: {q}\nA: {a}" for q, a, _ in refs])

        prompt = f"""<|system|>
You are a friendly and knowledgeable assistant that explains xAct concepts simply.
Use ONLY the reference information below to answer accurately.
<|context|>
{context}
<|user|>
{question}
<|assistant|>
"""

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.2,
                repetition_penalty=1.15,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        text = self.tokenizer.decode(out[0], skip_special_tokens=False)
        answer = text.split("<|assistant|>")[-1].strip()
        if "<|end|>" in answer:
            answer = answer.split("<|end|>")[0].strip()
        return answer


# ----------------------------
# Flask app
# ----------------------------
app = Flask(__name__)

KB_FILE = os.environ.get("KB_FILE", "xAct_humanized.jsonl")

# IMPORTANT: load only the lightweight RAG at boot
rag = xActRAG(KB_FILE)

# Lazy assistant (model loads only when /ask is called)
assistant = xActAssistant(rag)

@app.get("/")
def home():
    return "xAct service is up ✅"

@app.get("/chat")
def chat():
    return render_template("index.html")

@app.post("/ask")
def ask():
    data = request.get_json(force=True) or {}
    q = (data.get("question") or "").strip()
    if not q:
        return jsonify({"error": "Send JSON like {'question': '...'}"}), 400
    ans = assistant.answer(q)
    return jsonify({"question": q, "answer": ans})
