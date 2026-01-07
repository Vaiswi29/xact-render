import os
import json
import zipfile
import numpy as np
from flask import Flask, request, jsonify, render_template

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


# ----------------------------
# RAG system: loads Q/A jsonl and retrieves similar items
# ----------------------------
class xActRAG:
    def __init__(self, jsonl_file: str):
        if not os.path.exists(jsonl_file):
            raise FileNotFoundError(
                f"Missing knowledge base file: {jsonl_file}. "
                f"Put it in the same folder as app.py or set KB_FILE env var."
            )

        self.knowledge_base = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    msgs = data.get("messages", [])

                    user_q, assistant_a = "", ""
                    for m in msgs:
                        if m.get("role") == "user":
                            user_q = (m.get("content") or "").strip()
                        elif m.get("role") == "assistant":
                            assistant_a = (m.get("content") or "").strip()

                    if user_q and assistant_a:
                        self.knowledge_base.append((user_q, assistant_a))
                except Exception:
                    continue

        self.questions = [q for q, _ in self.knowledge_base]
        self.answers = [a for _, a in self.knowledge_base]

        print(f"[RAG] Loaded {len(self.questions)} Q/A pairs from {jsonl_file}")

        print("[RAG] Loading embedding model (all-MiniLM-L6-v2)...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        if self.questions:
            print("[RAG] Computing embeddings...")
            self.embeddings = self.embedder.encode(self.questions, convert_to_numpy=True, show_progress_bar=True)
        else:
            # 384 dims for all-MiniLM-L6-v2
            self.embeddings = np.zeros((0, 384), dtype=np.float32)

        print("[RAG] Ready")

    def retrieve(self, query: str, top_k: int = 3):
        query = (query or "").strip()
        if not query or self.embeddings.shape[0] == 0:
            return []

        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        sims = cosine_similarity(q_emb, self.embeddings)[0]
        top_idx = sims.argsort()[-top_k:][::-1]
        return [(self.questions[i], self.answers[i], float(sims[i])) for i in top_idx]


# ----------------------------
# Assistant: uses RAG + fine-tuned model (if present)
# ----------------------------
class xActAssistant:
    def __init__(self, rag: xActRAG, model_dir="xact_finetuned_model", model_zip="xact_finetuned_model.zip"):
        self.rag = rag

        # Unzip the trained model if zip exists and folder doesn't
        if os.path.exists(model_zip) and not os.path.exists(model_dir):
            print(f"[Model] Unzipping {model_zip}...")
            with zipfile.ZipFile(model_zip, "r") as z:
                z.extractall(".")
            print("[Model] Unzip complete")

        # Load fine-tuned model if present; otherwise fallback
        if os.path.exists(model_dir):
            print(f"[Model] Loading fine-tuned model from: {model_dir}")
            tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)

            # If pad token missing, set to eos to avoid warnings/errors
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = tokenizer.eos_token_id

            self.pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1,
            )
            print("[Model] Fine-tuned model loaded")
        else:
            # Fallback (keeps app running even if model zip isn't included)
            fallback = "EleutherAI/gpt-neo-125M"
            print(f"[Model] Fine-tuned model not found. Falling back to: {fallback}")
            self.pipe = pipeline(
                "text-generation",
                model=fallback,
                device=0 if torch.cuda.is_available() else -1,
            )
            print("[Model] Fallback model loaded")

    def answer(self, question: str, top_k: int = 3, temperature: float = 0.2):
        question = (question or "").strip()
        if not question:
            return "Please ask a question."

        refs = self.rag.retrieve(question, top_k=top_k)
        context = "\n\n---\n\n".join([f"Q: {q}\nA: {a}" for q, a, _ in refs]) if refs else "No relevant references found."

        prompt = f"""<|system|>
You are a friendly and knowledgeable assistant that explains xAct concepts simply.
Use ONLY the reference information below to answer accurately.
<|context|>
{context}
<|user|>
{question}
<|assistant|>
"""

        out = self.pipe(
            prompt,
            max_new_tokens=250,
            temperature=temperature,
            do_sample=True,
            repetition_penalty=1.15,
            pad_token_id=self.pipe.tokenizer.eos_token_id,
        )[0]["generated_text"]

        # Extract everything after the last assistant tag
        ans = out.split("<|assistant|>")[-1].strip()
        if "<|end|>" in ans:
            ans = ans.split("<|end|>")[0].strip()
        return ans


# ----------------------------
# Flask app setup
# ----------------------------
app = Flask(__name__)

KB_FILE = os.environ.get("KB_FILE", "xAct_humanized.jsonl")

print("[Boot] Starting service...")
print(f"[Boot] KB_FILE = {KB_FILE}")

rag = xActRAG(KB_FILE)
assistant = xActAssistant(rag)

print("[Boot] Service ready")


@app.get("/")
def home():
    return "xAct service running"


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/chat")
def chat():
    # Requires templates/index.html
    return render_template("index.html")


@app.post("/ask")
def ask():
    data = request.get_json(force=True, silent=True) or {}
    q = (data.get("question") or "").strip()
    if not q:
        return jsonify({"error": "Send JSON like {'question': '...'}"}), 400

    ans = assistant.answer(q)
    return jsonify({"question": q, "answer": ans})


# Local run (Render uses gunicorn; this is for your laptop)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
