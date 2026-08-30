# SETUP & DEPLOY WALKTHROUGH (start here)

This is the hands-on guide to get RAGmind running on your computer and then
live on the internet. It assumes the same comfort level as the password tool
you already deployed. Take it one numbered step at a time.

Two documents sit beside this one:
- `README.md` — the polished overview recruiters read.
- `STUDY-GUIDE.md` — the deep explanation of how every file works. Read that
  one to actually learn the system and be able to talk about it. This file is
  just "how do I run it."

---

## Part 1 — Run it on your Mac (about 10 minutes)

### Step 1: Open the folder in VS Code
Open the `ragmind` folder in VS Code (File → Open Folder). Same as before, do
your editing and running here, never in TextEdit.

### Step 2: Open a terminal inside VS Code
Terminal → New Terminal. You should see the prompt sitting inside the ragmind
folder.

### Step 3: Create a virtual environment (keeps this project's packages separate)
```bash
python3 -m venv .venv
source .venv/bin/activate
```
Your prompt should now start with `(.venv)`. That means you are "inside" the
environment. (If `python3` is not found, install Python 3 from python.org first.)

### Step 4: Install the dependencies
```bash
pip install -r requirements.txt
```
This pulls in Streamlit, sentence-transformers, and NumPy. It takes a couple
of minutes because the ML libraries are large. Normal.

### Step 5: Set your Gemini API key
You already have a Gemini key from the website assistant. Use the same one (or
make a fresh one at aistudio.google.com). In the terminal:
```bash
export GEMINI_API_KEY="paste-your-key-here"
```
Important: this only sets the key for *this* terminal window. If you close it,
you set it again. (Part 2 shows the permanent way for deployment.)
Security reminder: never paste this key into a file you will commit to GitHub.

### Step 6: Run the app
```bash
streamlit run app/streamlit_app.py
```
Your browser opens to a local address (like http://localhost:8501). The first
question you ask will pause ~30 seconds while it downloads the embedding model
once. After that it is fast.

### Step 7: Try it
Ask: "What is Kerberoasting and how do I mitigate it?"
You should see an answer, the sources it cited, and a "How it reasoned" trace
showing the retrieval attempts. Then try a question the corpus cannot answer,
like "What is the capital of France?" — it should refuse rather than guess.
That refusal is the system working correctly, not a bug.

### Step 8: Run the evaluation (this is your portfolio gold)
Open a new terminal (and re-do steps 3's `source` and step 5's `export` in it),
then:
```bash
python -m eval.evaluate 0.5
```
It prints a report table: retrieval recall, answer correctness, hallucination
rate, citation validity. Then run `python -m eval.evaluate 1.0` and compare.
Write down the numbers — that comparison is what you talk about in interviews.

---

## Part 2 — Put it live (Streamlit Community Cloud, free)

This is the same path as your password analyzer, so it will feel familiar.

### Step 1: Put the project on GitHub
Create a new repository (suggested name: `ragmind` or
`agentic-rag-evaluation`). Push the whole `ragmind` folder to it. The
`.gitignore` is already set up to keep your virtual environment and any secrets
out of the repo.

DOUBLE-CHECK before pushing: your API key must NOT be in any file. It lives
only in the environment / secrets, never in the code. The `.gitignore` blocks
the secrets file, but always glance at what you are committing.

### Step 2: Deploy on Streamlit Community Cloud
1. Go to share.streamlit.io and sign in with GitHub.
2. Click "New app", pick your `ragmind` repo.
3. Set the main file path to: `app/streamlit_app.py`
4. Before clicking deploy, open "Advanced settings" → "Secrets".

### Step 3: Add your key as a secret (the permanent, safe way)
In the Secrets box, paste this one line:
```
GEMINI_API_KEY = "paste-your-key-here"
```
Streamlit stores this securely and makes it available to the app as an
environment variable — exactly what `os.getenv("GEMINI_API_KEY")` in `llm.py`
reads. This is the cloud equivalent of the Netlify environment variable you
set for the website.

### Step 4: Deploy
Click deploy. First build takes a few minutes (installing the ML libraries).
When it is live you get a URL like `https://your-app.streamlit.app`.

### Step 5: Rename the URL (optional, looks cleaner)
In the app settings you can customize the subdomain, the same way you renamed
the password tool to `ambarvaldez-password-tool.streamlit.app`. Something like
`ambarvaldez-ragmind` reads well on a resume.

---

## Part 3 — Add it to the portfolio website

Once it is live, it becomes your flagship project. When you are ready, send me
the live URL and I will:
- Add a prominent project card to `home.html` and `work.html` (framed as the
  flagship, above or alongside the password tool), with a "Try the live app"
  button and one line of the headline metric.
- Add a short "how the AI on this site works" note near the chat widget so the
  existing assistant reads as engineering, not gimmick.

---

## If something breaks

- **"GEMINI_API_KEY is not set"**: you opened a new terminal and forgot to
  `export` the key again, or in the cloud you did not add the secret. Re-do
  step 5 (local) or step 3 (cloud).
- **First question hangs ~30s**: normal, it is downloading the embedding model
  once. Only happens the first time.
- **Rate limit error**: the free Gemini tier has limits; wait a minute. The
  eval harness already sleeps between questions to be gentle.
- **"No documents found"**: you must run from the `ragmind` folder so the
  `data/` paths resolve. Check your terminal is in the right directory.

---

## The honest framing for your resume

Do not call it more than it is, that is the fastest way to lose a sharp
interviewer. It is a single-corpus agentic RAG system with an evaluation
harness, built to demonstrate retrieval, grounding, agentic self-correction,
and measurement. That is genuinely impressive for a junior candidate precisely
because the evaluation piece is so rare. Lead with "I can prove it works and
show how I improved it," not with buzzwords. The STUDY-GUIDE has your 30-second,
2-minute, and 10-minute talking scripts.
