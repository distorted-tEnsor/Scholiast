# try_embeddings.py
# Step 3 of RAG: turn text into "meaning coordinates" and measure closeness.

from sentence_transformers import SentenceTransformer

# Load the local embedding model. The FIRST time you run this, it downloads
# the model (~130 MB) once, then caches it. After that it's instant & offline.
print("Loading the embedding model (first run downloads it)...\n")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Four sentences. Notice: the first two MEAN the same thing but SHARE NO
# keywords. The last one is about a totally different topic.
sentences = [
    "The method achieves high accuracy in detection.",   # A
    "How well did their approach perform?",              # B  (same meaning as A)
    "The system detects deepfake audio in real time.",   # C  (related topic)
    "The authors are based in Pune, India.",             # D  (unrelated topic)
]

# Turn each sentence into its 384-number "coordinates" (its embedding).
embeddings = model.encode(sentences)

print(f"Each sentence became a list of {len(embeddings[0])} numbers.\n")
print("Here are the first 8 numbers of sentence A's coordinates:")
print(embeddings[0][:8], "...\n")

# --- Now measure MEANING as DISTANCE ---
# We use "cosine similarity": 1.0 = identical meaning, 0 = unrelated.
# (Think of it as "how close on the meaning-map", scored 0 to 1.)
from sentence_transformers import util

print("How similar is sentence A to each other sentence?")
print("(1.00 = same meaning, lower = less related)\n")

for i, other in enumerate(sentences):
    score = util.cos_sim(embeddings[0], embeddings[i]).item()
    label = "ABCD"[i]
    print(f"  A vs {label}:  {score:.3f}   \"{other}\"")

print("\nLook at the scores: A should be closest to B (same meaning,")
print("different words!), somewhat close to C, and furthest from D.")   