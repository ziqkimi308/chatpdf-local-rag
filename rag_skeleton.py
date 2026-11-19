import ollama
import chromadb

# instance of chromadb
client = chromadb.Client()
collection = client.create_collection("skeleton_rag")

# sample documents
docs = [
	"Project Titan deployment completed on August 15, 2026. CapEx was $4.25M.",
    "The primary security lead for audits is Chief Inspector Sarah Chen.",
    "Production uptime across 2026 was measured at 99.982%."
]

# 1. Convert data into embed vector then add into chromadb
for i, doc in enumerate(docs):
	# prepare data
	emb = ollama.embed(model="nomic-embed-text", input=doc)["embeddings"][0]
	# add - putting data in
	collection.add(ids=[str(i)], embeddings=[emb], documents=[doc])

# 2. Convert question into embed vector then fetch answer using query
question = "Who is the security lead for audits?"
# prepare data
q_emb = ollama.embed(model="nomic-embed-text", input=question)["embeddings"][0]
# query - retrieves data out
results = collection.query(query_embeddings=[q_emb], n_results=1)
context = results["documents"][0][0]


prompt = f"Answer the question using ONLY this context:\n{context}\n\nQuestion: {question}\nAnswer:"
res = ollama.generate(model="qwen2.5-coder:3b", prompt=prompt)
print(f"Context found: {context}\n")
print(f"Answer: {res['response'].strip()}")