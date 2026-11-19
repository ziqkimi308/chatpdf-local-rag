from pathlib import Path
import chromadb
import ollama
from pypdf import PdfReader
import sys

LLM_MODEL="qwen2.5-coder:3b"
EMBED_MODEL="nomic-embed-text"
DB_DIR="./chroma_db"
SYSTEM = (
    "You answer questions using ONLY the provided context. "
    "You have no other knowledge. If the context does not contain the answer, "
    "reply exactly: I couldn't find that in the document."
)

# create instances
chroma = chromadb.PersistentClient(path=DB_DIR)
collection = chroma.get_or_create_collection("documents")

# 1. Read pdf file and insert embed vectors by chunks into db.
def ingest_pdf(pdf_path):
	# Safety guard
	pdf_file = Path(pdf_path)
	if not pdf_file.is_file():
		print(f"File not found or is not a file: {pdf_path}")
		return

	reader = PdfReader(pdf_path)
	print(f"Ingesting '{pdf_path}' ({len(reader.pages)} pages)...")

	count = 0
	for page_idx, page in enumerate(reader.pages, start=1):
		# all text in 1 page
		text = (page.extract_text() or "").strip()
		if not text:
			continue

		# word in chunks
		for i in range(0, len(text), 450):
			chunk = text[i:i+500]
			emb = ollama.embed(model=EMBED_MODEL, input=chunk)["embeddings"][0]
			# upsert() is better than add() in terms of idempotent write operation
			# (Idempotent = safe to re-run multiple times without creating duplicates or crashing)
			collection.upsert(
				ids=[f"{pdf_file.name}_p{page_idx}_c{count}"],
				embeddings=[emb],
				documents=[chunk],
				metadatas=[{"source":pdf_file.name, "page": page_idx}]
			)
			count += 1

	print(f"Done! {count} chunks indexed.")

def ask(question):
	# embedding
	q_emb = ollama.embed(model=EMBED_MODEL, input=question)["embeddings"][0]
	# query
	results = collection.query(query_embeddings=[q_emb], n_results=2)
	docs = results["documents"][0]
	metas = results["metadatas"][0]

	if not docs:
		print("No matching information found.")
		return

	# prepare context
	formatted_chunks = []
	for doc, meta in zip(docs, metas):
		formatted_chunks.append(f"[{meta['source']} Page {meta['page']}]: {doc}")
	context = "\n--\n".join(formatted_chunks)

	# prepare prompt
	prompt = (
        f"Answer the question based strictly on the context below. Always cite the page number.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )

	# ai answer
	print(f"\nAI > ", end="", flush=True)
	stream = ollama.generate(model=LLM_MODEL, system=SYSTEM, prompt=prompt, stream=True)
	for chunk in stream:
		print(chunk["response"], end="", flush=True)
	print()

if __name__ == "__main__":
	if len(sys.argv) > 1 and sys.argv[1].endswith(".pdf"):
		ingest_pdf(sys.argv[1])

	print("--- Local ChatPDF Ready (Type 'exit' to quit) ---")
	while True:
		q = input("\nAsk > ").strip()
		if q.lower() in ("exit", "quit"):
			break
		if q:
			ask(q)
	