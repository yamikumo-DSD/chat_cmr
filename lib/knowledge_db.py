import chromadb
from lib.rag import Embedding
from chromadb import EmbeddingFunction, Documents

class MultilingualE5Small(EmbeddingFunction):
    def __init__(self, max_length: int = 1000):
        self.max_length = max_length
    
    def __call__(self, input: Documents):
        from lib.rag import MultilingualE5Small

        embedding: Embedding = MultilingualE5Small()
        results = embedding.embed_documents(input, max_length=self.max_length)
        return [r.tolist() for r in results]
        
class JinaEmbeddingV3(EmbeddingFunction):
    def __init__(self) -> None:
        from lib.rag import JinaEmbeddingV3
        self._embedding = JinaEmbeddingV3()
    def load_model(self) -> None:
        from lib.rag import JinaEmbeddingV3
        self._embedding = JinaEmbeddingV3()
    def release_model(self) -> None:
        self._embedding = None
    def __call__(self, input: Documents):
        results = self._embedding.embed_documents(input, normalize=True)
        return results


def load_documents_in(directory_path: str) -> list[dict]:
    """
    Args:
        directory_path (str): directory to walk.
    Returns:
        list: list of items {'filename': '...', 'content': '...'}
    """
    import os
    import glob
    import pdfplumber
    from tqdm import tqdm
    
    result = []

    pattern = os.path.join(directory_path, "**", "*.pdf")
    pdf_files = glob.glob(pattern, recursive=True)

    pattern = os.path.join(directory_path, "**", "*.txt")
    txt_files = glob.glob(pattern, recursive=True)

    all_files = pdf_files + txt_files

    for file_path in tqdm(all_files):
        try:
            filename = os.path.basename(file_path)
            content = ""

            if file_path.endswith(".pdf"):
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            content += text + "\n"

            elif file_path.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

            result.append({"filename": filename, "content": content.strip()})

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return result



def build_knowledge_db(
    directory_path: str, 
    db_path: str,
    name: str = "kdb",
    chunk_size: int = 100,
    prefix_chunk_with_filename: bool = False,
):
    from lib.text_utils import split_text
    from lib.utils import split_list
    from tqdm import tqdm
    import hashlib
    import os

    if prefix_chunk_with_filename:
        raise NotImplementedError("prefix_chunk_with_filename")
    
    print(f"Loading documents in {directory_path}")
    document_files = load_documents_in(directory_path)

    print("Chunking documents")
    chunk_list: list[dict[str, str]] = []
    for document in document_files:
        print(f"Chunking \"{document["filename"]}\"")
        chunks = split_text(document["content"], max_length=chunk_size)
        for chunk_idx, chunk in enumerate(chunks):
            chunk_list.append({
                "filename": document["filename"], 
                "index": chunk_idx,
                "text": chunk,
            })

    # Set collection.
    chroma_client = chromadb.PersistentClient(
        path=db_path, 
        settings=chromadb.Settings(allow_reset=True)
    )
    collection = chroma_client.get_or_create_collection(
        name=name, 
        embedding_function=JinaEmbeddingV3()
    )

    documents = [item["text"] for item in chunk_list]
    metadatas = [{"filename": item["filename"], "index": item["index"]} for item in chunk_list]
    ids = [f"id_{hashlib.md5(d.encode()).hexdigest()}" for d in documents]

    BATCH_SIZE = 10
    n = len(documents) // BATCH_SIZE
    document_batches = split_list(documents, n)
    metadata_batches = split_list(metadatas, n)
    id_batches = split_list(ids, n)

    print("Embedding documents")

    for document_batch, metadata_batch, id_batch in tqdm(list(zip(document_batches, metadata_batches, id_batches))):
        collection.add(
            documents=document_batch,
            metadatas=metadata_batch,
            ids=id_batch,
        )

    print("Process completed")



def load_knowledge_db(db_path: str, name:str = "kdb") -> chromadb.Collection:
    chroma_client = chromadb.PersistentClient(
        path=db_path, 
        settings=chromadb.Settings(allow_reset=True)
    )
    return chroma_client.get_collection(name=name, embedding_function=JinaEmbeddingV3())




    
def pick_relevant_local_documents(
    query: str,
    db_path: str,
    name: str = "kdb",
    reranker: Embedding|None = None,
    n_relevant_chunks: int = 3,
    n_search_results: int = 10,
    score_thresh = 0,
):
    """
    Returns:
        list: list of {"filename": "...", "content": "...", "score": ...}
    """
    
    collection = load_knowledge_db(db_path, name=name)
    results = collection.query(query_texts=query, n_results=n_search_results)
    results["rerank_score"] = reranker.score_passages(query, results["documents"][0])
    
    simple_results = []
    for content, metadata, score in zip(results["documents"][0], results["metadatas"][0], results["rerank_score"]):
        simple_results.append({"content": content, "filename": metadata["filename"], "score": score})

    # Filter/Sort.
    filtered_results = [
        x
        for x in sorted(simple_results, key=lambda x: x["score"], reverse=False)
        if x["score"] >= score_thresh
    ]
    relevant_results = filtered_results[-n_relevant_chunks:]
    relevant_results.reverse()
    
    return relevant_results