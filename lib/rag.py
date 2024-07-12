import torch
from abc import ABC, abstractmethod
from torch import Tensor


class Embedding(ABC):
    @abstractmethod
    def embed_documents(self, docs: list[str], normalize: bool, max_length: int):
        pass
    @abstractmethod
    def score_passages(self, query: str, passages: list[str]):
        pass

    

class MultilingualE5Small(Embedding):
    def __init__(self):
        from transformers import AutoTokenizer, AutoModel
        
        if torch.cuda.is_available():
            self._device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")
            
        self._tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
        self._model = AutoModel.from_pretrained('intfloat/multilingual-e5-small').to(self._device)
        
    def _average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    @torch.no_grad()
    def embed_documents(self, docs: list[str], normalize: bool = True, max_length: int = 200):
        """
        # Each input text should start with "query: " or "passage: ", even for non-English texts.
        # For tasks other than retrieval, you can simply use the "query: " prefix.
        docs = [
            "query: how much protein should a female eat",
            "query: 南瓜的家常做法",
            "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
            "passage: 1.清炒南瓜丝 原料:嫩南瓜半个 调料:葱、盐、白糖、鸡精 做法: 1、南瓜用刀薄薄的削去表面一层皮,用勺子刮去瓤 2、擦成细丝(没有擦菜板就用刀慢慢切成细丝) 3、锅烧热放油,入葱花煸出香味 4、入南瓜丝快速翻炒一分钟左右,放盐、一点白糖和鸡精调味出锅 2.香葱炒南瓜 原料:南瓜1只 调料:香葱、蒜末、橄榄油、盐 做法: 1、将南瓜去皮,切成片 2、油锅8成热后,将蒜末放入爆香 3、爆香后,将南瓜片放入,翻炒 4、在翻炒的同时,可以不时地往锅里加水,但不要太多 5、放入盐,炒匀 6、南瓜差不多软和绵了之后,就可以关火 7、撒入香葱,即可出锅"
        ]
        """
        import torch.nn.functional as F
        batch_dict = self._tokenizer(
            docs, 
            max_length=max_length, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        ).to(self._device)

        outputs = self._model(**batch_dict)
        embeddings = self._average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        del outputs
        del batch_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

        return F.normalize(embeddings, p=2, dim=1) if normalize else embeddings
    
    def score_passages(self, query: str, passages: list[str]):
        input_texts = [f"query: {query}"] + [f"passage: {passage}" for passage in passages]
        embeddings = self.embed_documents(input_texts)
        scores = (embeddings[:1] @ embeddings[1:].T).tolist()[0]
        return scores



class JinaRerankerMultilingual(Embedding):
    def __init__(self, device: str|None = None):
        from lib.backends import suggest_device
        from transformers import AutoModelForSequenceClassification
        
        self._device = device if device else suggest_device()
        self._model = AutoModelForSequenceClassification.from_pretrained(
            'jinaai/jina-reranker-v2-base-multilingual',
            torch_dtype="auto",
            trust_remote_code=True,
        )
        self._model.to(self._device)
        self._model.eval()
        
    def embed_documents(self, docs: list[str], normalize: bool, max_length: int):
        pass
        
    def score_passages(self, query: str, passages: list[str]):
        sentence_pairs = [[query, doc] for doc in passages]
        scores = self._model.compute_score(sentence_pairs, max_length=1024)
        return scores


        
def pick_relevant_web_documents(
    query: str,
    embedding: Embedding,
    engine: str,
    n_relevant_chunks: int,
    google_api_key: str|None = None,
    google_cse_id: str|None = None,
    n_search_results: int = 10,
    chunk_length: int = 700,
    proxies: dict|None = None,
):
    from lib.utils import split_text
    from lib.scraping import (
        fetch_contents_of, 
        google_search_results_df, 
        ddg_search_results_df, 
        torch_results_df,
        DEFAULT_ONION_PROXIES,
    )
    import numpy as np


    # Scrape web contents.
    if engine == "google":
        df = google_search_results_df(query, google_api_key, google_cse_id, n_results=n_search_results)
    elif engine == "duckduckgo":
        df = ddg_search_results_df(query, n_results=n_search_results)
    elif engine == "torch":
        if not proxies:
            proxies = DEFAULT_ONION_PROXIES
        df = torch_results_df(query, n_results=n_search_results)
    else:
        raise ValueError("Search engine must be one of ['google', 'duckduckgo', 'torch'].")
    if len(df) == 0:
        return []


    def fetcher_without_proxies(url):
        return fetch_contents_of(url, use_chrome=False)
    def fetcher_with_proxies(url):
        return fetch_contents_of(url, use_chrome=False, proxies=proxies, timeout=(60, 60))

    df["content"] = df["url"].apply(fetcher_with_proxies if proxies else fetcher_without_proxies)

    # Split into chunks.
    chunks = []
    for _, item in df.iterrows():
        split_content = split_text(item["content"], max_length=chunk_length)
        chunks += [{"title": item["title"], "url": item["url"], "content": text} for text in split_content]
    texts = [chunk["content"] for chunk in chunks]

    # Vector search.
    scores = embedding.score_passages(query, texts)
    top_indices = np.argsort(scores)[-n_relevant_chunks:][::-1]

    del embedding
    
    return [chunks[idx] for idx in top_indices]