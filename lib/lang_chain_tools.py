class GoogleSearchResults:
    def run(self, query: str) -> str:
        from lib.scraping import google_search_results
        try:
            results = google_search_results(query, n_results=5)
            results_as_string = ''
            for idx in results.index:
                item = results.loc[idx]
                title, url, snippet = item['title'], item['url'], item['snippet']
                item_as_text = f'結果番号={idx+1}, url="{url}", title="{title}", description="{snippet}"'
                results_as_string = results_as_string + item_as_text + '\n'
            return results_as_string
        except:
            return 'Googleサーチが利用できない。別の検索エンジンを使用してください。'
            



import pandas as pd
from lib.scraping import google_search_results_df, fetch_contents_of


def add_embeddings_to_results(embeddings, results):
    documents = []
    for i in range(len(results)):
       item = results.loc[i]
       documents.append(item['title'] + item['snippet'])
    embds = embeddings.embed_documents(documents)
    results['embedding'] = embds

def get_most_suited_index(embeddings, results, query):
    import numpy as np
    if not 'embedding' in results.columns:
        add_embeddings_to_results(embeddings, results)
    query_embd = embeddings.embed_query(query)
    similarities = [np.dot(query_embd, item_embd) for item_embd in results['embedding']]
    return np.argmax(similarities)

class GoogleSearchOpenable:
    """
    Be careful for the indexing. For LLMs best result, using index starting with 1 not zero.
    """
    result: pd.DataFrame
    def __init__(
        self, 
        api_key: str,
        cse_id: str,
        n_results: int = 5, 
        embeddings=None, 
        summarizer=None, 
        custom_summarizing_function=None
    ) -> None:
        """
        embeddings is an embedding model that used for fuzzy result match.
        summarizer is a model used to summarize long contents of web pages.
        """
        from lib.tools import summarize
        
        self.embeddings = embeddings
        self.summarizer = summarizer
        self.n_results = n_results
        self.opened_flag = [False] * self.n_results
        self.result = pd.DataFrame()
        self.current_search_query = ''
        self.referred_urls = []
        self.api_key = api_key
        self.cse_id = cse_id
        self.summarizing_function = custom_summarizing_function if custom_summarizing_function else summarize
        
    def unset(self) -> None:
        self.opened_flag = [False] * self.n_results
        self.current_search_query = ''
        self.result = pd.DataFrame()
        self.referred_urls = []
        
    def not_opened(self) -> list[int]:
        return [index+1 for index, flag in enumerate(self.opened_flag) if not flag]

    def set(self, query: str) -> str:
        if not self.result.empty and not (True in self.opened_flag):
            return '次の検索に移る前に、ひとつはページを開いて中身を確認してください。'
        if query == self.current_search_query:
            return '前回と同じ検索ワードを指定することはできません。'
        self.current_search_query = query
        q = self.current_search_query.replace(' ', '+').replace('　', '+')
        self.referred_urls.append(f'https://www.google.com/search?q={q}')
        try:
            self.result = google_search_results_df(
                query, 
                self.api_key,
                self.cse_id,
                self.n_results,
            )
        except:
            return 'エラーにより検索が実行できませんでした。再度やり直してください。'
        self.opened_flag = [False] * self.n_results
        as_str = '\n'
        for i in range(self.n_results):
            item = self.result.loc[i]
            as_str = as_str + f'[検索結果{i+1}]\nタイトル:{item["title"]}\n内容:{item["snippet"]}\n'
        return as_str + f'\n*** 検索番号を指定して、内容を確認してください(ツールselectを使用) ***'
    def open_index(self, i: int) -> str:
       item = self.result.loc[i-1]
       contents = fetch_contents_of(item['url'], use_chrome=False)
       self.opened_flag[i-1] = True
       self.referred_urls.append(item['url'])

       if self.summarizer and len(contents) > 512:
           summary = self.summarizing_function(
               self.summarizer, 
               text=contents[:2000], 
               query=self.current_search_query,
               max_tokens=500,
           )
           contents = summary.replace('\n\n', '\n')
       else:
           contents = contents[:2000]

       text = f'\n[検索結果{i}を開きました]\n{contents}'
       if self.not_opened():
           text += f'\n\n検索結果{self.not_opened()}はまだ開かれていません'
       else:
           text += '\n\n全てのページが開かれました。別のキーワードで検索を行ってください。'
       return text

    def match_index(self, text) -> None|int:
        import re
        match = re.search(r'結果([0-9]+)', text)
        if match: 
            index = int(match.group(1))
            if index < self.n_results + 1:
                return index
        return None

    def check_flag_and_open(self, i) -> str:
       if self.opened_flag[i-1]:
           return f'既に開かれたページです。検索結果{self.not_opened()}の中から選択してください。'
       return self.open_index(i)
        
    def open(self, open_request_index: str):
        """
        open_request_index is a string containing an index number corresponding to one of the articles.
        """
        if self.result.empty: 
            return 'まずは、検索を行ってリストを取得してください。'
        if not False in self.opened_flag:
            return '全てのページが開かれました。別のキーワードで検索を行ってください。'
        if 'http' in open_request_index:
            return f'URLではなく、検索結果{self.not_opened()}の中から選択してください。'
        explicit_request = self.match_index(open_request_index)
        if explicit_request != None:
           return self.check_flag_and_open(explicit_request)
        if len(open_request_index) > 10:
            if not self.embeddings:
                return f'余計な情報が多すぎます。タイトル等含めず検索結果{self.not_opened()}だけ指定してください。'
            i = get_most_suited_index(self.embeddings, self.result, query=open_request_index) + 1
            return self.check_flag_and_open(i)
            
        for i in range(1, self.n_results + 1):
           if str(i) in open_request_index:
               return self.check_flag_and_open(i)
        return f'有効な検索結果{self.not_opened()}の中から選択してください。'
        
    def references(self, tiny: bool = False):
        urls = list(set(self.referred_urls)) # Remove duplications.
        if tiny:
            import pyshorteners
            shortener = pyshorteners.Shortener()
            return [shortener.tinyurl.short(url) for url in urls if 'http' in url]
        return [url for url in urls if 'http' in url]