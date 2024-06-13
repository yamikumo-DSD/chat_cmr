import pandas as pd				


def youtube_url2id(url: str) -> str:
    import re
    match = re.search(r"https://www\.youtube\.com/watch\?v=(.+)", url)
    if not match:
        return None
    id_ = match.group(1).split("&")[0]
    return id_
    
def is_youtube_url(url: str) -> bool:
    return youtube_url2id(url) is not None
    

def fetch_youtube_transcript(video_url: str) -> str:
    from youtube_transcript_api import YouTubeTranscriptApi
    import re
    video_id = youtube_url2id(video_url) 
    if not video_id:
        raise ValueError("Invalid Youtube URL.")
    
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    transcript_doc = ""
    for transcript in transcript_list:
        for tr in transcript.fetch():
            transcript_doc += tr["text"]
    transcript_doc = re.sub(r"\[(.+?)\]", r"",transcript_doc)
    return transcript_doc


def fetch_5ch_posts_df(url: str) -> pd.DataFrame:
    from bs4 import BeautifulSoup
    from pathlib import Path
    import requests
    
    response = requests.get(url)
    html_data = response.content
    soup = BeautifulSoup(html_data, "html.parser")
    
    articles = soup.find_all("article", class_="clear post")
    post_data = []
    
    for article in articles:
        details = article.find("details")
    
        postid = int(details.find("span", class_="postid").text)
        username = details.find("b").text.strip()
        userid = details.find("span", class_="uid").text.split(":")[-1].strip()
        date = details.find("span", class_="date").text
    
        content = article.find("section", class_="post-content").get_text("\n", strip=True)
    
        post_data.append([postid, username, userid, date, content])
    
    return pd.DataFrame(
        post_data, columns=["postid", "user_name", "user_id", "date", "content"]
    )


def fetch_5ch_posts_str(url: str) -> str:
    items = []
    for post in fetch_5ch_posts_df(url).iterrows():
        post_id = post[1][0]
        user_name = post[1][1]
        user_id = post[1][2]
        date = post[1][3]
        content = post[1][4].strip()
        item = f"""{post_id} 名前:{user_name} 投稿日:{date} ID:{user_id}
{content}"""
        items.append(item)
    return "\n---\n".join(items)

def is_bbspink_board(url: str) -> bool:
    import re
    pattern = r"https://\w+\.bbspink\.com/test/read\.cgi/.+/\d+/.+"
    match = re.search(pattern, url)
    return bool(match)

def is_5ch_board(url: str, involve_bbspink: bool = False) -> bool:
    import re
    pattern = r"https://\w+.5ch.net/test/read.cgi/.+/\d+/.+"
    match = re.search(pattern, url)
    
    if involve_bbspink:
        return bool(match) or is_bbspink_board(url)
    else:
        return bool(match)


def fetch_source_by_chrome(url: str, get_iframes: bool = False) -> str:
    from selenium import webdriver
    from bs4 import BeautifulSoup
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager

    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Enables headless mode
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    driver.get(url)

    if get_iframes:
        main_page_source = driver.page_source
        soup = BeautifulSoup(main_page_source, 'html.parser')
        iframes = soup.find_all('iframe')
        for index, _ in enumerate(iframes):
            iframe_elements = driver.find_elements(By.TAG_NAME, 'iframe')
            if len(iframe_elements) == 0: continue
            driver.switch_to.frame(iframe_elements[index])
            iframe_content = driver.page_source
            iframe_soup = BeautifulSoup(iframe_content, 'html.parser')
            iframe_text = ' '.join(iframe_soup.stripped_strings)
            new_tag = soup.new_tag("div")
            new_tag.string = iframe_text
            iframes[index].replace_with(new_tag)
            driver.switch_to.default_content()
        page_source = str(soup)
    else:
        page_source = driver.page_source
        
    driver.quit()
    return page_source
    
def fetch_contents_of(
    url: str, 
    join: str = ' ', 
    use_chrome: bool = False, 
    timeout: tuple = (5.0, 5.0),
    **kwargs,
) -> str:
    """
    Args:
        url: URL to fetch contents of.
        join (str: default " "): Texts are joined by the token.
        use_chrome (bool: default True): Flag to use Selenium Chrome driver to fetch contents.
        timeout (tuple: default (5.0, 5.0)): (time until connection established, time until download completes)
        **kwargs: Additional kwargs passed to requests or Selenium.
    """
    from bs4 import BeautifulSoup
    import requests
    import os
    import pdfplumber


    if use_chrome:
        try: content = fetch_source_by_chrome(url, get_iframes=False, **kwargs)
        except: return f'Access denied by the webpage ({url})'
    else:
        try: 
            if is_youtube_url(url):
                return fetch_youtube_transcript(url)
            if is_5ch_board(url, involve_bbspink=True):
                return fetch_5ch_posts_str(url)
            
            response = requests.get(url, timeout=timeout, **kwargs)
            
            if 'application/pdf' in response.headers.get('Content-Type', ''):
                temp_file_name = 'fetched.pdf'
                with open(temp_file_name, 'wb') as f:
                    f.write(response.content)
                with pdfplumber.open(temp_file_name) as pdf:
                    text = ''
                    for page in pdf.pages:
                        text += page.extract_text()
                os.remove(temp_file_name)
                return text
        except: 
            return f'Access denied by the webpage ({url})'
        content = response.content
    
    soup = BeautifulSoup(content, 'html.parser')
    removed_tags = ['script', 'style', 'button',  
                    'header', 'footer', 'nav', 'iframe']
    for script_or_style in soup(removed_tags):
        script_or_style.decompose()
    return join.join(soup.stripped_strings)

def get_google_search_results(
    query, 
    api_key: str,
    cse_id: str,
    start_index: int = 1, 
    n_results: int = 10,
):
    from googleapiclient.discovery import build				
    service = build("customsearch",	
                    "v1", 
                    cache_discovery=False, 
                    developerKey=api_key)
    result = service.cse().list(
        q=query, 
        cx=cse_id,
        num=n_results, 
#        lr='lang_ja',
        start=start_index
    ).execute()				
    return result				
				
def google_search_results_df(
    query, 
    api_key: str,
    cse_id: str,
    n_results: int = 10,
) -> pd.DataFrame:
    from datetime import datetime, timedelta, timezone				
				
    data = get_google_search_results(
        query, 
        api_key, 
        cse_id,
        1, 
        n_results, 
    )
    
    total_results = int(data['searchInformation']['totalResults'])				
    if total_results == 0:
        # Returns DataFrame with zero items.
        return pd.DataFrame(columns=["rank", "title", "url", "snippet"])

    items = data['items']
				
    result = []				
    num_items = len(items) if len(items) < n_results else n_results
    for i in range(num_items):				
        title = items[i]['title'] if 'title' in items[i] else ''				
        link = items[i]['link'] if 'link' in items[i] else ''				
        snippet = items[i]['snippet'] if 'snippet' in items[i] else ''				
        result.append('\t'.join([str(i+1), title, link, snippet]))				
				
    # List->DataFrame				
    df_search_results = pd.DataFrame(result)[0].str.split('\t', expand=True)				
    df_search_results.columns = ['rank', 'title', 'url', 'snippet']				
    return df_search_results



def ddg_search_results_df(
    query, 
    n_results: int = 10,
    region='wt-wt',
    safesearch='off',
    timelimit=None,
    **kwargs,
) -> pd.DataFrame:
    """
    Args:
        region: "jp-jp"(Japan), "wt-wt"(Not-specified)
        safesearch: "off", "on", or "moderate"
        timelimit: None, "d"(1 day), "w"(1 week), "m"(1 month), "y"(1 year)
        kwargs: Other kwargs for DDGS.
    """
    
    import nest_asyncio
    import duckduckgo_search
    nest_asyncio.apply()
    
    with duckduckgo_search.DDGS() as ddgs:
        items = list(ddgs.text(
            keywords=query,
            region='wt-wt', 
            safesearch='off',
            timelimit=timelimit,
            max_results=n_results
        ))
    if len(items) == 0:
        # Returns DataFrame with zero items.
        return pd.DataFrame(columns=["rank", "title", "url", "snippet"])

        
    result = []				
    num_items = len(items) if len(items) < n_results else n_results
    for i in range(num_items):				
        title = items[i]['title']
        link = items[i]['href']
        snippet = items[i]['body']
        result.append('\t'.join([str(i+1), title, link, snippet]))				
				
    # List->DataFrame				
    df = pd.DataFrame(result)[0].str.split('\t', expand=True)				
    df.columns = ['rank', 'title', 'url', 'snippet']				

    return df

DEFAULT_ONION_PROXIES = {
    'http' : "socks5h://localhost:9150",
    'https' : "socks5h://localhost:9150"
}


def torch_results_df(
    query, 
    n_results: int = 10,
    tor_port: int = 9150,
    timeout: tuple = (60, 60),
    **kwargs,
) -> pd.DataFrame:
    """
    Dark web search (TORCH). Launch Tor Browser before use this.
    I made this for experiment but ofc don't recommend to use this simply because of security.
    Args:
        tor_port (int, default: 9150): Not sure this is configurable. Anyway, default port for Tor connection is 9150.
        timeout (tuple, default:(60, 60)): Timeout per page. (time_until_connection_established, time_until_fetch_empletes.).
    Raises:
        ConnectionError: When failed to establish connection maybe due to Tor Browser not launched.
    Onion URL:
        http://supportvojao2z6taveolgpvgcz5k4v7smwgjcuzz5ahs5ctnscuejyd.onion/index.php?topic=7071.0
    Reference:
        https://tarenagashi.hatenablog.jp/entry/2023/10/06/000641
    """
    import pandas as pd
    from numpy import ceil
    
    N_ITEMS_PER_PAGE = 10

    def render_url(query: str, page: int) -> str:
        BASE = "http://xmh57jrknzkhv6y3ls3ubitzfqnkrwxhopf5aygthi7d6rplyvk3noyd.onion/cgi-bin/omega/omega"
        P = "+".join(query.split())
        url = f"{BASE}?P={P}&DEFAULTOP=and&[={page}]"
        return url

    def fetch_items_from_page(query: str, page: int) -> list:
        import requests
        from bs4 import BeautifulSoup
        proxies = {
            "http": f"socks5h://localhost:{tor_port}",
            "https": f"socks5h://localhost:{tor_port}",
        }
        
        res = requests.get(render_url(query, page), proxies=proxies, timeout=timeout)
        html = res.text
    
        soup = BeautifulSoup(html, 'html.parser')
        rows = soup.find_all('tr')
        items = []
        for i, row in enumerate(rows):
            url = row.find('a')['href']
            snippet = row.find('small').get_text()
            title = row.find('a').get_text()
            rank = N_ITEMS_PER_PAGE * page + i
            items.append((rank, title, url, snippet))

        return items
        
        
    n_pages_to_scrape = int(ceil(n_results / N_ITEMS_PER_PAGE))
    items = []
    for i in range(n_pages_to_scrape):
        new_items = fetch_items_from_page(query, page=i)
        if len(new_items) == 0: break
        items += new_items
        
    if len(items) == 0:
        # Returns DataFrame with zero items.
        return pd.DataFrame(columns=["rank", "title", "url", "snippet"])

    df = pd.DataFrame(items, columns=['rank', 'title', 'url', 'snippet'])
    return df[:n_results]