import pandas as pd				

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
    use_chrome: bool = True, 
    timeout: tuple = (5.0, 5.0)
) -> str:
    from bs4 import BeautifulSoup
    import requests
    import os
    import pdfplumber


    if use_chrome:
        try: content = fetch_source_by_chrome(url, get_iframes=False)
        except: return f'Access denied by the webpage ({url})'
    else:
        try: 
            response = requests.get(url, timeout=timeout)
            
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