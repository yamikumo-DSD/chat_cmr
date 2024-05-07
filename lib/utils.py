import re


def now() -> str:
    import datetime
    now = datetime.datetime.now()
    weekday = ['月', '火', '水', '木', '金', '土', '日']
    return f'{now.year}年{now.month}月{now.day}日 {weekday[now.weekday()]}曜日 {now.hour}時{now.minute}分'

def unbox_bracket(bracket: str, text: str) -> str:
    """ 
    Unbox bracket in the most outer position. Type of bracket can be given as a pair like '[]'.
    For example, '[hello]' is unboxed into 'hello'.
    Spaces before beginning bracket and after ending bracket are ignored.
    """
    if len(bracket) != 2: raise RuntimeError('The argument bracket only contains beginning and ending bracket.')
    begin, end = map(re.escape, bracket)  # Escape the bracket characters to use in regex pattern
    pattern = rf"^\s*{begin}(.*?){end}\s*$"
    content = re.search(pattern, text, flags=re.DOTALL)
    return content.group(1) if content else text


def caps2katakana(text: str) -> str:
    import re
    alphabet_to_katakana = {
        'A': 'エー', 'B': 'ビー', 'C': 'シー', 'D': 'ディー', 'E': 'イー',
        'F': 'エフ', 'G': 'ジー', 'H': 'エイチ', 'I': 'アイ', 'J': 'ジェー',
        'K': 'ケー', 'L': 'エル', 'M': 'エム', 'N': 'エヌ', 'O': 'オー',
        'P': 'ピー', 'Q': 'キュー', 'R': 'アール', 'S': 'エス', 'T': 'ティー',
        'U': 'ユー', 'V': 'ヴィー', 'W': 'ダブリュー', 'X': 'エックス', 'Y': 'ワイ', 'Z': 'ゼット'
    }
    def convert_to_katakana(match):
        return ''.join(alphabet_to_katakana[char] for char in match.group())
    pattern = r'[A-Z]{2,}'
    return re.sub(pattern, convert_to_katakana, text)

def english2katakana(word):
    import alkana
    return alkana.get_kana(word)

def mixed2katakana(text):
    import re
    pattern = re.compile(r'[A-Za-z]+')
    text = caps2katakana(text)
    text = pattern.sub(lambda x: english2katakana(x.group()), text)
    return text
    
    
def split_text(text, delimiters = [' ', ';', ':', '—', '。', '　', '♡', '', '!', '?', '！', '？', '\n'], max_length=100):
    import re
    # Create a pattern that matches the delimiters and keeps them in the results
    pattern = f"({'|'.join([re.escape(d) for d in delimiters if d])})"
    # Use findall to include delimiters in the results
    words_and_delimiters = re.findall(f"([^{''.join([re.escape(d) for d in delimiters if d])}]+|{'|'.join([re.escape(d) for d in delimiters if d])})", text)
    chunks = []
    current_chunk = ''
    
    for wd in words_and_delimiters:
        if len(current_chunk) + len(wd) > max_length:
            # If adding the next word exceeds max_length, start a new chunk
            chunks.append(current_chunk)
            current_chunk = wd
        else:
            # Else, add the word/delimiter to the current chunk
            current_chunk += wd

    if current_chunk:
        chunks.append(current_chunk)

    # For the case there are no deliminaters in the text.
    def force_breakdown_strings(strings, thresh: int):
        return [s[i:i+thresh] for s in strings for i in range(0, len(s), thresh)]

    return force_breakdown_strings(chunks, thresh=max_length)
    
def change_directory(new_directory):
    import os
    import functools
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            prev_directory = os.getcwd()
            os.chdir(new_directory)
            try:
                result = func(*args, **kwargs)
            finally:
                os.chdir(prev_directory)
            return result
        return wrapper
    return decorator

    
def get(x: dict, key: str):
    try: return x[key]
    except KeyError: return None



def fix_indentation(code: str, indentation: int = 4):
    """
    Replace 4*n+1(n=0, 1, 2, ...) spaces by 4*n
    Number of spaces for an indentation can be specified by argument "indentation".
    This function is mainly utilized to fix output of quantized command-r-plus, and hence in most cases, this is unnecessary.
    """
    lines = code.splitlines()
    new_lines = []
    for line in lines:
        space_count = len(re.match(r"^ *", line).group())

        if space_count % 4 == 1:
            line = line[1:] # Remove single ' '.

        new_lines.append(line)

    return "\n".join(new_lines)