import re

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
    pattern = re.compile(r'[A-Za-z]+')
    text = caps2katakana(text)
    text = pattern.sub(lambda x: english2katakana(x.group()), text)
    return text
    
    
def split_text(text, delimiters = [' ', ';', ':', '—', '。', '　', '♡', '', '!', '?', '！', '？', '\n'], max_length=100):
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



def fix_indentation(code: str, indentation: int = 4):
    """
    Replace 4*n+1(n=0, 1, 2, ...) spaces by 4*n
    Number of spaces for an indentation can be specified by argument "indentation".
    This function is mainly utilized to fix output of quantized command-r-plus, and hence in most cases, this is unnecessary.
    """
    def fix(line):
        import re
        n_spaces = len(re.match(r"^ *", line).group())
        return line[1:] if n_spaces % indentation == 1 else line

    return "\n".join(map(fix, code.splitlines()))


def strip_fence(code: str) -> str:
    import re
    pattern = re.compile(r'```(\w+)?\n(.*?)\n```', re.DOTALL)
    match = re.search(pattern, code)
    return match.group(2).strip() if match else code


def reformat_python_code(code: str) -> str:
    import black
    code = fix_indentation(code)
    code = strip_fence(code)
    try:  code = black.format_str(code, mode=black.Mode())
    except:  pass
    return code


def replace_text(text, replacement_dict):
    for key, value in replacement_dict.items():
        text = text.replace(key, value)
    return text




def indexed_placeholders(target: str, pattern: str, placeholder: str, key: str = "num"):
    """
    Replace substrings in target matched to the pattern by indexed placeholder.
    
    Args:
        target: Target string containing some pattern.
        pattern: Regex string.
        placeholder: Format string containing "{num}" replaced by indexes.
        key: Used if you don't want to use "num" as a key.

    Retuned:
        Replace string.
    
    Example:
        target="**alphabeta**"
        pattern=r"(alpha|beta)"
        placeholder="<|{num}|>"
        key="num"
        -> Retuned string is gonna be "**<|0|><|1|>**"
    """
    import re
    
    matches = list(re.finditer(pattern, target))

    offset = 0
    for i, match in enumerate(matches):
        start, end = match.start() + offset, match.end() + offset
        current_placeholder = placeholder.format(**{key: str(i)})
        target = target[:start] + current_placeholder + target[end:]
        offset += len(current_placeholder) - (end - start)

    return target



def rstrip_seq(text: str, seq: str) -> str:
    """
    Example:
        rstrip_seq("hello worl", "world") -> "hello "
    """
    for i in range(len(seq)):
        incomplete_seq = seq[:len(seq) - i]
        if text.endswith(incomplete_seq):
            return text[:-len(incomplete_seq)]
    return text
    
def lstrip_seq(text: str, seq: str) -> str:
    """
    Example:
        lstrip_seq("hel world", "hello") -> " world"
    """
    for i in range(len(seq) + 1):
        incomplete_seq = seq[:len(seq) - i]
        if text.startswith(incomplete_seq):
            return text[len(incomplete_seq):]
    return text



def parse_streaming_xml(xml_fragment, no_root_tag: bool = True):
    from lxml import etree
    import xmltodict
    try:
        if no_root_tag:
            xml_fragment = "<root>" + xml_fragment
        parser = etree.XMLParser(recover=True)
        root = etree.fromstring(xml_fragment, parser=parser)
        if root:
            return xmltodict.parse(
                etree.tostring(root, pretty_print=True).decode()
            )["root"]
        else:
            return {}
    except etree.XMLSyntaxError as e:
        return f"Error parsing XML: {e}"

        
def dict_to_html_list(d: dict|list|int|float|str, str_cutoff: int = 100):
    html_list = '<ul>'

    if isinstance(d, dict):
        for key, value in d.items():
            value = dict_to_html_list(value, str_cutoff)
            html_list += f'<li>{key}: {value}</li>'
    elif isinstance(d, list):
        for value in d:
            value = dict_to_html_list(value, str_cutoff)
            html_list += f'<li>{value}</li>'
    elif isinstance(d, str):
        if len(d) > str_cutoff:
            d = d[:str_cutoff] + "..."
        d = replace_text(d, {">": "&gt;", "<": "&lt;"})
        html_list += d
    else:
        html_list += "Cannot display this data"
            
    html_list += '</ul>'
    return html_list