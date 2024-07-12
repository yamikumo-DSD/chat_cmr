import functools
import re
from lib.utils import indexed_placeholders


def replace_text(text, replacement_dict):
    for key, value in replacement_dict.items():
        text = text.replace(key, value)
    return text



def _guard_code(replace_func):
    @functools.wraps(replace_func)
    def wrapper(text: str, **kwargs):
        # Replace with placeholders.

        # Fences.
        regex_fenced = re.compile(r'(```(\w+)?\n(.*?)\n```)', re.DOTALL)
        fences = [fence[0] for fence in re.findall(regex_fenced, text)] # Captured fenced codes.
        text = indexed_placeholders(text, regex_fenced, "[|fence_{num}|]")
        
        # Inlines.
        regex_inline = r"(`[^`\n]+?`)"
        inlines = re.findall(regex_inline, text)
        text = indexed_placeholders(text, regex_inline, "[|inline_{num}|]")
        
        text = replace_func(text, **kwargs)

        for i in range(len(fences)):
            placeholder = f"[|fence_{i}|]"
            text = text.replace(placeholder, fences[i])
        for i in range(len(inlines)):
            placeholder = f"[|inline_{i}|]"
            text = text.replace(placeholder, inlines[i])
            
        return text
        
    return wrapper

@_guard_code
def _remove_excess_brs(text, tag_name):
    text = text.replace(f"<br><{tag_name}>", f"<{tag_name}>")
    text = text.replace(f"</{tag_name}><br>", f"</{tag_name}>")
    return text


def replace_charref(text) -> str:
    return replace_text(text, {
        '<': '&lt;',
        '>': '&gt;',
        ' ': '&nbsp;',
    })
    
@_guard_code
def replace_table(text) -> str:
    def split_items(text):
        text = text.replace('｜', '|')
        return text.split('|')
    
    # Regex to capture Markdown tables
    #table_pattern = r"(\|.*\|[\r\n]+\|[-|: ]*\|[\r\n]+(\|.*\|[\r\n]*)+)"
    table_pattern = r"([|｜].*[|｜][\r\n]+[|｜][ -|｜:]*\|[\r\n]+([|｜].*[|｜][\r\n]*)+)"
    html_output = text

    # Find all Markdown tables in the text
    tables = re.findall(table_pattern, text)
    for table in tables:
        # Extract the table (removing potential tuple match from groups in regex)
        table = table[0]
        lines = table.strip().split('\n')
        header = lines[0]
        rows = lines[2:]  # Skip the separator line

        # Start converting to HTML
        html_table = '<table>'
        html_table += '<tr>' + ''.join(f'<th>{col.strip()}</th>' for col in split_items(header) if col.strip()) + '</tr>'
        for row in rows:
            html_table += '<tr>' + ''.join(f'<td>{col.strip()}</td>' for col in split_items(row) if col.strip()) + '</tr>'
        html_table += '</table>\n'

        # Replace the Markdown table with HTML table in the output
        html_output = html_output.replace(table, html_table)

    return html_output

def replace_code(text) -> str:
    # Fenced.
    regex = re.compile(r'```(\w+)?\n(.*?)\n?```', re.DOTALL)
    def replacer(match) -> str:
        lang = match.group(1)
        content = replace_charref(match.group(2))
        return f'<div style="background-color: #999999; color: black;">{lang}</div><pre><code>{content}</code></pre>'
    text = re.sub(regex, replacer, text)
    
    # Inline.
    regex = r"`([^`]+?)`"
    def replacer(match) -> str:
        content = replace_charref(match.group(1))
        return f"<code>{content}</code>"
    text = re.sub(regex, replacer, text)
    
    return text

@_guard_code
def replace_bold_and_italic(text) -> str:
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"(?<!\*)\*\*(?!\*)(.+?)(?<!\*)\*\*(?!\*)", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*\*\*(?!\*)(.+?)(?<!\*)\*\*\*(?!\*)", r"<b><i>\1</i></b>", text)
    return text
    
@_guard_code
def replace_strike(text) -> str:
    return re.sub(r"(?<!~)~~(?!~)(.+?)(?<!~)~~(?!~)", r"<s>\1</s>", text)
    
@_guard_code
def replace_heading(text) -> str:
    text = re.sub(r"^\s*######(.+)", r"<h6>\1</h6>", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*#####(.+)", r"<h5>\1</h5>", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*####(.+)", r"<h4>\1</h4>", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*###(.+)", r"<h3>\1</h3>", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*##(.+)", r"<h2>\1</h2>", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*#(.+)", r"<h1>\1</h1>", text, flags=re.MULTILINE)
    return text

@_guard_code
def replace_newline(text) -> str:
    return text.replace("\n", "<br>")

@_guard_code
def replace_horizon(text) -> str:
    text = re.sub(r"^\s{,3}\*{3,}\s*", r"<hr>", text, flags=re.MULTILINE)
    text = re.sub(r"^\s{,3}-{3,}\s*", r"<hr>", text, flags=re.MULTILINE)
    text = re.sub(r"^\s{,3}_{3,}\s*", r"<hr>", text, flags=re.MULTILINE)
    return text

@_guard_code
def replace_blockquote(text) -> str:
    # This regex captures blockquotes and ensures multiline blockquotes are treated as a single blockquote
    def replacer(match):
        lines = match.group(0).strip().split("\n")
        processed_lines = "".join(f"<p>{line[1:].strip()}</p>" for line in lines)
        return f"<blockquote>{processed_lines}</blockquote>" + match.group(4) # group(4) is (\n|$)
    text = re.sub(r"((^|\n)\s*(>.+\n?)+)(\n|$)", replacer, text, flags=re.MULTILINE)
    return text


@_guard_code
def replace_unordered_list(text):
    
    def replace(markdown_text):
        lines = markdown_text.split('\n')
        html_output = ""
        indent_level = 0
        stack = []
    
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith('- '):
                current_level = (len(line) - len(stripped_line)) // 2
                if current_level > indent_level:
                    html_output += "<ul>\n"
                    stack.append(indent_level)
                    indent_level = current_level
                while current_level < indent_level:
                    html_output += "</ul>\n"
                    indent_level = stack.pop()
                html_output += f"  <li>{stripped_line[2:]}</li>\n"
        
        while indent_level > 0:
            html_output += "</ul>\n"
            indent_level = stack.pop()
    
        return html_output
        
    import re
    pattern = re.compile(r"((^|\n)(\s*-.+\n?)+)(\n|$)", flags=re.MULTILINE)
    match = re.findall(pattern, text)
    md_lists = [item[0].strip() for item in match]
    for md_list in md_lists:
        text = text.replace(md_list, replace(md_list))

    return text


def convert(text) -> str:
    #text = replace_charref(text)
    text = replace_table(text)
    text = replace_bold_and_italic(text)
    text = replace_strike(text)
    text = replace_heading(text)
    text = replace_horizon(text)
    text = replace_blockquote(text)
    text = replace_unordered_list(text)
    text = replace_newline(text)
    
    text = replace_code(text)

    tags = [
        "h1", "h2", "h3", "h4", "h5", "h6",
        "table", "hr", "blockquote", "pre", "p",
        "ul", "ol", "li",
    ]
    for tag in tags:
        text = _remove_excess_brs(text=text, tag_name=tag)
    
    return text


