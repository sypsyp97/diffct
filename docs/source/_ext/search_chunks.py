"""Write the section chunks that the in-browser semantic search embeds and displays.

After the HTML build, every page is split into chunks of at most MAX_CHARS characters.
A chunk keeps the page title, the section path, the URL of its anchor, the prose (`t`), the code
(`c`) and the sentence start offsets of the prose (`o`).
`docs/search/embed.mjs` reads `_static/search-chunks.json` and writes the vectors.
"""

import json
import re
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

MAX_CHARS = 1800
MAX_CODE_CHARS = 1500
MIN_SENTENCE = 40
MAX_SENTENCES = 16
MIN_CHARS = 60
MIN_PAGE_CHARS = 200  # shorter pages are redirect stubs
# The changelog repeats words from every page and outranks the guides. Sphinx search still covers it.
SKIP_DOCS = {'search', 'genindex', 'py-modindex', 'changelog'}
HEADINGS = {'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}


def _clean(node):
    for junk in node.select('.headerlink, button.copybtn, script, style, .linenos'):
        junk.decompose()


def _text(node):
    return ' '.join(node.get_text(' ', strip=True).split())


def _prose_and_code(node):
    """Return the text of a node without its code blocks, and the text of those blocks."""
    code = []
    for pre in node.find_all('pre'):
        code.append(' '.join(pre.get_text().split()))
        pre.decompose()
    return _text(node), ' '.join(code)


def _blocks(section):
    """Yield (anchor id or None, prose, code) for the content directly under one section."""
    for child in section.children:
        if isinstance(child, NavigableString):
            continue
        if child.name == 'section' or child.name in HEADINGS:
            continue
        if child.name == 'dl' and 'py' in (child.get('class') or []):
            for dt in child.find_all('dt', recursive=False):
                dd = dt.find_next_sibling('dd')
                prose, code = _prose_and_code(dd) if dd is not None else ('', '')
                yield dt.get('id'), prose, f'{_text(dt)} {code}'.strip()
            continue
        prose, code = _prose_and_code(child)
        if prose or code:
            yield None, prose, code


def _sentence_starts(text):
    """Offsets where sentences start. Fragments shorter than MIN_SENTENCE merge into the next."""
    starts = [0]
    for match in re.finditer(r'(?<=[.!?])\s+(?=[A-Z0-9`"(])', text):
        if match.end() - starts[-1] >= MIN_SENTENCE:
            starts.append(match.end())
    return starts[:MAX_SENTENCES]


def _split(pieces, section_id):
    """Pack (anchor, prose, code) pieces into chunks of at most MAX_CHARS characters."""
    chunks, prose_parts, code_parts, size, anchor = [], [], [], 0, section_id

    def flush():
        nonlocal prose_parts, code_parts, size
        if prose_parts or code_parts:
            chunks.append((anchor, '\n'.join(prose_parts), ' '.join(code_parts)[:MAX_CODE_CHARS]))
        prose_parts, code_parts, size = [], [], 0

    for piece_anchor, prose, code in pieces:
        while len(prose) > MAX_CHARS:
            cut = prose.rfind(' ', 0, MAX_CHARS)
            cut = cut if cut > 0 else MAX_CHARS
            head, prose = prose[:cut].strip(), prose[cut:].strip()
            flush()
            chunks.append((piece_anchor or section_id, head, ''))
            piece_anchor = None
        if (prose_parts or code_parts) and size + len(prose) + len(code) // 2 > MAX_CHARS:
            flush()
            anchor = piece_anchor or section_id
        if not (prose_parts or code_parts):
            anchor = piece_anchor or anchor
        if prose:
            prose_parts.append(prose)
        if code:
            code_parts.append(code)
        size += len(prose) + len(code) // 2 + 1
    flush()
    return chunks


def _page_chunks(html_path, url):
    soup = BeautifulSoup(html_path.read_text(encoding='utf-8'), 'html.parser')
    article = soup.select_one('article.bd-article')
    if article is None:
        return []
    _clean(article)
    top = article.find('section')
    if top is None:
        return []
    page_title = _text(top.find(HEADINGS))
    out = []
    for section in [top, *top.find_all('section')]:
        trail = [s for s in [*section.find_parents('section'), section] if s is not top]
        title = ' > '.join(_text(s.find(HEADINGS, recursive=False)) for s in trail)
        for anchor, prose, code in _split(list(_blocks(section)), section.get('id')):
            if len(prose) + len(code) < MIN_CHARS:
                continue
            if anchor == top.get('id'):
                anchor = None
            out.append({
                'u': url + (f'#{anchor}' if anchor else ''),
                'p': page_title,
                's': '' if section is top else title,
                't': prose,
                'c': code,
                'o': _sentence_starts(prose) if prose else [],
            })
    return out


def write_chunks(app, exception):
    if exception is not None or app.builder.format != 'html':
        return
    chunks = []
    for docname in sorted(app.env.found_docs):
        if docname in SKIP_DOCS:
            continue
        html_path = Path(app.outdir) / f'{docname}.html'
        if html_path.exists():
            page = _page_chunks(html_path, f'{docname}.html')
            if sum(len(c['t']) + len(c['c']) for c in page) >= MIN_PAGE_CHARS:
                chunks.extend(page)
    target = Path(app.outdir) / '_static' / 'search-chunks.json'
    target.write_text(json.dumps(chunks, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')


def setup(app):
    app.connect('build-finished', write_chunks)
    return {'parallel_read_safe': True, 'parallel_write_safe': True}
