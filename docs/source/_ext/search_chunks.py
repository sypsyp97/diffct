"""Write the section chunks that the in-browser semantic search embeds and displays.

After the HTML build, every page is split into chunks of at most MAX_CHARS characters.
A chunk keeps the page title, the section title and the URL of its anchor.
`docs/search/embed.mjs` reads `_static/search-chunks.json` and writes the vectors.
"""

import json
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

MAX_CHARS = 1800
MIN_CHARS = 60
MIN_PAGE_CHARS = 200  # shorter pages are redirect stubs
# The changelog repeats words from every page and outranks the guides. Sphinx search still covers it.
SKIP_DOCS = {'search', 'genindex', 'py-modindex', 'changelog'}
HEADINGS = {'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}


def _clean(node):
    for junk in node.select('.headerlink, button.copybtn, script, style, .linenos'):
        junk.decompose()
    for pre in node.find_all('pre'):
        pre.replace_with(NavigableString(' ' + pre.get_text() + ' '))


def _text(node):
    return ' '.join(node.get_text(' ', strip=True).split())


def _blocks(section):
    """Yield (anchor id or None, text) for the content directly under one section."""
    for child in section.children:
        if isinstance(child, NavigableString):
            continue
        if child.name == 'section' or child.name in HEADINGS:
            continue
        if child.name == 'dl' and 'py' in (child.get('class') or []):
            for dt in child.find_all('dt', recursive=False):
                dd = dt.find_next_sibling('dd')
                body = _text(dd) if dd is not None else ''
                yield dt.get('id'), f'{_text(dt)} {body}'.strip()
            continue
        text = _text(child)
        if text:
            yield None, text


def _split(pieces, section_id):
    """Pack (anchor, text) pieces into chunks of at most MAX_CHARS characters."""
    chunks, current, size, anchor = [], [], 0, section_id
    for piece_anchor, text in pieces:
        while len(text) > MAX_CHARS:
            cut = text.rfind('\n', 0, MAX_CHARS)
            cut = text.rfind(' ', 0, MAX_CHARS) if cut < MAX_CHARS // 2 else cut
            cut = cut if cut > 0 else MAX_CHARS
            head, text = text[:cut].strip(), text[cut:].strip()
            if current:
                chunks.append((anchor, '\n'.join(current)))
                current, size = [], 0
            chunks.append((piece_anchor or section_id, head))
            piece_anchor = None
        if current and size + len(text) > MAX_CHARS:
            chunks.append((anchor, '\n'.join(current)))
            current, size, anchor = [], 0, piece_anchor or section_id
        if not current:
            anchor = piece_anchor or anchor
        current.append(text)
        size += len(text) + 1
    if current:
        chunks.append((anchor, '\n'.join(current)))
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
        for anchor, text in _split(list(_blocks(section)), section.get('id')):
            if len(text) < MIN_CHARS:
                continue
            if anchor == top.get('id'):
                anchor = None
            out.append({
                'u': url + (f'#{anchor}' if anchor else ''),
                'p': page_title,
                's': '' if section is top else title,
                't': text,
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
            if sum(len(c['t']) for c in page) >= MIN_PAGE_CHARS:
                chunks.extend(page)
    target = Path(app.outdir) / '_static' / 'search-chunks.json'
    target.write_text(json.dumps(chunks, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')


def setup(app):
    app.connect('build-finished', write_chunks)
    return {'parallel_read_safe': True, 'parallel_write_safe': True}
