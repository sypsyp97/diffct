/*
 * Ranking core of the documentation search. It has no DOM access, so the build script and the
 * browser use the same code.
 *
 * Search units, in the order of the vector file:
 *   chunks    one per documentation section piece (prose `t`, code `c`, sentence offsets `o`)
 *   sentences every prose sentence of every chunk, to pick the best-matching sentence
 *   pages     one per page: title, section titles and the opening prose
 * Keyword search is BM25 over chunks. Semantic search is cosine similarity over chunk vectors.
 * Reciprocal rank fusion merges both lists. A page-level semantic score lifts the chunks of the
 * best pages, and code-only chunks count less, so that code does not outrank prose.
 */

export const RRF_K = 60;
export const POOL = 50;
const PAGE_TOP = 10;
const TAG_TOP = 15;
const STOP = new Set(('a an and are as at be by can do does for from how i in is it of on or ' +
  'the this to use what when where which with').split(' '));

export function pageOf(chunk) {
  return chunk.u.split('#')[0];
}

function sentences(chunk) {
  const o = chunk.o;
  return o.map((start, i) => chunk.t.slice(start, i + 1 < o.length ? o[i + 1] : undefined).trim());
}

// ---- embedding texts ----------------------------------------------------------------------

function heading(chunk) {
  return chunk.s ? `${chunk.p} > ${chunk.s}` : chunk.p;
}

/** Texts to embed, in vector-file order: chunks, then sentences, then pages. */
export function embeddingTexts(chunks) {
  const chunkTexts = chunks.map((c) => {
    const code = c.c && c.t.length < 600 ? `${c.t ? ' ' : ''}Code: ${c.c.slice(0, 300)}` : '';
    return `title: ${heading(c)} | text: ${c.t}${code}`;
  });
  const sentenceTexts = chunks.flatMap((c) => sentences(c).map((s) => `title: ${heading(c)} | text: ${s}`));
  const pages = new Map();
  for (const c of chunks) {
    const key = pageOf(c);
    if (!pages.has(key)) pages.set(key, { title: c.p, sections: [], intro: '' });
    const page = pages.get(key);
    if (c.s && !page.sections.includes(c.s)) page.sections.push(c.s);
    if (!page.intro && c.t) page.intro = c.t.slice(0, 400);
  }
  const pageTexts = [...pages.values()].map((p) =>
    `title: ${p.title} | text: ${p.intro} Sections: ${p.sections.join('; ')}`.slice(0, 1200));
  return { chunkTexts, sentenceTexts, pageTexts };
}

// ---- tokens and BM25 ----------------------------------------------------------------------

export function tokens(text) {
  const out = [];
  for (const m of text.toLowerCase().matchAll(/[a-z0-9]+|[぀-ヿ㐀-鿿가-힯]+/g)) {
    const word = m[0];
    if (/^[a-z0-9]/.test(word)) {
      if (STOP.has(word)) continue;
      out.push(word.length > 3 && word.endsWith('s') ? word.slice(0, -1) : word);
    } else {
      for (let i = 0; i < word.length; i++) {
        out.push(word[i]);
        if (i + 1 < word.length) out.push(word.slice(i, i + 2));
      }
    }
  }
  return out;
}

export function highlightTerms(query) {
  const words = query.toLowerCase().match(/[a-z0-9_]{2,}/g) || [];
  return [...new Set(words.filter((w) => !STOP.has(w)))];
}

function buildBm25(chunks) {
  const docs = chunks.map((c) => {
    const counts = new Map();
    const title = tokens(`${c.p} ${c.s}`);
    const body = tokens(`${c.t} ${c.c}`);
    for (const t of title) counts.set(t, (counts.get(t) || 0) + 3);
    for (const t of body) counts.set(t, (counts.get(t) || 0) + 1);
    return { counts, length: title.length * 3 + body.length };
  });
  const df = new Map();
  for (const d of docs) for (const t of d.counts.keys()) df.set(t, (df.get(t) || 0) + 1);
  const avg = docs.reduce((s, d) => s + d.length, 0) / docs.length;
  return { docs, df, avg, n: docs.length };
}

// ---- index --------------------------------------------------------------------------------

/** Build the search index from the chunk list, the metadata and the vector file. */
export function createIndex(chunks, meta, buffer) {
  const n = chunks.length;
  const sentenceStart = [];
  let m = 0;
  for (const c of chunks) { sentenceStart.push(m); m += c.o.length; }
  const pageNames = [...new Set(chunks.map(pageOf))];
  const pageIndex = new Map(pageNames.map((p, i) => [p, i]));
  const total = n + m + pageNames.length;
  if (buffer.byteLength !== total * 4 + total * meta.dim) throw new Error('search index is out of date');
  return {
    chunks, meta, n, m, pageNames, sentenceStart,
    pageOfChunk: chunks.map((c) => pageIndex.get(pageOf(c))),
    scales: new Float32Array(buffer, 0, total),
    codes: new Int8Array(buffer, total * 4, total * meta.dim),
    bm25: buildBm25(chunks),
  };
}

// ---- ranking ------------------------------------------------------------------------------

/** Keyword ranking: ids of the best chunks. */
export function keywordSearch(index, query) {
  const terms = [...new Set(tokens(query))];
  const { docs, df, avg, n } = index.bm25;
  const k1 = 1.2;
  const b = 0.75;
  const scored = [];
  docs.forEach((d, i) => {
    let score = 0;
    for (const t of terms) {
      const f = d.counts.get(t);
      if (!f) continue;
      const idf = Math.log(1 + (n - df.get(t) + 0.5) / (df.get(t) + 0.5));
      score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * d.length / avg));
    }
    if (score > 0) scored.push([i, score]);
  });
  scored.sort((x, y) => y[1] - x[1]);
  return scored.slice(0, POOL).map((s) => s[0]);
}

function dot(index, unit, q) {
  const { dim } = index.meta;
  const base = unit * dim;
  let sum = 0;
  for (let j = 0; j < dim; j++) sum += q[j] * index.codes[base + j];
  return sum * index.scales[unit];
}

/** Normalize the query vector to the stored dimension. */
export function queryVector(index, full) {
  const { dim } = index.meta;
  const q = Float32Array.from(full.subarray(0, dim));
  const norm = Math.hypot(...q);
  return q.map((x) => x / norm);
}

/** Semantic ranking: ids of the best chunks and the best pages. */
export function semanticSearch(index, q) {
  const chunkScores = [];
  for (let i = 0; i < index.n; i++) chunkScores.push([i, dot(index, i, q)]);
  chunkScores.sort((x, y) => y[1] - x[1]);
  const pageBase = index.n + index.m;
  const pageScores = index.pageNames.map((_, i) => [i, dot(index, pageBase + i, q)]);
  pageScores.sort((x, y) => y[1] - x[1]);
  return {
    q,
    ids: chunkScores.slice(0, POOL).map((s) => s[0]),
    pages: pageScores.slice(0, PAGE_TOP).map((s) => s[0]),
  };
}

/**
 * Merge keyword ids and a semantic result (or null) into groups of results by page.
 * Each group has up to `perPage` visible items and up to `extra` more behind a toggle.
 */
export function combine(index, keywordIds, semantic, {
  maxGroups, perPage = 2, extra = 3, pageWeight = 1, codeOnlyWeight = 0.85, referenceWeight = 0.8,
}) {
  const score = new Map();
  const add = (id, value) => score.set(id, (score.get(id) || 0) + value);
  const kwRank = new Map(keywordIds.map((id, r) => [id, r]));
  keywordIds.forEach((id, r) => add(id, 1 / (RRF_K + r + 1)));
  const semRank = new Map();
  if (semantic) {
    semantic.ids.forEach((id, r) => { semRank.set(id, r); add(id, 1 / (RRF_K + r + 1)); });
    const pageRank = new Map(semantic.pages.map((p, r) => [p, r]));
    for (const id of score.keys()) {
      const r = pageRank.get(index.pageOfChunk[id]);
      if (r !== undefined) add(id, pageWeight / (RRF_K + r + 1));
    }
  }
  for (const id of score.keys()) {
    const chunk = index.chunks[id];
    if (chunk.t.length < 80) score.set(id, score.get(id) * codeOnlyWeight);
    if (chunk.u.startsWith('generated/')) score.set(id, score.get(id) * referenceWeight);
  }
  const ranked = [...score.entries()].sort((x, y) => y[1] - x[1]);
  const groups = new Map();
  const seen = new Set();
  for (const [id, value] of ranked) {
    const chunk = index.chunks[id];
    if (seen.has(chunk.u)) continue;
    seen.add(chunk.u);
    const page = pageOf(chunk);
    if (!groups.has(page)) groups.set(page, { page, title: chunk.p, items: [], more: [] });
    const g = groups.get(page);
    const inKw = kwRank.has(id) && kwRank.get(id) < TAG_TOP;
    const inSem = semRank.has(id) && semRank.get(id) < TAG_TOP;
    let tag = inKw && inSem ? 'both' : inKw ? 'keyword' : inSem ? 'semantic' : (kwRank.has(id) ? 'keyword' : 'semantic');
    const item = { id, tag, score: value };
    if (g.items.length < perPage) g.items.push(item);
    else if (g.more.length < extra) g.more.push(item);
  }
  return [...groups.values()].slice(0, maxGroups);
}

// ---- snippets -----------------------------------------------------------------------------

function hits(text, terms) {
  const lower = text.toLowerCase();
  return terms.filter((t) => lower.includes(t)).length;
}

function clip(text, terms, width) {
  if (text.length <= width) return text;
  const lower = text.toLowerCase();
  let at = -1;
  for (const t of terms) {
    const i = lower.indexOf(t);
    if (i >= 0 && (at < 0 || i < at)) at = i;
  }
  const start = Math.max(0, at - 50);
  return (start > 0 ? '... ' : '') + text.slice(start, start + width).trim() + (start + width < text.length ? ' ...' : '');
}

/** The sentence to show for a result: the best keyword sentence, else the best semantic one. */
export function snippet(index, item, terms, semantic) {
  const chunk = index.chunks[item.id];
  const list = sentences(chunk);
  if (!list.length) return { text: clip(chunk.c, terms, 200), code: true };
  let best = 0;
  let bestHits = 0;
  list.forEach((s, i) => { const h = hits(s, terms); if (h > bestHits) { best = i; bestHits = h; } });
  if (!bestHits && semantic && (item.tag === 'semantic' || item.tag === 'both')) {
    let top = -Infinity;
    list.forEach((_, i) => {
      const value = dot(index, index.n + index.sentenceStart[item.id] + i, semantic.q);
      if (value > top) { top = value; best = i; }
    });
  }
  return { text: clip(list[best], terms, 240), code: false };
}
