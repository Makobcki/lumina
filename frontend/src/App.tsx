import { FormEvent, useMemo, useState } from 'react';

const PAGE_SIZE = 10;
const DEFAULT_API_URL = 'http://localhost:8000';
const API_BASE_URL = (import.meta.env.VITE_API_URL || DEFAULT_API_URL).replace(/\/$/, '');
const SEARCH_ENDPOINT = `${API_BASE_URL}/search`;

interface SearchResult {
  id: string;
  title: string;
  score: number;
  url: string;
  snippet: string;
  source?: string;
}

interface SearchResponse {
  query: string;
  results: SearchResult[];
}

function escapeRegExp(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function highlightSnippet(snippet: string, query: string) {
  const terms = Array.from(new Set(query.trim().split(/\s+/).filter(Boolean)));
  if (!terms.length) {
    return snippet;
  }

  const pattern = new RegExp(`(${terms.map(escapeRegExp).join('|')})`, 'gi');
  const termSet = new Set(terms.map((term) => term.toLowerCase()));
  const parts = snippet.split(pattern);

  return parts.map((part, index) =>
    termSet.has(part.toLowerCase()) ? (
      <mark key={`${part}-${index}`} className="rounded bg-sky-400/30 px-0.5 text-sky-100">
        {part}
      </mark>
    ) : (
      <span key={`${part}-${index}`}>{part}</span>
    ),
  );
}

function App() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [visibleLimit, setVisibleLimit] = useState(PAGE_SIZE);
  const [lastSearchQuery, setLastSearchQuery] = useState('');

  const hasResults = results.length > 0;
  const canLoadMore = results.length >= visibleLimit;
  const containerClassName = useMemo(
    () =>
      hasResults
        ? 'min-h-screen px-4 py-8 sm:px-6 lg:px-8'
        : 'flex min-h-screen items-center justify-center px-4 py-8',
    [hasResults],
  );

  async function performSearch(searchQuery: string, topK: number) {
    const response = await fetch(SEARCH_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query: searchQuery, top_k: topK }),
    });

    if (!response.ok) {
      throw new Error(`Search request failed with status ${response.status}`);
    }

    return (await response.json()) as SearchResponse;
  }

  async function handleSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmedQuery = query.trim();
    if (!trimmedQuery) {
      return;
    }

    setIsSearching(true);
    setError(null);
    setHasSearched(true);

    try {
      const payload = await performSearch(trimmedQuery, PAGE_SIZE);
      setVisibleLimit(PAGE_SIZE);
      setLastSearchQuery(trimmedQuery);
      setResults(payload.results);
    } catch (requestError) {
      const message = requestError instanceof Error ? requestError.message : 'Unknown search error';
      setResults([]);
      setError(message);
    } finally {
      setIsSearching(false);
    }
  }

  async function handleLoadMore() {
    const nextLimit = visibleLimit + PAGE_SIZE;
    setIsSearching(true);
    setError(null);

    try {
      const payload = await performSearch(lastSearchQuery, nextLimit);
      setVisibleLimit(nextLimit);
      setResults(payload.results);
    } catch (requestError) {
      const message = requestError instanceof Error ? requestError.message : 'Unknown search error';
      setError(message);
    } finally {
      setIsSearching(false);
    }
  }

  return (
    <div className={containerClassName}>
      <div className={hasResults ? 'mx-auto w-full max-w-4xl' : 'w-full max-w-3xl'}>
        <div className={hasResults ? 'mb-8' : 'mb-10 text-center'}>
          <p className="mb-3 text-sm font-semibold uppercase tracking-[0.4em] text-sky-300/80">Lumina</p>
          <h1 className={hasResults ? 'text-3xl font-semibold text-white' : 'text-5xl font-semibold text-white'}>
            Distributed Search Engine
          </h1>
          <p className="mt-4 text-sm text-slate-300 sm:text-base">
            Semantic retrieval for documentation and knowledge bases.
          </p>
        </div>

        <form onSubmit={handleSearch} className="rounded-3xl border border-white/10 bg-slate-900/70 p-3 shadow-2xl shadow-sky-950/30 backdrop-blur">
          <div className="flex flex-col gap-3 sm:flex-row">
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search FastAPI, React hooks, queues, documentation..."
              className="min-h-14 flex-1 rounded-2xl border border-transparent bg-slate-950/80 px-5 text-base text-slate-100 outline-none transition focus:border-sky-400"
            />
            <button
              type="submit"
              disabled={isSearching}
              className="min-h-14 rounded-2xl bg-sky-500 px-6 font-medium text-slate-950 transition hover:bg-sky-400 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-300"
            >
              {isSearching ? 'Searching…' : 'Search'}
            </button>
          </div>
        </form>

        {error ? (
          <div className="mt-6 rounded-2xl border border-rose-500/30 bg-rose-950/40 px-4 py-3 text-sm text-rose-200">
            {error}
          </div>
        ) : null}

        {hasResults ? (
          <div className="mt-8 space-y-4">
            {results.map((result) => (
              <article key={result.id} className="rounded-3xl border border-white/10 bg-slate-900/60 p-6 shadow-lg shadow-slate-950/20 backdrop-blur">
                <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
                  <div>
                    <a
                      href={result.url}
                      target="_blank"
                      rel="noreferrer"
                      className="text-xl font-semibold text-sky-300 transition hover:text-sky-200"
                    >
                      {result.title || result.url}
                    </a>
                    <p className="mt-1 break-all text-sm text-slate-400">{result.url}</p>
                  </div>
                  <div className="text-sm text-slate-400">
                    <span>Score: {result.score.toFixed(4)}</span>
                    {result.source ? <span className="ml-3">Source: {result.source}</span> : null}
                  </div>
                </div>
                <p className="mt-4 whitespace-pre-line text-sm leading-7 text-slate-200">
                  {highlightSnippet(result.snippet, lastSearchQuery)}
                </p>
              </article>
            ))}

            {canLoadMore ? (
              <div className="flex justify-center pt-2">
                <button
                  type="button"
                  disabled={isSearching}
                  onClick={handleLoadMore}
                  className="rounded-2xl border border-sky-400/40 bg-slate-900/80 px-6 py-3 text-sm font-medium text-sky-200 transition hover:border-sky-300 hover:text-sky-100 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {isSearching ? 'Loading…' : 'Load more'}
                </button>
              </div>
            ) : null}
          </div>
        ) : hasSearched && !isSearching ? (
          <div className="mt-8 rounded-3xl border border-white/10 bg-slate-900/50 p-6 text-center text-slate-300">
            No results found. Try indexing documents with the crawler and search again.
          </div>
        ) : (
          <div className="mt-8 text-center text-sm text-slate-400">
            Enter a query to search indexed chunks from documentation and crawled pages.
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
