import React, { useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import Papa from "papaparse";
import "./styles.css";

const LOCAL_API_URL = import.meta.env.VITE_LOCAL_API_URL || "http://localhost:8000";
const MAX_ROWS = 500;
const PLATFORMS = ["All Shops", "Shopee", "TikTok", "Lazada"];
const ASPECT_KEYWORDS = {
  Delivery: ["delivery", "post", "courier", "shipped", "arrive", "sampai", "penghantaran"],
  Packaging: ["packaging", "pack", "bungkus", "kotak", "kemas", "kemik"],
  Price: ["price", "harga", "murah", "mahal", "value", "worth"],
  Quality: ["quality", "kualiti", "barang", "product", "item", "rosak"],
  "Customer service": ["customer", "service", "seller", "chat", "reply", "refund"],
  "Size / fit": ["size", "saiz", "fit", "muat", "besar", "kecil"],
};

function normalizeRow(row, index) {
  const value = (keys, fallback = "") => {
    const key = keys.find((candidate) => row[candidate] !== undefined);
    return key ? String(row[key] ?? "").trim() : fallback;
  };
  return {
    id: value(["review_id", "id"], String(index + 1)),
    date: value(["date", "review_date"]),
    platform: value(["platform"], "Unknown"),
    text: value(["review_text", "text", "review"]),
  };
}

function detectAspects(text) {
  const lower = text.toLowerCase();
  return Object.entries(ASPECT_KEYWORDS)
    .filter(([, words]) => words.some((word) => new RegExp(`\\b${word}\\b`).test(lower)))
    .map(([aspect]) => aspect);
}

function download(name, content, type) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  URL.revokeObjectURL(url);
}

function App() {
  const [rows, setRows] = useState([]);
  const [results, setResults] = useState([]);
  const [platform, setPlatform] = useState("All Shops");
  const [withAspects, setWithAspects] = useState(true);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState("");
  const [error, setError] = useState("");

  const filteredRows = useMemo(
    () =>
      rows.filter(
        (row) => platform === "All Shops" || row.platform.toLowerCase() === platform.toLowerCase(),
      ),
    [rows, platform],
  );

  const metrics = useMemo(() => {
    const counts = { Positive: 0, Negative: 0, Neutral: 0 };
    results.forEach((result) => {
      counts[result.sentiment] += 1;
    });
    return {
      total: results.length,
      counts,
      negativeRatio: results.length ? counts.Negative / results.length : 0,
    };
  }, [results]);

  const negativeKeywords = useMemo(() => {
    const words = {};
    results
      .filter((result) => result.sentiment === "Negative")
      .forEach((result) =>
        result.text
          .toLowerCase()
          .replace(/[^a-z0-9\s'-]/g, " ")
          .split(/\s+/)
          .filter((word) => word.length > 2)
          .forEach((word) => {
            words[word] = (words[word] || 0) + 1;
          }),
      );
    return Object.entries(words)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 8)
      .map(([word]) => word);
  }, [results]);

  function readCsv(file) {
    setError("");
    setResults([]);
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: ({ data }) => {
        const parsed = data.map(normalizeRow).filter((row) => row.text);
        if (!parsed.length) {
          setError("CSV must contain a review_text, text, or review column.");
        } else if (parsed.length > MAX_ROWS) {
          setError(`CSV has ${parsed.length} rows. The limit is ${MAX_ROWS}.`);
        } else {
          setRows(parsed);
        }
      },
      error: (parseError) => setError(`Could not parse CSV: ${parseError.message}`),
    });
  }

  async function analyze() {
    if (!filteredRows.length) {
      setError("Upload a CSV with reviews first.");
      return;
    }
    setError("");
    setRunning(true);
    setResults([]);
    console.groupCollapsed("[sentiment] Starting local CPU analysis");
    console.info("[sentiment] Rows queued:", filteredRows.length);
    console.info("[sentiment] Endpoint:", `${LOCAL_API_URL}/classify`);
    try {
      setProgress(`Analyzing ${filteredRows.length} reviews on your CPU...`);
      const response = await fetch(`${LOCAL_API_URL}/classify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ texts: filteredRows.map((row) => row.text) }),
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(detail.detail || `API returned HTTP ${response.status}`);
      }
      const { predictions } = await response.json();
      const nextResults = filteredRows.map((row, index) => ({
          ...row,
          sentiment: predictions[index].sentiment,
          score: predictions[index].score,
          aspects: withAspects ? detectAspects(row.text) : [],
        }));
      setResults(nextResults);
      console.info("[sentiment] Analysis completed:", nextResults.length);
    } catch (analysisError) {
      const message = analysisError instanceof Error ? analysisError.message : String(analysisError);
      console.error("[sentiment] Analysis failed", {
        message,
        error: analysisError,
        endpoint: `${LOCAL_API_URL}/classify`,
      });
      setError(`The local CPU model could not be reached: ${message}`);
    } finally {
      console.groupEnd();
      setRunning(false);
      setProgress("");
    }
  }

  function exportResults(format) {
    if (!results.length) return;
    if (format === "json") {
      download("sentiment-results.json", JSON.stringify(results, null, 2), "application/json");
      return;
    }
    download("sentiment-results.csv", Papa.unparse(results), "text/csv");
  }

  return (
    <main className="app-shell">
      <header className="brand-header">
        <div className="brand-mark">SA</div>
        <div>
          <div className="eyebrow">INSIGHT STUDIO</div>
          <h1>Review sentiment, made actionable.</h1>
          <p>Understand what customers love, where they struggle, and why.</p>
        </div>
      </header>
      <p className="intro-copy">
        Analyze code-mixed Malay/English e-commerce reviews with sentiment, aspect insights,
        and exportable reports.
      </p>

      <section className="control-panel">
        <h2>Start an analysis</h2>
        <div className="controls">
          <label className="upload-box">
            <span>Review dataset</span>
            <strong>{rows.length ? `${rows.length} reviews loaded` : "Drop a CSV here or click to upload"}</strong>
            <input type="file" accept=".csv,text/csv" onChange={(event) => readCsv(event.target.files[0])} />
            <small>CSV must include: review_text (optional: review_id, date, platform)</small>
          </label>
          <div className="settings">
            <label>
              Platform
              <select value={platform} onChange={(event) => setPlatform(event.target.value)}>
                {PLATFORMS.map((item) => <option key={item}>{item}</option>)}
              </select>
            </label>
            <label className="checkbox">
              <input type="checkbox" checked={withAspects} onChange={(event) => setWithAspects(event.target.checked)} />
              <span>Aspect-based analysis<small>Delivery, price, quality...</small></span>
            </label>
          </div>
        </div>
        <button className="primary-button" disabled={running || !rows.length} onClick={analyze}>
          {running ? progress : "Analyze reviews"}
        </button>
        {error && <p className="error">{error}</p>}
      </section>

      <section className="snapshot">
        <h2>Snapshot</h2>
        <div className="metric-grid">
          <Metric label="Reviews analyzed" value={metrics.total} caption="Current selection" />
          <Metric label="Positive" value={metrics.counts.Positive} caption="Customer sentiment" tone="positive" />
          <Metric label="Negative" value={metrics.counts.Negative} caption="Needs attention" tone="negative" />
          <Metric label="Neutral" value={metrics.counts.Neutral} caption={`${(metrics.negativeRatio * 100).toFixed(1)}% negative ratio`} tone="neutral" />
        </div>
        <div className="insight-grid">
          <Chart title="Sentiment distribution">
            {["Positive", "Negative", "Neutral"].map((label) => (
              <div className="bar-row" key={label}>
                <span>{label}</span><div className={`bar ${label.toLowerCase()}`} style={{ width: `${metrics.total ? (metrics.counts[label] / metrics.total) * 100 : 0}%` }} /><b>{metrics.counts[label]}</b>
              </div>
            ))}
          </Chart>
          <Chart title="Top complaint keywords">
            <div className="keyword-list">{negativeKeywords.length ? negativeKeywords.map((word) => <span key={word}>{word}</span>) : <em>Negative-review keywords will appear here.</em>}</div>
          </Chart>
        </div>
      </section>

      <section className="results-panel">
        <div className="results-heading">
          <h2>Results</h2>
          <div>
            <button onClick={() => exportResults("csv")} disabled={!results.length}>Download CSV</button>
            <button onClick={() => exportResults("json")} disabled={!results.length}>Download JSON</button>
          </div>
        </div>
        {results.length ? (
          <div className="table-wrap"><table><thead><tr><th>Review</th><th>Platform</th><th>Sentiment</th><th>Aspects</th></tr></thead><tbody>{results.map((result) => <tr key={result.id}><td>{result.text}</td><td>{result.platform}</td><td><span className={`pill ${result.sentiment.toLowerCase()}`}>{result.sentiment}</span></td><td>{result.aspects.join(", ") || "—"}</td></tr>)}</tbody></table></div>
        ) : <p className="empty">Upload a CSV and analyze it to see the results.</p>}
      </section>
      <footer>Powered by your local ModernBERT CPU model</footer>
    </main>
  );
}

function Metric({ label, value, caption, tone = "" }) {
  return <div className={`metric-card ${tone}`}><span>{label}</span><strong>{value.toLocaleString()}</strong><small>{caption}</small></div>;
}

function Chart({ title, children }) {
  return <div className="chart-card"><h3>{title}</h3>{children}</div>;
}

createRoot(document.getElementById("root")).render(<App />);
