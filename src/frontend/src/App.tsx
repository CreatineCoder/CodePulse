import { useEffect, useMemo, useRef, useState } from "react";

type RepoMeta = {
  repo: string;
  total_files: number;
  total_commits: number;
  distinct_authors: number;
  last_update: string | null;
  prs: number;
  avg_age_days: number;
  avg_lines_changed: number;
  bug_prone_labeled: number;
  total_lines_changed: number;
};

type Contribution = { feature: string; value: number; shap: number };

type TopFile = {
  file_path: string;
  probability: number;
  num_commits: number;
  num_authors: number;
  age_days: number;
  total_lines_changed: number;
  is_bug_prone_labeled: number;
  contributions: Contribution[];
};

type Insights = {
  summary: {
    repo: string;
    total_files: number;
    bug_prone_labeled: number;
    bug_prone_predicted: number;
    total_commits: number;
    total_lines_changed: number;
    avg_age_days: number;
    max_probability: number;
    mean_probability: number;
  };
  top_files: TopFile[];
  feature_importance: { feature: string; importance: number }[];
  probability_distribution: { bin: string; count: number }[];
  model_tag: string;
  features: string[];
};

const SAMPLES = ["httpie/cli", "pallets/flask", "scrapy/scrapy"];

function pulsePath(severity: number) {
  const W = 800, H = 220, mid = H / 2;
  const segs = 4;
  const pts: [number, number][] = [];
  for (let s = 0; s < segs; s++) {
    const baseX = s * (W / segs);
    for (let i = 0; i < 30; i++) {
      pts.push([baseX + i * 2, mid + (Math.random() - 0.5) * 3]);
    }
    const spikeX = baseX + 80;
    const amp = 60 + severity * 70;
    pts.push([spikeX, mid]);
    pts.push([spikeX + 6, mid - amp * 0.35]);
    pts.push([spikeX + 12, mid + amp]);
    pts.push([spikeX + 18, mid - amp * 0.95]);
    pts.push([spikeX + 24, mid + amp * 0.25]);
    pts.push([spikeX + 30, mid]);
    for (let i = 0; i < 30; i++) {
      pts.push([spikeX + 30 + i * 2, mid + (Math.random() - 0.5) * 3]);
    }
  }
  return "M" + pts.map((p) => p[0].toFixed(1) + "," + p[1].toFixed(1)).join(" L");
}

function fmt(n: number) {
  if (!isFinite(n)) return "—";
  return n >= 1000 ? n.toLocaleString() : String(Math.round(n));
}

function useClock() {
  const [t, setT] = useState(() => new Date().toISOString().slice(11, 19) + " UTC");
  useEffect(() => {
    const id = setInterval(() => setT(new Date().toISOString().slice(11, 19) + " UTC"), 1000);
    return () => clearInterval(id);
  }, []);
  return t;
}

function useCountUp(target: number, deps: any[] = []) {
  const [val, setVal] = useState(0);
  useEffect(() => {
    let raf = 0;
    const start = performance.now();
    const from = 0;
    const dur = 900;
    const step = (now: number) => {
      const k = Math.min(1, (now - start) / dur);
      const v = Math.round(from + (target - from) * (1 - Math.pow(1 - k, 3)));
      setVal(v);
      if (k < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return val;
}

function labelFor(p: number) {
  if (p >= 0.7) return { text: "BUG-PRONE", cls: "label-bad" };
  if (p >= 0.4) return { text: "ELEVATED", cls: "label-mid" };
  return { text: "WATCH", cls: "label-watch" };
}

function verdictFor(score: number) {
  if (score >= 25) return "/// elevated risk";
  if (score >= 15) return "// moderate risk";
  return "/ low–moderate";
}

export default function App() {
  const clock = useClock();
  const [repos, setRepos] = useState<RepoMeta[]>([]);
  const [repoInput, setRepoInput] = useState("");
  const [labelWindow, setLabelWindow] = useState("90D");
  const [insights, setInsights] = useState<Insights | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<TopFile | null>(null);
  const [animateBars, setAnimateBars] = useState(false);

  useEffect(() => {
    fetch("/api/repos")
      .then((r) => r.json())
      .then((d) => setRepos(d))
      .catch(() => {});
  }, []);

  // auto-load flask for demo, like the prototype
  useEffect(() => {
    const t = setTimeout(() => {
      if (!insights && !loading) {
        setRepoInput("pallets/flask");
        runAnalysis("pallets/flask");
      }
    }, 400);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function runAnalysis(nameArg?: string) {
    const name = (nameArg ?? repoInput).trim();
    if (!name) return;
    setLoading(true);
    setError(null);
    setInsights(null);
    setAnimateBars(false);
    try {
      const r = await fetch("/api/insights?repo=" + encodeURIComponent(name));
      if (!r.ok) {
        setError(name);
        setLoading(false);
        return;
      }
      const data: Insights = await r.json();
      setInsights(data);
      setLoading(false);
      requestAnimationFrame(() => setAnimateBars(true));
    } catch {
      setError(name);
      setLoading(false);
    }
  }

  const repoMeta = useMemo(
    () => repos.find((r) => r.repo === insights?.summary.repo),
    [repos, insights],
  );

  const score = insights
    ? Math.round((insights.summary.bug_prone_predicted / Math.max(1, insights.summary.total_files)) * 100)
    : 0;
  const scoreVal = useCountUp(score, [score]);

  const severity = insights ? Math.min(1, score / 40) : 0;
  const pathD = useMemo(() => (insights ? pulsePath(severity) : ""), [insights, severity]);

  const maxFeatureImp = insights
    ? Math.max(...insights.feature_importance.map((f) => f.importance))
    : 1;

  const highRiskCount = insights ? insights.top_files.filter((f) => f.probability >= 0.7).length : 0;

  return (
    <div className="wrap">
      <TopBar clock={clock} reposIndexed={repos.length} />
      <Hero reposIndexed={repos.length} />
      <InputRow
        repoInput={repoInput}
        setRepoInput={setRepoInput}
        labelWindow={labelWindow}
        setLabelWindow={setLabelWindow}
        onRun={() => runAnalysis()}
      />
      <Examples
        onPick={(name) => {
          setRepoInput(name);
          runAnalysis(name);
        }}
      />

      <div className={"loading" + (loading ? " on" : "")} />

      {!insights && !loading && !error && (
        <div className="empty">
          <span className="big">
            awaiting <em>input</em>
          </span>
          enter a repository above to compute its <em>bug-pulse</em> signal
        </div>
      )}

      {error && (
        <div className="empty">
          <span className="big">no signal</span>
          repository <em>{error}</em> is not indexed &nbsp;·&nbsp; try one of the sample buttons
        </div>
      )}

      {insights && (
        <div className="results show">
          <RepoStrip insights={insights} repoMeta={repoMeta} />

          <div className="grid">
            <div>
              <RiskPanel
                insights={insights}
                pathD={pathD}
                scoreVal={scoreVal}
                highRiskCount={highRiskCount}
              />
              <FilesPanel files={insights.top_files.slice(0, 8)} onSelect={setSelectedFile} animate={animateBars} />
            </div>

            <div>
              <FeaturesPanel
                features={insights.feature_importance}
                max={maxFeatureImp}
                animate={animateBars}
              />
              <DiagnosisPanel insights={insights} repoMeta={repoMeta} />
            </div>
          </div>
        </div>
      )}

      <Footer />

      <Drawer file={selectedFile} onClose={() => setSelectedFile(null)} />
    </div>
  );
}

function TopBar({ clock, reposIndexed }: { clock: string; reposIndexed: number }) {
  return (
    <div className="bar">
      <div className="left">
        <span>
          <span className="pulse-dot" />
          <b>SIGNAL ONLINE</b>
        </span>
        <span>
          BUILD <b>a14.05.15</b>
        </span>
        <span>
          REPOS <b>{reposIndexed || "—"}</b>
        </span>
      </div>
      <div>
        <b>CODEPULSE</b> &nbsp;//&nbsp; DEVELOPER INTELLIGENCE
      </div>
      <div className="right">
        <span>{clock}</span>
        <span>v0.1 / alpha</span>
      </div>
    </div>
  );
}

function Hero({ reposIndexed }: { reposIndexed: number }) {
  return (
    <div className="hero">
      <div>
        <div className="wordmark">
          code<span className="slash">/</span>pulse
        </div>
        <div className="tag">
          predict <em>bug-prone files</em> from git history &nbsp;·&nbsp; ml + explainability
        </div>
      </div>
      <div className="hero-meta">
        <div>
          <span>model</span>
          <b>lightgbm + treeshap</b>
        </div>
        <div>
          <span>label window</span>
          <b>90 days</b>
        </div>
        <div>
          <span>repos indexed</span>
          <b>{reposIndexed || "—"}</b>
        </div>
        <div>
          <span>status</span>
          <b className="ok">nominal</b>
        </div>
      </div>
    </div>
  );
}

function InputRow({
  repoInput,
  setRepoInput,
  labelWindow,
  setLabelWindow,
  onRun,
}: {
  repoInput: string;
  setRepoInput: (v: string) => void;
  labelWindow: string;
  setLabelWindow: (v: string) => void;
  onRun: () => void;
}) {
  return (
    <div className="input-row">
      <div className="input-label">repo /</div>
      <input
        className="repo-input"
        placeholder="owner/name  ·  e.g. pallets/flask"
        spellCheck={false}
        value={repoInput}
        onChange={(e) => setRepoInput(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") onRun();
        }}
      />
      <select
        className="select"
        value={labelWindow}
        onChange={(e) => setLabelWindow(e.target.value)}
        title="label window"
      >
        <option>30D</option>
        <option>90D</option>
        <option>180D</option>
      </select>
      <button className="analyze" onClick={onRun}>
        analyze ↗
      </button>
    </div>
  );
}

function Examples({ onPick }: { onPick: (name: string) => void }) {
  return (
    <div className="examples">
      <span>try /</span>
      {SAMPLES.map((s) => (
        <button key={s} onClick={() => onPick(s)}>
          {s}
        </button>
      ))}
    </div>
  );
}

function RepoStrip({ insights, repoMeta }: { insights: Insights; repoMeta?: RepoMeta }) {
  const last = repoMeta?.last_update ? repoMeta.last_update.slice(0, 10) : "—";
  return (
    <div className="repo-strip">
      <Cell k="repository" v={insights.summary.repo} />
      <Cell k="source files" v={fmt(insights.summary.total_files)} />
      <Cell k="commits indexed" v={fmt(insights.summary.total_commits)} />
      <Cell k="authors" v={fmt(repoMeta?.distinct_authors ?? 0)} />
      <Cell k="last update" v={last} />
    </div>
  );
}

function Cell({ k, v }: { k: string; v: string }) {
  return (
    <div className="cell">
      <div className="k">{k}</div>
      <div className="v">{v}</div>
    </div>
  );
}

function RiskPanel({
  insights,
  pathD,
  scoreVal,
  highRiskCount,
}: {
  insights: Insights;
  pathD: string;
  scoreVal: number;
  highRiskCount: number;
}) {
  const score = Math.round(
    (insights.summary.bug_prone_predicted / Math.max(1, insights.summary.total_files)) * 100,
  );
  const avgChurn = insights.summary.total_files
    ? Math.round(insights.summary.total_lines_changed / insights.summary.total_files)
    : 0;
  return (
    <div className="panel risk">
      <svg className="pulse-svg" preserveAspectRatio="none" viewBox="0 0 800 220">
        <path d={pathD} />
      </svg>
      <div className="panel-head">
        <span>
          <span className="idx">01/</span> pulse signal · 90-day forecast
        </span>
        <span>{insights.model_tag}</span>
      </div>
      <div className="panel-body">
        <div className="risk-grid">
          <div>
            <span className="score-num">{String(scoreVal).padStart(2, "0")}</span>
            <span className="score-suffix">%</span>
          </div>
          <div className="score-meta">
            <span className="verdict">{verdictFor(score)}</span>
            <b>{insights.summary.bug_prone_predicted}</b> of <b>{insights.summary.total_files}</b>{" "}
            files flagged
            <br />
            ranked by <b>shap × lightgbm</b>
            <br />
            base rate &nbsp;·&nbsp; <b>6.7%</b>
          </div>
        </div>
      </div>
      <div className="stats">
        <Stat k="avg age" v={`${Math.round(insights.summary.avg_age_days)} d`} />
        <Stat k="avg churn" v={`${fmt(avgChurn)} L`} />
        <Stat k="high risk" v={`${highRiskCount} files`} warn />
        <Stat k="max prob" v={`${Math.round(insights.summary.max_probability * 100)}%`} />
      </div>
    </div>
  );
}

function Stat({ k, v, warn }: { k: string; v: string; warn?: boolean }) {
  return (
    <div className="stat">
      <div className="k">{k}</div>
      <div className={"v" + (warn ? " warn" : "")}>{v}</div>
    </div>
  );
}

function FilesPanel({
  files,
  onSelect,
  animate,
}: {
  files: TopFile[];
  onSelect: (f: TopFile) => void;
  animate: boolean;
}) {
  return (
    <div className="panel" style={{ marginTop: 24 }}>
      <div className="panel-head">
        <span>
          <span className="idx">02/</span> highest-risk files
        </span>
        <span>top {files.length} · pred. bug probability</span>
      </div>
      <table className="files">
        <thead>
          <tr>
            <th style={{ width: 36 }}>#</th>
            <th>path</th>
            <th>risk</th>
            <th>commits</th>
            <th>authors</th>
            <th>signal</th>
          </tr>
        </thead>
        <tbody>
          {files.map((f, i) => {
            const lbl = labelFor(f.probability);
            return (
              <FileRow key={f.file_path} idx={i} file={f} label={lbl} onSelect={onSelect} animate={animate} />
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function FileRow({
  idx,
  file,
  label,
  onSelect,
  animate,
}: {
  idx: number;
  file: TopFile;
  label: { text: string; cls: string };
  onSelect: (f: TopFile) => void;
  animate: boolean;
}) {
  const [shown, setShown] = useState(false);
  useEffect(() => {
    if (!animate) return;
    const t = setTimeout(() => setShown(true), 80 + idx * 60);
    return () => clearTimeout(t);
  }, [animate, idx]);
  const w = `${Math.round(file.probability * 100)}%`;
  return (
    <tr
      onClick={() => onSelect(file)}
      style={{
        opacity: shown ? 1 : 0,
        transform: shown ? "none" : "translateY(4px)",
        transition: "all .35s ease",
      }}
    >
      <td className="rank">{String(idx + 1).padStart(2, "0")}</td>
      <td className="path">{file.file_path}</td>
      <td>
        <span className="meter" style={{ ["--w" as any]: w }} />
        <span className="pct">{Math.round(file.probability * 100)}%</span>
      </td>
      <td>{file.num_commits}</td>
      <td>{file.num_authors}</td>
      <td>
        <span className={label.cls}>{label.text}</span>
      </td>
    </tr>
  );
}

function FeaturesPanel({
  features,
  max,
  animate,
}: {
  features: { feature: string; importance: number }[];
  max: number;
  animate: boolean;
}) {
  return (
    <div className="panel">
      <div className="panel-head">
        <span>
          <span className="idx">03/</span> feature attribution
        </span>
        <span>mean |shap|</span>
      </div>
      <div className="panel-body">
        {features.map((f, i) => (
          <FeatureBar key={f.feature} idx={i} f={f} max={max} animate={animate} top={i === 0} />
        ))}
      </div>
    </div>
  );
}

function FeatureBar({
  idx,
  f,
  max,
  animate,
  top,
}: {
  idx: number;
  f: { feature: string; importance: number };
  max: number;
  animate: boolean;
  top: boolean;
}) {
  const [w, setW] = useState("0%");
  useEffect(() => {
    if (!animate) {
      setW("0%");
      return;
    }
    const t = setTimeout(() => setW(`${((f.importance / max) * 100).toFixed(1)}%`), 120 + idx * 90);
    return () => clearTimeout(t);
  }, [animate, idx, f.importance, max]);
  return (
    <div className={"feature" + (top ? " top" : "")}>
      <div className="name">{f.feature}</div>
      <div className="bar" style={{ ["--w" as any]: w }} />
      <div className="val">{f.importance.toFixed(2)}</div>
    </div>
  );
}

function DiagnosisPanel({ insights, repoMeta }: { insights: Insights; repoMeta?: RepoMeta }) {
  const top = insights.feature_importance[0]?.feature.replace(/_/g, " ") ?? "—";
  const avgAge = Math.round(insights.summary.avg_age_days);
  return (
    <div className="panel" style={{ marginTop: 24 }}>
      <div className="panel-head">
        <span>
          <span className="idx">04/</span> diagnosis
        </span>
        <span>auto</span>
      </div>
      <div
        className="panel-body"
        style={{ fontSize: 12.5, lineHeight: 1.75, color: "var(--ink-mid)" }}
      >
        <b style={{ color: "var(--ink)" }}>{insights.summary.bug_prone_predicted}</b> of{" "}
        <b style={{ color: "var(--ink)" }}>{insights.summary.total_files}</b> source files flagged as bug-prone in
        the next 90 days. dominant signal is{" "}
        <b style={{ color: "var(--hazard)" }}>{top}</b> — high-churn files accumulate latent defects faster than
        the team can resolve them.
        <br />
        <br />
        average file age is <b style={{ color: "var(--ink)" }}>{avgAge}d</b>; long-lived files with frequent edits
        carry the highest probability. consider routing new contributors away from the top-{Math.min(5, insights.top_files.length)}{" "}
        paths above without senior review.
        <br />
        <br />
        model <b style={{ color: "var(--ink)" }}>{insights.model_tag}</b> · base rate{" "}
        <b style={{ color: "var(--ink)" }}>6.7%</b> · attribution via treeshap{repoMeta?.prs ? ` · ${repoMeta.prs} PRs indexed` : ""}.
      </div>
    </div>
  );
}

function Footer() {
  return (
    <div className="footer">
      <div className="row">
        <span>© codepulse</span>
        <span>devansh / 2026</span>
      </div>
      <div className="row">
        <span>weak supervision · noisy-or</span>
        <span>temporal split · 90d</span>
        <span>lightgbm 4.6.0</span>
      </div>
    </div>
  );
}

function Drawer({ file, onClose }: { file: TopFile | null; onClose: () => void }) {
  const on = !!file;
  const maxAbs = useMemo(
    () => (file ? Math.max(...file.contributions.map((c) => Math.abs(c.shap)), 0.001) : 1),
    [file],
  );
  return (
    <>
      <div className={"drawer-backdrop" + (on ? " on" : "")} onClick={onClose} />
      <aside className={"drawer" + (on ? " on" : "")} aria-hidden={!on}>
        {file && (
          <>
            <div className="drawer-head">
              <div>
                <div className="k">file</div>
                <div className="path">{file.file_path}</div>
              </div>
              <button className="drawer-close" onClick={onClose}>
                close ×
              </button>
            </div>
            <div className="drawer-body">
              <div className="k" style={{ fontSize: 10, color: "var(--ink-mid)", letterSpacing: ".22em", marginBottom: 8 }}>
                pred. probability
              </div>
              <div className="drawer-prob">
                {Math.round(file.probability * 100)}
                <span className="pct">%</span>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "repeat(2,1fr)", gap: 14, marginTop: 22, marginBottom: 26 }}>
                <MiniStat k="commits" v={String(file.num_commits)} />
                <MiniStat k="authors" v={String(file.num_authors)} />
                <MiniStat k="age" v={`${file.age_days} d`} />
                <MiniStat k="lines changed" v={fmt(file.total_lines_changed)} />
              </div>

              <div className="k" style={{ fontSize: 10, color: "var(--ink-mid)", letterSpacing: ".22em", marginBottom: 12 }}>
                shap contributions
              </div>
              {file.contributions.map((c) => {
                const pct = (Math.abs(c.shap) / maxAbs) * 50;
                const pos = c.shap > 0;
                return (
                  <div className="contrib" key={c.feature}>
                    <div className="name">{c.feature}</div>
                    <div className="bar-wrap">
                      <div className="mid" />
                      {pos ? (
                        <div className="pos" style={{ width: `${pct}%` }} />
                      ) : (
                        <div className="neg" style={{ width: `${pct}%` }} />
                      )}
                    </div>
                    <div className="val">{(c.shap >= 0 ? "+" : "") + c.shap.toFixed(2)}</div>
                  </div>
                );
              })}
              <div style={{ fontSize: 10, color: "var(--ink-dim)", marginTop: 22, letterSpacing: ".2em", textTransform: "uppercase" }}>
                red = pushes toward bug-prone &nbsp;·&nbsp; green = pushes toward clean
              </div>
            </div>
          </>
        )}
      </aside>
    </>
  );
}

function MiniStat({ k, v }: { k: string; v: string }) {
  return (
    <div style={{ borderLeft: "1px solid var(--line)", paddingLeft: 12 }}>
      <div style={{ fontSize: 10, color: "var(--ink-mid)", letterSpacing: ".22em", textTransform: "uppercase" }}>
        {k}
      </div>
      <div style={{ fontSize: 18, color: "var(--ink)", marginTop: 4 }}>{v}</div>
    </div>
  );
}
