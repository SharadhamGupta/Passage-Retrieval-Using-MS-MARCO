"""Streamlit demo for the MS MARCO passage retrieval project (Sprint 6).

Run:  streamlit run app.py

This file only handles presentation. Retrieval comes from src.retrieval.registry
(the same code the CLI and the evaluation use); metrics are read from the files
written by scripts/evaluate_all.py. Nothing here computes or invents scores.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import time

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CONFIG_ENV_VAR, VALID_MODES, load_config  # noqa: E402
from src.data.loader import load_qrels_tsv, load_queries_tsv  # noqa: E402
from src.retrieval.registry import (  # noqa: E402
    METHOD_LABELS,
    METHODS,
    SCORE_DESCRIPTIONS,
    IndexNotFoundError,
    load_retriever,
)
from src.ui.helpers import (  # noqa: E402
    METHOD_COLORS,
    QueryValidationError,
    eval_artifacts,
    index_mtime,
    index_status,
    judged_queries,
    metrics_long_table,
    overlap_matrix,
    result_card_html,
    shared_methods,
    validate_query,
)

st.set_page_config(page_title="MS MARCO passage search", page_icon="🔎", layout="wide")

CONFIG_PATH = os.environ.get(CONFIG_ENV_VAR)
CFG = load_config(CONFIG_PATH)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');
html, body, [class*="css"], .stMarkdown, .stTextInput input, button { font-family: 'IBM Plex Sans', system-ui, sans-serif; }
.block-container { padding-top: 2.2rem; max-width: 1200px; }
.app-title { font-size: 2.1rem; font-weight: 600; letter-spacing: -0.02em; color: #1c2333; margin: 0; line-height: 1.15; }
.app-sub { color: #52607a; font-size: 1rem; margin: .35rem 0 1.2rem 0; max-width: 70ch; }
.method-key { display: flex; gap: 1.1rem; flex-wrap: wrap; margin: 0 0 1.2rem 0; font-size: .9rem; color: #3a4459; }
.method-key span::before { content: ""; display: inline-block; width: .7rem; height: .7rem; border-radius: 2px;
  background: var(--c); margin-right: .4rem; vertical-align: -1px; }
.card { display: grid; grid-template-columns: 2.6rem 1fr; gap: .9rem; background: #ffffff; border: 1px solid #e1e5ec;
  border-left: 4px solid var(--method); border-radius: 10px; padding: .95rem 1.1rem; margin-bottom: .7rem; }
.card .rank { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.9rem; font-weight: 600; color: var(--method);
  line-height: 1; text-align: right; font-variant-numeric: tabular-nums; }
.card .meta { display: flex; flex-wrap: wrap; align-items: center; gap: .55rem .9rem; font-size: .85rem; color: #52607a; }
.card .pid { font-weight: 500; color: #1c2333; }
.card .score { display: inline-flex; align-items: center; gap: .45rem; font-variant-numeric: tabular-nums; }
.card .bar { width: 64px; height: 6px; background: #edf0f5; border-radius: 3px; overflow: hidden; display: inline-block; }
.card .bar > span { display: block; height: 100%; background: var(--method); border-radius: 3px; }
.card .passage { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.04rem; line-height: 1.62; color: #1c2333;
  margin: .45rem 0 0 0; max-width: 78ch; }
.card mark { background: transparent; color: inherit; box-shadow: inset 0 -0.42em 0 rgba(255, 205, 64, .55); padding: 0 1px; }
.badge { border: 1px solid #cfd6e2; border-radius: 999px; padding: 1px 9px; font-size: .78rem; color: #3a4459; background: #f6f7f9; }
.badge-rel { border-color: #1baf7a; color: #0f6e4c; background: #e8f7f0; }
.col-head { display: flex; align-items: baseline; gap: .5rem; border-bottom: 3px solid var(--c); padding-bottom: .35rem; margin-bottom: .7rem; }
.col-head b { font-size: 1.1rem; color: #1c2333; }
.col-head small { color: #52607a; }
.compare .card { grid-template-columns: 1.8rem 1fr; padding: .75rem .8rem; }
.compare .card .rank { font-size: 1.35rem; }
.compare .card .passage { font-size: .95rem; line-height: 1.55; }
.hint { color: #52607a; font-size: .9rem; }
@media (max-width: 640px) { .app-title { font-size: 1.6rem; } .card { grid-template-columns: 1.8rem 1fr; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner=False, max_entries=9)
def _cached_retriever(method: str, mode: str, config_path, mtime: float):
    """Loads an index (and the embedding model for dense) once per (method, mode, index version)."""
    return load_retriever(method, mode, load_config(config_path))


def get_retriever(method: str, mode: str):
    """Returns (retriever, None) or (None, error message)."""
    try:
        label = METHOD_LABELS[method]
        with st.spinner(f"Loading the {label} index" + (" and embedding model" if method == "dense" else "") + "..."):
            return _cached_retriever(method, mode, CONFIG_PATH, index_mtime(CFG, method, mode)), None
    except IndexNotFoundError as exc:
        return None, f"The {METHOD_LABELS[method]} index for “{mode}” has not been built yet.\n\n```bash\n{str(exc).split('Build it with:')[-1].strip()}\n```"
    except Exception as exc:  # DenseModelError, DenseIndexError, corrupt files ...
        kind = "Model loading failed" if type(exc).__name__ == "DenseModelError" else "Could not open the index"
        return None, f"**{kind}** for {METHOD_LABELS[method]}: {exc}"


@st.cache_data(show_spinner=False)
def _cached_judged_queries(queries_path: str, qrels_path: str, mtime: float):
    return judged_queries(load_queries_tsv(queries_path), load_qrels_tsv(qrels_path), limit=300)


def judged_query_list(mode: str):
    qp, rp = CFG.data.get_queries_path(mode), CFG.data.get_qrels_path(mode)
    if not (qp.exists() and rp.exists()):
        return []
    return _cached_judged_queries(str(qp), str(rp), qp.stat().st_mtime + rp.stat().st_mtime)


@st.cache_data(show_spinner=False)
def _read_csv(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"query_id": str, "relevant_pid": str})


def read_csv(path: Path) -> pd.DataFrame:
    return _read_csv(str(path), path.stat().st_mtime)


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Settings")
    default_mode = CFG.data.mode if CFG.data.mode in VALID_MODES else "dev"
    mode = st.selectbox(
        "Dataset",
        VALID_MODES,
        index=VALID_MODES.index(default_mode),
        format_func=lambda m: {"sample": "Sample (5 passages, for testing)",
                               "dev": "Dev subset (prepared MS MARCO sample)",
                               "full": "Full collection (8.8M passages)"}[m],
    )
    top_k = st.slider("Results per method", min_value=1, max_value=50, value=10)
    st.markdown("### Indexes")
    status = index_status(CFG, mode)
    for m in METHODS:
        s = status[m]
        if s["exists"]:
            st.markdown(f"✅ **{METHOD_LABELS[m]}** ready")
        else:
            st.markdown(f"⬜ **{METHOD_LABELS[m]}** not built")
            st.code(s["command"], language="bash")
    passages_path = CFG.data.get_passages_path(mode)
    if not passages_path.exists():
        st.warning(f"Passages file for “{mode}” not found:\n\n`{passages_path}`")


# ----------------------------------------------------------------------------- header
st.markdown('<h1 class="app-title">MS MARCO passage search</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-sub">Ask a question in plain English and see which passages each retrieval method '
    "ranks highest. Every result and every metric on this page comes from the project's own indexes and "
    "evaluation files.</p>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="method-key">' + "".join(
        f'<span style="--c:{METHOD_COLORS[m]}">{METHOD_LABELS[m]}</span>' for m in METHODS
    ) + "</div>",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------- query state
if "query_text" not in st.session_state:
    st.session_state.query_text = ""
if "picked" not in st.session_state:
    st.session_state.picked = None


def _on_pick():
    choice = st.session_state.get("dev_pick")
    if choice and choice.get("qid"):
        st.session_state.query_text = choice["text"]
        st.session_state.picked = choice


def judged_for(query: str):
    """Judged-relevant passage IDs when the query is an unmodified dev query, else empty set."""
    p = st.session_state.picked
    return set(p["relevant"]) if p and p["text"] == query else set()


def query_controls():
    dev = judged_query_list(mode)
    if dev:
        options = [{"qid": None, "text": "", "relevant": []}] + dev
        st.selectbox(
            "Or start from a dev query that has a judged answer",
            options,
            key="dev_pick",
            format_func=lambda o: "Choose a query..." if not o["qid"] else f"{o['text']}  (qid {o['qid']})",
            on_change=_on_pick,
        )
    st.text_input(
        "Your question",
        key="query_text",
        placeholder="e.g. what is the manhattan project",
        max_chars=500,
    )


query_controls()  # one shared query for both the Search and Compare tabs

tab_search, tab_compare, tab_eval, tab_about = st.tabs(
    ["Search", "Compare methods", "Evaluation results", "How it works"]
)

# ----------------------------------------------------------------------------- search tab
with tab_search:
    method = st.radio("Method", METHODS, format_func=lambda m: METHOD_LABELS[m], horizontal=True, key="method")
    run = st.button("Search", type="primary", key="search_btn")
    if run:
        try:
            q = validate_query(st.session_state.query_text)
        except QueryValidationError as exc:
            st.warning(str(exc))
        else:
            retriever, err = get_retriever(method, mode)
            if err:
                st.error(err)
            else:
                t0 = time.perf_counter()
                try:
                    results = retriever.search(q, top_k=top_k)
                except Exception as exc:
                    st.error(f"Search failed: {type(exc).__name__}: {exc}")
                    results = None
                ms = (time.perf_counter() - t0) * 1000
                if results is not None:
                    judged = judged_for(q)
                    n_rel = sum(r.passage_id in judged for r in results)
                    line = (f"{len(results)} passages from {retriever.info['num_passages']:,} in {ms:.1f} ms. "
                            f"{SCORE_DESCRIPTIONS[method]}")
                    if judged:
                        line += f" Judged-relevant passages in this list: {n_rel} of {len(judged)}."
                    st.markdown(f'<p class="hint">{line}</p>', unsafe_allow_html=True)
                    if not results:
                        st.info("No passage shares a word with this query. Try other wording, or the dense method, "
                                "which matches meaning rather than exact words.")
                    top = results[0].score if results else 0.0
                    st.markdown("".join(
                        result_card_html(r, method, q, top, judged_relevant=r.passage_id in judged) for r in results
                    ), unsafe_allow_html=True)

# ----------------------------------------------------------------------------- compare tab
with tab_compare:
    st.markdown('<p class="hint">Runs the same query through all three methods. Scores use different scales, '
                "so compare positions, not numbers.</p>", unsafe_allow_html=True)
    if st.button("Compare", type="primary", key="compare_btn"):
        try:
            q = validate_query(st.session_state.query_text)
        except QueryValidationError as exc:
            st.warning(str(exc))
        else:
            results_by, timing, errors = {}, {}, {}
            for m in METHODS:
                retriever, err = get_retriever(m, mode)
                if err:
                    errors[m] = err
                    continue
                t0 = time.perf_counter()
                try:
                    results_by[m] = retriever.search(q, top_k=top_k)
                except Exception as exc:
                    errors[m] = f"Search failed: {type(exc).__name__}: {exc}"
                timing[m] = (time.perf_counter() - t0) * 1000
            judged = judged_for(q)
            if len(results_by) >= 2:
                ov = overlap_matrix(results_by)
                parts = [f"{METHOD_LABELS[a]} and {METHOD_LABELS[b]} share {v['shared']}"
                         for a, b, v in ((*k.split("|"), v) for k, v in ov["pairs"].items())]
                st.markdown(f'<p class="hint">Overlap in the top {top_k}: ' + "; ".join(parts)
                            + f". Returned by every working method: {len(ov['in_all'])}.</p>", unsafe_allow_html=True)
            cols = st.columns(len(METHODS))
            for col, m in zip(cols, METHODS):
                with col:
                    sub = f"{timing[m]:.1f} ms" if m in timing else ""
                    if judged and m in results_by:
                        hits = sum(r.passage_id in judged for r in results_by[m])
                        sub += f", {hits} judged relevant"
                    st.markdown(f'<div class="col-head" style="--c:{METHOD_COLORS[m]}"><b>{METHOD_LABELS[m]}</b>'
                                f"<small>{sub}</small></div>", unsafe_allow_html=True)
                    if m in errors:
                        st.error(errors[m])
                        continue
                    rs = results_by[m]
                    if not rs:
                        st.info("No matching passages.")
                        continue
                    st.markdown('<div class="compare">' + "".join(
                        result_card_html(r, m, q, rs[0].score, r.passage_id in judged,
                                         shared_methods(r.passage_id, m, results_by)) for r in rs
                    ) + "</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------- evaluation tab
with tab_eval:
    eval_dir = CFG.evaluation.get_output_dir(mode)
    art = eval_artifacts(eval_dir)
    if not art["available"]:
        st.info(
            f"No evaluation results for “{mode}” yet (missing: {', '.join(art['missing'])}). "
            "Run the evaluation, then reload this page:"
        )
        st.code(f"python scripts/evaluate_all.py --mode {mode}", language="bash")
    else:
        agg, settings = art["aggregate"], art["settings"]
        es = agg["evaluation_set"]
        st.markdown(f'<p class="hint">Measured {agg["timestamp"]} on {es["queries_evaluated"]:,} judged queries over '
                    f'{es["corpus_size"]:,} passages. Source: <code>{eval_dir}</code></p>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Queries evaluated", f"{es['queries_evaluated']:,}")
        c2.metric("Queries excluded", f"{es['queries_excluded']:,}", help=str(es.get("exclusion_reasons") or "none"))
        c3.metric("Passages in corpus", f"{es['corpus_size']:,}")
        c4.metric("Judged relevant missing from corpus", f"{es['judged_relevant_missing_from_corpus']:,}")

        long = pd.DataFrame(metrics_long_table(agg))
        try:
            import altair as alt

            color = alt.Color("label:N", title="Method",
                              scale=alt.Scale(domain=[agg["labels"][m] for m in agg["methods"]],
                                              range=[METHOD_COLORS.get(m, "#888") for m in agg["methods"]]))
            base = alt.Chart(long).encode(
                x=alt.X("metric:N", title=None, sort=["Recall@5", "Recall@10", "MRR@10", "nDCG@10"],
                        axis=alt.Axis(labelAngle=0)),
                xOffset=alt.XOffset("label:N", sort=[agg["labels"][m] for m in agg["methods"]]),
            )
            bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=26).encode(
                y=alt.Y("value:Q", title="Mean over judged queries", scale=alt.Scale(domain=[0, 1])),
                color=color,
                tooltip=[alt.Tooltip("label:N", title="Method"), "metric:N",
                         alt.Tooltip("value:Q", format=".4f", title="Mean"),
                         alt.Tooltip("ci_low:Q", format=".3f", title="95% CI low"),
                         alt.Tooltip("ci_high:Q", format=".3f", title="95% CI high")],
            )
            rules = base.mark_rule(color="#3a4459").encode(y="ci_low:Q", y2="ci_high:Q")
            st.altair_chart((bars + rules).properties(height=340), use_container_width=True)
        except ImportError:
            st.bar_chart(long.pivot(index="metric", columns="label", values="value"))
        st.caption("Bars are means of the per-query scores in per_query_metrics.csv; lines show 95% bootstrap "
                   "confidence intervals.")

        table = pd.DataFrame([
            {"Method": agg["labels"][m], **{k: agg["aggregates"][m]["means"][k]
                                             for k in ("Recall@5", "Recall@10", "MRR@10", "nDCG@10", "Recall@100")},
             "Mean latency (ms)": agg["latency"][m]["mean_ms"], "p95 latency (ms)": agg["latency"][m]["p95_ms"]}
            for m in agg["methods"]
        ])
        st.dataframe(table.style.format({c: "{:.4f}" for c in table.columns if "@" in c}
                                        | {"Mean latency (ms)": "{:.2f}", "p95 latency (ms)": "{:.2f}"}),
                     hide_index=True, use_container_width=True)

        st.markdown("#### Are the differences real?")
        sig = pd.DataFrame(agg["significance"])
        if not sig.empty:
            sig["Comparison"] = sig["method_a"].map(METHOD_LABELS) + " minus " + sig["method_b"].map(METHOD_LABELS)
            sig = sig.rename(columns={"metric": "Metric", "mean_a_minus_b": "Mean difference", "p_value": "p-value",
                                      "significant_at_0.05": "Significant (p < 0.05)"})
            st.dataframe(sig[["Comparison", "Metric", "Mean difference", "p-value", "Significant (p < 0.05)"]],
                         hide_index=True, use_container_width=True)
            st.caption("Paired randomization test on the same queries. A non-significant difference means this "
                       "query set cannot tell the methods apart.")

        st.markdown("#### Where the methods disagree")
        dpath = art["paths"]["disagreements"]
        if dpath.exists():
            dis = read_csv(dpath)
            if dis.empty:
                st.write("The methods agree on every evaluated query.")
            else:
                st.write(f"{len(dis):,} queries with an RR@10 gap of at least 0.5, or a top-10 hit for some "
                         "methods but not others. Tags are hypotheses to check by reading the passages.")
                show = ["query_id", "query", "winners", "losers", "rr_gap", "lexical_overlap_query_vs_relevant",
                        "hypothesis_tags"]
                st.dataframe(dis[[c for c in show if c in dis.columns]], hide_index=True,
                             use_container_width=True, height=300)
                pick = st.selectbox("Inspect a query", dis["query"].tolist(), key="dis_pick")
                row = dis[dis["query"] == pick].iloc[0]
                st.markdown(f"**Judged relevant passage {row['relevant_pid']}:** {row.get('relevant_text', '')}")
                for m in agg["methods"]:
                    if f"{m}_top1_pid" in row:
                        judged_note = "judged" if row.get(f"{m}_top1_judged") else "unjudged"
                        st.markdown(f"**{METHOD_LABELS[m]}** (first relevant rank: "
                                    f"{row.get(f'{m}_first_relevant_rank') if pd.notna(row.get(f'{m}_first_relevant_rank')) else 'not found'}"
                                    f") top hit `{row[f'{m}_top1_pid']}` ({judged_note}): {row.get(f'{m}_top1_text', '')}")

        st.markdown("#### Files")
        dl = st.columns(4)
        for col, key in zip(dl, ("per_query", "aggregate", "significance", "chart")):
            p = art["paths"][key]
            if p.exists():
                col.download_button(f"Download {p.name}", p.read_bytes(), file_name=p.name, key=f"dl_{key}")
        with st.expander("Execution settings"):
            st.json(settings)
        if art["paths"]["report"].exists():
            with st.expander("Full evaluation report"):
                text = art["paths"]["report"].read_text(encoding="utf-8")
                st.markdown(text.replace("![comparison chart](comparison_chart.png)", ""))

# ----------------------------------------------------------------------------- about tab
with tab_about:
    st.markdown(f"""
**TF-IDF** scores a passage by the cosine between its word-weight vector and the query's. Rare words weigh more
(inverse document frequency); repeated words count with diminishing returns (1 + log tf).

**BM25** (k1 = {CFG.bm25.k1}, b = {CFG.bm25.b}) also rewards rare query words, but caps how much repetition helps and
discounts long passages. It is the standard lexical baseline for MS MARCO.

**Dense retrieval** encodes the query and every passage with the same neural model
(`{CFG.dense.model_name}`) into vectors whose direction captures meaning. Passages were encoded once, offline; a
search encodes only the query and finds the nearest passage vectors with FAISS (cosine similarity). It can match
passages that use different words from the query, and can miss exact names or numbers.

**Reading the evaluation.** MS MARCO usually marks one passage per query as relevant. Other passages may answer the
question just as well, but count as wrong. On a sampled corpus every judged passage is present while most hard
distractors are not, so absolute scores are higher than on the full collection. Treat a difference as real only when
the significance test says so.
""")
