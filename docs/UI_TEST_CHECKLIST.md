# Streamlit UI - manual test checklist

Run before the demo, on the machine you will present from. Tick each box.

## Fresh start
- [ ] New virtual environment; `pip install -r requirements.txt -r requirements-optional.txt` finishes without errors.
- [ ] `pytest -q` passes (dense and app tests run, not skipped).
- [ ] `streamlit run app.py` opens http://localhost:8501 with no traceback in the terminal.

## Sidebar
- [ ] Dataset selector shows Sample / Dev subset / Full collection; switching it re-renders without errors.
- [ ] With all indexes built, all three methods show "ready".
- [ ] Temporarily rename `data/indices/tfidf_dev_100k.pkl`: TF-IDF shows "not built" with the exact build command. Rename it back.
- [ ] "Results per method" slider changes the number of cards (try 1 and 50).

## Search tab
- [ ] Click Search with an empty box -> warning "Type a question or a few keywords to search."
- [ ] Query `???` -> warning about letters or numbers.
- [ ] `what is the manhattan project` with BM25 -> ranked cards 1..k, each with rank, passage ID, score bar + value, passage text.
- [ ] Query words are highlighted in passage text; stopwords like "what"/"is" are not.
- [ ] Same query with TF-IDF and Dense: scores look different in scale (TF-IDF 0-1, dense cosine), the hint line explains it.
- [ ] Pick a dev query from the dropdown: the query box fills, and any judged-relevant passage in the results shows a green "Judged relevant" badge and the hint line reports how many were found.
- [ ] Edit the picked query text: judged badges disappear (judgments belong to the original query only).
- [ ] A query with no shared words (e.g. `xqzvw`) on BM25 -> "No passage shares a word..." message, no crash.
- [ ] Second search with Dense is noticeably faster than the first (model and index cached).

## Compare tab
- [ ] Compare runs the current query through all three methods in three coloured columns.
- [ ] Overlap line reports how many passages each pair shares; cards show "Also in ..." badges for shared passages.
- [ ] Each column header shows that method's latency.
- [ ] Stop the dense index (rename `data/indices/dense_dev_100k/metadata.json`): the dense column shows an error, the other two still show results. Restore the file.
- [ ] Disconnect from the internet on a machine without the cached model (or set `dense.model_name` to a wrong name in a copy of the config with `MSMARCO_CONFIG`): dense shows "Model loading failed" with fix suggestions.

## Evaluation results tab
- [ ] Before running the evaluation for a mode: an info box with the exact `evaluate_all.py` command (no fake numbers).
- [ ] After `python scripts/evaluate_all.py --mode dev`: query counts, exclusions and corpus size match `reports/eval/dev/run_settings.json`.
- [ ] Chart bars and the table match `reports/eval/dev/aggregate_metrics.csv` exactly (spot-check MRR@10 for each method).
- [ ] Hovering a bar shows method, metric, mean and confidence interval.
- [ ] Significance table matches `significance.csv`.
- [ ] Disagreement explorer lists queries; choosing one shows the judged passage and each method's top hit.
- [ ] Download buttons save `per_query_metrics.csv`, `aggregate_metrics.json`, `significance.csv`, `comparison_chart.png`.
- [ ] "Execution settings" and "Full evaluation report" expanders open and render.

## Presentation checks
- [ ] Browser zoom 125% and a narrow window (~400 px): cards stay readable, no horizontal scrolling of the page.
- [ ] Projector test: method colours (blue TF-IDF, orange BM25, green dense) are distinguishable and every colour also has a text label.
