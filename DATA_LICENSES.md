# Data sources and licenses

Checked 2026-11-24 by reading each publisher's own policy page. This is a record of what was read, not legal advice. Items marked "unconfirmed" must be settled before any redistribution beyond this repository.

Every source record in `data/sources/sources.jsonl` carries `origin_url`, `fetched` (date), `version` and a `license_note`, and can be re-fetched with `plainline build-pilot`.

## Used in the pilot

| Source | What is used | Terms found | Status |
|---|---|---|---|
| MedlinePlus health topic summaries (NLM, NIH) | Health topic summary text, English, cut at a sentence boundary to at most 350 words | NLM states that summaries on health topic pages are public domain and may be reproduced and redistributed (https://medlineplus.gov/about/using/usingcontent/). NLM asks users to credit it as the source ("Courtesy of the U.S. National Library of Medicine"). | Confirmed for health topic summaries only |
| DailyMed SPL drug labels (NLM, FDA labeling) | Short excerpts (boxed warning, indications, dosage, contraindications, warnings; each section cut at a sentence boundary) from one manufacturer label per drug | DailyMed is a public NLM service and the labeling is FDA-regulated. The label text is authored and submitted by the drug manufacturer, so it is not clearly a U.S. government work. I did not find a DailyMed-specific reuse statement on the pages I read (https://www.nlm.nih.gov/copyright.html covers NLM government works and says third-party content may be copyrighted). | Unconfirmed. Kept to short excerpts with `setid` and version so the full label can be re-fetched. Treat redistribution of the excerpts as unresolved until checked with NLM or counsel. |

## Rules the fetchers follow

- MedlinePlus: only health topic summaries. Never fetched: A.D.A.M. encyclopedia articles, ASHP drug monographs, Natural Medicines supplement monographs, and images. NLM lists these as copyrighted.
- DailyMed: Rx labels from non-repackager labelers where the title allows it, one label per drug.
- No UMLS, SNOMED CT, DrugBank, RxNorm or MIMIC content is used anywhere in the core pipeline.

## Considered, not used in the pilot

| Source | Terms found | Plan |
|---|---|---|
| CDC and ATSDR web pages | CDC states most of its web content is not subject to copyright and is public domain, with exceptions for third-party licensed material, state and local government works, and foreign copyright. Attribution to CDC or HHS is required (https://www.cdc.gov/other/agencymaterials.html). | Planned for the next data round. A fetcher must check each page for a third-party copyright notice before inclusion. |
| NICE and most society guidelines | Not open in general | Excluded unless a specific license permits it |
| Newly written clinician vignettes | Authored for this project | Planned held-out test split, never published as raw text |

## Contamination note

All pilot sources are public web text and are probably in model training data. The pilot split is therefore labelled `dev`. The roadmap in README.md covers a held-out split. Source files carry a canary string.
