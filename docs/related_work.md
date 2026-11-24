# Related work

This page places Plainline next to the work it builds on and says, for each closest neighbour, what is different here. Citations are author-year; the full list is at the end and in [references.bib](../references.bib). Every entry in that file was checked against the arXiv API, Crossref, the ACL Anthology or the publisher record. We read abstracts and, where noted in the text, the parts of papers that bear on the comparison. We did not re-run any of these systems. Where a claim about a paper's scope depends on something we did not read in full, the sentence says so.

## 1. General medical LLM evaluation

Early medical evaluation was multiple choice. MedQA (Jin et al., 2021) collects licensing-exam questions, and MultiMedQA with Med-PaLM (Singhal et al., 2023) combined several question sets with a human evaluation of long-form answers. Med-PaLM 2 (Singhal et al., 2025) pushed multiple-choice accuracy further and added pairwise physician preference. Exam accuracy tells us little about whether a model preserves a specific warning when it rewrites a document, and it is close to saturated for frontier models.

Rubric-based evaluation of open-ended answers is now the common lab format. HealthBench (Arora et al., 2025) scores multi-turn health conversations against physician-written rubrics with a model grader, and HealthBench Professional (Hicks et al., 2026) extends the idea to real clinician chats. MedHELM (Bedi et al., 2026) organises medical tasks into a clinician-validated taxonomy and uses an LLM jury for open-ended outputs. These benchmarks include communication and patient-education tasks, but communication quality is one rubric axis among many, and criteria are written per conversation. They do not hold the clinical content fixed while varying the reader, so they cannot isolate what changes when the audience changes.

Interactive and agentic evaluations test other skills. AgentClinic (Schmidgall et al., 2026) simulates clinical encounters, CRAFT-MD (Johri et al., 2025) evaluates models in conversational patient-interaction settings rather than static vignettes, and MedAgentBench (Jiang et al., 2025) places agents in a virtual EHR. They study what the model does in a task. Plainline studies one narrow transformation, rewriting a given source for a reader.

**What is new here.** A controlled, paired design. The source is fixed, the reader is the only manipulated variable, and the score is the survival of a pre-tiered list of safety-critical facts rather than a holistic or per-conversation rubric score.

## 2. Do people actually benefit?

Bean et al. (2026) ran a randomised trial with 1,298 participants in which people used LLMs to assess medical scenarios. The models identified the condition well when tested alone, but participants using them did not do better than a control group. The authors point to the human-model interaction, not model knowledge, as the failure point, and to the weakness of standard benchmarks as predictors of it. This motivates studying communication directly. It is also a limit on what we can claim: Plainline uses simulated readers and a judge, and says nothing about real comprehension (see the Teachback item in the roadmap).

## 3. Plain-language adaptation and factuality

This is the closest technical neighbour. PLABA (Attal et al., 2023) provides adaptations of biomedical abstracts into plain language. Devaraj et al. (2021) study paragraph-level simplification of medical texts and the factual errors it introduces. CELLS and the retrieval-augmented lay-language work of Guo et al. (2024) cover lay summaries of scientific articles. FactPICO (Joseph et al., 2024) collects expert factuality ratings of LLM plain-language summaries of randomised-trial abstracts and organises errors around PICO elements (population, intervention, comparator, outcome). PlainQAFact and its PlainFact dataset (You and Guo, 2026) propose a factual-consistency metric for plain-language summaries.

Domain-general factuality tools are the building blocks of such checks. SummaC (Laban et al., 2022) and AlignScore (Zha et al., 2023) score consistency between a text and a source, and FActScore (Min et al., 2023) decomposes a generation into atomic facts and checks each. Our per-fact judge uses the same decomposition idea in the opposite direction: the facts come from the source and we ask whether each one survives in the output. That asks about recall of required content, which consistency scores do not measure, since an output can be fully consistent and still omit the warning.

**What is new here.** Four things. (1) The target is source-to-reader adaptation of safety content in drug labels and emergency-care topics, not abstract-to-lay summarisation. (2) Facts carry criticality tiers, so an omitted boxed warning counts differently from an omitted background sentence, and we report retention of the critical tier on its own. (3) The same source is rewritten for several readers plus a neutral control, so the Adaptation Tax is a paired difference in which source difficulty cancels. (4) A reader-cue slice with identical task text across cues measures whether stated reader identity changes safety content. We do not claim novelty for per-fact verification or for plain-language factuality checking in general.

## 4. Patient communication benchmarks

DischargeSim (Yao et al., 2025) simulates patient education at discharge and tests comprehension of the simulated patient. DischargeBench (Jang et al., 2026) builds persona-grounded patient simulations, including health-literacy axes, to evaluate discharge education by LLMs; according to its abstract, validation against real patients is still open. RephQA (Qiu et al., 2025) measures how well LLM answers to public-health questions match target reading levels. Berkman et al. (2011) summarise the link between low health literacy and worse outcomes, which is why reader-aware explanation matters at all.

**What is new here.** Those works centre on dialogue, simulated patients, or reading level as the outcome. Plainline fixes a source document and a checklist and measures the content that survives. Reading level appears only as a pass or fail band, so a model cannot look good by being easy to read and wrong. We do not use patient-record data, so there is no credentialed-access requirement for the core data.

## 5. Readability measurement

We use Flesch-Kincaid grade (Flesch, 1948; Kincaid et al., 1975) and keep SMOG (McLaughlin, 1969) as an alternative. These formulas are crude proxies for comprehension, and the Plainline bands are working values not validated against real readers. They are used as a gate, not a quality score.

## 6. LLM-as-judge reliability

LLM judges agree with humans on many tasks but show biases (Zheng et al., 2023), including favouring their own outputs (Panickssery et al., 2024). We use a narrow three-way question per fact instead of a holistic score, recommend a judge from a different family than the model under test, and the CLI warns when they match. The judge is not yet validated against clinician labels. The validation protocol, with Cohen's kappa (Cohen, 1960), is in [judge_validation.md](judge_validation.md).

## 7. Audience cues and sociodemographic bias

Omiye et al. (2023) show that LLMs repeat debunked race-based medical claims, Zack et al. (2024) find that GPT-4 can perpetuate racial and gender biases in clinical vignettes and recommendations, and Pfohl et al. (2024) propose a toolbox for surfacing health-equity harms. Those studies vary patient attributes and look at clinical content or recommendations. Our cue slice varies the reader's stated profession, literacy, language, age or caregiver role, with identical task text, and measures retention of the same fixed safety facts. It is a narrow audit of one behaviour, not a bias taxonomy.

## 8. Contamination, canaries, harness

All pilot sources are public web text and probably in training data, so the current split is labelled dev. We follow the practice of Jacovi et al. (2023) of avoiding plain-text publication of held-out data and including canary strings, as in BIG-bench (Srivastava et al., 2023). The held-out split is a roadmap item. The benchmark is packaged as an Inspect AI task (UK AI Security Institute, 2024). Uncertainty is reported with percentile bootstrap intervals over items (Efron, 1979).

## Summary table

| Closest work | What it does | Difference in Plainline |
|---|---|---|
| HealthBench, HealthBench Professional | Physician rubrics over health conversations | Fixed source and reader as the manipulated variable; tiered fact survival |
| MedHELM | Taxonomy of medical tasks, LLM jury | One narrow task with a per-fact judge, no jury average |
| FactPICO, PlainQAFact | Factuality of plain-language summaries of abstracts | Safety-fact recall with tiers, paired readers, labels and emergency-care text |
| Devaraj et al., PLABA, CELLS | Plain-language adaptation datasets and methods | Evaluation protocol for frontier models, not a training corpus |
| DischargeSim, DischargeBench | Simulated patient dialogue at discharge | Source-grounded, single-turn, no patient records |
| RephQA | Reading level of public-health answers | Reading level is a gate; fact survival is the score |
| Bean et al. | Randomised trial of public use of LLMs | We do not test real readers; a later study is planned |
| Omiye, Zack, Pfohl | Sociodemographic bias in clinical outputs | Cue slice on reader identity, same safety facts for every cue |

## References

- Arora et al. (2025). HealthBench: Evaluating Large Language Models Towards Improved Human Health. [arXiv:2505.08775](https://arxiv.org/abs/2505.08775)
- Attal et al. (2023). A dataset for plain language adaptation of biomedical abstracts. Scientific Data. [doi:10.1038/s41597-022-01920-3](https://doi.org/10.1038/s41597-022-01920-3)
- Bean et al. (2026). Reliability of LLMs as medical assistants for the general public: a randomized preregistered study. Nature Medicine. [doi:10.1038/s41591-025-04074-y](https://doi.org/10.1038/s41591-025-04074-y)
- Bedi et al. (2026). Holistic evaluation of large language models for medical tasks with MedHELM. Nature Medicine. [doi:10.1038/s41591-025-04151-2](https://doi.org/10.1038/s41591-025-04151-2)
- Berkman et al. (2011). Low Health Literacy and Health Outcomes: An Updated Systematic Review. Annals of Internal Medicine. [doi:10.7326/0003-4819-155-2-201107190-00005](https://doi.org/10.7326/0003-4819-155-2-201107190-00005)
- Cohen (1960). A Coefficient of Agreement for Nominal Scales. Educational and Psychological Measurement. [doi:10.1177/001316446002000104](https://doi.org/10.1177/001316446002000104)
- Devaraj et al. (2021). Paragraph-level Simplification of Medical Texts. Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. [doi:10.18653/v1/2021.naacl-main.395](https://doi.org/10.18653/v1/2021.naacl-main.395)
- Efron (1979). Bootstrap Methods: Another Look at the Jackknife. The Annals of Statistics. [doi:10.1214/aos/1176344552](https://doi.org/10.1214/aos/1176344552)
- Flesch (1948). A new readability yardstick. Journal of Applied Psychology. [doi:10.1037/h0057532](https://doi.org/10.1037/h0057532)
- Guo et al. (2024). Retrieval augmentation of large language models for lay language generation. Journal of Biomedical Informatics. [doi:10.1016/j.jbi.2023.104580](https://doi.org/10.1016/j.jbi.2023.104580)
- Hicks et al. (2026). HealthBench Professional: Evaluating Large Language Models on Real Clinician Chats. [arXiv:2604.27470](https://arxiv.org/abs/2604.27470)
- Jacovi et al. (2023). Stop Uploading Test Data in Plain Text: Practical Strategies for Mitigating Data Contamination by Evaluation Benchmarks. Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. [doi:10.18653/v1/2023.emnlp-main.308](https://doi.org/10.18653/v1/2023.emnlp-main.308)
- Jang et al. (2026). DischargeBench: Pre-Deployment Evaluation of LLM Discharge Educators through Persona-Grounded Patient Simulation. [arXiv:2609.20827](https://arxiv.org/abs/2609.20827)
- Jiang et al. (2025). MedAgentBench: A Virtual EHR Environment to Benchmark Medical LLM Agents. NEJM AI. [doi:10.1056/AIdbp2500144](https://doi.org/10.1056/AIdbp2500144)
- Jin et al. (2021). What Disease Does This Patient Have? A Large-Scale Open Domain Question Answering Dataset from Medical Exams. Applied Sciences. [doi:10.3390/app11146421](https://doi.org/10.3390/app11146421)
- Johri et al. (2025). An evaluation framework for clinical use of large language models in patient interaction tasks. Nature Medicine. [doi:10.1038/s41591-024-03328-5](https://doi.org/10.1038/s41591-024-03328-5)
- Joseph et al. (2024). FactPICO: Factuality Evaluation for Plain Language Summarization of Medical Evidence. Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). [doi:10.18653/v1/2024.acl-long.459](https://doi.org/10.18653/v1/2024.acl-long.459)
- Kincaid et al. (1975). Derivation of New Readability Formulas (Automated Readability Index, Fog Count and Flesch Reading Ease Formula) for Navy Enlisted Personnel. Naval Technical Training Command, Research Branch. [doi:10.21236/ada006655](https://doi.org/10.21236/ada006655)
- Laban et al. (2022). SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization. Transactions of the Association for Computational Linguistics. [doi:10.1162/tacl_a_00453](https://doi.org/10.1162/tacl_a_00453)
- McLaughlin (1969). SMOG grading: a new readability formula. Journal of Reading. [link](https://ogg.osu.edu/media/documents/health_lit/WRRSMOG_Readability_Formula_G._Harry_McLaughlin__1969_.pdf)
- Min et al. (2023). FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation. Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. [doi:10.18653/v1/2023.emnlp-main.741](https://doi.org/10.18653/v1/2023.emnlp-main.741)
- Omiye et al. (2023). Large language models propagate race-based medicine. npj Digital Medicine. [doi:10.1038/s41746-023-00939-z](https://doi.org/10.1038/s41746-023-00939-z)
- Panickssery et al. (2024). LLM Evaluators Recognize and Favor Their Own Generations. Advances in Neural Information Processing Systems 37. [arXiv:2404.13076](https://arxiv.org/abs/2404.13076)
- Pfohl et al. (2024). A toolbox for surfacing health equity harms and biases in large language models. Nature Medicine. [doi:10.1038/s41591-024-03258-2](https://doi.org/10.1038/s41591-024-03258-2)
- Qiu et al. (2025). RephQA: Evaluating Readability of Large Language Models in Public Health Question Answering. [arXiv:2509.16360](https://arxiv.org/abs/2509.16360)
- Schmidgall et al. (2026). AgentClinic: a multimodal benchmark for tool-using clinical AI agents. npj Digital Medicine. [doi:10.1038/s41746-026-02674-7](https://doi.org/10.1038/s41746-026-02674-7)
- Singhal et al. (2023). Large language models encode clinical knowledge. Nature. [doi:10.1038/s41586-023-06291-2](https://doi.org/10.1038/s41586-023-06291-2)
- Singhal et al. (2025). Toward expert-level medical question answering with large language models. Nature Medicine. [doi:10.1038/s41591-024-03423-7](https://doi.org/10.1038/s41591-024-03423-7)
- Srivastava et al. (2023). Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models. Transactions on Machine Learning Research. [arXiv:2206.04615](https://arxiv.org/abs/2206.04615)
- UK AI Security Institute (2024). Inspect AI: Framework for Large Language Model Evaluations. [link](https://github.com/UKGovernmentBEIS/inspect_ai)
- Yao et al. (2025). DischargeSim: A Simulation Benchmark for Educational Doctor-Patient Communication at Discharge. Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. [arXiv:2509.07188](https://arxiv.org/abs/2509.07188)
- You and Guo (2026). PlainQAFact: Retrieval-augmented factual consistency evaluation metric for biomedical plain language summarization. Journal of Biomedical Informatics. [doi:10.1016/j.jbi.2026.105019](https://doi.org/10.1016/j.jbi.2026.105019)
- Zack et al. (2024). Assessing the potential of GPT-4 to perpetuate racial and gender biases in health care: a model evaluation study. The Lancet Digital Health. [doi:10.1016/s2589-7500(23)00225-x](https://doi.org/10.1016/s2589-7500(23)00225-x)
- Zha et al. (2023). AlignScore: Evaluating Factual Consistency with A Unified Alignment Function. Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). [doi:10.18653/v1/2023.acl-long.634](https://doi.org/10.18653/v1/2023.acl-long.634)
- Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. Advances in Neural Information Processing Systems 36 (Datasets and Benchmarks Track). [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)
