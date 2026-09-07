# Track A — CEBRA dataset audit (motor / movement only)

**Date:** 2026-08-13
**Scope:** every dataset CEBRA 0.6.1 ships under `cebra_exploration/third_party/cebra/cebra/datasets/`. Loader code is the source of truth; docs are secondary.
**Question:** can any CEBRA-shipped dataset exercise our paper's claim — source-trained permutation-invariant decode + closed-form per-unit encoding-signature carrier on a *new session whose unit set has changed*, with no backward pass on the target?

---

## Bottom-line verdict

**No. Adopting any CEBRA-shipped dataset adds no evidence our paper does not already have from its four datasets. Do not download or integrate them.**

The only motor-relevant recording CEBRA ships is NLB Area2_Bump: **one rhesus macaque (Han), one recording session, 65 units that do not change**. CEBRA's `session=` argument is *active vs passive trial type*, not a recording session. A train/valid/test split of the same session with the same units cannot test cross-session calibration. The only unique behavioural axis — active reach vs unexpected bump — is a within-session context shift, not a unit-set change.

Rat hippocampus is the only CEBRA dataset with differing unit counts across recordings (120 / 48 / 55 / 66). It is movement (1.6 m linear track) but CA1 place cells, not motor cortex; the natural carrier is place+direction, not our cosine or velocity basis. Using three rats as source and one as target would be a cross-species, cross-area, cross-encoding experiment, not a BCI session-calibration result. We already have multi-session unit-set change (000688, FALCON) and cross-animal confirmation (subject-M). A new independent *motor* cohort would help; this is not one.

Area2_Bump shares the Miller / Northwestern lab with `dandi_000688` but **not** the animal: local 000688 subjects are M, C, J (T exists on DANDI, not held locally); Han does not appear. Same-lab S1 vs M1 is a confound if treated as independent motor evidence, and an opportunity only if we wanted a same-lab area comparison — which still fails the unit-set-change test.

---

## Per-dataset table

| Dataset (CEBRA register name) | Class | Sessions / subjects | Unit set changes? | Task basis | Verdict | One-sentence reason |
|---|---|---|---|---|---|---|
| Area2_Bump (`area2-bump*`) | motor/movement (S1 area 2, not M1) | **1 session / 1 subject (Han)** | **No** | 8-dir discrete + hand pos/vel | **NOT_USABLE** | Single-session, fixed 65-unit set; cannot define a source/target split that changes identities. |
| Rat hippocampus (`rat-hippocampus-*`) | movement (CA1 place cells) | 4 subjects × 1 packaged session each | Yes, across animals | 1-D position + binary L/R; no 8-dir | **NOT_USABLE** | Multi-animal unit-set change is real, but the encoding, species, area, and claim are not ours; would not strengthen the paper. |
| Allen Ca / Neuropixels movie (`allen-*`) | non-motor | n/a | n/a | movie-frame features | **NOT_USABLE** | Mouse visual cortex, passive movie watching. |
| Synthetic / GMM (`continuous-label-*`, `continuous-gaussian-mixture-*`) | non-motor | 1 generated stream | No | synthetic continuous label | **NOT_USABLE** | No animals, no sessions, no motor behaviour. |
| Poisson utilities (`cebra/datasets/poisson.py`) | non-motor | none (not a dataset) | n/a | n/a | **NOT_USABLE** | Spike-count simulator, not a recording. |
| Demo (`demo-*`) | non-motor | synthetic tensors | fake | random | **NOT_USABLE** | Unit-test noise; docstring says it will not yield useful embeddings. |

Scoring on the five criteria is in each detailed section. Nothing is `USABLE` or `USABLE_WITH_CAVEAT`.

---

## 1. Enumeration of everything CEBRA ships

Read from `cebra/datasets/__init__.py` L90–99 (the import list) plus every file in that directory. Registered names come from `@register` / `@parametrize` in the loader files cited below.

| File | What it is | Motor? |
|---|---|---|
| `monkey_reaching.py` | NLB Area2_Bump, monkey S1 | **Yes (movement; S1 not M1)** |
| `hippocampus.py` | Grosmark & Buzsáki CA1 linear track, 4 rats | **Movement, not motor cortex** |
| `allen/*.py` | Allen visual cortex Ca / Neuropixels, MOVIE1 | No |
| `synthetic_data.py` | Generated continuous-label observations | No |
| `gaussian_mixture.py` | Generated 2-D latents + 100-D noisy obs | No |
| `poisson.py` | `PoissonNeuronTransform` utilities | No (not registered) |
| `demo.py` | Random tensors for tests | No |
| `generate_synthetic_data.py`, `make_neuropixel.py`, `allen/make_neuropixel.py`, `save_dataset.py` | Generators / helpers, not datasets | No |

Docs (`docs/source/api/pytorch/datasets.rst` L7–33) list the same four families: synthetic, rat hippocampus, monkey S1, Allen. Nature 2023 methods confirm the same four (paper HTML, “We benchmarked and demonstrate the abilities of CEBRA on four datasets”).

---

## 2. Area2_Bump — `cebra/datasets/monkey_reaching.py`

### 2.1 What the recording is

- **Species / area / task:** rhesus macaque, Brodmann area 2 of somatosensory cortex (S1), delayed 8-direction centre-out reaching on a manipulandum; on a random 50% of trials a 2 N bump is applied during centre-hold (`monkey_reaching.py` L22–35; Nature 2023 methods: “electrophysiological recordings were performed in Area 2 of somatosensory cortex (S1) in a rhesus macaque”; DANDI 000127 `about.identifier` UBERON:0013533 “Somatosensory area 2”).
- **Provenance:** Chowdhury, Glaser & Miller, *eLife* 2020 (`10.7554/eLife.48198`); packaged for NLB’21 (Pei et al. 2021); DANDI **000127** version `0.220113.0359`, DOI `10.48324/dandi.000127/0.220113.0359` (`monkey_reaching.py` L25–34).
- **Subject:** **Han only.** Raw NWB path hardcoded as `s1_reaching/sub-Han_desc-train_behavior+ecephys.nwb` (`monkey_reaching.py` L69–72). DANDI assets are `sub-Han/sub-Han_desc-train_behavior+ecephys.nwb` and `sub-Han/sub-Han_desc-test_ecephys.nwb` ([DANDI 000127 assets API](https://api.dandiarchive.org/api/dandisets/000127/versions/0.220113.0359/assets/)).
- **Sessions / subjects:** **1 / 1.** DANDI `assetsSummary.numberOfSubjects = 1`, `numberOfFiles = 2` ([version API](https://api.dandiarchive.org/api/dandisets/000127/versions/0.220113.0359/)). NLB dataset card: “We release a **single recording session** for this dataset, provided by Raeed Chowdhury and Lee Miller at Northwestern” ([neurallatents.github.io/datasets.html](https://neurallatents.github.io/datasets.html)).

**Critical loader fact.** `session` ∈ {`active`, `passive`, `all`, `active-passive`} selects *trial type*, not a recording session (`monkey_reaching.py` L132–139, L274–275, L445–450). `active-passive` remaps directions 0–7 (active) vs 8–15 (passive); `all` keeps 0–7 for both.

### 2.2 Units, behaviour, trial structure

- **Units:** NLB’21 paper: **65 neurons, 462 total trials** ([arxiv 2109.04463](https://arxiv.org/abs/2109.04463), Area2_Bump paragraph). One session ⇒ the unit set is fixed. I did **not** open the `.jl` preload, so 65/462 is from NLB, not re-counted in CEBRA’s joblib.
- **Behavioural fields in the loader** (`monkey_reaching.py` L90–116, L349–361):
  - discrete 8-dir: `cond_dir/45` (active) or `bump_dir/45` (passive) → integers 0–7. **Our cosine basis `φ = [cos θ, sin θ]` applies directly** with `θ = k · π/4`.
  - continuous `hand_pos` (x, y) and `hand_vel` (x, y). **Velocity basis also applies.**
  - binary `passive` flag.
- **Trial structure:** aligned to `move_onset_time`, window **(−100, +500) ms**, **1 ms bins**, spikes smoothed with a Gaussian **σ = 40 ms** (`monkey_reaching.py` L57–59, L143–145; Nature 2023 methods, same numbers). Implied `trial_len = 600` samples/trial (`trial_len = len(spikes)/n_trials`, L115).
- **Splits:** NLB `train` / `val`; CEBRA uses NLB train as train, then **splits NLB val in half** into valid and test (`monkey_reaching.py` L310–313, L121–128, L148–153). These are trial splits of the **same session, same units**.

### 2.3 Download, size, licence

| Artefact | URL | Size (HEAD `Content-Length`) | Licence |
|---|---|---|---|
| Raw NWB train | DANDI 000127 `sub-Han_desc-train_behavior+ecephys.nwb` | **1,822,876,234 B (~1.82 GB)** — over the 200 MB stop line; **not downloaded** | DANDI `spdx:CC-BY-4.0` |
| Raw NWB test | DANDI 000127 `sub-Han_desc-test_ecephys.nwb` | 492,576 B | same |
| CEBRA preload `all_all.jl.gz` | `https://cebra.fra1.digitaloceanspaces.com/data/monkey_reaching_preload_smth_40/all_all.jl.gz` | 48,175,026 B | code Apache-2.0 (`LICENSE.md` L13–16); data follows DANDI CC-BY-4.0 |
| `active_all.jl.gz` | same prefix | 26,187,656 B | same |
| `passive_all.jl.gz` | same prefix | 21,998,090 B | same |
| `active_train.jl.gz` | same prefix | 19,157,931 B | same |
| remaining split `.jl.gz` | same prefix | 2.5–36 MB each (HEAD’d; see §6) | same |

HEAD via `curl -I -x http://127.0.0.1:7890` returned HTTP/2 200 for every URL above. Nothing ≥ 200 MB was pulled. Figshare mirrors exist in `tests/test_datasets.py` L311–319 but the runtime loader uses DigitalOcean.

### 2.4 Overlap with what we already hold

| Local holding | What it is | Overlap with Area2_Bump? |
|---|---|---|
| `sua_exploration/data/000128/sub-Jenkins` (662 M, 2 NWB) | NLB **MC_Maze**, Jenkins, M1+PMd, Churchland/Kaufman/Shenoy (`000128/dandiset.yaml` L41–44, L103–104) | **None.** Different dandiset, lab, animal, area. |
| `sua_exploration/data/000129/sub-Indy` (49 M, 2 NWB) | NLB **MC_RTT**, Indy, M1, O’Doherty/UCSF (`000129/dandiset.yaml` L42–43, L66–72) | **None.** |
| `sua_exploration/data/dandi_000688/` (12 G; sub-M 22 NWB, sub-C 53 CO + 15 RT, sub-J 3 NWB) | Miller lab M1/PMd centre-out + RT, subjects M/C/J (`04_experiments.tex` L39–48; local NWB `subject_id` = M/C/J, `lab` = Miller, `institution` = Northwestern University) | **Same lab, different animal, different area.** DANDI 000688 also has **sub-T** (12 NWB on the archive, not held locally). No `sub-Han` path in 000688 assets. |
| `SPINT-main/data/000941` FALCON M1 | Monkey L, Rouse/Schieber, reach-to-grasp | None. |
| `SPINT-main/data/000953` FALCON M2 | Monkey N, Chestek, finger | None. |
| `SPINT-main/data/000954` FALCON H1 | Human Pitt; held-in only in scope | None. |

Lee E. Miller is a contributor on both DANDI 000127 (ORCID `0000-0001-8675-7140`) and DANDI 000688 (same ORCID). Local 000688 NWB `session_description` is “Monkey {M,C,J} performing center-out reaching task” — not Han. **Inference, marked:** published Miller-lab naming often maps C→Chewie, M→Mihili, J→Jango (e.g. Zenodo 8271239 README); I did **not** verify those full names inside 000688 NWB. Regardless of full names, the subject IDs are M/C/J/T, not Han.

Same-lab reaching + manipulandum is a **confound** if Area2_Bump were treated as an independent motor cohort, and a **same-lab S1 vs M1 opportunity** only if we wanted that comparison. It still cannot test unit-set change.

### 2.5 Scores

| Criterion | Score | Evidence |
|---|---|---|
| (a) motor/movement behaviour | **Yes** | Centre-out reach + bump; hand pos/vel present. Area is S1, not M1. |
| (b) multiple sessions/subjects with differing unit sets | **No — decisive fail** | 1 subject, 1 session, 65 units. `session=` is trial type. |
| (c) task basis our carrier can consume | **Yes** | Discrete 8-dir (`target` 0–7) and continuous vel/pos. |
| (d) source/target split for the three-phase protocol | **Not definable without cheating** | Train/val/test = trial split, same units. Active→passive = same units, context shift. Neither is “new session, new unit set, no backprop.” |
| (e) practical | Preload 22–48 MB, URL live; raw NWB 1.82 GB, do not pull. CC-BY-4.0. | |

### 2.6 Verdict

**NOT_USABLE** — a beautifully curated single-session S1 recording cannot test cross-session calibration.

Active-vs-passive is the one axis our four datasets lack. It is the wrong axis: it tests whether a carrier fitted on volitional reaches transfers to proprioceptive bumps **on the same units**, not whether a closed-form identity token recalibrates a changed unit set. That is a Chowdhury/London-style encoding question, not this paper.

No integration path. If someone later wanted a *within-session* bump ablation (out of paper scope), the loader would be `cebra.datasets.init("area2-bump-target-active")` / `...-passive` with `φ = [cos(kπ/4), sin(kπ/4)]`. That is recorded here only so it is not rediscovered as a “usable” split.

---

## 3. Rat hippocampus — `cebra/datasets/hippocampus.py` (judgement call)

### 3.1 What the recording is

- **Species / area / task:** Long-Evans rat, hippocampal CA1, water-rewarded running on a **1.6 m linear track**, leftward vs rightward (`hippocampus.py` L22–31, L92–96; Nature 2023 ED: “the animal transverse a 1.6 meter linear track leftwards or rightwards”).
- **Provenance:** Grosmark & Buzsáki, *Science* 2016; Chen et al., *Sci. Rep.* 2016; CRCNS **hc-11**, DOI `10.6080/K0862DC5` (`hippocampus.py` L25–30).
- **CEBRA packaging:** four files, one per rat — `achilles`, `buddy`, `cicero`, `gatsby` (`hippocampus.py` L50–82, L86–90). Labels are `position` ∈ R³ = `[position_m, right, left]` with `right,left ∈ {0,1}` (loader L94–96; verified in the probe below).
- **CRCNS parent set:** eight concatenated PRE / MAZE / POST sessions from the **same four rats** (some rats have two novelty sessions). CEBRA ships **one `.jl` per rat**, i.e. one maze-running excerpt each — not the eight-session CRCNS corpus. Which CRCNS session was packaged per rat is **unverified**.

### 3.2 Sessions, units, samples (verified by probe)

Tiny preloads (72–414 KB gz) were downloaded as a metadata probe by `cebra_exploration/scripts/probe_cebra_dataset_shapes.py` (well under the 200 MB stop). Shapes of the extracted joblib:

| Rat | `spikes` shape | Units | Samples | ≈ duration at 25 ms | `position` |
|---|---|---|---|---|---|
| Achilles | (10178, 120) | **120** | 10178 | 254 s | (10178, 3), pos ∈ [0, 1.6], L/R one-hot |
| Buddy | (6577, 48) | **48** | 6577 | 164 s | same layout |
| Cicero | (47431, 55) | **55** | 47431 | 1186 s | same layout |
| Gatsby | (17026, 66) | **66** | 17026 | 426 s | same layout |

Bin size **25 ms** is in the loader docstring (`hippocampus.py` L94). Unit counts **differ across animals**, so a cross-animal source/target split is *structurally* definable. That is the only CEBRA dataset for which criterion (b) is true.

Multi-subject wrapper: `rat-hippocampus-multisubjects-3fold-trial-split-{0,1,2}` concatenates the four rats (`hippocampus.py` L289–315). Within-rat “trials” are round-trips parsed from direction changes (`hippocampus.py` L207–210), then 3-fold nested CV — **trial splits of one recording**, not new implant days.

### 3.3 Download, size, licence

| File | URL | gz size (HEAD) | unzipped (probe) |
|---|---|---|---|
| `achilles.jl.gz` | `https://cebra.fra1.digitaloceanspaces.com/data/rat_hippocampus/achilles.jl.gz` | 167,761 B | 10,015,448 B |
| `buddy.jl.gz` | `.../buddy.jl.gz` | 71,947 B | 2,683,669 B |
| `cicero.jl.gz` | `.../cicero.jl.gz` | 414,118 B | 22,008,237 B |
| `gatsby.jl.gz` | `.../gatsby.jl.gz` | 211,955 B | 9,398,605 B |

All four HEAD’d HTTP/2 200 and downloaded successfully through `127.0.0.1:7890`. Figshare mirrors in `tests/test_datasets.py` L299–309.

**Licence:** CRCNS hc-11 requires citing Grosmark & Buzsáki 2016 and the dataset DOI (`hippocampus.py` L25–30; [crcns.org hc-11 description PDF](https://buzsakilab.nyumc.org/datasets/GrosmarkAD/_extra/crcns_hc-11_data_description.pdf)). The CRCNS HTML pages did not render a licence body in this environment; the full CRCNS data-use agreement text is **unverified**. CEBRA code is Apache-2.0.

### 3.4 Scores

| Criterion | Score | Evidence |
|---|---|---|
| (a) motor/movement behaviour | **Movement, not motor** | Position on a linear track. Area is CA1, not motor cortex. Place-cell encoding, not cosine-tuned reach. |
| (b) multiple sessions/subjects, differing unit sets | **Yes, across animals only** | 4 rats, 120/48/55/66 units. Each CEBRA file is one session. Not chronic implant drift. |
| (c) task basis our carrier can consume | **No, not without redesign** | No discrete target direction. Position is 1-D on [0, 1.6] m plus binary heading. `φ = [cos θ, sin θ]` does not apply. A place-field / position+direction basis would be a different carrier (Track C, not “use their data”). Velocity is not in the packaged `.jl` (keys are only `spikes`, `position`). |
| (d) source/target split | **Definable, but the wrong experiment** | e.g. Achilles+Buddy+Gatsby source → Cicero target; calibration prefix = first N running laps. That tests cross-animal place-cell transfer, not implanted-decoder session calibration. |
| (e) practical | Trivial size, URLs live. Licence citation-only as far as verified. | |

### 3.5 Where I land

**NOT_USABLE** for this paper.

I am not dismissing it because it is “not motor cortex” as a slogan. I am dismissing it because the paper’s claim is a conjunction: *changed unit set + short labelled prefix + no target backprop + motor/BCI decode*. Hippocampus supplies only the first. The carrier would have to be rewritten around place fields; the result, if positive, would be “closed-form signatures transfer across rats in CA1,” which we would not be allowed to cite as support for the BCI claim (same-data/same-split/same-metric rule in the brief §4). We already have a stronger, on-claim cross-animal check (000688 subject-M, 15 CO sessions).

A new *independent motor* cohort would strengthen cross-animal generalization. Four Buzsáki-lab rats are independent of Miller/FALCON, but they are not a motor cohort.

---

## 4. Non-motor datasets — one-line dismissals

- **Allen Ca / Neuropixels (`allen/*`).** Mouse primary visual cortex during Allen MOVIE1; labels are DINO embeddings of movie frames (`allen/ca_movie.py` L22–34, L59–64; `allen/neuropixel_movie.py` L47–50). Passive viewing, not movement. Out of scope.
- **Synthetic / Gaussian mixture (`synthetic_data.py`, `gaussian_mixture.py`).** Generated 2-D latents → 100-D noisy observations with Poisson/Gaussian/Laplace/uniform/t noise (`gaussian_mixture.py` L42–48; `synthetic_data.py` L105–108). No animal, no session, no motor behaviour.
- **Poisson (`poisson.py`).** `PoissonNeuronTransform` / `PoissonNeuron` sampling utilities. Not a registered dataset.
- **Demo (`demo.py`).** `torch.randn` / `randint` tensors; “none of the datasets will yield useful embeddings” (`demo.py` L21–26).

---

## 5. Does anything add evidence we lack?

| What the paper still wants | Do we already have it? | Would a CEBRA dataset add it? |
|---|---|---|
| Multi-session, changing unit set, same implant | Yes: 000688 subject-C / subject-M; FALCON M2/H1 | **No.** Area2_Bump is one session. Hippocampus is one session per animal. |
| Cross-animal generalization, motor | Yes: 000688 subject-M (15 CO sessions) vs subject-C; FALCON is a different lab/animal | **No.** Area2_Bump is same Miller lab, different area, one session. Hippocampus is the wrong species/area/encoding. |
| Discrete 8-dir cosine basis | Yes: 000688 CO | Area2_Bump has it, on one session — redundant. |
| Continuous velocity basis | Yes: 000688 RT, FALCON M2 | Area2_Bump has hand vel, on one session — redundant. |
| Active vs passive / bump | **No** | Area2_Bump has it, **same units**. Wrong axis for the claim. |
| Independent motor lab / animal | Partially: FALCON (Chestek, Pitt, Rouse) vs Miller 000688 | Area2_Bump is **not** independent of 000688 (same PI, same building). |
| Human 7-DoF | Yes: FALCON H1 held-in | Neither CEBRA set is human. |

**Honest answer:** they add nothing we lack *on the claim*. The bump structure is scientifically interesting and out of scope. Skipping these datasets saves the download (1.82 GB raw / ~48 MB preload) and weeks of a protocol that would come back `UNDEFINED` on criterion (b).

---

## 6. HEAD sizes for remaining Area2_Bump preloads

All under `https://cebra.fra1.digitaloceanspaces.com/data/monkey_reaching_preload_smth_40/`, HTTP/2 200:

| File | Bytes |
|---|---|
| `all_train.jl.gz` | 36,044,362 |
| `all_test.jl.gz` | 6,107,089 |
| `all_valid.jl.gz` | 6,062,796 |
| `active_test.jl.gz` | 3,583,703 |
| `active_valid.jl.gz` | 3,473,393 |
| `passive_train.jl.gz` | 16,884,379 |
| `passive_test.jl.gz` | 2,543,198 |
| `passive_valid.jl.gz` | 2,577,465 |

---

## 7. What I could not verify

1. **Exact trial counts per CEBRA Area2_Bump split** (`num_trials` inside each `.jl`). NLB states 462 trials / 65 neurons for the whole session; CEBRA’s train / half-val / half-val cut was not opened (19–48 MB; URL and size verified, content not).
2. **Exact unit count inside the Area2_Bump `.jl`** — cited as 65 from NLB’21, not re-counted from CEBRA’s preload.
3. **Which of the eight CRCNS hc-11 sessions** was packaged into each of `achilles.jl` / `buddy.jl` / `cicero.jl` / `gatsby.jl`.
4. **Whether CEBRA’s hippocampus `.jl` is speed-filtered or maze-epoch-only.** Durations (2.7–20 min) are shorter than the CRCNS ~45 min maze epoch; the filter rule is not in the loader.
5. **Full CRCNS hc-11 licence text.** Citation requirement is documented; the HTML licence body did not render.
6. **Full names of 000688 monkeys** (Chewie / Mihili / Jango / MrT). Local NWB only stores IDs M/C/J. Mapping is an inference from other Miller-lab releases, not from these files.
7. **Whether Han appears in any unpublished Miller-lab session we do not hold.** On DANDI 000688 the subjects are J, T, C, M only (111 assets, no `sub-Han`).
8. **FALCON M2/H1 held-out private eval sets** — not opened; H1 held-out is out of scope (brief §3.1). Not needed for this audit.
9. **NLB Area2_Bump train vs val trial counts** in the official NLB split table. The 462 figure is the session total.
10. **Whether 000129 “15 sessions” in the brief** is a pipeline slice of the single Indy NLB file or a pointer at 000688 RT. Irrelevant to CEBRA overlap: 000129 is Indy/UCSF, not Han/Miller.

---

## 8. Probe artefact

`cebra_exploration/scripts/probe_cebra_dataset_shapes.py` — downloads the four hippocampus `.jl.gz` files (< 1 MB total) through the clash proxy and prints shapes. Re-runnable. No Area2_Bump bytes were written to disk.
