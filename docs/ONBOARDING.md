# OpenInterBias — Resoconto per il team (onboarding)

> Documento di ingresso per chi vede il progetto **per la prima volta**. Spiega cos'è, cosa è
> stato cambiato dopo la fork, com'è organizzato il cluster, cosa è già stato eseguito e quali
> sono i prossimi passi. Per i dettagli operativi rimanda ai documenti citati in fondo (§10).
>
> Aggiornato al **2026-06-02**. Repo: `https://github.com/giampyhatesyou/OpenInterBias.git`.

---

## 0. TL;DR (lo stato in 5 righe)

- **OpenInterBias** è una fork di **OpenBias** (CVPR 2024) che vogliamo estendere alla **bias
  intersezionale** (es. *gender × race* invece dei singoli attributi).
- La **baseline OpenBias completa è già girata end-to-end su baldo**: **6.384 immagini SDXL →
  VQA con LLaVA-1.5-13B → plot**. Gli artefatti (`vqa_answers.json`, `data_counts.json`, i due
  PNG) sono su baldo; copie locali dei plot sono sul Desktop.
- L'estensione intersezionale (lo "Stage 5") è **progettata ma non ancora implementata**: il piano
  è in [`intersectional/STAGE5_PLAN.md`](../intersectional/STAGE5_PLAN.md) e attende la review del team.
- Tutto lo Stage 5 sarà **post-hoc, CPU-only, additivo**: legge gli output già in cache, **non
  ritocca** la pipeline upstream e non rigenera immagini.
- Prima cosa da fare per il team: **leggere e validare il piano Stage 5** (decisioni Q1–Q5, §8).

---

## 1. Cos'è OpenBias e cosa vogliamo aggiungere

**OpenBias** (D'Incà et al., CVPR 2024 — UniTN) rileva bias *open-set* nei modelli text-to-image,
senza una lista di bias predefinita. Pipeline a 3 stage + quantificazione:

```
Stage 1  Bias Proposal     un LLM (Llama-2) propone i bias possibili da un set di caption
Stage 2  Image Generation  il modello T2I target (SD-XL) genera immagini dalle stesse caption
Stage 3  Bias Detection    un VQA (LLaVA-1.5-13B) riconosce quali classi di bias sono presenti
Stage 4  Quantification    make_plots.py calcola le metriche e disegna i grafici
```

Due metriche upstream:
- **Context-free**: skew della distribuzione di classe aggregata su *tutte* le immagini → `1 − entropia`.
- **Context-aware**: media, per-prompt, dell'entropia delle immagini dello stesso prompt.

**Il nostro contributo (la fork): lo Stage 5 — analisi intersezionale.** Invece di misurare un
attributo alla volta, misuriamo la **distribuzione congiunta di due attributi** sullo stesso
soggetto (stesso `refer_to`, es. `person gender × person race`), con due metriche:
- **Joint Intensity** = `1 − entropia(distribuzione congiunta normalizzata)`;
- **Normalized Mutual Information** = dipendenza tra i due attributi.

Lo Stage 5 è **post-hoc**: parte da `vqa_answers.json` (output dello Stage 3) e non tocca gli stage 1–4.

---

## 2. Cosa è stato cambiato dopo la fork

Le modifiche si dividono in tre categorie. **Principio guida: l'upstream resta intatto; l'estensione
intersezionale è solo additiva** (vedi [`ARCHITECTURE_NOTE.md`](../ARCHITECTURE_NOTE.md) §5).

### 2a. Modifiche minime ai file upstream (servono solo per far girare la pipeline)

| File | Cosa è cambiato | Perché |
|---|---|---|
| `utils/config.py` | I path hard-coded (`/<insert>/<path>/<here>/…`) ora si leggono da **variabili d'ambiente** via `get_path('OPENBIAS_*', default)`. | Rendere la repo multi-macchina senza editare il codice (vedi `.env.example` / [`SETUP.md`](../SETUP.md)). |
| `utils/config.py` | `max_prompts_per_bias: 100 → 2` e `n-images: 10 → 1`. | Ridurre la baseline a una dimensione fattibile sul cluster (**6.384 immagini** invece di centinaia di migliaia). ⚠️ Vedi la nota in §6 sul context-aware. |
| `make_plots.py` | `--vqa_model` ora prende le scelte da `VQA_SETTING` (prima era una lista fissa). | Coerenza, nessun cambio di logica. |
| `requirements.txt` | Pin aggiornati/allentati. | Far combaciare le versioni con torch/CUDA del cluster. |
| `README.md`, `.gitignore` | Header fork + sezione configurazione; ignore di output pesanti (`sd_generated_dataset/`, `results/`, `runs/*`). | Documentazione e igiene del repo. |

> Nota: nel repo è committato anche `utils/config.py.bak.pre_smoke` (backup dell'originale) e dei
> **log di generazione molto grossi** in `logs/stage2_*.out/.err`. Sono utili come traccia, ma sono
> candidati alla pulizia (vedi §9, punto 5).

### 2b. Scaffolding aggiunto (additivo, non tocca l'upstream)

| Cartella / file | A cosa serve |
|---|---|
| [`intersectional/`](../intersectional/) | Cuore dell'estensione. `SCHEMA.md` (schema + metriche), `ARCHITECTURE_NOTE.md` (come si innesta post-hoc), **`STAGE5_PLAN.md`** (piano implementativo dettagliato). Il codice (`pairing.py`, `scoring.py`, `run_analysis.py`, `make_plots.py`) è **ancora da scrivere**. |
| [`docs/`](.) | `SETUP_BALDO.md` (checklist per girare sul cluster), `SCHEMA_DECISION.md` (le domande aperte Q1–Q5 per il gruppo), questo `ONBOARDING.md`. |
| [`cluster/`](../cluster/) | Template SLURM `.sbatch` per gli stage 1–4 + `05_intersectional_analysis.sbatch` **placeholder** (esce con codice 2 finché non esiste `run_analysis.py`). |
| [`configs/`](../configs/) | Template YAML di run (`baseline`, `pilot`, `intersectional`) per la riproducibilità. Sono documentazione: **non** sostituiscono `utils/config.py` a runtime. |
| [`tools/`](../tools/) | Helper read-only: `check_paths.py` (verifica i path), `inspect_proposed_biases.py` (conta le coppie candidate), `smoke_baseline.py`, `snapshot_run.sh`. |
| [`tests/`](../tests/) | Pytest. 5 test sugli invarianti upstream passano oggi; `tests/intersectional/` è il posto (ancora vuoto) per i test dello Stage 5. |
| [`runs/`](../runs/) | Cartelle per-run (gitignored tranne il README): convenzione di bookkeeping degli esperimenti. |
| `ARCHITECTURE_NOTE.md`, `SETUP.md`, `AGENT.md` | Mappa dell'architettura, setup multi-macchina, istruzioni operative (AGENT.md è gitignored). |

### 2c. Tre bug che impedivano alla baseline di funzionare — tutti risolti

Durante i run su baldo sono emersi tre blocchi (dettaglio completo nella memoria di progetto e in
[`RESOCONTO_OPENBIAS.md`](../../RESOCONTO_OPENBIAS.md)):

1. **ConceptNet (timeout mascherato).** `utils/utils.py:filter_caption_generated` interrogava l'API
   web `api.conceptnet.io` per ogni classe assente da `synonyms.json`; rispondeva **HTTP 502 ~8.000
   volte**, bruciando l'intero limite SLURM *prima* di generare. → **Risolto**: il `synonyms.json`
   committato (8.224 voci) copre tutte le classi → post-processing **offline in ~14 s**. *Non
   cancellare mai `synonyms.json`.*
2. **peft / accelerate (crash al caricamento SDXL).** `peft 0.17.1` richiede `accelerate>=0.34`, ma
   nel venv c'è `accelerate 0.23` → `diffusers.from_pretrained` crashava importando peft. →
   **Risolto**: `pip uninstall peft` (la baseline non lo usa: LLaVA-13B è merged, SDXL salta peft).
   *Reversibile*: `pip install peft==0.17.1`.
3. **PYTHONHASHSEED (crash della VQA).** Bug latente in `utils/utils.py:merge_class_clusters`: un
   `del` su una chiave potenzialmente assente → `KeyError` intermittente (dipende dall'ordine di
   `list(set(...))`, cioè dall'hash seed casuale). → **Workaround senza toccare il codice**:
   `export PYTHONHASHSEED=0` negli script di run (verificato: seed 0–5 tutti OK). *Fix definitivo
   consigliato al team*: cambiare quei `del` in `dict.pop(key, None)`.

---

## 3. Com'è organizzato il server (baldo)

Cluster GPU del DISI (UniTN): `baldo.disi.unitn.it`, utente `andrea.giampietro`, alias SSH `baldo`.

| Aspetto | Dettaglio |
|---|---|
| **Repo** | `~/OpenInterBias` (branch `main`). |
| **venv** | `~/openbias` (Python 3.9, torch 2.5.1+cu121). **Usa `~/openbias/bin/python` direttamente** — `source openbias/bin/activate` dalla repo fallisce (il venv è in `$HOME`, non nella repo). |
| **SLURM** | account `foundation.models25`, QOS `gpuedu`. Partizioni (tutte `gpu:l40s:4`): `edu-short`=5 min, `edu-medium`=2 h, `edu-long`=24 h, `edu-thesis`=24 h. **I nodi di calcolo hanno internet.** |
| **sbatch obbligatori** | Ogni job DEVE avere `--nodes=1 --ntasks=1` (altrimenti la submission fallisce) + `--account=foundation.models25 --qos=gpuedu --gres=gpu:N`. |
| **Cache HF** | I pesi stanno nella cache **di default** `~/.cache/huggingface` (SDXL base+refiner, llava-v1.5-13b, bert/bart/roberta, SBERT all-mpnet-base-v2). ⚠️ `cluster/_common.env` imposta `HF_HOME=$repo/.hf_cache` che è **VUOTA** → ri-scaricherebbe tutto. **Lascia `HF_HOME` al default; non sourcare quella riga.** |
| **Pesi modelli** | Llama-2-7b-chat, LLaVA-1.5-13B (`utils/llava/weights/llava-v1.5-13b`), SDXL base+refiner. Memoria: LLaVA-13B fp16 ≈ 26 GB, SDXL fp16 ≈ 12 GB → una L40S (48 GB) ne regge uno alla volta. |

**Realtà di scheduling (importante per chi lancia i job).** Le L40S su baldo sono molto contese: un
job a 2 GPU resta "affamato" dietro a tanti job a 1 GPU. Trucco usato per la VQA: eseguirla **dentro
un'allocazione Jupyter già attiva** (2 L40S idle) via `srun --jobid=<jupyter> --overlap` → coda zero.

---

## 4. La pipeline e la struttura del codice

```
Dataset captions
   │  Stage 1  bias_proposals.py            (Llama-2)         → proposed_biases/<dataset>/3/*.json
   ▼
proposed_biases/  ── Stage 2  generate_images.py  (SD-XL DDP) → sd_generated_dataset/<…>/<cap_id>/0.jpg
   │
   ▼
sd_generated_dataset/ ── Stage 3  run_VQA.py  (LLaVA-1.5-13B) → results/VQA/.../{vqa_answers,data_counts}.json
   │
   ▼
results/  ── Stage 4  make_plots.py  (CPU)                    → context_free.png + context_aware.png
   │
   ▼
[Stage 5  intersectional/run_analysis.py  (CPU, DA SCRIVERE)] → joint_answers.json + intersectional_results.json
```

File chiave (tutti documentati in [`ARCHITECTURE_NOTE.md`](../ARCHITECTURE_NOTE.md) §2–3):
`utils/config.py` (configurazione centrale), `utils/datasets.py` (loader dataset),
`utils/VQA.py` (interfaccia LLaVA + SBERT), `utils/generative_models.py` (SDXL/SD), `utils/utils.py`
(filtri e clustering — è qui che vivevano i bug #1 e #3).

---

## 5. Stato del repo git (cosa trovate al clone)

- **Remote**: `origin → https://github.com/giampyhatesyou/OpenInterBias.git`.
- **Branch**: `main` (allineato a `origin/main`) e `chore/repo-prep` (lo scaffolding è arrivato da
  qui via PR #1). `main` è il branch di riferimento.
- **Working tree**: pulito, **tranne** un file non tracciato: `intersectional/STAGE5_PLAN.md`. È il
  piano dello Stage 5, lasciato volutamente *non committato* — va revisionato e committato dal team.
- **History**: nessuna firma "AI/Co-Authored-By" nei commit (regola di progetto: zero tracce AI nel
  git/GitHub; nessun commit/push automatico).

---

## 6. Cosa è stato eseguito — la baseline completa

Pipeline OpenBias upstream (stage 1→4), su **COCO**, generatore **SD-XL**, VQA **LLaVA-1.5-13B**:

| Stage | Cosa | Esito |
|---|---|---|
| 1 Bias proposal | Llama-2 (già fatto) | ✅ `proposed_biases/coco/3/coco_train.json` |
| 2 Generazione | SD-XL base+refiner, 40 step, 1024×1024, ~7 s/img su L40S | ✅ **6.384 immagini** |
| 3 VQA | LLaVA-1.5-13B su 2× L40S (DDP), ~1 h | ✅ **6.384 risposte** |
| 4 Plot | make_plots.py (CPU) | ✅ `context_free.png` + `context_aware.png` |

**Artefatti su baldo** (`~/OpenInterBias/results/VQA/coco/generated/sd-xl/llava-1.5-13b/`):
- `vqa_answers.json` (~1,3 MB) — predizione per-immagine, multi-attributo. È **l'input pronto per lo
  Stage 5**. Es.: `…/827/0.jpg → {person race: caucasian, person age: middle-aged}`.
- `data_counts.json` (~1,5 MB) — conteggi aggregati per classe.
- `context_free.png`, `context_aware.png`.

**Copie locali** (`code/`): `baseline_context_free.png`, `baseline_context_aware.png`, e un'immagine
d'esempio `sample_sdxl_coco_26.jpg`. Gli script usati sono `ob_vqa_smoke.sbatch` (smoke test VQA su
1 immagine) e `ob_vqa_plots.sbatch` (VQA + plot, edu-long, 2× L40S, con `PYTHONHASHSEED=0`).

> `generate_images.py` **auto-riprende** (salta le cartelle caption già popolate), quindi job
> killati/parziali non perdono nulla.

### ⚠️ Note di lettura dei risultati (importanti)

- **Con `n-images=1` la metrica context-aware è degenere** (la distribuzione per-prompt è un singolo
  punto). La metrica valida nella baseline attuale è la **context-free** (aggrega su migliaia di
  caption). Per un context-aware sensato bisogna rigenerare con `n-images > 1` (costo ×N).
- Il plot `context_free.png` è **visivamente affollato**: OpenBias propone migliaia di bias
  fine-grained → le etichette si sovrappongono. La sostanza è nei JSON; per figure leggibili si
  filtra ai top-bias o a una shortlist demografica.
- I messaggi `Entropy is nan or inf` nei log sono i casi attesi di bias con **una sola classe**
  superstite (entropia ÷log(1)=0) → gestiti/saltati da make_plots (chiude con `rc=0`).

---

## 7. Modifiche all'ambiente baldo (tutte reversibili)

- venv `~/openbias`: **rimosso `peft`** (reversibile: `pip install peft==0.17.1`). Nessun'altra
  libreria toccata.
- Cache HF: scaricati **SDXL refiner** e **SBERT all-mpnet-base-v2** (pesi mancanti).
- `PYTHONHASHSEED=0` è impostato **solo dentro gli script di run**, non è una modifica persistente.
- Script helper creati in `~` su baldo (`ob_*.sbatch`, `seed_probe.py`, ecc.): **niente di committato**.

---

## 8. Prossimi passi — lo Stage 5 (intersezionale)

Il piano completo e motivato è in **[`intersectional/STAGE5_PLAN.md`](../intersectional/STAGE5_PLAN.md)**.
In sintesi è **post-hoc, CPU-only, additivo**: legge `vqa_answers.json`, forma le coppie con stesso
`refer_to` (es. *person × person*), calcola **Joint Intensity** e **MI normalizzata** riusando la
stessa `entropy()` e le stesse esclusioni (`unknown/other/non-binary`) di `make_plots.py` → così i
numeri sono confrontabili con la baseline. Coppie candidate ben supportate su COCO: **age×gender
~50k**, gender×race ~11,5k, age×race ~10,9k, gender×occupation ~3,4k.

**Da implementare** (additivo): `intersectional/pairing.py`, `scoring.py`, `run_analysis.py`,
`make_plots.py` + test in `tests/intersectional/`. Lo `cluster/05_intersectional_analysis.sbatch` è
già un placeholder pronto da cablare.

**Decisioni da confermare col team prima di scrivere il codice** (dettaglio in
[`docs/SCHEMA_DECISION.md`](SCHEMA_DECISION.md)):

| # | Domanda | Default proposto |
|---|---|---|
| Q1 | Solo coppie o anche triple/N-way? | **Solo coppie** (v1); le triple sarebbero data-starved. |
| Q2 | Stesso `refer_to`? | **Sì** (coppie semanticamente intersezionali, non co-occorrenza). |
| Q3 | Filtrare su `present_in_prompt`? | **Sì**, entrambi gli attributi `false`. |
| Q4 | File di output separato? | **Sì**, `joint_answers.json` nuovo (non si tocca l'upstream). |
| Q5 | Escludere `unknown/other/non-binary`? | **Sì**, come `make_plots.py`. |
| — | **Normalizzazione MI** | `min(H(A),H(B))` (limita NMI a 1). **Da confermare.** |
| — | **MI context-aware** | ⚠️ Con `n-images=1` è ≈0 per costruzione → **proposta: MI solo context-free**, Joint Intensity anche context-aware. *Questa è la scelta metodologica più importante.* |

**Altri passi opzionali**:
- Figure pulite della baseline (filtro top-bias / shortlist demografica).
- Rigenerare con `n-images > 1` se serve un context-aware significativo.
- Fix definitivo del bug #3 in `utils/utils.py` (`dict.pop(key, None)`).

---

## 9. Da dove iniziare (per un nuovo membro)

1. Leggi **questo file**, poi [`ARCHITECTURE_NOTE.md`](../ARCHITECTURE_NOTE.md) (architettura) e
   [`intersectional/STAGE5_PLAN.md`](../intersectional/STAGE5_PLAN.md) (cosa costruiamo).
2. Per girare sul cluster: segui [`docs/SETUP_BALDO.md`](SETUP_BALDO.md) passo-passo. Configura i
   path via `.env` (vedi `.env.example`) e verifica con `python tools/check_paths.py` (exit 0 = ok).
3. Controlla gli invarianti upstream: `pytest tests/test_upstream_schema.py -v` (5 test devono passare).
4. Esplora le coppie candidate: `python tools/inspect_proposed_biases.py proposed_biases/coco/3/coco_train.json`.
5. **Review del piano Stage 5**: rispondete a Q1–Q5 e alla scelta su MI context-free; poi si scrive
   il codice (pairing+scoring+plot, ~1,5–2 giorni stimati) e si lancia (CPU, minuti) su `vqa_answers.json`.
6. (Igiene repo) valutare la rimozione dei log pesanti committati (`logs/stage2_*.out/.err`) e di
   `utils/config.py.bak.pre_smoke`.

---

## 10. Mappa dei documenti

| Documento | Contenuto |
|---|---|
| [`README.md`](../README.md) | README upstream OpenBias + header fork. |
| [`ARCHITECTURE_NOTE.md`](../ARCHITECTURE_NOTE.md) | Architettura pipeline + strategia di estensione + layout repo. |
| [`SETUP.md`](../SETUP.md) | Configurazione multi-macchina (variabili d'ambiente). |
| [`docs/SETUP_BALDO.md`](SETUP_BALDO.md) | Checklist operativa per girare la baseline su baldo (step 1–12). |
| [`docs/SCHEMA_DECISION.md`](SCHEMA_DECISION.md) | Domande aperte Q1–Q5 sullo schema intersezionale. |
| [`intersectional/SCHEMA.md`](../intersectional/SCHEMA.md) | Schema candidato/output + metriche. |
| [`intersectional/ARCHITECTURE_NOTE.md`](../intersectional/ARCHITECTURE_NOTE.md) | Come lo Stage 5 si innesta post-hoc. |
| [`intersectional/STAGE5_PLAN.md`](../intersectional/STAGE5_PLAN.md) | **Piano implementativo dettagliato dello Stage 5** (da committare). |
| [`cluster/README.md`](../cluster/README.md) | Uso dei template SLURM. |
| `code/RESOCONTO_OPENBIAS.md` | Resoconto della sessione di run (2026-05-31) — fuori dal repo. |
