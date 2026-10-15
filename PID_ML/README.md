# P&ID Digitization: from drawings to a connected knowledge graph

*Part of the **Synthesis Engineering** skills series. More at [synthesisengineer.ai](http://synthesisengineer.ai).*

Piping & Instrumentation Diagrams (P&IDs) hold a plant's ground truth: every pump, valve, instrument and pipe, and how they connect.
In most companies that knowledge is locked in PDFs and scanned images. This tutorial notebook,
[`PID_Digitization_Transformer.ipynb`](PID_Digitization_Transformer.ipynb), turns P&IDs into a structured, connected graph.

## What it does

- **Equipment & symbols**: a Relationformer-style transformer detector finds valves, instruments, arrows and major equipment;
  tag names are read with OCR (optionally refined by a multimodal LLM).
- **Connectivity**: an OpenCV skeleton line tracer follows the pipes. It tells real junctions from lines that merely cross and handles
  dashed signal lines.
- **Flow direction**: a multimodal LLM reads each flow arrow (it found 99 % of the arrows on real plans). Direction is propagated
  along the traced lines, with a confidence flag where arrows disagree.
- **Engineering semantics**: ISA-5.1 tag decoding; alarms, trips and SIS functions are linked to the transmitters they act on.
- **Entity types from model internals**: the frozen vision encoder of the open-weight
  [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) re-scores every detection and assigns its entity type (valve,
  instrument, flow arrow, inlet/outlet, general, tank, pump), which also decides what counts as major equipment. This more than doubled
  entity-type accuracy on unseen real plans (mAP 0.31 → 0.70).
- **Real-world accuracy**: fine-tuning on just 6 labelled real drawings raised connection mAP from 0.46 to 0.74 on unseen plans
  (0.76 with the Qwen refiner).

## Why it matters for context graphs and context engines

LLMs are only as good as the context you give them, and a P&ID graph is ideal context:

- **Nodes**: equipment and instruments with tags, types, coordinates and safety functions (e.g. a high-high level alarm linked to its
  transmitter).
- **Edges**: pipe and signal connections with line type, confidence, and upstream → downstream flow.

Load that into a graph store or context engine, and an assistant can answer questions a document search can't:

- "What's downstream of valve V-1012 if it fails closed?"
- "Which instruments monitor the feed line to tank T-101?"
- "List every safety function tied to this pump's discharge."

The context engine retrieves only the relevant subgraph (a few hops around an asset) instead of entire drawings. That keeps prompts
small, grounded and traceable back to the source diagram, which helps MOC reviews, HAZOP preparation, maintenance planning and
digital twins.

## How to use it

1. **Install**: see §0 of the notebook. Install PyTorch with CUDA for your driver, then `pip install -r requirements.txt`.
   Trained checkpoints are stored with Git LFS (`git lfs install && git lfs pull`).
2. **Point the notebook at your drawings**: set `DRAWINGS_DIR` in §12, or export `PID_DRAWINGS_DIR`, to a folder of P&IDs
   (PNG/JPG/TIFF/PDF; subfolders included).
3. **Run it**: each drawing produces a JSON file and an overlay image, and all drawings are combined into a plant-level graph
   (`plant_graph.graphml` / `plant_graph.json`) with cross-sheet links, plus CSV exports of equipment, connections and safety
   functions.
4. **Feed your context engine**: load the graph into your graph database or context engine and retrieve subgraphs around the assets a
   question mentions as LLM context.
5. **Adapt to your drawing style**: label a few of your sheets (PID2Graph `.graphml` format) and run `adapt_to_real()` (§8c). That was
   the single biggest accuracy gain.

Optional: configure a multimodal LLM (Anthropic, OpenAI, Azure or Amazon Bedrock) in §12 for tag reading, connection refinement and
flow direction. Image calls are capped per drawing and cached.

Recommended: run §14 once. It downloads only the vision tower of Qwen3.8-Flash-Next (shard 1 of 131, 1.0 GB, needs `transformers`
≥ 5.17), trains a small refiner head (or loads `runs/pidformer/qwen_refiner.pt`) and defines `QWEN_REFINER`. Pass it as
`digitize(..., refiner=QWEN_REFINER)`, or re-run §12, which then uses it for every drawing. It needs ~1 GB of extra GPU memory.

## Output at a glance

```json
{"sheet": "plant_150.png",
 "equipment": [{"id": 3, "tag": "LT-101", "role": "instrument", "super_class": "instrumentation", "type_confidence": 0.98,
                "bbox": [x1, y1, x2, y2], "connected_to": [7, 12], "upstream": [7], "downstream": [12]}],
 "connections": [{"source": 3, "target": 7, "style": "solid", "score": 0.82, "methods": ["line_trace", "relation_head"]}],
 "flow": [{"from": 7, "to": 3, "source": "arrow", "consistent": true, "votes": "2:0"}],
 "safety_functions": [{"tag": "LAHH-101", "linked_tag": "LT-101", "link_basis": "loop_number"}]}
```

## Results and recent improvements

Fused edge mAP, node AP@0.5 and 7-class entity-type mAP from the notebook (§9, §9b, §13, §14). OPEN100 "after" and "+ Qwen refiner"
numbers are on the 6 held-out plans that were never used for training or for choosing settings; the other 6 were used for the few-shot
adaptation. "Before" is the first published version, scored on all 12 plans with no real labels.

| Benchmark | Metric | Before | After | + Qwen refiner (§14) |
|---|---|---|---|---|
| PID2Graph Synthetic | Node AP@0.5 | 97.9 % | 97.5 % | **98.9 %** |
| | Entity type (7-class mAP) | 22.9 % | 7.6 % | **97.5 %** |
| | Edge recall | 81.7 % | 91.5 % | **92.7 %** |
| | Edge AP (class-agnostic) | 67.0 % | 87.0 % | **89.6 %** |
| | Edge mAP | 47.7 % | 75.2 % | **76.8 %** |
| | Tanks / pumps found | 155/155, 89/89 | 155/155, 89/89 | 155/155, 89/89 |
| OPEN100 (real plans) | Node AP@0.5 | 71.0 % | 91.1 % | **93.9 %** |
| | Entity type (7-class mAP) | 37.3 % | 31.0 % | **70.4 %** |
| | Edge mAP | 14.7 % | 74.3 % | **76.4 %** |
| Dataset-P&ID | Edge mAP | 63.4 % | 89.2 % | – (symbols already ~100 %) |

What made the difference:

1. **Classical vision for connectivity.** A skeleton line tracer (OpenCV + scikit-image) follows the drawn pipes, classifies junctions by
   branch geometry (a straight 4-way crossing is not a connection), and bridges dashed lines and gaps at crossings. With correct symbols
   it reaches 78 % edge mAP on real plans (the earlier tracer: 14 %). The learned relation head now re-ranks the tracer's edges.
2. **Diagnosis before optimisation.** Most remaining errors came from symbol detection, not line following. A zoomed second detection
   pass for tiny symbols (flow arrows, small valves) and a rule that stops text next to a symbol being read as a dashed line each gave
   measurable gains.
3. **A few real examples beat clever rules.** Fine-tuning with 6 labelled real drawings (`adapt_to_real()`, §8c) raised real-plan edge
   mAP from 0.46 to 0.74 on unseen plans. Label-free geometry (contours, distance transforms, topology) helped too (0.47 → 0.51), but
   far less.
4. **Model internals as features.** A frozen vision-language encoder knows what drawn symbols look like. Its block-19 hidden states,
   read at a resolution where every symbol spans ~3 visual tokens, feed a 0.9 M-parameter head that re-scores each detection and names its
   entity type (§14). That type now also decides which large-equipment candidates are tanks, pumps or inlet/outlet connectors.
   Entity-type mAP on unseen real plans rose 0.31 → 0.70 and edge mAP 0.74 → 0.76.

### Flow direction: why and how accurate

The benchmark graphs (PID2Graph) are undirected, but engineering reasoning is not. Knowing a valve *connects to* a pump is useful;
knowing it is *upstream* of the pump is what you need to trace the impact of a failure, plan an isolation, or walk through a HAZOP node.
Flow direction was therefore added as a new capability (§8d, benchmark in §13):

| Step | Result |
|---|---|
| Flow arrows identified by a multimodal LLM on numbered crops (≤ 12 images per drawing, cached) | 99 % of arrows found, 93 % precision (geometry only: 52 % / 55 %) |
| Arrow direction | 94 % correct on a 72-arrow blind test set (geometry only: 49 %) |
| Direction propagated along the traced pipes, through valves and fittings | ~75 % of directed connections correct; 85–90 % where all arrows agree (`consistent=True`) |
| Side effect: confirmed arrows relabelled as `flow_arrow` | real-plan 7-class symbol mAP 0.30 → 0.44 |

The datasets have no direction labels, so direction accuracy is measured against blind visual labels (arrows labelled from crops without
seeing any method's answer) and a reference walked on the ground-truth pipe graph from those arrows. Connections that arrows claim in
opposite directions, mostly T-junctions where two branches feed one header, are flagged `consistent=False` rather than hidden.

### Model internals: the Qwen3.8-Flash-Next vision encoder (§14)

Prompting a vision-language model treats it as a black box. §14 instead reads its **internal visual representations** and feeds them to a
small custom model. That is where page-specific information lives: activations depend on the drawing, whereas the base weights are the same
for every drawing.

- **What fits on one 24 GB GPU.** The checkpoint has 180 B parameters (360 GB). Its MoE decoder does not fit, but the 449 M-parameter vision
  encoder sits entirely in shard 1 of 131 (1.0 GB). The notebook downloads only that shard: 0.9 GB in bf16, 70 ms per 1024² tile, so
  quantization was not needed.
- **Resolution first.** One merged visual token covers 32×32 px, so a small arrow on a full sheet would vanish. Every candidate is cropped
  and resampled so the symbol spans ~100 px (about 3×3 merged tokens), whatever its size on the sheet.
- **The refiner.** It combines the symbol-region features of vision block 19 (the encoder stops there) with detector cues, and is trained
  on all four 90° rotations of each symbol. It predicts *is this a real symbol?* and *which entity type?* (valve, instrument, flow arrow,
  inlet/outlet, general, tank, pump), and each item's `type_confidence` is exported for review. It plugs into
  `digitize(..., refiner=QWEN_REFINER)`.
- **Equipment arbitration.** The large-equipment finder used to turn every big detection into a tank: on real plans it relabelled half of
  the inlet/outlet connectors and invented ~50 tanks. The refined entity type now decides each candidate.
- **Protocol.** Settings were chosen on a 2-fold swap of the 12 real OPEN100 plans; the held-out plans were scored once, at the end.
  Training candidates are out-of-sample (each plan recorded with the detector that never saw it).

| Idea from model internals | Result | Adopted |
|---|---|---|
| Block-19 features + rotation augmentation → rescoring + entity type, which also arbitrates the equipment finder | real-plan entity mAP 0.31 → 0.70, node AP 0.91 → 0.94, edge mAP 0.74 → 0.76; synthetic entity mAP 0.08 → 0.98, edge mAP 0.75 → 0.77; +0.4 s per drawing | ✓ |
| First version: six layers (blocks 4–26) + merged tokens, no arbitration | real-plan entity mAP 0.61, node AP 0.93, edge mAP 0.77; synthetic entity mAP 0.95 | replaced |
| Equipment arbitration alone (first version's features, selection fold) | entity mAP 0.79 → 0.93, edge mAP 0.81 → 0.82 | ✓ |
| Layer study: each of the 27 blocks, top 3 / 4 / 6 blocks, all 27, learned scalar mix, ± merged tokens | block 19 alone is best (proxy 0.938 vs 0.927); the last block alone is worst of the deep ones (0.88, one outlier channel dominates its statistics) | block 19 ✓ |
| "Significant features": ANOVA top-256…4,096 channels, per-channel z-scoring | 0.90–0.92, never better than the full layer | ✗ |
| Other pooling (max, learned attention) and sites (MLP hidden activations, attention outputs, whole crop) | 0.90–0.93, not better | ✗ |
| Rotation test-time averaging; tighter or higher-resolution crops; bigger head / longer training | equal or lower | ✗ (rotation *training* augmentation ✓) |
| Topological data analysis: persistent homology of each symbol crop (GUDHI cubical complexes) | helps a head without image features (0.71 → 0.75), hurts with Qwen features (0.93 → 0.79–0.88) | ✗ |
| Edge verifier: Qwen features of a crop spanning both symbols, TDA connection persistence, cue recalibration | best on the selection fold (0.79 → 0.82), not on held-out plans (0.768 → 0.722–0.765) | ✗ |
| Same head without image features (control) | most of the real-plan gain disappears (proxy 7-class 0.60 vs 0.73) | – |
| Page-style embedding + FiLM conditioning | hurt the selection fold (7-class 0.93 → 0.88) | ✗ |
| LoRA adapters on vision blocks 20–26 + merger | best candidate-level scores, but in the full pipeline entity mAP 0.73 vs 0.79 on the selection fold | ✗ |
| Gradient embeddings (gradient w.r.t. Qwen features, BADGE) for uncertainty | same error-detection AUROC as softmax confidence (≈0.86–0.91) | ✗ (confidence exported instead) |
| Query-by-example (page prototypes matched along drawn lines) | recovered 0 of 38 missed symbols: most are mis-boxed, the rest are 14 px fragments | ✗ |
| Qwen feature maps fused into the detector's backbone (dense P&ID backbone) | edge mAP 0.76 vs 0.80 for the same fine-tune without Qwen; detection 4× slower | ✗ |
| Decoder-side state (prefix KV / Gated-DeltaNet cache of legend + title block, MoE routing, fused VL hidden states) | not runnable: the 360 GB decoder does not fit on one 24 GB GPU | not tested |

### Tried and not adopted

Each is documented with its numbers in the notebook: Segment Anything (Meta SAM) box refinement, flip and multi-scale test-time
augmentation, ensemble voting across passes, OCR-based text suppression, self-similarity template matching, contour input channels for
the detector, contour junction areas around arrows, and several conflict-resolution rules for flow direction. From §14: page-style FiLM
conditioning, LoRA adapters on the Qwen vision encoder, gradient embeddings, query-by-example, dense backbone fusion, multi-layer / top-k /
selected-channel features, other pooling and encoder sites, persistent-homology (TDA) features for symbols and connections, and an edge
verifier. Several helped synthetic drawings or candidate-level scores while not helping real drawings end to end.

## Lessons learned

Classical computer vision (contours, skeletons, topology) and modern ML/LLMs work best together, and a handful of real labelled
examples beats clever rules. The internals of a large vision-language model are useful features, but only after the resolution problem
is solved, and simple frozen features beat heavier adaptation when real labels are scarce. Score the *pipeline* per class, not only
the model: twice a better model looked worse end to end because a later rule overrode its decisions. Every experiment, including the ideas that did
not work, is documented with its numbers in the notebook (§9c–§9f, §13, §14).

## Links

- Repository: [github.com/fastflair/tutorials](https://github.com/fastflair/tutorials) (this folder: `PID_ML/`)
- Synthesis Engineering: [synthesisengineer.ai](http://synthesisengineer.ai)
- Reference paper: Stürmer, Graumann, Koch, *From Engineering Diagrams to Graphs: Digitizing P&IDs with Transformers*
  ([arXiv:2411.13929](https://arxiv.org/abs/2411.13929))
- Datasets: [Digitize-PID (Hugging Face)](https://huggingface.co/datasets/hamzas/digitize-pid-yolo),
  [Zenodo 8028570](https://zenodo.org/records/8028570), [PID2Graph](https://zenodo.org/records/14803338)
