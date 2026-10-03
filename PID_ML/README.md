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
- **Real-world accuracy**: fine-tuning on just 6 labelled real drawings raised connection mAP from 0.46 to 0.74 on unseen plans.

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

## Output at a glance

```json
{"sheet": "plant_150.png",
 "equipment": [{"id": 3, "tag": "LT-101", "role": "instrument", "bbox": [x1, y1, x2, y2],
                "connected_to": [7, 12], "upstream": [7], "downstream": [12]}],
 "connections": [{"source": 3, "target": 7, "style": "solid", "score": 0.82, "methods": ["line_trace", "relation_head"]}],
 "flow": [{"from": 7, "to": 3, "source": "arrow", "consistent": true, "votes": "2:0"}],
 "safety_functions": [{"tag": "LAHH-101", "linked_tag": "LT-101", "link_basis": "loop_number"}]}
```

## Results and recent improvements

Fused edge mAP and node AP@0.5 from the notebook (§9, §9b, §13). OPEN100 "after" numbers are on the 6 held-out plans that were never
used for training; the other 6 were used for the few-shot adaptation. "Before" is the first published version, scored on all 12 plans
with no real labels.

| Benchmark | Metric | Before | After |
|---|---|---|---|
| PID2Graph Synthetic | Node AP@0.5 | 97.9 % | 97.5 % |
| | Edge recall | 81.7 % | 91.5 % |
| | Edge AP (class-agnostic) | 67.0 % | 87.0 % |
| | Edge mAP | 47.7 % | 75.2 % |
| | Tanks / pumps found | 155/155, 89/89 | 155/155, 89/89 |
| OPEN100 (real plans) | Node AP@0.5 | 71.0 % | 91.1 % |
| | Edge mAP | 14.7 % | 74.3 % |
| Dataset-P&ID | Edge mAP | 63.4 % | 89.2 % |

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

### Tried and not adopted

Each is documented with its numbers in the notebook: Segment Anything (Meta SAM) box refinement, flip and multi-scale test-time
augmentation, ensemble voting across passes, OCR-based text suppression, self-similarity template matching, contour input channels for
the detector, contour junction areas around arrows, and several conflict-resolution rules for flow direction. Several helped synthetic
drawings while hurting real ones.

## Lessons learned

Classical computer vision (contours, skeletons, topology) and modern ML/LLMs work best together, and a handful of real labelled
examples beats clever rules. Every experiment, including the ideas that did not work, is documented with its numbers in the notebook
(§9c–§9f, §13).

## Links

- Repository: [github.com/fastflair/tutorials](https://github.com/fastflair/tutorials) (this folder: `PID_ML/`)
- Synthesis Engineering: [synthesisengineer.ai](http://synthesisengineer.ai)
- Reference paper: Stürmer, Graumann, Koch, *From Engineering Diagrams to Graphs: Digitizing P&IDs with Transformers*
  ([arXiv:2411.13929](https://arxiv.org/abs/2411.13929))
- Datasets: [Digitize-PID (Hugging Face)](https://huggingface.co/datasets/hamzas/digitize-pid-yolo),
  [Zenodo 8028570](https://zenodo.org/records/8028570), [PID2Graph](https://zenodo.org/records/14803338)
