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
