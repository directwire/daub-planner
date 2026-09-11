# reference — the reference planner for daub

A complete, production-derived **image → plan** planner: it reads a
reference image and emits a daub *plan* — the JSON stroke catalog that
[daub](https://github.com/directwire/Daub) rasterizes into a finished
painting (PNG, layered `.kra`, layered `.psd`, reveal video).

Two pure-Python files, deterministic by construction (seeded PRNG, no
clock): the same reference image always plans the same strokes, on any
machine, byte for byte.

## How it plans

The planner measures everything from the reference itself — no neural
network, no style transfer. Three passes, all calibrated:

1. **Ink linework (X1 layers)** — black-top-hat finds the ink marks,
   thinning gives centrelines; each mark's width is the measured
   distance-transform width, its pressure is inverse-mapped from the
   measured ink depth through a calibrated alpha table, and each mark
   picks its brush from its own measured geometry (width, taper,
   tonal life).
2. **Flat colour bed (F1)** — the reference is median-cut to its own
   colour strata; every stratum is traced ring by ring as overlapping
   broad strokes, ring width anchored on the linework's measured
   brush scale.
3. **Banded tonal refill (L layers)** — a distance-transform scale
   field picks stroke width per location; an error-driven loop plans
   coarse-to-fine bands, simulates each pass with a calibrated ink
   deposit model, and refills wherever the simulation still deviates
   from the reference.

Calibration tables (`ink_calib.json`, `fork_ink_calib.json`) ship
alongside: they are the measured deposit response of the real brushes
(coverage width and ink alpha as functions of pressure and nominal
size), which is what lets the planner predict coverage instead of
guessing.

## Usage

```bash
pip install -r requirements.txt

# plan: reference image -> plan JSON (+ a preview PNG)
python stroke_engine.py plan reference.jpg my_plan.json

# optional: force one ink brush for every linework mark
python stroke_engine.py plan reference.jpg my_plan.json --pen "d) Ink-3 Gpen"

# render with daub
daub render my_plan.json --out painting.png --kra painting.kra
```

Point `DAUB_KRMCP_TOOLS` at this directory to light up the planner side
of daub's own tooling (`daub_paint.py`, the MCP server):

```bash
export DAUB_KRMCP_TOOLS=/path/to/daub-planner/reference
```

## Files

| File | What it is |
|---|---|
| `stroke_engine.py` | the planner: scale field, banded refill, layer ruler, `plan()` CLI |
| `stroke_trace.py` | tracing passes (loaded as a sibling module): ink centrelines + flat colour bed |
| `ink_calib.json` | measured deposit table, renderer-side brush response |
| `fork_ink_calib.json` | measured deposit table used by the ink tracing passes |
| `requirements.txt` | numpy · pillow · scipy · scikit-image |

`stroke_trace.py` prefers scikit-image's skeletonize and only falls
back to a slower numpy Zhang-Suen without it — install scikit-image.

## Embedding

```python
import stroke_engine as se

se.plan("reference.jpg", "my_plan.json")           # file to file

# or drive the pieces yourself
from PIL import Image
ref = Image.open("reference.jpg").convert("RGB")
strokes = se._auto_pen_pass(ref)                    # ink marks
se._finalize_layers(strokes)                        # band -> layer/preset/opacity
```

Anything the planner emits is a plain plan JSON — feed it to
`daub render` and it either renders or fails loud on a contract
violation. That round-trip is the whole integration test.
