# daub-planner

Planner implementations for [daub](https://github.com/directwire/Daub).

[daub](https://github.com/directwire/Daub) renders a *plan* — a JSON catalog
of pressure-tagged brush strokes — into a finished painting: PNG, layered
`.kra`, layered `.psd`, reveal video. daub ships the renderer; this
repository is the home for the other half of the pipeline: planners that
turn an image into a plan.

Anything that emits a valid plan plugs into daub as-is.

## The plan contract

A plan is a JSON document:

```json
{
  "canvas": [2000, 2000],
  "strokes": [
    { "layer": "X1", "preset": "b) Basic-2 Opacity", "size": 12,
      "opacity": 0.9, "color": "#2244cc",
      "points": [[120, 90, 0.4], [480, 300, 0.8], [840, 90, 0.3]] }
  ],
  "count": 1
}
```

- `canvas` — `[width, height]` in pixels.
- `strokes` — the painting, stroke by stroke. `layer` names the layer
  (first-appearance order builds the stack); `preset` must be one of daub's
  calibrated brushes (`daub presets` lists them); `size`, `opacity` and
  `color` set each stroke; `points` are `[x, y, pressure]` triples.
- `count` — number of strokes.

Full semantics, export details and downstream integration notes:
[daub · docs/DOWNSTREAM_GUIDE.md](https://github.com/directwire/Daub/blob/main/docs/DOWNSTREAM_GUIDE.md).

## Contributing

Built a planner — or an adapter that turns your image pipeline's output
into plans? Open a pull request. One directory per implementation, with a
README covering inputs, outputs and dependencies. daub itself is the
validator: `daub render your-plan.json --out check.png` fails loud on any
contract violation.

## License

Same terms as [daub](https://github.com/directwire/Daub): non-commercial
use freely granted; commercial use requires the copyright holder's prior
written consent. See [LICENSE](LICENSE).
