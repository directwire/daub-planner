"""Stroke tracing: copy the reference's own strokes, measured.

It is the tracing half of the reference planner (see stroke_engine.py).
It replaces the whole error-repair lineage for the painting body: flat
colour masses are traced as contour rings of broad strokes, ink linework
is traced as skeleton centrelines - everything planned from the
reference itself, nothing from a canvas-vs-reference error map.

Two passes, both measured:

  flat ("F1", underneath)  the reference is median-cut to its own colour
        strata; paper-tint strata are left as paper; every other stratum
        is traced ring by ring (distance-transform contours), ring
        spacing overlapping so adjacent rings knit into a wash. Ring
        stroke width is ANCHORED ON THE REFERENCE: the median measured
        width of the linework pass - the original's own brush scale.
        Colour = the stratum's measured colour; pressure = the stratum's
        measured depth against paper, inverse-mapped through the
        calibrated alpha table.

  line ("X1", on top)      black-top-hat finds the gongbi ink lines;
        Zhang-Suen thinning gives centrelines; each stroke's width is
        the measured EDT width (continuous - no band table), its
        heaviest point's width is inverted through the calibrated
        width table into a nominal size, and per-point pressure is
        inverse-mapped from measured ink depth through the calibrated
        alpha table - so width AND darkness fade in and out along each
        stroke exactly as the original's ink does.

Calibration honesty: pressure and size come from ink_calib.json
(the measured response of the real Krita preset); the size KEY that
selects the response curve snaps to the calibrated grid because that is
what the renderer does - each stroke's nominal size itself stays a
continuous measurement. The only craft constants here are process ones
(contour overlap, minimum stratum area), not stroke design.

Usage (standalone, renders flat+line tracing on paper for inspection):
  python tools/stroke_trace.py <ref.jpg> <out.png> [--size 2000]
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
DAUB = os.environ.get("DAUB_BIN", "daub")  # or put daub on PATH
CAL = os.path.join(HERE, "ink_calib.json")
# Extraction measures against the calibrated table of the
# painting engine's own brush responses (fork_ink_calib.json);
# the two shipped tables agree byte-for-byte on the shared
# presets (only Sumi-e differs), and ink_calib.json stays the
# renderer-side table.
CAL_FORK = os.path.join(HERE, "fork_ink_calib.json")

# Brush pool (all real Krita presets, calibrated in the shipped
# tables). Layer classes get DIFFERENT brushes on purpose (a
# ground/wash layer and a fine ink-pass layer want different
# tools):
#   FLAT (F1, flat colour bed)   b) Basic-2 Opacity - opaque body wash
#   LINE (X1, ink linework)      d) Ink-* family, chosen per stroke by
#                                measured width (see line_preset()).
PRESET_FLAT = "b) Basic-2 Opacity"
LINE_FINE = "d) Ink-2 Fineliner"      # hairline ink, uniform
LINE_GONG = "d) Ink-3 Gpen"           # gongbi ink with taper
LINE_WIDE = "d) Ink-8 Sumi-e"         # broad soft ink masses
PAPER = np.array([232, 238, 208], dtype=np.float32)
SIZE_KEYS = [4, 8, 16, 32, 64, 120]

# process constants (how rings knit, what counts as a stratum) - NOT
# stroke design; stroke size/force/colour are always measured.
RING_OVERLAP = 0.6        # ring step = ring width * this (<1 = knit)
MIN_STRATUM_AREA = 600    # px, smaller colour strata stay untouched
MAX_FLAT_STROKES = 120000


def load_calib(path=CAL):
    with open(path) as fh:
        return json.load(fh)


def inv_pressure(cal, target_alpha, preset=PRESET_FLAT):
    """Inverse of the calibrated alpha table: what pressure deposits
    `target_alpha` ink (mirror of calib.rs lerp + alpha_at). Accepts a
    scalar or an array. Per-preset: each brush's response is its own."""
    t = cal["presets"].get(preset, cal["presets"][PRESET_FLAT])["alpha"]
    return np.interp(target_alpha, t["v"], t["p"])


def nominal_size(cal, w_px, p, preset=PRESET_FLAT):
    """Nominal stroke size whose calibrated coverage width at pressure
    `p` equals the measured `w_px`. Two fixed-point rounds because daub
    picks the NEAREST calibrated size key and that key's response curve
    feeds back into the required nominal size."""
    wmax = w_px / 0.92
    for _ in range(2):
        key = min(SIZE_KEYS, key=lambda k: abs(k - wmax))
        sc = cal["presets"].get(preset, cal["presets"][PRESET_FLAT])["sizes"][str(key)]
        wf = float(np.interp(p, sc["p"], sc["w"]))
        wmax = w_px / max(wf, 1e-6)
    return wmax


def _stroke(color, size, pts, layer, preset=PRESET_FLAT):
    """pts: (n, 3) array of x, y, pressure."""
    return {
        "layer": layer, "preset": preset,
        "size": round(float(size), 1), "opacity": 1.0,
        "color": "#%02x%02x%02x" % tuple(int(v) for v in color),
        "points": [[float(a), float(b), round(float(pv), 3)]
                   for a, b, pv in pts],
    }


def inv_width(cal, preset, size, w_frac):
    """Inverse width table: what pressure deposits width `w_frac * size`
    (mirror of calib.rs). For brushes whose alpha is pressure-flat (ink
    pens: alpha ~1.0 at every pressure), darkness cannot carry the
    original's stroke force - WIDTH must (pressure -> width only)."""
    key = min(SIZE_KEYS, key=lambda k: abs(k - size))
    sc = cal["presets"].get(preset, cal["presets"][PRESET_FLAT])["sizes"][str(key)]
    out = np.interp(np.clip(w_frac, 0.0, sc["w"][-1]), sc["w"], sc["p"])
    return out.item() if np.ndim(out) == 0 else out


def _alpha_flat(cal, preset):
    """True when the preset's calibrated alpha is pressure-independent
    (ink-pen class): force then has to ride the width profile instead."""
    t = cal["presets"].get(preset, cal["presets"][PRESET_FLAT])["alpha"]
    return max(t["v"]) - min(t["v"]) < 0.05


LINE_MARKER = "b) Basic-5 Size"       # uniform hard round (marker class)

# Expression solver: pick each mark's pen from what the mark NEEDS,
# not from shape similarity. A uniform-geometry mark with
# real intra-mark depth life needs the TONAL channel - Basic-2 Opacity's
# calibrated alpha (0.30 -> 1.0) expresses it; every ink pen here is
# alpha-pinned at 1.0, and Fineliner's width band (0.64 -> 0.77) clips
# pressure into sticks. Gates measured on dense sample linework:
TONAL_DD = 0.20     # depth p90-p10 >= 51 grey levels -> tonal life
TONAL_DMAX = 80.0   # >= one genuinely dark point along the mark
UNIFORM_WR = 1.9    # width p90/p10 <= this -> geometry "uniform"
P_FLOOR = 0.75      # relative tonal floor: p >= 0.75 * p_heavy per mark
                    # (width swing stays <= ~1.6x, tonality mark-relative)


def line_preset(wmax_px):
    """Classic width-only brush choice (kept for compatibility): the
    original's own line width drives the tool. See select_pen() for the
    refined character-aware rule."""
    if wmax_px < 5.0:
        return LINE_FINE
    if wmax_px <= 14.0:
        return LINE_GONG
    return LINE_WIDE


def select_pen(w, depth):
    """Refined per-mark brush choice: character x width, not a bare
    width table. A mark's own measured geometry decides its tool:

      - wide  (>= 16px)          -> Sumi-e (broad ink mass)
      - translucent wash (depth
        shallow vs its width)    -> Sumi-e (soft ink; hard pens cannot
                                     make a shallow wide mark)
      - pen dynamics (width
        tapers along the mark)   -> Gpen (the taper carries the force),
                                     Fineliner only below ~3.5px where
                                     taper is noise
      - tonal life on uniform
        geometry                 -> Basic-2 Opacity (the expression
                                     solver: depth dynamics need the
                                     alpha channel, which no ink pen
                                     here has)
      - uniform stroke (marker/
        hard round even width)   -> Fineliner < 4px (hairline),
                                     Basic-5 4-16px (even hard round at
                                     the measured size)
    w: measured width profile along the mark (px), depth: ridge depth.
    """
    wmax = float(w.max())
    if wmax >= 16.0:
        return LINE_WIDE
    nw = len(w)
    # pen dynamics live at the ENDS (attack/release), so measure taper as
    # how much the mark thins toward its tips vs its peak - a middle-60%
    # std/mean misses exactly the taper that distinguishes pen from marker
    if nw >= 6:
        k = max(1, int(0.2 * nw))
        ends = np.concatenate([w[:k], w[nw - k:]])
    else:
        ends = w
    taper = 1.0 - float(ends.mean()) / max(wmax, 1e-6)
    dmax = float(depth.max()) if depth.size else 0.0
    # translucent wash: wide shallow mark a hard pen cannot reproduce
    if dmax < 0.10 * wmax and wmax >= 8.0:
        return LINE_WIDE
    if taper > 0.14:                    # real pen dynamics (thins at tips)
        return LINE_FINE if wmax < 3.5 else LINE_GONG
    # expression solver: the mark's geometry is uniform but its ink
    # deepens/shallows along the way - only an alpha-graded pen can say
    # that (all ink pens here are alpha-flat; see TONAL_* above)
    if depth.size >= 5:
        dlo, dhi = np.percentile(depth, 10), np.percentile(depth, 90)
    elif depth.size:
        dlo, dhi = float(depth.min()), float(depth.max())
    else:
        dlo, dhi = 0.0, 0.0
    wlo, whi = np.percentile(w, 10), np.percentile(w, 90)
    if (float(whi) / max(float(wlo), 0.5) <= UNIFORM_WR
            and (float(dhi) - float(dlo)) / 255.0 >= TONAL_DD
            and dmax >= TONAL_DMAX):
        return PRESET_FLAT              # Basic-2 Opacity: alpha carries
    if wmax < 4.0:                      # uniform hairline
        return LINE_FINE
    return LINE_MARKER                  # uniform hard round


def chain_pixels(mask):
    """8-connected greedy walks over a 1px path mask -> chains of (x, y).
    Deterministic: seeds in raster order; next pixel = the unvisited
    neighbour closest to straight ahead, so rings close naturally."""
    ys, xs = np.nonzero(mask)
    unvis = set(zip(xs.tolist(), ys.tolist()))
    nb8 = [(1, 0), (1, 1), (0, 1), (-1, 1),
           (-1, 0), (-1, -1), (0, -1), (1, -1)]
    chains = []
    for seed in sorted(unvis):
        if seed not in unvis:
            continue
        unvis.discard(seed)
        chain = [seed]
        prev = None
        cur = seed
        while True:
            px, py = cur
            best = None
            best_dot = -10.0
            for dx, dy in nb8:
                q = (px + dx, py + dy)
                if q not in unvis:
                    continue
                dot = 1.0 if prev is None else float(dx * prev[0] + dy * prev[1])
                if dot > best_dot:
                    best, best_dot = q, dot
            if best is None:
                break
            prev = (best[0] - px, best[1] - py)
            unvis.discard(best)
            chain.append(best)
            cur = best
        if len(chain) >= 3:
            chains.append(chain)
    return chains


def _decimate(path, step=2.5):
    """One point per ~`step` px of arc length (keeps the path's shape
    without flooding daub with sub-pixel dabs)."""
    path = np.asarray(path, dtype=np.float64)
    if len(path) <= 2:
        return path
    arc = np.hypot(*np.diff(path, axis=0).T).cumsum()
    bins = np.floor(np.r_[0.0, arc] / step).astype(int)
    keep = np.r_[True, bins[1:] != bins[:-1]]
    out = path[keep]
    if len(out) < 3:
        out = path[:3]
    return out


def luminance(arr):
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.226 * arr[..., 2]


def ridge_mask(lum, thresh):
    """Black top-hat: luminance minus its morphological closing. High
    where the painting carries ink darker than its surround."""
    closed = ndimage.grey_closing(lum, size=(15, 15))
    depth = closed - lum
    return depth, depth > thresh


def zhang_suen(mask):
    """Skeleton to real 1px centrelines.

    skimage.morphology.skeletonize is a C-accelerated one-shot and the
    primary path (the hand-rolled vectorised Zhang-Suen below is
    pathological on dense masks: measured 400+ CPU-seconds on a 33%
    ridge-density image that skeletonize does in ~2s). The numpy loop is
    kept as a fallback when scikit-image is unavailable.
    """
    try:
        from skimage.morphology import skeletonize
        skel = skeletonize(mask)
        dist = ndimage.distance_transform_edt(mask)
        lab, n = ndimage.label(skel, structure=np.ones((3, 3)))
        return skel, dist, lab, n
    except ImportError:
        pass

    I = np.pad(mask.astype(np.uint8), 1)
    nb = [(-1, 0), (-1, 1), (0, 1), (1, 1),
          (1, 0), (1, -1), (0, -1), (-1, -1)]
    H, W = mask.shape
    while True:
        changed = False
        for step in (0, 1):
            c = I[1:-1, 1:-1]
            P = [I[1 + dy:1 + dy + H, 1 + dx:1 + dx + W] for dy, dx in nb]
            B = sum(P[k] for k in range(8))
            S = np.stack(P + [P[0]])
            A = ((S[:-1] == 0) & (S[1:] == 1)).sum(axis=0)
            cond = (c == 1) & (B >= 2) & (B <= 6) & (A == 1)
            if step == 0:
                cond &= (P[0] * P[2] * P[4] == 0) & (P[2] * P[4] * P[6] == 0)
            else:
                cond &= (P[0] * P[2] * P[6] == 0) & (P[0] * P[4] * P[6] == 0)
            if cond.any():
                I[1:-1, 1:-1][cond] = 0
                changed = True
        if not changed:
            break
    skel = I[1:-1, 1:-1].astype(bool)
    dist = ndimage.distance_transform_edt(mask)
    lab, n = ndimage.label(skel, structure=np.ones((3, 3)))
    return skel, dist, lab, n


def order_component(pts, tree):
    """Greedy nearest-neighbour chains through one skeleton component.
    On a gap, restart from the farthest remaining pixel.

    The next skeleton pixel is 8-adjacent to the current one, so a
    k=16 neighbourhood query is exhaustive here. The historical
    k=len(rest)+1 made a single large connected component quadratic
    (measured multi-hour stalls on dense linework); k=16 is exact for
    the <=2.5px acceptance radius and keeps chaining O(m log m).

    Seed-pick equivalence (T9): the old loop re-sorted the whole
    remaining set per path and argmaxed the copy - i.e. "alive point
    with the largest second-neighbour distance, ties to the lowest
    index". A single stable descending argsort scanned with a monotone
    cursor returns exactly that point every round (alive only shrinks,
    so the cursor never moves back), without the per-path O(R log R)
    sort + list->array rebuild. Byte-identical output verified against
    the previous implementation on fine + coarse configurations.
    """
    n_pts = len(pts)
    d, _ = tree.query(pts, k=2)
    alive = np.ones(n_pts, dtype=bool)
    alive_n = n_pts
    # Stable on the negated key: equal distances keep ascending original
    # index order, so the first alive hit in this order is the argmax
    # point the sorted copy would have picked - bit-for-bit.
    by_far = np.argsort(-d[:, 1], kind="stable")
    pos = 0
    k16 = min(16, n_pts)
    paths = []
    while alive_n:
        while not alive[by_far[pos]]:
            pos += 1
        seed = int(by_far[pos])
        pos += 1
        alive[seed] = False
        alive_n -= 1
        path = [seed]
        cur = seed
        while alive_n:
            # k must NOT shrink with remaining: at an endpoint the two
            # nearest can be self + an already-used neighbour, hiding the
            # one usable point (measured: x=299 stranded at k=2 while its
            # only candidate ranked 3rd). Fixed 16 covers self + up to 8
            # used 8-neighbours + the next unused one.
            dd, ii = tree.query(pts[cur], k=k16)
            nxt = None
            for dist_v, idx in zip(dd[1:], ii[1:]):
                if alive[idx] and dist_v <= 2.5:
                    nxt = int(idx)
                    break
            if nxt is None:
                break
            path.append(nxt)
            alive[nxt] = False
            alive_n -= 1
            cur = nxt
        if len(path) >= 3:
            paths.append(pts[path])
    return paths


def extract_line_strokes(ref, cal, depth_thresh=10.0, min_px=12,
                         max_strokes=6000, layer="X1", pen=None,
                         pen_auto=True, ink_density=0.0,
                         exclude_density=0.0, min_depth=0.0):
    """Ink linework -> plan strokes with measured width / depth /
    colour. Returns (strokes, widths_px) - the width list is the flat
    pass's scale anchor (the original's own brush size).

    min_depth: drop marks whose median ridge depth along the path falls
              below this many grey levels. The density percentile is
              GLOBAL, so on photo references the mask swallows smooth
              skin (skin is darker than the wall) and skeleton
              fragments of those plateaus become opaque speckles.
              Measured separation: photo junk marks sit at median
              depth <=42 while dense linework marks are >=73.
              Illustrated texture spreads lower (5-50, p50 ~29),
              and the floor still verified safe by render A/B - the
              dropped marks were speckle noise, real texture lives in
              the band layers.

    pen:      explicit preset name for EVERY mark (CLI override), or None
              for automatic per-mark choice.
    pen_auto: True  -> select_pen() (character x width, refined);
              False -> line_preset() (classic width-only table).
    ink_density: >0 -> adaptive threshold: keep only the darkest
              `ink_density`-fraction of the depth map as ink (masks the
              penumbra/tonal gradients a fixed depth threshold floods;
              measured: 44.9% canvas @ thresh 7 -> ~13% @ density .13).
              <=0 -> use the explicit depth_thresh (legacy behaviour).
    exclude_density: >0 -> ALSO drop the darkest `exclude_density`
              fraction (the band the fine pass already took). Lets a
              coarser backing pass emit ONLY the marks between the fine
              level and its own, so it pads the linework instead of
              re-drawing it (double-inking otherwise).

    Size anchor is the chain's p70 width (not the peak: junctions and
    penumbra inflate the max and every line paints chunky). Automatic
    mode layers the marks by brush class: Sumi-e -> X1W (bottom),
    Gpen/Basic-5 -> X1M, Fineliner -> X1F (top, fine over broad)."""
    t0 = time.time()
    rgb = np.asarray(ref, dtype=np.float32)
    lum = luminance(rgb)
    depth, _ = ridge_mask(lum, depth_thresh if ink_density <= 0 else 0.0)
    if ink_density > 0:
        thr = float(np.percentile(depth, 100.0 * (1.0 - ink_density)))
        mask = depth > thr
        if exclude_density > 0:
            # exclude the deepest band (already painted by the fine pass)
            thr_x = float(np.percentile(depth,
                                        100.0 * (1.0 - exclude_density)))
            mask = np.logical_and(mask, depth <= thr_x)
    else:
        mask = depth > depth_thresh
        print("trace/line: FIXED depth_thresh=%.0f (adaptive path off) - "
              "on photos this historically flooded the mask; pass "
              "ink_density>0 unless you mean it" % depth_thresh, flush=True)
    print("trace/line: thinning ridge mask (%.1f%% of canvas)..."
          % (mask.mean() * 100), flush=True)
    skel, dist, lab, n = zhang_suen(mask)

    depth_s = np.where(skel, depth, 0.0)
    width_s = np.where(skel, dist * 2.0, 0.0)
    ys, xs = np.nonzero(skel)
    comp = lab[ys, xs]
    order = np.argsort(comp)
    ys, xs, comp = ys[order], xs[order], comp[order]
    bounds = np.searchsorted(comp, np.arange(1, n + 2))

    from scipy.ndimage import median_filter
    strokes = []
    widths = []
    warned = set()
    for ci in range(n):
        lo, hi = bounds[ci], bounds[ci + 1]
        if hi - lo < min_px:
            continue
        pts = np.stack([xs[lo:hi], ys[lo:hi]], axis=1).astype(np.float64)
        tree = cKDTree(pts)
        for path in order_component(pts, tree):
            if len(path) < max(min_px, 8):
                continue
            path2 = _decimate(path)
            px = np.clip(path2[:, 0], 0, ref.width - 1).astype(int)
            py = np.clip(path2[:, 1], 0, ref.height - 1).astype(int)
            w = width_s[py, px]
            d = depth_s[py, px]
            if min_depth > 0.0 and float(np.median(d)) < min_depth:
                # a mark whose own ridge depth is this weak is invisible
                # in the reference - on photos these are skeleton
                # fragments of smooth-skin plateau the global percentile
                # mask swallowed, later amplified into opaque speckles
                continue
            if len(w) >= 5:
                ws = median_filter(w, 5)
            else:
                ws = w
            w_anchor = float(np.percentile(ws, 70))
            if w_anchor < 1.2:
                continue
            # brush from the original's own measured line character
            if pen is not None:
                preset = pen
            elif pen_auto:
                preset = select_pen(ws, d)
            else:
                preset = line_preset(w_anchor)
            if preset not in cal["presets"]:
                if preset not in warned:
                    print("trace/line: %r not calibrated yet -> "
                          "Basic-2 fallback" % preset, flush=True)
                    warned.add(preset)
                preset = PRESET_FLAT
            # size anchored at the mark's typical width (p70 of the
            # smoothed profile), NOT the peak - junctions and penumbra
            # inflate wmax and paint every line chunky otherwise
            if _alpha_flat(cal, preset):
                p_heavy = 1.0
                size = nominal_size(cal, w_anchor, p_heavy, preset)
                p = np.clip(inv_width(cal, preset, size,
                                      w / max(size, 1e-6)),
                            0.0, 1.0)
            else:
                p_heavy = inv_pressure(
                    cal, min(float(d.max()) / 255.0, 1.0), preset)
                size = nominal_size(cal, w_anchor, p_heavy, preset)
                p = inv_pressure(cal,
                                 np.clip(d / 255.0, 0.0, 1.0),
                                 preset)
                if preset == PRESET_FLAT:
                    # relative tonal floor (P_FLOOR): pressure rides no
                    # lower than 75% of the mark's own peak - width swing
                    # stays bounded while tonality stays mark-relative
                    # (an absolute floor dragged faint marks darker than
                    # the reference; measured A/B). Scoped
                    # to the solver's tonal channel: other alpha-branch
                    # presets (e.g. Basic-5's 0.07 range) keep the
                    # pressures the width tables were anchored with.
                    p = np.maximum(p, P_FLOOR * p_heavy)
            col = rgb[py, px].mean(axis=0) * 0.92
            if pen is None and pen_auto and ink_density > 0:
                # automatic adaptive mode layers by brush class: broad
                # under, fine over (X1W < X1M < X1F in the layer stack).
                # Legacy depth_thresh callers keep the single `layer`.
                # The expression solver's Basic-2 keeps layer = width
                # class (it can surface at either size).
                if preset == PRESET_FLAT:
                    lname = "X1F" if w_anchor < 4.0 else "X1M"
                else:
                    lname = {"d) Ink-8 Sumi-e": "X1W",
                             "d) Ink-3 Gpen": "X1M",
                             "b) Basic-5 Size": "X1M",
                             "d) Ink-2 Fineliner": "X1F"}.get(preset, layer)
            else:
                lname = layer
            strokes.append(_stroke(col, min(size, 64.0),
                                   np.stack([px, py, p], axis=1).astype(float),
                                   lname, preset))
            strokes[-1]["_vw"] = float(w_anchor)  # visual width (sorting)
            widths.append(w_anchor)
    # painter order: broad bands first, fine lines last (stack order);
    # sort by VISUAL width (_vw = measured px), not nominal size
    # (Basic-5's nominal is inflated by its calibration curve)
    rank = {"X1W": 0, "X1M": 1, "X1F": 2}
    strokes.sort(key=lambda s: (rank.get(s["layer"], 1),
                                -s.get("_vw", s["size"])))
    widths = [w for _, w in sorted(
        zip(strokes, widths), key=lambda sw: (rank.get(sw[0]["layer"], 1),
                                              -sw[0].get("_vw",
                                                         sw[0]["size"])))]
    if len(strokes) > max_strokes:
        # over budget: shed the broadest marks first, keep the fine ones
        strokes = strokes[len(strokes) - max_strokes:]
        widths = widths[len(widths) - max_strokes:]
        print("trace/line: capped to %d finest strokes" % len(strokes),
              flush=True)
    if widths:
        warr = np.array(widths)
        by_brush = {}
        for s in strokes:
            by_brush[s["preset"]] = by_brush.get(s["preset"], 0) + 1
        print("trace/line: %d strokes in %.1fs; width px p10 %.1f / "
              "median %.1f / p90 %.1f; brushes %s"
              % (len(strokes), time.time() - t0, np.percentile(warr, 10),
                 np.median(warr), np.percentile(warr, 90), by_brush),
              flush=True)
    else:
        print("trace/line: no ink ridges found", flush=True)
    return strokes, widths


def extract_flat_strokes(ref, cal, anchor_width, colors=24, layer="F1",
                         preset=PRESET_FLAT, paper=None):
    """Colour strata -> overlapping contour rings of broad strokes.

    Ring width is MEASURED, not chosen: each stratum's equivalent layer
    thickness (area / perimeter - how thick the original's colour band
    actually is) sets the ring width and spacing, clamped below by the
    linework's measured median (a wash is never finer than the
    original's own brush) and above by the calibrated size grid.
    Thick bands take wide brushes - the painter's own economy.

    `preset` is the flat-class brush (flat colour bed, distinct from
    the ink line class by design); it must already be in the calibration.

    `paper` overrides the leave-untouched tint (default PAPER): callers
    whose canvas ground is the reference's own surround pass that colour
    so "leave the paper" becomes "leave the ground" - a photo whose
    background is deep must not have near-paper strata skipped into
    showing that deep ground."""
    t0 = time.time()
    rgb = np.asarray(ref, dtype=np.float32)
    # smooth before quantising: gradient regions (skin, washes) otherwise
    # classify into salt-and-pepper strata and the contour rings shatter
    smooth = ref.filter(ImageFilter.GaussianBlur(2))
    qi = smooth.quantize(colors=colors, method=Image.MEDIANCUT)
    labels = np.asarray(qi)                      # (H, W) palette indices
    pal = np.array(qi.getpalette()[: colors * 3],
                   dtype=np.float32).reshape(-1, 3)
    # merge palette entries closer than the pipeline's colour gate (45,
    # the same gate the groundfill planner and paper test use): fine
    # quantiser splits inside a wash are noise, and painting each split
    # separately shatters the skin/gradient strata into speckle
    reps = []                                    # [rep index, [members]]
    for ci in range(colors):
        for r in reps:
            if float(np.abs(pal[ci] - pal[r[0]]).sum()) < 45.0:
                r[1].append(ci)
                break
        else:
            reps.append([ci, [ci]])
    remap = np.zeros(colors, dtype=int)
    for new, (_, members) in enumerate(reps):
        remap[members] = new
    labels = remap[labels]
    pal = np.array([pal[r[0]] for r in reps], dtype=np.float32)
    ground = PAPER if paper is None else np.asarray(paper, dtype=np.float32)
    depth = np.abs(rgb - ground).sum(axis=2)
    dmax = max(float(depth.max()), 1.0)

    strokes = []
    strata = []
    for ci in range(len(pal)):
        m = labels == ci
        area = int(m.sum())
        if area < MIN_STRATUM_AREA:
            continue
        if float(np.abs(pal[ci] - ground).sum()) < 45.0:
            continue  # paper stays paper - a copyist doesn't paint the void
        strata.append((area, ci))
    strata.sort(reverse=True)  # broad masses first, small details last

    n_rings = 0
    w_layers = []
    k3 = np.ones((3, 3))
    for area, ci in strata:
        m = ndimage.binary_opening(labels == ci, np.ones((5, 5)))
        if m.sum() < MIN_STRATUM_AREA:
            continue  # opening dissolved this stratum: it was speckle
        dist = ndimage.distance_transform_edt(m)
        # measured layer thickness: area over boundary length
        per = float(np.abs(ndimage.binary_dilation(m, np.ones((3, 3)))
                           .astype(int) - m.astype(int)).sum())
        w_layer = min(max(area / max(per, 1.0), anchor_width), 64.0)
        w_layers.append(w_layer)
        step = max(w_layer * RING_OVERLAP, 3.0)
        lo = 0.0
        while True:
            hi = lo + step
            band = m & (dist > lo) & (dist <= hi)
            if not band.any():
                break
            # the ring is the band's outer boundary - the path a brush
            # rides; neighbouring rings step < ring width, so they knit
            ring = band & ~ndimage.binary_erosion(band, k3)
            col = rgb[band].mean(axis=0)  # the wash's own colour, as-is
            # flat fills are BODY COLOUR: laid opaque, at full pressure -
            # a wash's light and shade lives in its stratum colours (each
            # band carries its measured tone), NOT in deposit alpha.
            # Mapping tone to alpha would let the paper bleed through
            # and grey out every saturated area.
            size = nominal_size(cal, w_layer, 1.0, preset)
            n_rings += 1
            for chain in chain_pixels(ring):
                if len(chain) < 8:
                    continue  # speckle, not a brush path
                pts = _decimate(chain)
                if len(pts) < 3:
                    continue
                strokes.append(_stroke(col, min(size, 120.0),
                                       np.concatenate(
                                           [pts, np.ones((len(pts), 1))],
                                           axis=1), layer, preset))
            lo = hi
            if lo > max(dist.max(), 1.0):
                break
    print("trace/flat: %d strata -> %d rings, %d ring strokes in %.1fs "
          "(layer width p50 %.1fpx)"
          % (len(strata), n_rings, len(strokes), time.time() - t0,
             float(np.median(w_layers)) if w_layers else 0), flush=True)
    if len(strokes) > MAX_FLAT_STROKES:
        strokes = strokes[:MAX_FLAT_STROKES]
        print("trace/flat: capped to %d strokes" % len(strokes), flush=True)
    return strokes


def trace_reference(ref, colors=24, cal=None):
    """Both passes, in paint order (flat beneath, linework on top).

    The line pass runs at the pipeline-standard adaptive density
    (0.13, same as stroke_engine._auto_pen_pass's fine pass). The
    all-defaults call falls into fixed depth_thresh=10 territory,
    which on photos floods 86% of the canvas into the ink mask
    and rides the 6000-stroke cap."""
    if cal is None:
        cal = load_calib(CAL_FORK)
    line, widths = extract_line_strokes(ref, cal, ink_density=0.13)
    anchor = float(np.median(widths)) if widths else 10.0
    flat = extract_flat_strokes(ref, cal, anchor, colors=colors)
    return flat + line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("out_png")
    ap.add_argument("--size", type=int, default=2000)
    ap.add_argument("--colors", type=int, default=24)
    ap.add_argument("--cal", default=None,
                    help="calibration JSON (default fork_ink_calib.json, "
                         "the painting-side table; ink_calib.json is the "
                         "renderer-side table)")
    args = ap.parse_args()
    t0 = time.time()

    ref = Image.open(args.ref).convert("RGB")
    if max(ref.size) != args.size:
        s = args.size / max(ref.size)
        ref = ref.resize((int(ref.width * s + 0.5),
                          int(ref.height * s + 0.5)), Image.LANCZOS)
    W, H = ref.size

    strokes = trace_reference(ref, colors=args.colors,
                              cal=load_calib(args.cal or CAL_FORK))
    plan = {"kind": "plan", "source": "stroke_trace",
            "ref": os.path.abspath(args.ref), "canvas": [W, H],
            "strokes": strokes}
    plan_path = args.out_png.replace(".png", "_trace.json")
    with open(plan_path, "w") as fh:
        json.dump(plan, fh)

    # cal/tips resolve from daub's own vendored defaults (the renderer's
    # tables ship inside the daub repo); nothing else to configure
    r = subprocess.run([DAUB, "render", plan_path, "--out", args.out_png],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("daub failed:\n%s\n%s" % (r.stdout, r.stderr))
    print(r.stdout.strip().splitlines()[-1], flush=True)
    print("trace render done in %.1fs -> %s"
          % (time.time() - t0, args.out_png), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
