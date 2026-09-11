"""stroke_engine - calibrated stroke planner for the daub renderer.

Plans a *stroke catalog* (a daub plan JSON) that reproduces a reference
image with a small set of calibrated brushes. Everything is computed
FROM the reference image, per stroke, per point:

  width(x,y)   : distance transform of the edge map -> local structure
                 scale. Flat background = wide strokes, near a contour
                 line = thin. NO fixed width tiers: each stroke carries
                 its own width profile w[i] along its path (smoothed,
                 end-tapered).
  pressure[i]  : the image-derived width profile normalized to the
                 stroke's max width, times an attack/release envelope.
                 Pressure GRADATES WITHIN one stroke; the calibrated
                 brush pipeline renders it:
                   - "b) Basic-5 Size"    -> pressure = width
                                             (thick-thin form)
                   - "b) Basic-2 Opacity" -> pressure = ink weight
                                             (wash)
  color        : blurred-reference color under the stroke center
  opacity      : per stroke, from local contrast (firm where crisp)

On top of the banded refill pass, an ink-linework pass and a flat
colour bed are traced from the reference itself (stroke_trace.py,
loaded as a sibling module).

The plan JSON is the single source of truth: every stroke is
{layer, preset, size, opacity, color, points:[[x,y,pressure]...]}.
Deterministic: seeded PRNG, no clock - the same reference always plans
the same strokes.

Usage:
  python stroke_engine.py plan REFERENCE_IMAGE OUT_PLAN.json
  python stroke_engine.py plan REFERENCE_IMAGE OUT_PLAN.json --pen "d) Ink-3 Gpen"

Render the plan with daub:
  daub render OUT_PLAN.json --out painting.png
"""

import json
import math
import os
import random
import sys
import time

import numpy as np

from PIL import Image, ImageDraw, ImageFilter

import numpy as np

from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))


# Width bands, coarse first. A seed's band comes from the image scale field:
#   band k takes seeds whose image-derived width falls in [lo, hi).
# (preset, opacity) chosen per band: washes get opacity dynamics,
# structure/detail get size dynamics (true thick-thin). roi restricts where
# the band may fire (None = whole canvas).
BANDS = [
    # (w_lo, w_hi, thresh, preset,             base_opacity, roi, mag_gate)
    # Wash bands run near-opaque: Basic-2 Opacity multiplies dab opacity by
    # pressure (0.45..1.0), so nominal 0.5 painted like a watercolor glaze
    # and the whole canvas went pastel (v4.0 lesson).
    # mag_gate is ONLY for the finest face pass: a low gate on L7 locked flat
    # but COLORED regions (petal bodies, cheeks) out of the paint entirely -
    # they sit 7-22px from edges, land in L7, and "no gradient" read as "no
    # ink needed" (v4.1 flower was beige canvas with red outline lines).
    (52, 120, 48, "b) Basic-2 Opacity", 0.85, None, 0),
    (26, 52, 40, "b) Basic-2 Opacity", 0.90, None, 0),
    (13, 26, 34, "b) Basic-5 Size", 0.92, None, 0),
    (7, 13, 30, "b) Basic-5 Size", 0.94, None, 0),
    # w_lo 4->5: calibrated 4px strokes deposit almost no ink (49.5% landing).
    # roi "auto": _auto_detail_rois() nominates the detail boxes per
    # image (it keeps a whole-canvas fallback: a starved fine pass
    # is the worse failure mode).
    (5, 7, 26, "b) Basic-5 Size", 1.0, "auto", 100),
]
EDGE_T = 50          # sobel |gx|+|gy| > EDGE_T*3 counts as a contour
DIST_CAP = 130.0     # distance transform cap (px)
DIST_TO_W = 0.9      # dist px -> stroke width px
W_MIN, W_MAX = 4.0, 120.0
TAPER_FRACTION = 0.14   # attack/release as fraction of arc length
TAPER_MIN = 0.45        # end pressure fraction
MAX_PASSES = 5          # error-driven refill passes per band (converge-stop)

# Auto detail-ROI detector (_auto_detail_rois below). Tile-score currency
# matches the fine pass's own sobel gates, so a nomination is somewhere the
# gates could actually pass; the budget keeps the detail pass bounded.
DET_TILE = 32           # score tile, px
DET_STRUCT = 25.0       # per-pixel min-sobel counting as structure
DET_FRAC = 0.04         # tile candidate floor (fraction of structure px)
DET_BUDGET = 0.35       # max canvas fraction the fine bands may nominate
DET_MAX_BOXES = 16      # connected nominations kept, largest first

# Ink calibration (ink_calib.json, shipped alongside): how much ink a
# stroke REALLY leaves per (preset, size, pressure). Without it the sim
# believes one pass = solid coverage, so error-driven refill stops
# early and the render lands starved and pale.
_CALIB = None
_calib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "ink_calib.json")
try:
    with open(_calib_path) as _fh:
        _CALIB = json.load(_fh)
except OSError:
    print("WARNING: %s missing - sim uses naive full-ink model" % _calib_path)


def _load_stroke_trace():
    """Load the sibling stroke_trace module (flat bed + ink tracing).

    stroke_trace ships as a data-file sibling loaded via importlib so
    the planner stays a two-file drop-in; it prefers scikit-image's
    skeletonize and only falls back to the numpy Zhang-Suen when
    scikit-image is unavailable.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "stroke_trace", os.path.join(HERE, "stroke_trace.py"))
    stt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stt)
    return stt


def dist(c1, c2):
    return abs(c1[0] - c2[0]) + abs(c1[1] - c2[1]) + abs(c1[2] - c2[2])


def blurred(im, r):
    return im.filter(ImageFilter.GaussianBlur(r))


class ScaleField:
    """Edge map + chamfer distance transform -> local stroke width.

    Pure python (no numpy on the plugin side constraint carries here):
    two-pass chamfer on a half-res grid, bilinear-ish upscale, gaussian
    smoothing. width(x,y) = clamp(dist*DIST_TO_W, W_MIN, W_MAX).
    """

    def __init__(self, gray, W, H):
        w, h = gray.size
        p = gray.load()
        # sobel magnitude on full res, max-pooled into half-res edge mask
        hw, hh = W // 2, H // 2
        edge = [[False] * hh for _ in range(hw)]
        for y in range(1, h - 1, 2):
            for x in range(1, w - 1, 2):
                a = p[x - 1, y - 1]; b = p[x, y - 1]; c = p[x + 1, y - 1]
                d = p[x - 1, y];                         e = p[x + 1, y]
                f = p[x - 1, y + 1]; g = p[x, y + 1]; hh2 = p[x + 1, y + 1]
                gx = (c + 2 * e + hh2) - (a + 2 * d + f)
                gy = (f + 2 * g + hh2) - (a + 2 * b + c)
                if abs(gx) + abs(gy) > EDGE_T * 3:
                    edge[x // 2][y // 2] = True

        INF = DIST_CAP + 10
        d0 = [[0.0 if edge[x][y] else INF for y in range(hh)] for x in range(hw)]

        def up(d):
            for y in range(1, hh):
                for x in range(1, hw):
                    v = d[x][y]
                    for nx, ny, cost in ((x - 1, y, 1.0), (x, y - 1, 1.0),
                                         (x - 1, y - 1, 1.414),
                                         (x + 1, y - 1, 1.414)):
                        if 0 <= nx < hw:
                            c = d[nx][ny] + cost
                            if c < v:
                                v = c
                    d[x][y] = v
            return d

        def down(d):
            for y in range(hh - 2, -1, -1):
                for x in range(hw - 2, -1, -1):
                    v = d[x][y]
                    for nx, ny, cost in ((x + 1, y, 1.0), (x, y + 1, 1.0),
                                         (x + 1, y + 1, 1.414),
                                         (x - 1, y + 1, 1.414)):
                        if 0 <= nx < hw:
                            c = d[nx][ny] + cost
                            if c < v:
                                v = c
                    d[x][y] = v
            return d

        d0 = down(up(d0))
        # clamp + half-res -> full-res nearest upscale
        self.w_map = Image.new("L", (hw, hh))
        raw = bytearray(hw * hh)
        for y in range(hh):
            for x in range(hw):
                v = d0[x][y]
                raw[y * hw + x] = int(min(v, DIST_CAP) / DIST_CAP * 255)
        self.w_map.putdata(raw)
        self.w_map = self.w_map.resize((W, H), Image.BILINEAR)
        self.w_map = blurred(self.w_map, 4)
        self.w_px = self.w_map.load()

    def width(self, x, y):
        v = self.w_px[max(0, min(self.w_map.size[0] - 1, int(x))),
                      max(0, min(self.w_map.size[1] - 1, int(y)))]
        return max(W_MIN, min(W_MAX, v / 255.0 * DIST_CAP * DIST_TO_W))


def envelope(arc_fracs, taper=TAPER_FRACTION, floor=TAPER_MIN):
    """Attack/release over normalized arc-length positions (0..1)."""
    out = []
    for t in arc_fracs:
        p = 1.0
        if t < taper:
            p = floor + (1.0 - floor) * (t / taper)
        elif t > 1.0 - taper:
            p = floor + (1.0 - floor) * ((1.0 - t) / taper)
        out.append(p)
    return out


def smooth_list(vals, k=3):
    out = vals[:]
    for _ in range(k):
        for i in range(1, len(out) - 1):
            out[i] = (out[i - 1] + 2 * out[i] + out[i + 1]) / 4.0
    return out


_AUTO_DETAIL = [None]  # last auto-nominated detail boxes (plan() audits it)


def _sobel_mag_np(a):
    """3x3 sobel |grad| matching sobel_mag()'s currency, vectorized."""
    p = np.pad(a, 1, mode="edge")
    tl, t, tr = p[:-2, :-2], p[:-2, 1:-1], p[:-2, 2:]
    l, b, r = p[1:-1, :-2], p[2:, 1:-1], p[1:-1, 2:]
    bl, br = p[2:, :-2], p[2:, 2:]
    gx = (tr + 2 * r + br) - (tl + 2 * l + bl)
    gy = (bl + 2 * b + br) - (tl + 2 * t + tr)
    return np.hypot(gx, gy)


def _otsu_thresh(v):
    """Otsu split over a 64-bin histogram; returns a bin-center value."""
    hist, edges = np.histogram(v, bins=64)
    centers = (edges[:-1] + edges[1:]) / 2
    w = np.cumsum(hist).astype(np.float64)
    m = np.cumsum(hist * centers)
    total_w, total_m = w[-1], m[-1]
    best, best_t = -1.0, float(centers[0])
    for i in range(len(hist) - 1):
        w0, w1 = w[i], total_w - w[i]
        if w0 == 0 or w1 == 0:
            continue
        m0, m1 = m[i] / w0, (total_m - m[i]) / w1
        between = w0 * w1 * (m0 - m1) ** 2
        if between > best:
            best, best_t = between, float(centers[i])
    return best_t


def _morph3(m, dilate):
    """3x3 dilate/erode on a tile grid; pad ring keeps borders honest."""
    p = np.pad(m, 1, constant_values=(False if dilate else True))
    acc = np.zeros_like(p) if dilate else np.ones_like(p)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            s = np.roll(np.roll(p, dy, 0), dx, 1)
            acc = (acc | s) if dilate else (acc & s)
    return acc[1:-1, 1:-1]


def _tiles_to_boxes(keep, score, tile, W, H):
    """Connected tile components -> pixel boxes, strongest first."""
    th, tw = keep.shape
    seen = np.zeros_like(keep, dtype=bool)
    comps = []
    for sy in range(th):
        for sx in range(tw):
            if not keep[sy, sx] or seen[sy, sx]:
                continue
            stack = [(sy, sx)]
            seen[sy, sx] = True
            ys, xs, energy, area = [], [], 0.0, 0
            while stack:
                cy, cx = stack.pop()
                area += 1
                energy += float(score[cy, cx])
                ys.append(cy)
                xs.append(cx)
                for ny, nx in ((cy - 1, cx), (cy + 1, cx),
                               (cy, cx - 1), (cy, cx + 1)):
                    if (0 <= ny < th and 0 <= nx < tw
                            and keep[ny, nx] and not seen[ny, nx]):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append((energy, area, min(xs), min(ys), max(xs), max(ys)))
    comps.sort(key=lambda c: (-c[0], -c[1]))
    return [(max(0, x0 * tile), max(0, y0 * tile),
             min(W, (x1 + 1) * tile), min(H, (y1 + 1) * tile))
            for _e, _a, x0, y0, x1, y1 in comps[:DET_MAX_BOXES]]


def _auto_detail_rois(gray_raw, W, H, tag=""):
    """Nominate the detail boxes for the finest band + detail pass.

    Constant hand-drawn boxes were tried first and rejected: any
    fixed pixel region is image-specific and confines the
    sub-7px bands instead of following the picture's own
    detail.

    Score = per-tile fraction of "structure" pixels, structure = the
    MINIMUM of sobel|grad| on blur-2 and blur-6 gray. The min is the
    noise killer: real detail survives blur 6, JPEG mosquito noise and
    checker aliasing do not (textured fabric self-nominates loudly
    without it). Threshold = Otsu over tile scores with an absolute
    without it). Threshold = Otsu over tile scores with an absolute
    floor, then a hard canvas-fraction budget (texture-rich photos would
    nominate everything; the detail pass is 5 grid rounds and must stay
    bounded). Tile close->open->dilate merges gaps and drops speckle;
    connected components become boxes, strongest first.

    Deterministic. Returns pixel boxes, or None when nothing clears the
    floor (flat image) - None means the fine pass runs whole-canvas:
    a starved fine pass is the worse failure mode (v4.1 flower lesson).
    """
    t0 = time.time()
    g2 = _sobel_mag_np(np.asarray(blurred(gray_raw, 2), dtype=np.float32))
    g6 = _sobel_mag_np(np.asarray(blurred(gray_raw, 6), dtype=np.float32))
    struct = np.minimum(g2, g6) >= DET_STRUCT
    th, tw = H // DET_TILE, W // DET_TILE
    if th < 4 or tw < 4:
        return None
    frac = struct[:th * DET_TILE, :tw * DET_TILE].reshape(
        th, DET_TILE, tw, DET_TILE).mean(axis=(1, 3))
    cand = frac >= max(_otsu_thresh(frac.ravel()), DET_FRAC)
    if cand.sum() > DET_BUDGET * cand.size:
        # texture-rich image: keep only the strongest nominations in budget
        k = int(DET_BUDGET * cand.size)
        order = np.argsort(frac.ravel(), kind="stable")[::-1]
        keep = np.zeros(cand.size, dtype=bool)
        got = 0
        for i in order:
            if not cand.flat[i]:
                continue
            keep[i] = True
            got += 1
            if got >= k:
                break
        cand = keep.reshape(th, tw)
    if cand.mean() < 0.01:
        print("%sdetail-rois: nothing clears the floor (%.1f%% of tiles) "
              "-> fine pass whole-canvas (%.1fs)"
              % (tag, 100 * cand.mean(), time.time() - t0))
        return None
    keep = _morph3(_morph3(_morph3(cand, True), False), True)
    boxes = _tiles_to_boxes(keep, frac, DET_TILE, W, H)
    print("%sdetail-rois: %d boxes, %.0f%% of tiles (%.1fs)"
          % (tag, len(boxes), 100 * float(keep.mean()), time.time() - t0))
    for x0, y0, x1, y1 in boxes:
        print("%s  detail-rois: (%d, %d, %d, %d)" % (tag, x0, y0, x1, y1))
    return boxes


def _plan_strokes(ref, seed, regions, rng, t_start, tag="", max_passes=None,
                  detail_rois=None, measure_hook=None, progress=None):
    """Error-driven refill planning onto `seed` (mutated in place).

    Shared by plan() (full canvas, regions=None) and by any patch-box
    or additive-deficit caller (regions=[...]). Every planned stroke is
    stamped onto `seed` with the calibrated sim so the next pass
    measures real ink; the returned strokes still carry "band" for
    _finalize_layers.

    regions: None -> whole canvas, bands keep their own ROIs;
             [(x0, y0, x1, y1), ...] -> every band AND the fine pass
             plan only inside these boxes (refine-mode patch repaint).
    max_passes: refill pass budget per band; default MAX_PASSES. The
             flower patch proved 5 is a ceiling, not a convergence
             (pass 5 still emitted 367 strokes, sim plateaued at 142).
    progress: optional callback(list_so_far), invoked after each band
             pass and after the fine pass - the live-growth feed for
             the GUI (plan() dumps finalized partial plans through it).
    """
    W, H = ref.size
    gray_raw = ref.convert("L")
    gray = blurred(gray_raw, 2)
    scale = ScaleField(gray, W, H)
    # finest band + detail pass ROI: explicit detail_rois wins, else the
    # image's own nominations (an explicit regions override per band below)
    auto_rois = detail_rois
    if auto_rois is None:
        auto_rois = _auto_detail_rois(gray_raw, W, H, tag)
        _AUTO_DETAIL[0] = auto_rois
    _GP[0] = gray.load()
    _GS[0], _GS[1] = gray.size
    print("%sscale field ready (%.0fs)" % (tag, time.time() - t_start))

    canvas = seed
    draw = ImageDraw.Draw(canvas)
    strokes = []
    sim_book = [[]]  # per-pass stroke list when measure_hook is active

    def record(s, preset, op):
        # sim bookkeeping is the pass-to-pass truth by default; with a
        # measure_hook a real renderer can re-measure each pass instead
        # (sim deposit != renderer deposit - trusting the sim alone
        # leaves paper speckle where it guessed "covered").
        # _preset/_op ride on the stroke so the caller can project the
        # same layer fields _finalize_layers will assign later.
        if measure_hook is None:
            _sim_draw(draw, s, preset, op)
        else:
            s["_preset"] = preset
            s["_op"] = op
            sim_book[0].append(s)

    def err(x, y, refb):
        return dist(canvas.getpixel((int(x), int(y))),
                    refb.getpixel((int(x), int(y))))

    def make_stroke(x0, y0, refb, thresh, max_len, w_cap=None):
        w0 = scale.width(x0, y0)
        if w_cap:
            w0 = min(w0, w_cap)
        color = refb.getpixel((x0, y0))
        # deterministic flat-area direction: hash the seed, avoid parallelism
        hsh = (x0 * 73856093) ^ (y0 * 19349663)
        ang = ((hsh >> 8) & 1023) / 1024.0 * math.pi
        init_d = (math.cos(ang), math.sin(ang))
        # follow the blurred-gray orientation field, step scaled to width
        pts = [[float(x0), float(y0)]]
        ws = [w0]
        for sign in (1, -1):
            cur = [float(x0), float(y0)]
            d = None
            local = []
            while math.hypot(cur[0] - x0, cur[1] - y0) < max_len:
                w_here = scale.width(cur[0], cur[1])
                if w_cap:
                    w_here = min(w_here, w_cap)
                step = max(4.0, w_here * 0.45)
                g = sobel_dir(cur[0], cur[1]) or init_d
                if d is not None and g[0] * d[0] + g[1] * d[1] < 0:
                    g = (-g[0], -g[1])
                d = g
                nxt = [cur[0] + d[0] * sign * step, cur[1] + d[1] * sign * step]
                if not (1 < nxt[0] < W - 2 and 1 < nxt[1] < H - 2):
                    break
                if dist(refb.getpixel((int(nxt[0]), int(nxt[1]))), color) > thresh * 1.7:
                    break
                if err(nxt[0], nxt[1], refb) < thresh * 0.45:
                    break
                if sign == 1:
                    pts.append(nxt)
                    ws.append(w_here)
                else:
                    local.append([nxt, w_here])
                cur = nxt
            if sign == -1:
                pairs = local[::-1]
                pts = [p for p, _ in pairs] + pts
                ws = [w for _, w in pairs] + ws

        if len(pts) < 2:
            return None
        # image-derived width profile: smooth + end taper -> pressure per point
        ws = smooth_list(ws, 3)
        # decimate: paintLine interpolates pressure linearly per segment, so
        # keep points only where the profile bends (width step or corner).
        keep = [0]
        for i in range(1, len(pts) - 1):
            a, b, c = keep[-1], i, i + 1
            w_jump = abs(ws[b] - ws[a])
            vx, vy = pts[b][0] - pts[a][0], pts[b][1] - pts[a][1]
            wx, wy = pts[c][0] - pts[b][0], pts[c][1] - pts[b][1]
            la, lb = math.hypot(vx, vy) or 1.0, math.hypot(wx, wy) or 1.0
            dot = max(-1.0, min(1.0, (vx * wx + vy * wy) / (la * lb)))
            if w_jump / max(ws[a], 1.0) > 0.10 or math.acos(dot) > math.radians(28):
                keep.append(i)
        keep.append(len(pts) - 1)
        pts = [pts[i] for i in keep]
        ws = [ws[i] for i in keep]
        # arc-length fractions for the envelope
        arcs = [0.0]
        for i in range(1, len(pts)):
            arcs.append(arcs[-1] + math.hypot(pts[i][0] - pts[i - 1][0],
                                              pts[i][1] - pts[i - 1][1]))
        total = arcs[-1] or 1.0
        env = envelope([a / total for a in arcs])
        wmax = max(ws)
        points = [[round(pts[i][0], 1), round(pts[i][1], 1),
                   round(max(0.10, min(1.0, ws[i] / wmax * env[i])), 3)]
                  for i in range(len(pts))]
        gcx = sum(p[0] for p in pts) / len(pts)
        gcy = sum(p[1] for p in pts) / len(pts)
        contrast = min(1.0, sobel_mag(gcx, gcy) / 220.0)
        return {"points": points, "size": round(wmax, 1),
                "width_range": [round(min(ws) * TAPER_MIN, 1), round(wmax, 1)],
                "color": "#%02x%02x%02x" % color[:3],
                "contrast": round(contrast, 3)}

    # error-driven multi-pass: paint a pass, re-measure the simulated
    # canvas, and keep refilling wherever it still deviates from the
    # reference until a pass converges. One grid pass on an empty canvas
    # leaves inter-stroke gaps that nothing ever revisited (v4.2 disease:
    # "one thin coat"); passes >= 2 offset the grid half a step so refill
    # seeds land between pass-1 strokes.
    for w_lo, w_hi, thresh, _preset, _op, roi, gate in BANDS:
        # Mid bands (26/13/7) are where dark-speckle damage concentrates
        # (forensics on dense linework: every dark px sat in these
        # three tiers, widest/finest tiers clean). Each tier
        # colour pick from photo-grain dark specks, the fine tier's short
        # arc keeps one bad pick from becoming a long constant-colour worm.
        mid = 7 <= w_lo <= 26
        blur_r = max(2, int(w_lo / 3))
        if mid:
            blur_r = max(6, blur_r)
        refb = blurred(ref, blur_r)
        if mid:
            # 3x3 median: a 1-2px dark speck no longer survives as an
            # anchor colour, and the crawl's divergence test reads the
            # same de-speckled reference
            refb = refb.filter(ImageFilter.MedianFilter(size=3))
        step = max(3, int(w_lo * 0.85))
        first = None
        band_total = 0
        if regions:
            roi = regions
        elif roi == "auto":
            roi = auto_rois
        for pass_no in range(max_passes or MAX_PASSES):
            off = (step // 2) if (pass_no % 2) else 0
            grid = [(x, y) for y in range(2 + off, H, step)
                    for x in range(2 + off, W, step)]
            if roi:
                grid = [p for p in grid
                        if any(x0 <= p[0] < x1 and y0 <= p[1] < y1
                               for x0, y0, x1, y1 in roi)]
            rng.shuffle(grid)
            count = 0
            for (x, y) in grid:
                if gate and sobel_mag(x, y) < gate:
                    continue
                w_here = scale.width(x, y)
                if not (w_lo <= w_here < w_hi):
                    continue
                if err(x, y, refb) < thresh:
                    continue
                s = make_stroke(x, y, refb, thresh,
                                max_len=int(w_hi * (2.5 if mid else 5)))
                if not s:
                    continue
                s["band"] = [w_lo, w_hi]
                strokes.append(s)
                count += 1
                record(s, _preset, _op)
            band_total += count
            if first is None:
                first = count
            print("%sband %2d-%3d pass %d: %d strokes (%.0fs)"
                  % (tag, w_lo, w_hi, pass_no + 1, count,
                     time.time() - t_start))
            if progress is not None:
                progress(strokes)
            if measure_hook is not None and sim_book[0]:
                canvas = measure_hook(sim_book[0])
                draw = ImageDraw.Draw(canvas)
                sim_book[0] = []
            if count == 0 or count < first * 0.04:
                break
        print("%sband %2d-%3d total: %d strokes"
              % (tag, w_lo, w_hi, band_total))

    # fine pass in the detail ROIs regardless of scale (eyes, flower, beads);
    # refine mode reuses the same machinery inside the patch boxes;
    # no boxes -> the image's own auto nominations (None = whole canvas)
    refb = blurred(ref, 2)
    step = 5
    first = None
    fine_rois = regions or auto_rois
    for pass_no in range(max_passes or MAX_PASSES):
        off = (step // 2) if (pass_no % 2) else 0
        grid = [(x, y) for y in range(2 + off, H, step)
                for x in range(2 + off, W, step)]
        if fine_rois:
            grid = [p for p in grid
                    if any(x0 <= p[0] < x1 and y0 <= p[1] < y1
                           for x0, y0, x1, y1 in fine_rois)]
        rng.shuffle(grid)
        count = 0
        for (x, y) in grid:
            if err(x, y, refb) < 26:
                continue
            if sobel_mag(x, y) < 70:
                continue
            s = make_stroke(x, y, refb, 26, max_len=40, w_cap=8.5)
            if not s:
                continue
            s["band"] = [5, 7]
            strokes.append(s)
            count += 1
            record(s, "b) Basic-5 Size", 1.0)
        if first is None:
            first = count
        print("%sdetail ROI pass %d: %d strokes (%.0fs)"
              % (tag, pass_no + 1, count, time.time() - t_start))
        if measure_hook is not None and sim_book[0]:
            canvas = measure_hook(sim_book[0])
            draw = ImageDraw.Draw(canvas)
            sim_book[0] = []
        if count == 0 or count < first * 0.04:
            break

    if progress is not None:
        progress(strokes)
    return strokes


def _finalize_layers(strokes):
    """Band order (coarse first) is the paint order: assign layers,
    presets and per-stroke opacity, then sort into paint order."""
    layer_of_band = {}
    for s in strokes:
        key = tuple(s["band"])
        if key not in layer_of_band:
            layer_of_band[key] = "L%d" % int(key[0])
        s["layer"] = layer_of_band[key]
        for w_lo, w_hi, _t, preset, base_op, _roi, _g in BANDS:
            if [w_lo, w_hi] == s["band"]:
                s["preset"] = preset
                s["opacity"] = round(min(1.0, base_op + 0.35 * s["contrast"]), 3)
                break
        del s["band"], s["contrast"]

    order = {"b) Basic-2 Opacity": 0, "b) Basic-5 Size": 1}
    strokes.sort(key=lambda s: (-s["size"], order[s["preset"]]))


# One width ruler for every layer: disjoint intervals of MEASURED
# visual width; the edges reuse the nominal BANDS edges plus a
# finest tier. Layer bitmaps composite in first-appearance order,
# and the bands are disjoint, so the stack order is monotone by
# construction. Inside a band, ink strokes (generation-time X*
# layer names) split into X layers stacked above that band's bed
# layer - the capsule engine routes tips by layer name, so an X
# name restores the ink tip look; ink >= 13px folds into its bed
# band (a width domain the tip engine never covered).
_WIDTH_BANDS = ((52.0, 120.0), (26.0, 52.0), (13.0, 26.0),
                (7.0, 13.0), (5.0, 7.0), (0.0, 5.0))
_BAND_X_NAME = {3: "X1C", 4: "X1M", 5: "X1F"}   # band idx -> ink layer


def _assign_layers_by_width(strokes):
    """Re-group strokes into disjoint measured-width bands, after merge;
    ink strokes (generation X* names) split into X layers inside their
    band.

    Generation bands assign layer from the feature's nominal width, but
    _merge_fragments rewrites size to the combined arc's widest point
    (an L26 mark measured 114.7) and X pens' nominal size diverges from
    painted width (X1C size 64 -> 17.9 visual). Layer bitmaps composite
    in first-seen order, so stale bands let wide marks bury thinner
    ones across layers (measured on one dense reference: 4.3M such
    pairs; a single merged 114.7px mark alone covered 3313 thinner
    strokes). Banding on the SAME key
    the z-order sort uses makes every smaller mark paint above every
    wider one - within and across layers - by construction. Presets and
    opacities keep their generation-band values; only the compositing
    group changes. F1 (the bed) is exempt: prefixed bottom by design.

    The X split (band idx 3/4/5 -> X1C/X1M/X1F) keeps that ruler intact:
    an X layer holds exactly one band's ink, so it sits above the same
    band's bed and below every narrower band. >=13px ink folds into its
    bed band (the tip engine never covered that width domain).
    """
    for s in strokes:
        gen = s["layer"]
        w = float(s.get("_vw", s["size"]))
        for i, (lo, hi) in enumerate(_WIDTH_BANDS):
            if lo <= w < hi:
                if gen.startswith("X") and i in _BAND_X_NAME:
                    s["layer"] = _BAND_X_NAME[i]
                else:
                    s["layer"] = "L%d" % int(lo)
                break
        else:
            s["layer"] = "L52"    # >= 120: widest band, bottom-most


def _zorder_key(s):
    """Sort key for the one width ruler: (band order, bed before
    ink within a band, wide to narrow within the group). Between
    bands, smaller strokes paint above bigger ones; within a band
    the bed paints before the ink lines (ink over bed is the
    painting semantic), so the layer stack's first-appearance
    order is always bed-below-ink. Relative stroke order inside
    each layer is unchanged from the global wide-to-narrow order."""
    w = float(s.get("_vw", s["size"]))
    band = next((i for i, (lo, hi) in enumerate(_WIDTH_BANDS)
                 if lo <= w < hi), 0)
    return (band, 1 if s["layer"].startswith("X") else 0, -w)


def _merge_fragments(strokes, gap=7.0, ang_deg=35.0, passes=4):
    """Merge chopped strokes back into continuous marks.

    The walkers stop early (error/colour guards, per-arm max_len), so a
    single feature in the reference comes out as several fragments; each
    fragment then gets its own attack/release envelope and paints as a
    string of beads instead of one mark. Merge same-layer strokes whose
    endpoints sit within `gap` px and whose end/start tangents are within
    `ang_deg`, then re-apply ONE envelope over the combined arc.

    Deterministic: consumes in stroke order, first (lowest-index) match
    wins. Spatial grid keeps the search O(n * neighbours), not O(n^2):
    a naive full scan took 428s on 24.7k strokes.
    """
    def tang(points, back):
        p = points[-2:] if back else points[:2]
        if len(p) < 2:
            return (1.0, 0.0)
        dx, dy = p[-1][0] - p[0][0], p[-1][1] - p[0][1]
        L = math.hypot(dx, dy) or 1.0
        return (dx / L, dy / L)

    merged = list(strokes)
    for _ in range(passes):
        n0 = len(merged)
        used = [False] * len(merged)
        out = []
        cell = max(2.0, gap)
        # index of every not-yet-consumed stroke START by (layer, cell)
        grid = {}
        for j, b in enumerate(merged):
            s = b["points"][0]
            key = (b["layer"], int(s[0] // cell), int(s[1] // cell))
            grid.setdefault(key, []).append(j)
        for i, a in enumerate(merged):
            if used[i]:
                continue
            cur = a
            cur_color = a.get("color")
            while True:
                end = cur["points"][-1]
                tx, ty = tang(cur["points"], back=True)
                cx, cy = int(end[0] // cell), int(end[1] // cell)
                cands = []
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for j in grid.get((cur["layer"], cx + dx,
                                           cy + dy), []):
                            if not used[j]:
                                cands.append(j)
                best = None
                for j in sorted(cands):
                    if j == i:
                        continue
                    b = merged[j]
                    if abs(float(b.get("size", 0)) - float(cur["size"])) > 0.6:
                        continue
                    st = b["points"][0]
                    if (end[0] - st[0]) ** 2 + (end[1] - st[1]) ** 2 > gap * gap:
                        continue
                    bx, by = tang(b["points"], back=False)
                    if tx * bx + ty * by < math.cos(math.radians(ang_deg)):
                        continue
                    best = j
                    break
                if best is None:
                    break
                used[best] = True
                b = merged[best]
                bp = b["points"]
                if (cur["points"][-1][0] - bp[0][0]) ** 2 + \
                   (cur["points"][-1][1] - bp[0][1]) ** 2 < 4.0:
                    bp = bp[1:]
                if not bp:
                    break
                pts = cur["points"] + bp
                ws = [max(0.1, p[2]) * cur["size"] for p in pts]
                wmax = max(ws)
                arcs = [0.0]
                for k in range(1, len(pts)):
                    arcs.append(arcs[-1] + math.hypot(
                        pts[k][0] - pts[k - 1][0],
                        pts[k][1] - pts[k - 1][1]))
                total = arcs[-1] or 1.0
                env = envelope([a_ / total for a_ in arcs])
                newpts = [[round(pts[k][0], 1), round(pts[k][1], 1),
                           round(max(0.10, min(1.0, ws[k] / wmax * env[k])), 3)]
                          for k in range(len(pts))]
                cur = {"layer": cur["layer"], "preset": cur["preset"],
                       "size": round(wmax, 1),
                       "opacity": cur.get("opacity", 1.0),
                       "color": cur_color, "points": newpts}
            out.append(cur)
        merged = out
        if len(merged) == n0:
            break
    return merged


def _border_bg(ref):
    """Canvas paper colour = the reference's border (its surround), NOT a
    hardcoded tint. A hardcoded pale-foliage constant made any
    white-background reference paint green through the gaps."""
    a = np.asarray(ref.convert("RGB"), dtype=np.int64)
    edge = np.concatenate([a[0, :], a[-1, :], a[:, 0], a[:, -1]])
    m = np.median(edge, axis=0).astype(int)
    return (int(m[0]), int(m[1]), int(m[2]))


def _auto_pen_pass(ref, cal_path=None, pen=None, pen_auto=True):
    """Append a full-resolution ink pass to a v4 tonal plan (eyes,
    lashes, hair strands, lace: sub-5px marks the half-res band engine
    cannot measure). Reuses the trace extractor: full-res EDT widths,
    per-mark brush by measured character (refined select_pen by default;
    --pen forces one preset), pressure from the width profile.
    One pass, deterministic, ANY image.
    """
    stt = _load_stroke_trace()
    if cal_path is None:
        cal_path = os.path.join(HERE, "fork_ink_calib.json")
    cal = json.load(open(cal_path))
    if pen is not None and pen not in cal.get("presets", {}):
        print("--pen %r not in the calibration table (%s) -> automatic"
              % (pen, ", ".join(sorted(cal.get("presets", {})))), flush=True)
        pen = None
    strokes, widths = stt.extract_line_strokes(
        ref, cal, depth_thresh=7.0, min_px=6, max_strokes=25000,
        layer="X1", pen=pen, pen_auto=pen_auto, ink_density=0.13)
    if pen is None and pen_auto:
        # coarse backing layer: full 0.30 density mask, marks >= 6px
        # visual - the configuration that render A/B validated.
        # (exclude_density is available on extract_line_strokes as an
        # opt-in knob; not used here - the validated look wins.)
        cstrokes, _ = stt.extract_line_strokes(
            ref, cal, depth_thresh=7.0, min_px=6, max_strokes=30000,
            layer="X1C", pen=None, pen_auto=True, ink_density=0.30,
            min_depth=40.0)
        # min_depth: the 0.30 percentile mask is GLOBAL - on photos it
        # swallows smooth skin (skin is darker than the surround wall)
        # and its skeleton fragments become opaque speckles. Median
        # ridge depth separates photo junk (measured <= 42) from real
        # ink marks (>= 73 on dense linework). Soft illustrated
        # texture sits in between (5-50, p50 ~ 29), and the
        # where the dropped marks were white speckle noise on flower
        # and hair and the kept bands carry all real texture.
        cstrokes = [s for s in cstrokes
                    if s.get("_vw", s["size"]) >= 6.0]
        for s in cstrokes:
            s["layer"] = "X1C"
            s.setdefault("opacity", 1.0)
        strokes = cstrokes + strokes
    for s in strokes:
        s.setdefault("opacity", 1.0)
    return strokes


def plan(ref_path, out_path, pen=None, pen_auto=True):
    t_start = time.time()
    ref = Image.open(ref_path).convert("RGB")
    W, H = ref.size
    bg = _border_bg(ref)
    canvas = Image.new("RGB", (W, H), bg)  # paper = reference's surround
    # ink pass first: F1's ring width anchors on the linework's own median
    # mark width (a wash is never finer than the original's brush)
    xstrokes = _auto_pen_pass(ref, pen=pen, pen_auto=pen_auto)  # ink (X1)
    # F1 flat-colour bed UNDER everything: median-cut strata traced as
    # opaque contour rings with measured colours. paper=bg: a photo's
    # surround is not paper - near-ground strata must still be painted.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "stroke_trace", os.path.join(HERE, "stroke_trace.py"))
    stt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stt)
    with open(os.path.join(HERE, "fork_ink_calib.json")) as fh:
        cal = json.load(fh)
    vws = [float(s.get("_vw", s["size"])) for s in xstrokes]
    anchor = float(np.median(vws)) if vws else 10.0
    f1 = stt.extract_flat_strokes(ref, cal, anchor, colors=24, layer="F1",
                                  paper=bg)
    # the L bands refine ON TOP of the laid bed: sim-stamp F1 first so the
    # refill loop measures the bed instead of bare paper (no double coat).
    # solid=True: the bed is opaque by design; the calibrated alpha
    # (0.92 @ p1.0) is a real-brush deposit fraction and blended paper
    # into the bed colour, leaving it pale - the detail pass then doubled
    # its refill (L5 4240 -> 8850) trying to cover the shortfall.
    draw = ImageDraw.Draw(canvas)
    for s in f1:
        _sim_draw(draw, s, s["preset"], s["opacity"], solid=True)

    # live-growth feed (the painting should grow from the first
    # second, not arrive as a post-hoc slideshow): after each band pass, dump a
    # FINALIZED copy of strokes-so-far for the GUI to render while
    # planning continues. Copies only - the real strokes keep their
    # band keys until the one true _finalize_layers below. Torn reads
    # are impossible: tmp file + os.replace.
    partial = out_path[:-5] + "_partial.json"
    _last_dump = [0.0]
    # Incremental partial feed (profiling: the full re-serialization
    # per beat was 9s of a 33s instrumented plan - O(n^2) in stroke count).
    # _finalize_layers' per-stroke work is pure (layer/preset/opacity from
    # the stroke's own band) and its partial sort key is per-stroke too, so
    # each stroke's compact JSON text is computed once at first sight and
    # reused every later beat: a beat then only finalizes the NEW strokes,
    # re-sorts (key, text) pairs and joins cached strings. Same stable-sort
    # tie behaviour (cache keeps insertion order). The doc shape reproduces
    # json.dump(..., separators=(",", ":")) by hand - verified byte-identical
    # against the full-reserialization path.
    _PRESET_ORDER = {"b) Basic-2 Opacity": 0, "b) Basic-5 Size": 1}
    _fin_cache = []                  # (sortkey, text), insertion order
    _n_cached = [0]
    _f1_texts = [None]

    def _fin_text(s):
        c = dict(s)
        c["layer"] = "L%d" % int(tuple(c["band"])[0])
        for w_lo, w_hi, _t, preset, base_op, _roi, _g in BANDS:
            if [w_lo, w_hi] == c["band"]:
                c["preset"] = preset
                c["opacity"] = round(min(1.0, base_op + 0.35 * c["contrast"]), 3)
                break
        del c["band"], c["contrast"]
        # old path's two stacked sorts, flattened into one stable key:
        # _finalize_layers' (-size, preset order), then its re-sort by
        # -get("_vw", size); insertion order breaks full ties as before.
        return ((-float(c.get("_vw", c["size"])), -c["size"],
                 _PRESET_ORDER[c["preset"]]),
                json.dumps(c, separators=(",", ":")))

    def dump_partial(so_far, force=False):
        # 1.5s beat: W5 needed 4s because each beat re-serialized everything
        # (O(n^2)); with the per-stroke text cache (see _fin_cache) a beat is
        # a sort + join, so the preview can breathe faster for free.
        now = time.time()
        if not force and now - _last_dump[0] < 1.5:
            return
        _last_dump[0] = now
        for s in so_far[_n_cached[0]:]:
            _fin_cache.append(_fin_text(s))
        _n_cached[0] = len(so_far)
        if _f1_texts[0] is None:
            _f1_texts[0] = [json.dumps(s, separators=(",", ":")) for s in f1]
        parts = sorted(_fin_cache, key=lambda kv: kv[0])
        body = ",".join(_f1_texts[0] + [t for _k, t in parts])
        tmp = partial + ".%d.tmp" % os.getpid()
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                # compact separators: this file is a live preview feed
                # (GUI json.load's it), no byte contract - unlike the
                # final plan below, whose layout must not move
                fh.write('{"reference":%s,"canvas":[%d,%d],"seed":%d,'
                         '"bg":%s,"detail_rois":%s,"count":%d,'
                         '"strokes":[%s]}' % (
                             json.dumps(os.path.abspath(ref_path)),
                             W, H, 20260904,
                             json.dumps(list(bg), separators=(",", ":")),
                             json.dumps(list(_AUTO_DETAIL[0] or []),
                                        separators=(",", ":")),
                             len(_f1_texts[0]) + len(parts), body))
        except OSError:
            return                 # feed is best-effort; next beat retries
        # On Windows a concurrent reader of the target (a GUI preview
        # poller) can make os.replace land inside a read window
        # (WinError 32), which must never kill the plan mid-band.
        # Retry across one poll beat, then drop this frame - a
        # preview feed must never kill the plan it previews.
        for _ in range(4):
            try:
                os.replace(tmp, partial)
                return
            except OSError:
                time.sleep(0.08)
        try:
            os.remove(tmp)
        except OSError:
            pass

    dump_partial([])            # the bed alone = first growth frame
    strokes = _plan_strokes(ref, canvas, None, random.Random(20260904),
                            t_start, progress=dump_partial)
    dump_partial(strokes, force=True)   # last live frame, bands complete
    _finalize_layers(strokes)
    strokes = _merge_fragments(strokes)  # join chopped marks (bead fix)
    strokes = strokes + xstrokes
    _assign_layers_by_width(strokes)
    # Small above big, whole plan sorted by measured (visual) width.
    # Sort key = the one width ruler (_vw visual width, else size);
    # layer membership was re-assigned by _assign_layers_by_width on
    # the same ruler (disjoint bands), so the layer stack (first
    # appearance) and the per-stroke order agree: between bands any
    # smaller stroke paints above any bigger one; within a band the
    # bed paints before the ink lines (ink over bed = painting
    # semantic); X layers sit between their band's bed layer and
    # the narrower band above. Stack order is always
    # F1 L52 L26 L13 L7 X1C L5 X1M L0 X1F.
    strokes.sort(key=_zorder_key)
    # The F1 bed is the widest stroke class in the whole plan, so it
    # goes FIRST in the JSON = bottom layer (layers are built on
    # first appearance), keeping its own generation order (strata
    # big->small, ring outer->inner - the solid rings' coverage
    # order IS the gradient direction and must not be reordered);
    # every other stroke keeps the global small-first-big-last order.
    strokes = f1 + strokes

    doc = {"reference": os.path.abspath(ref_path),
           "canvas": [W, H], "seed": 20260904, "bg": list(bg),
           "detail_rois": list(_AUTO_DETAIL[0] or []),
           "count": len(strokes), "strokes": strokes}
    with open(out_path, "w") as fh:
        # json.dump() walks the chunk-by-chunk PYTHON iterencode (it can
        # never use the one-shot C encoder); dumps() does and emits
        # byte-identical text, ~5x faster on large plans
        # (0.46s -> 0.09s measured on a 33k-stroke plan).
        fh.write(json.dumps(doc))
    print("total %d strokes -> %s (%.0fs)"
          % (len(strokes), out_path, time.time() - t_start))
    canvas.save(out_path.replace(".json", "_preview.png"))
    return 0


def _erode(mask, r):
    """Boolean erosion, radius r (approx disc via r rounds of cross-min).
    Iterated numpy rolls over a small crop are effectively free; wrapping
    is neutralized by zeroing the border band first, so an eroded mask
    can never seed strokes that bleed past the patch box."""
    if r <= 0:
        return mask.copy()
    out = mask.copy()
    out[:r, :] = out[-r:, :] = False
    out[:, :r] = out[:, -r:] = False
    for _ in range(r):
        out = (out & np.roll(out, 1, 0) & np.roll(out, -1, 0)
               & np.roll(out, 1, 1) & np.roll(out, -1, 1))
    return out


GROUND_LADDER = (30, 20, 14, 10, 7, 5, 3)
"""Fill scale ladder, coarse->fine. Small patches are all corridor
(measured p75 radius 2-5); whole-canvas blocks (background, garment,
hair cores) have radii in the dozens, so real big strokes belong at
the top while the small scales survive at the bottom for the same
edge-safe behaviour in tight blocks."""


def plan_groundfill(ref, canvas, regions, size=20, colors=12, gate=45,
                    ladder=GROUND_LADDER, blur=20, skip_carried=True,
                    row_frac=0.45, along_frac=0.4, min_run=1.5):
    """Plan block-geometry fill strokes. Pure planning: no I/O; the
    single implementation behind whole-canvas and patch-region coats.

    ref/canvas: same-size RGB PIL images (canvas is the measured current
    state, used only by the skip-gate). Returns (paint, u_layer).

    Multi-scale fill: one global stroke size cannot fit a flower's
    morphology (petal corridors erode to nothing at size/2+2 - first
    attempt filled 0 strokes). Big discs fill block cores, smaller
    ones take what is left, and every boundary band thinner than the
    smallest feasible stroke stays untouched for the detail layers.

    blur: the reference is itself a textured painting - median-cut on
    raw pixels yields salt-and-pepper class masks that erosion shreds
    into confetti (whole-canvas dry run: 10% coverage of scattered
    dots). Classes are therefore cut from a heavily blurred copy so
    masks come out as masses (hair mass, foliage mass, skin); the wash
    hue still comes from the RAW crop mean per class, so the ground
    colour is not muddied, and within-mass texture is the detail
    pass's job, not the ground's.
    """
    can = np.asarray(canvas, dtype=np.int16)
    paint = []
    u_layer = "U%d" % size
    n_cls = 0
    for x0, y0, x1, y1 in regions:
        crop = ref.crop((x0, y0, x1, y1))
        geo = crop.filter(ImageFilter.GaussianBlur(blur))
        q = geo.quantize(colors=colors, method=Image.MEDIANCUT)
        cls = np.asarray(q)
        pal = q.getpalette()
        nc = (pal and len(pal) // 3) or colors
        for c in range(min(nc, colors)):
            sel = cls == c
            area = int(sel.sum())
            if area < 400:
                continue
            n_cls += 1
            rgb = [int(round(v)) for v in
                   np.asarray(crop, dtype=np.int32)[sel].mean(axis=0)]
            color = "#%02x%02x%02x" % tuple(rgb)
            covered = np.zeros_like(sel)
            # scale ladder from the measured morphology: distance
            # transform of the flower blocks gives p75 corridor radius
            # 2-5, only two classes have cores above radius 10 - big
            # discs physically do not exist inside a flower, so the
            # ladder must meet the geometry. Each scale erodes the
            # ORIGINAL block mask - punching `covered` holes into the
            # mask before re-eroding made every hole edge eat another
            # r+2 and collapsed the ladder to size-6 confetti.
            for r in ladder:
                s = 2 * r
                m = _erode(sel, r + 2)
                # row_frac/along_frac/min_run: stamp-grid density. The
                # defaults match the spacing of a blooming real brush
                # (the grid reads as a coat); a renderer without deposit
                # bloom does NOT bloom, so a sparse grid leaves paper
                # holes that nothing covers - pass tighter fractions
                # there for a solid coat.
                step_y = max(3, int(s * row_frac))
                for row in range(r + 2, m.shape[0] - r - 2, step_y):
                    xs = np.flatnonzero(m[row])
                    if not len(xs):
                        continue
                    breaks = np.flatnonzero(np.diff(xs) > 1)
                    starts = np.concatenate(([0], breaks + 1))
                    ends = np.concatenate((breaks, [len(xs) - 1]))
                    for a, b in zip(starts, ends):
                        xa, xb = int(xs[a]), int(xs[b])
                        if xb - xa < int(min_run * s):
                            continue  # too short to stamp twice: edge zone
                        cxl = (xa + xb) // 2
                        if covered[row, cxl]:
                            continue  # a bigger stroke already fills here
                        cx = x0 + (xa + xb) / 2.0
                        cy = y0 + row
                        cc = can[int(cy), int(cx)]
                        if skip_carried and sum(abs(int(cc[j]) - rgb[j])
                                                for j in range(3)) < gate:
                            continue  # canvas already carries this block
                        # NOTE: on a BARE PAPER canvas the gate starves every
                        # pale block - a pale mass mean sits within `gate` of
                        # the paper tint, reads as "already carried", and the
                        # mass never gets its ground. Pass skip_carried=False
                        # when planning onto bare paper.
                        half = s / 2.0
                        pts = [[float(x0 + xa + half), float(cy), 1.0]]
                        x = x0 + xa + half + s * along_frac
                        while x <= x0 + xb - half:
                            pts.append([float(x), float(cy), 1.0])
                            x += s * along_frac
                        if len(pts) == 1:
                            # tight along_frac can leave a run that fits
                            # one stamp - daub rejects single-point
                            # strokes, so give the capsule a second
                            # point; the slight spill past the eroded
                            # core is harmless (the default min_run 1.5
                            # guarantees >= 2 stamps anyway).
                            pts.append([float(pts[0][0] + s * 0.35),
                                        float(cy), 1.0])
                        paint.append({"layer": u_layer,
                                      "preset": "b) Basic-2 Opacity",
                                      "size": s, "opacity": 0.9,
                                      "color": color, "points": pts})
                        covered[max(0, row - step_y):row + step_y,
                                xa:xb + 1] = True
    print("plan_groundfill: %d fill strokes over %d block class(es)"
          % (len(paint), n_cls), flush=True)
    return paint, u_layer


def plan_repair_fill(ref, canvas, regions, size=26, thresh=60, op=0.9):
    """Measured-state grid fill for flat masses the block-geometry coat
    starves. plan_groundfill was built for flower morphology (classes cut
    from blur-20, erosion ladder, min-run gates) - on pale midtone masses
    (neck shadow, window recess) the blurred classes shred and the ladder
    erases them, so gap cells came back with confetti at best. This is
    the dumb strong version: a fixed half-step grid over the given
    regions, stamping horizontal chains wherever the MEASURED canvas
    still deviates from the reference by more than `thresh`. Pure
    planning: no I/O."""
    refb = ref.filter(ImageFilter.GaussianBlur(size // 2))
    rpx = refb.load()
    cpx = canvas.load()
    raw = np.asarray(ref, dtype=np.float32)
    s = size
    step = max(3, int(s * 0.45))
    half = s / 2.0
    paint = []

    def chain(run):
        # colour from the RAW reference under the chain footprint, not a
        # blurred point sample: blur(size/2) at a thin gap between dark
        # blocks mixes "hair + paper gap" into pale olive mud - paler
        # than the dark ground it lands on - and the repair stamped
        # visible pale dashes. The raw footprint mean can never be
        # lighter than the local average, so a chain on dark ground
        # stays dark; any midtone flattening it does cause is repaired
        # by the measured detail loop that runs after it.
        if len(run) == 1:
            # isolated single-sample gap: daub rejects one-point
            # strokes (engine pit) - extend the capsule a nudge.
            run.append([run[0][0] + s * 0.35, run[0][1], 1.0])
        xs = [p[0] for p in run]
        y = run[0][1]
        crop = raw[max(0, int(y - half)):int(y + half) + 1,
                   max(0, int(min(xs) - half)):int(max(xs) + half) + 1]
        if crop.size == 0:
            return
        rgb = crop.reshape(-1, 3).mean(axis=0)
        paint.append({"layer": "U%d" % s,
                      "preset": "b) Basic-2 Opacity",
                      "size": s, "opacity": op,
                      "color": "#%02x%02x%02x" % tuple(int(round(v)) for v in rgb),
                      "points": [[p[0], p[1], 1.0] for p in run]})

    for x0, y0, x1, y1 in regions:
        y = y0 + half
        while y < y1 - s // 2:
            run = []
            x = x0 + half
            while x < x1 - half + 1:
                rc = rpx[int(x), int(y)]
                cc = cpx[int(x), int(y)]
                if sum(abs(rc[j] - cc[j]) for j in range(3)) > thresh:
                    run.append([float(x), float(y)])
                else:
                    if run:
                        chain(run)
                    run = []
                x += step
            if run:
                chain(run)
            y += step
    print("plan_repair_fill: %d repair chains" % len(paint), flush=True)
    return paint


def plan_topup_mask(ref, holes, size=8, step=3, op=0.9, max_gap=6):
    """In-place hole patch for a FINISHED painting. The repair grid (and
    every coat before it) plans on its own scan lattice, so whatever the
    rounds leave uncovered sits in regular lanes - a visible grid of
    paper where the reference is not paper. This traces exactly those
    pixels: per-row runs of the boolean `holes` mask become short fine
    chains, coloured from the raw reference under the run - the liner
    brush pass over pinholes. Nothing else on the canvas is touched and
    no earlier stroke is re-planned. Pure planning: no I/O."""
    raw = np.asarray(ref, dtype=np.float32)
    h = holes.shape[0]
    paint = []
    claimed = np.zeros(holes.shape, dtype=bool)
    for y in range(h):
        row = holes[y]
        if not row.any():
            continue
        xs = np.flatnonzero(row)
        runs = []
        start = prev = int(xs[0])
        for x in xs[1:]:
            x = int(x)
            if x - prev <= max_gap:
                prev = x
            else:
                runs.append((start, prev))
                start = prev = x
        runs.append((start, prev))
        for xa, xb in runs:
            band = claimed[y:min(h, y + size // 2 + 1), xa:xb + 1]
            if band.size and band.mean() > 0.4:
                continue  # a chain laid on the rows above already covers
            pts = [[float(x), float(y) + 0.5, 1.0]
                   for x in range(xa, xb + 1, step)]
            if len(pts) == 1:
                pts.append([pts[0][0] + step, pts[0][1], 1.0])
            crop = raw[max(0, y - 4):y + 5, max(0, xa - 4):xb + 5]
            if crop.size == 0:
                continue
            rgb = crop.reshape(-1, 3).mean(axis=0)
            claimed[y:min(h, y + size // 2 + 1), max(0, xa - 2):xb + 3] = True
            paint.append({"layer": "UT", "preset": "b) Basic-2 Opacity",
                          "size": size, "opacity": op,
                          "color": "#%02x%02x%02x" % tuple(int(round(v)) for v in rgb),
                          "points": pts})
    print("plan_topup_mask: %d patch chains" % len(paint), flush=True)
    return paint


def _lerp_table(tbl, p):
    ps, vs = tbl["p"], tbl["v"]
    if p <= ps[0]:
        return vs[0]
    if p >= ps[-1]:
        return vs[-1]
    for i in range(1, len(ps)):
        if p <= ps[i]:
            u = (p - ps[i - 1]) / (ps[i] - ps[i - 1])
            return vs[i - 1] + u * (vs[i] - vs[i - 1])
    return vs[-1]


def _sim_draw(draw, s, preset, opacity, solid=False):
    """Stamp one stroke onto the simulated canvas with CALIBRATED ink.

    Real brushes deposit partial ink: coverage width is a fraction of the
    nominal size and one pass is translucent. Model both:
      width = nominal * w_cov_frac(pressure)   (coverage-equivalent)
      color = bg*(1-a) + color*a,  a = a_ctr(pressure) * opacity
    The blend treats the local canvas as bare background - right for the
    first coat, conservative for later ones; refill thresholds dominate
    either way, and the naive model (solid, full width) is what lied.

    solid=True: full-pressure full-coverage stamp (the F1 bed pre-stamp -
    that bed IS opaque by design, and blending its colour with the sim's
    paper tint left it pale, which the detail pass then paid for by
    doubling its refill).
    """
    pts = s["points"]
    wmax = s["size"]
    rgb = tuple(int(s["color"][j:j + 2], 16) for j in (1, 3, 5))
    cal = (_CALIB or {}).get("presets", {}).get(preset)
    if cal and not solid:
        bg = _CALIB.get("bg", [232, 238, 208])
        sizes = cal["sizes"]
        skey = min(sizes, key=lambda k: abs(float(k) - wmax))
    for i in range(len(pts) - 1):
        pm = (pts[i][2] + pts[i + 1][2]) / 2.0
        if cal and not solid:
            wf = _lerp_table({"p": sizes[skey]["p"], "v": sizes[skey]["w"]},
                             pm)
            a = min(1.0, _lerp_table(cal["alpha"], pm) * opacity)
        else:
            wf, a = pm, 1.0
        w = int(max(1, round(wmax * wf)))
        col = rgb if a >= 0.96 else tuple(
            int(bg[k] * (1 - a) + rgb[k] * a) for k in range(3))
        draw.line([(pts[i][0], pts[i][1]), (pts[i + 1][0], pts[i + 1][1])],
                  fill=col, width=w)
        r = w / 2.0
        for p in (pts[i], pts[i + 1]):
            draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r],
                         fill=col)


_GP = [None]  # hoisted gray pixel accessor (set in plan())
_GS = [0, 0]  # gray size


def _sobel(x, y):
    p = _GP[0]
    xi, yi = int(x), int(y)
    w, h = _GS
    if xi < 1 or yi < 1 or xi > w - 2 or yi > h - 2:
        return None
    a = p[xi - 1, yi - 1]; b = p[xi, yi - 1]; c = p[xi + 1, yi - 1]
    d = p[xi - 1, yi];                            e = p[xi + 1, yi]
    f = p[xi - 1, yi + 1]; g = p[xi, yi + 1]; hh = p[xi + 1, yi + 1]
    gx = (c + 2 * e + hh) - (a + 2 * d + f)
    gy = (f + 2 * g + hh) - (a + 2 * b + c)
    return gx, gy


def sobel_dir(x, y):
    gxy = _sobel(x, y)
    if gxy is None:
        return None
    gx, gy = gxy
    if abs(gx) + abs(gy) < 24:
        return None
    ln = math.hypot(gx, gy)
    return (-gy / ln, gx / ln)  # perpendicular to gradient


def sobel_mag(x, y):
    gxy = _sobel(x, y)
    if gxy is None:
        return 0.0
    return math.hypot(*gxy)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    cmd = sys.argv[1]
    if cmd == "plan":
        # plan REF OUT [--pen "d) Ink-3 Gpen"]   (--pen forces the ink
        # brush for every X1 mark; default = auto per-mark selection)
        pen = None
        pen_auto = True
        args = sys.argv[4:]
        if args and args[0] == "--pen" and len(args) > 1:
            pen = args[1]
            pen_auto = False
        sys.exit(plan(sys.argv[2], sys.argv[3], pen=pen, pen_auto=pen_auto))
    print(__doc__)
    return 1


if __name__ == "__main__":
    main()
