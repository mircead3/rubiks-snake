#!/usr/bin/env python3
"""
Compose all Rubik's Snake pseudo loops pairwise.

Usage:
  python3 compose_pseudo.py              # print results to stdout
  python3 compose_pseudo.py out.txt      # write results to file

For each ordered pair (A, B) of pseudo loops (including A == B):
  - Try B in 4 variants: forward, reversed, swap13, reversed+swap13
  - Try all (d1, d2, d3) in {0,1,2,3}^3
  - Full explicit digit sequence: A_digits + [d1, d2] + B_variant + [d3]
  - If that sequence is closeable (closing digit cd exists), the result is a
    closed loop with N_A + N_B + 2 segments.
  - Check for self-intersections; output all valid non-intersecting results.
  - Deduplicate by canonical form (rotation, reversal, swap13).

Composition structure:
  A_last --(d1)--> bridge1 --(d2)--> B_first ... B_last --(d3)--> bridge2 --(cd)--> A_first
"""

import sys, itertools, math
import numpy as np

PI = math.pi
INV_SQRT2 = 1.0 / math.sqrt(2)

# ─── Joint transforms ──────────────────────────────────────────────────────────

def _T(tx, ty, tz):
    m = np.eye(4); m[0,3]=tx; m[1,3]=ty; m[2,3]=tz; return m

def _build_joint(parity, d):
    theta = (2 - d) * PI / 2
    c, s  = math.cos(theta), math.sin(theta)
    Rx    = np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]], float)
    Ry    = np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]], float)
    Rx_pi = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]], float)
    Ry_pi = np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]], float)
    if parity == 0: return _T(0.5,0,0.5) @ Ry @ Rx_pi @ _T(-0.5,0,-0.5)
    else:           return _T(0,0.5,0.5) @ Rx @ Ry_pi @ _T(0,-0.5,-0.5)

JOINTS     = [[_build_joint(p, d) for d in range(4)] for p in range(2)]
INV_JOINTS = [[np.linalg.inv(JOINTS[p][d]) for d in range(4)] for p in range(2)]

_SWAP13 = (0, 3, 2, 1)

# ─── Self-intersection ─────────────────────────────────────────────────────────

_SAMPLES_LOCAL = np.array([
    [0.20, 0.20, 0.50], [0.35, 0.10, 0.25], [0.10, 0.35, 0.25],
    [0.35, 0.10, 0.75], [0.10, 0.35, 0.75],
], float)
_EPS = 0.08

def _wsamples(T):
    return (_SAMPLES_LOCAL @ T[:3,:3].T) + T[:3,3]

def _intersects(ws_i, inv_j):
    p = (ws_i @ inv_j[:3,:3].T) + inv_j[:3,3]
    return bool(np.any(
        (p[:,0] > _EPS) & (p[:,1] > _EPS) & (p[:,2] > _EPS) &
        (p[:,2] < 1 - _EPS) & (p[:,0] + p[:,1] < 1 - _EPS)
    ))

def _find_close_d(T, parity):
    for d in range(4):
        if np.allclose(T, INV_JOINTS[parity][d], atol=1e-5): return d
    return None

# ─── Build transforms ─────────────────────────────────────────────────────────

def build_transforms(digits):
    """Return list of len(digits)+1 transforms (one per segment)."""
    T = np.eye(4)
    result = [T]
    for i, d in enumerate(digits):
        T = T @ JOINTS[i % 2][d]
        result.append(T)
    return result

# ─── Closed-loop self-intersection ────────────────────────────────────────────

def has_self_intersection(transforms):
    """Check if a closed loop self-intersects.

    transforms[0..N-1] = the N segment transforms.
    Closing joint wraps segment N-1 back to segment 0.
    Two segments are non-adjacent when their circular distance >= 3.
    """
    N   = len(transforms)
    ws  = [_wsamples(T) for T in transforms]
    inv = [np.linalg.inv(T) for T in transforms]

    for i in range(N):
        for j in range(i + 3, N):
            if min(j - i, N - (j - i)) < 3:
                continue
            if _intersects(ws[i], inv[j]) or _intersects(ws[j], inv[i]):
                return True
    return False

# ─── Pseudo loop data ─────────────────────────────────────────────────────────

PSEUDO_LOOPS = [
    ('loop7_1',  [1,3,1,0,1,2]),
    ('loop7_2',  [1,3,2,0,3,2]),
    ('loop9_1',  [1,0,1,2,0,0,3,1]),
    ('loop9_2',  [1,0,1,2,0,1,1,2]),
    ('loop9_3',  [1,0,1,2,3,3,3,2]),
    ('loop9_4',  [1,0,1,3,1,1,0,2]),
    ('loop9_5',  [1,0,2,1,3,0,0,2]),
    ('loop9_6',  [1,2,0,1,2,0,0,2]),
    ('loop9_7',  [1,2,3,0,0,1,3,1]),
    ('loop9_8',  [1,2,3,0,0,2,1,3]),
    ('loop9_9',  [1,2,3,0,3,3,1,2]),
    ('loop9_10', [1,2,3,2,0,2,3,1]),
    ('loop9_11', [1,2,3,2,0,3,1,3]),
    ('loop9_12', [1,2,3,2,1,1,3,2]),
    ('loop9_13', [1,3,0,3,2,3,0,2]),
    ('loop9_14', [1,3,3,1,2,1,0,2]),
]

def get_variants(digits):
    """Return 4 variants of a digit sequence with labels."""
    d  = list(digits)
    r  = d[::-1]
    s  = [_SWAP13[x] for x in d]
    rs = [_SWAP13[x] for x in r]
    return [('fwd', d), ('rev', r), ('sw13', s), ('rev+sw13', rs)]

# ─── Main ─────────────────────────────────────────────────────────────────────

def canonical_form(full_seq):
    N = len(full_seq)
    best = None
    rev    = full_seq[::-1]
    sw     = tuple(_SWAP13[x] for x in full_seq)
    rev_sw = tuple(_SWAP13[x] for x in rev)
    for base in (full_seq, rev, sw, rev_sw):
        for k in range(N):
            c = base[k:] + base[:k]
            if best is None or c < best: best = c
    return best


def main():
    total_tried = 0
    total_closeable = 0

    # canon → (nameA, nameB, vname, d1, d2, d3, cd, N_result, code)
    seen_canon = {}
    # (nameA_base, nameB_base) → set of canons  (use sorted pair for unordered)
    pair_canons = {}

    for nameA, digA in PSEUDO_LOOPS:
        for nameB, digB in PSEUDO_LOOPS:
            for vname, B_var in get_variants(digB):
                for d1, d2, d3 in itertools.product(range(4), repeat=3):
                    total_tried += 1
                    digits = digA + [d1, d2] + B_var + [d3]

                    T = np.eye(4)
                    for k, d in enumerate(digits):
                        T = T @ JOINTS[k % 2][d]
                    close_d = _find_close_d(T, len(digits) % 2)
                    if close_d is None:
                        continue
                    total_closeable += 1

                    transforms = build_transforms(digits)
                    if has_self_intersection(transforms):
                        continue

                    full_seq = tuple(digits) + (close_d,)
                    canon = canonical_form(full_seq)
                    N_result = len(digits) + 1
                    code = ''.join(map(str, digits))

                    if canon not in seen_canon:
                        seen_canon[canon] = (nameA, nameB, vname, d1, d2, d3, close_d, N_result, code)
                        pair_key = tuple(sorted([nameA, nameB]))
                        pair_canons.setdefault(pair_key, set()).add(canon)

    print(f"# Pseudo loop compositions (canonical representatives only)")
    print(f"# Tried: {total_tried:,}  Closeable: {total_closeable:,}  Distinct: {len(seen_canon)}")
    print()

    # Pair breakdown table
    loop_names = [name for name, _ in PSEUDO_LOOPS]
    print("# Pair breakdown (unordered pairs, count of distinct compositions):")
    print("#")
    header = f"# {'':12s}" + "".join(f"{n:>10s}" for n in loop_names)
    print(header)
    for nA in loop_names:
        row = f"# {nA:12s}"
        for nB in loop_names:
            key = tuple(sorted([nA, nB]))
            cnt = len(pair_canons.get(key, set()))
            row += f"{cnt:>10d}"
        print(row)
    print()

    # Group canonical results by segment count
    by_size = {}
    for canon, rec in seen_canon.items():
        sz = rec[7]
        by_size.setdefault(sz, []).append((canon, rec))

    import argparse, sys
    out_path = sys.argv[1] if len(sys.argv) > 1 else None
    lines = []

    for sz in sorted(by_size):
        group = sorted(by_size[sz], key=lambda x: x[1][8])  # sort by code
        lines.append(f"# ── {sz}-segment compositions ({len(group)} total) ────────────────────────────────")
        lines.append("")
        for idx, (canon, (nameA, nameB, vname, d1, d2, d3, cd, N_result, code)) in enumerate(group, 1):
            name = f"comp{sz}_{idx}"
            lines.append(f"# {name}: {nameA} + {nameB}")
            lines.append(f"{name} {code}({cd})")
            lines.append("")

    text = "\n".join(lines) + "\n"

    if out_path:
        with open(out_path, 'w') as f:
            f.write(text)
        print(f"Written {len(seen_canon)} compositions to {out_path}")
    else:
        print(text)
    print(f"# Done. {len(seen_canon)} distinct compositions.")

if __name__ == '__main__':
    main()
