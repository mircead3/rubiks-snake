#!/usr/bin/env python3
"""
Analyze primitive closed loops: period, bounding box, palindrome, chirality.

Usage:
  python3 analyze_loops.py simple_loops_14.txt
  python3 analyze_loops.py simple_loops_4-12.txt simple_loops_14.txt
"""

import sys, math, numpy as np

PI        = math.pi
INV_SQRT2 = 1.0 / math.sqrt(2)
_SWAP13   = (0, 3, 2, 1)

def _T(tx, ty, tz):
    m = np.eye(4); m[0,3]=tx; m[1,3]=ty; m[2,3]=tz; return m

def _build_joint(parity, d):
    theta = (2 - d) * PI / 2; c, s = math.cos(theta), math.sin(theta)
    Rx    = np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]], float)
    Ry    = np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]], float)
    Rx_pi = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]], float)
    Ry_pi = np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]], float)
    if parity == 0: return _T(0.5,0,0.5) @ Ry @ Rx_pi @ _T(-0.5,0,-0.5)
    else:           return _T(0,0.5,0.5) @ Rx @ Ry_pi @ _T(0,-0.5,-0.5)

JOINTS     = [[_build_joint(p, d) for d in range(4)] for p in range(2)]
INV_JOINTS = [[np.linalg.inv(JOINTS[p][d]) for d in range(4)] for p in range(2)]\

LOCAL_CORNERS = [
    [0,0,0],[1,0,0],[0,1,0],[1,1,0],
    [0,0,1],[1,0,1],[0,1,1],[1,1,1],
]

def _find_close_d(T, parity):
    for d in range(4):
        if np.allclose(T, INV_JOINTS[parity][d], atol=1e-5): return d
    return None

def full_seq(code_str):
    """Return the full N-digit tuple (N-1 stored digits + closing digit)."""
    digits = tuple(int(c) for c in code_str)
    T = np.eye(4)
    for i, d in enumerate(digits): T = T @ JOINTS[i % 2][d]
    close_d = _find_close_d(T, len(digits) % 2)
    if close_d is None: return None
    return digits + (close_d,)

# ─── Period ───────────────────────────────────────────────────────────────────

def minimal_period(seq):
    N = len(seq)
    for p in range(1, N + 1):
        if N % p == 0 and all(seq[i] == seq[i % p] for i in range(N)):
            return p
    return N

# ─── Palindrome ───────────────────────────────────────────────────────────────

def is_palindrome(seq):
    """True if some cyclic rotation of seq equals seq reversed."""
    N   = len(seq)
    rev = seq[::-1]
    return any(rev[k:] + rev[:k] == seq for k in range(N))

# ─── Chirality ────────────────────────────────────────────────────────────────

def is_chiral(seq):
    """True if the swap13 mirror is NOT achievable by rotation or reversal alone."""
    N   = len(seq)
    sw  = tuple(_SWAP13[x] for x in seq)
    rev = seq[::-1]
    rotations_fwd = {seq[k:] + seq[:k] for k in range(N)}
    rotations_rev = {rev[k:] + rev[:k] for k in range(N)}
    return sw not in rotations_fwd and sw not in rotations_rev

# ─── Bounding box ─────────────────────────────────────────────────────────────

def bounding_box_ratio(code_str):
    """Ratio of smallest to largest bounding box dimension (1 = perfect cube)."""
    digits = tuple(int(c) for c in code_str)
    T = np.eye(4)
    transforms = [T.copy()]
    for i, d in enumerate(digits[:-1]):   # all but closing joint
        T = T @ JOINTS[i % 2][d]
        transforms.append(T.copy())

    pts = []
    for Ti in transforms:
        R, t = Ti[:3, :3], Ti[:3, 3]
        for c in LOCAL_CORNERS:
            pts.append(R @ np.array(c, float) + t)
    pts = np.array(pts)
    dims = pts.max(axis=0) - pts.min(axis=0)
    dims = np.sort(dims)
    return float(dims[0] / dims[2]) if dims[2] > 1e-9 else 0.0

# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_file(path):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split()
            if len(parts) >= 2:
                entries.append((parts[0], parts[1]))
    return entries

def analyze(entries):
    results = []
    for name, code in entries:
        seq = full_seq(code)
        if seq is None: continue   # pseudo_loop or invalid
        period = minimal_period(seq)
        palin  = is_palindrome(seq)
        chiral = is_chiral(seq)
        ratio  = bounding_box_ratio(code)
        results.append((name, code, period, palin, chiral, ratio))
    return results

def main():
    files = sys.argv[1:]
    if not files:
        print("Usage: analyze_loops.py <file> [file ...]")
        sys.exit(1)

    all_entries = []
    for path in files:
        all_entries.extend(parse_file(path))

    print(f"Analyzing {len(all_entries)} loops...\n")
    results = analyze(all_entries)

    # Summary
    n_chiral   = sum(1 for r in results if r[4])
    n_palin    = sum(1 for r in results if r[3])
    print(f"Palindromes : {n_palin} / {len(results)}")
    print(f"Chiral      : {n_chiral} / {len(results)}")
    print()

    # Sort by period (asc), then bbox ratio (desc)
    results.sort(key=lambda r: (r[2], -r[5]))

    print(f"{'Name':<20} {'Code':<16} {'N':>3} {'Period':>6} {'Palin':>5} {'Chiral':>6} {'BBox':>6}")
    print("-" * 70)
    for name, code, period, palin, chiral, ratio in results:
        N = len(code) + 1
        print(f"{name:<20} {code:<16} {N:>3} {period:>6} {'yes' if palin else 'no':>5} "
              f"{'yes' if chiral else 'no':>6} {ratio:>6.3f}")

if __name__ == '__main__':
    main()
