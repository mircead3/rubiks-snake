#!/usr/bin/env python3
"""
Optimized search for primitive closed Rubik's Snake loops.

Optimizations applied:
  1. Fix first digit = 1  (every primitive loop with N>4 has a representative
     starting with 1; reduces DFS branching by 4x at the root)
  2. Incremental rectangular-face check  (prune the moment any non-adjacent
     pair of already-placed segments shares a rectangular face)
  3. Incremental self-intersection check  (prune as soon as any two
     already-placed non-adjacent segments physically overlap)
  4. Distance pruning  (if the current tip is farther from all 4 possible
     closing targets than MAX_STEP * remaining_joints, prune)
  5. Multiprocessing  (split work by 2nd and 3rd digits → 16 tasks)
"""

import math, sys, time, numpy as np
from multiprocessing import Pool, cpu_count, Lock

PI       = math.pi
INV_SQRT2 = 1.0 / math.sqrt(2)
MAX_STEP  = math.sqrt(2)   # max single-joint tip displacement


# ─── Joint transforms ─────────────────────────────────────────────────────────

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
INV_JOINTS = [[np.linalg.inv(JOINTS[p][d]) for d in range(4)] for p in range(2)]


# ─── Rectangular face data ────────────────────────────────────────────────────
# Prism faces (centre_local, outward_normal_local):

RECT_FACES_LOCAL = [
    (np.array([0.5, 0.0, 0.5]), np.array([ 0.0,       -1.0,       0.0])),
    (np.array([0.0, 0.5, 0.5]), np.array([-1.0,        0.0,       0.0])),
    (np.array([0.5, 0.5, 0.5]), np.array([ INV_SQRT2,  INV_SQRT2, 0.0])),
]

def _faces_world(T):
    R, t = T[:3,:3], T[:3,3]
    return [(R @ c + t, R @ n) for c, n in RECT_FACES_LOCAL]


# ─── Self-intersection data ───────────────────────────────────────────────────

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


# ─── Canonical form ───────────────────────────────────────────────────────────
# Dihedral D_N equivalences on the cyclic joint sequence:
#   forward, reverse, swap13, reverse+swap13.

_SWAP13 = (0, 3, 2, 1)

def canonical_form(full_seq):
    N      = len(full_seq)
    best   = None
    fwd    = full_seq
    rev    = full_seq[::-1]
    sw     = tuple(_SWAP13[x] for x in full_seq)
    rev_sw = tuple(_SWAP13[full_seq[N-1-i]] for i in range(N))
    for base in (fwd, rev, sw, rev_sw):
        for k in range(N):
            c = base[k:] + base[:k]
            if best is None or c < best: best = c
    return best


# ─── Existing simple_loops ────────────────────────────────────────────────────

SIMPLE_LOOPS_CODES = [
    "222",
    "13131","12312",
    "0123032","1232123",
    "001230012","001310031","001310113","002101303","002133113",
    "011320331","012012102","012302321","013120131","013120213",
    "00012300032","00101200303","00120031002","00120113302","00120120013",
    "00120120331","00120121112","00123002123","00123202303","00123302101",
    "00130013001","00130323013","00130323331","00130331101","00132023203",
    "00132031021","00132033023","00200200200","00200210203","00201210101",
    "01101201101","01101233013","01101233331","01101303303","01101311031",
    "01101311113","01123033113","01123203101","01123211013","01123211331",
    "01131013023","01131021201","01132302102","01132332303","01133113303",
    "01133121031","01133121113","01201213321","01210132023","01210203230",
    "01213313023","01230201230","01233231303","01233313203","01233321021",
    "01303101303","01303133113","01311323113","02123202321","11213311331",
    "11331133113",
]

def _find_close_d(T, parity):
    for d in range(4):
        if np.allclose(T, INV_JOINTS[parity][d], atol=1e-5): return d
    return None

def compute_existing_canonical():
    existing = set()
    for code_str in SIMPLE_LOOPS_CODES:
        digits = tuple(int(c) for c in code_str)
        T = np.eye(4)
        for i, d in enumerate(digits): T = T @ JOINTS[i%2][d]
        close_d = _find_close_d(T, len(digits) % 2)
        if close_d is not None:
            existing.add(canonical_form(digits + (close_d,)))
    return existing


# ─── DFS worker ───────────────────────────────────────────────────────────────

_write_lock  = None
_output_path = None

def _init_worker(lock, path):
    global _write_lock, _output_path
    _write_lock  = lock
    _output_path = path

def search_worker(args):
    N, prefix, existing_canonical = args

    close_par  = (N - 1) % 2
    target_pos = np.array([INV_JOINTS[close_par][d][:3, 3] for d in range(4)])  # (4,3)

    seen  = set()
    found = []

    # Per-depth cached data (index = segment index 0..N-1)
    T_stack  = [None] * N
    digits   = [0]    * (N - 1)
    faces_c  = [None] * N
    wsamples = [None] * N
    inv_T    = [None] * N

    # Segment 0 = identity
    I4 = np.eye(4)
    T_stack[0]  = I4
    faces_c[0]  = _faces_world(I4)
    wsamples[0] = _wsamples(I4)
    inv_T[0]    = I4

    # ── Apply and validate prefix ─────────────────────────────────────────────
    for i, d in enumerate(prefix):
        parity      = i % 2
        T_new       = T_stack[i] @ JOINTS[parity][d]
        nd          = i + 1
        T_stack[nd] = T_new
        digits[i]   = d
        fw          = _faces_world(T_new)
        ws          = _wsamples(T_new)
        iv          = np.linalg.inv(T_new)
        faces_c[nd] = fw; wsamples[nd] = ws; inv_T[nd] = iv

        # Face check: nd vs 0..nd-2  (non-adjacent = distance ≥ 2)
        for j in range(nd - 1):
            for ci, ni in fw:
                for cj, nj in faces_c[j]:
                    if np.allclose(ci, cj, atol=1e-5) and np.allclose(ni + nj, 0.0, atol=1e-5):
                        return found   # invalid prefix

        # Intersection check: nd vs 0..nd-3  (distance ≥ 3)
        for j in range(nd - 2):
            if _intersects(ws, inv_T[j]) or _intersects(wsamples[j], iv):
                return found           # invalid prefix

    start_depth = len(prefix)

    # ── Recursive DFS ─────────────────────────────────────────────────────────
    def dfs(depth):
        if depth == N - 1:
            close_d = _find_close_d(T_stack[depth], close_par)
            if close_d is None: return
            full_seq = tuple(digits) + (close_d,)
            canon    = canonical_form(full_seq)
            if canon not in seen:
                seen.add(canon)
                if canon not in existing_canonical:
                    code = ''.join(map(str, canon[:N-1]))
                    found.append(code)
                    with _write_lock:
                        with open(_output_path, 'a') as _f:
                            _f.write(code + '\n')
            return

        parity  = depth % 2
        is_last = (depth == N - 2)     # next step places the final segment

        for d in range(4):
            T_new = T_stack[depth] @ JOINTS[parity][d]
            nd    = depth + 1

            # ── Distance pruning ──────────────────────────────────────────────
            # After placing nd, we have (N-2-depth) joints left to reach N-1.
            remaining = N - 2 - depth
            dists = np.linalg.norm(target_pos - T_new[:3, 3], axis=1)
            if dists.min() > remaining * MAX_STEP + 1e-4:
                continue

            T_stack[nd] = T_new
            digits[depth] = d
            fw = _faces_world(T_new);   faces_c[nd]  = fw
            ws = _wsamples(T_new);      wsamples[nd] = ws
            iv = np.linalg.inv(T_new);  inv_T[nd]    = iv

            # ── Rectangular face check (incremental) ──────────────────────────
            # nd vs j in [j_start .. nd-2]  (non-adjacent: distance ≥ 2)
            # When placing the last segment, segment 0 becomes adjacent via the
            # closing joint → skip j=0.
            j0_face = 1 if is_last else 0
            bad = False
            for j in range(j0_face, nd - 1):
                for ci, ni in fw:
                    for cj, nj in faces_c[j]:
                        if np.allclose(ci, cj, atol=1e-5) and np.allclose(ni + nj, 0.0, atol=1e-5):
                            bad = True; break
                    if bad: break
                if bad: break
            if bad: continue

            # ── Self-intersection check (incremental) ─────────────────────────
            # nd vs j in [j0_si .. nd-3]  (distance ≥ 3)
            # When placing the last segment, segments 0 and 1 become circular-
            # distance 1 and 2 → skip them.
            j0_si = 2 if is_last else 0
            for j in range(j0_si, nd - 2):
                if _intersects(ws, inv_T[j]) or _intersects(wsamples[j], iv):
                    bad = True; break
            if bad: continue

            dfs(nd)

    dfs(start_depth)
    return found


# ─── Parallel search ──────────────────────────────────────────────────────────

def search_N(N, existing_canonical, output_path):
    n_cores = cpu_count()
    # Fix digit 0 = 1; split by digits 1 and 2 → 16 tasks
    tasks = [(N, (1, d1, d2), existing_canonical)
             for d1 in range(4) for d2 in range(4)]

    print(f"Searching N={N}  ({4**(N-1):,} raw candidates, "
          f"{len(tasks)} tasks on {n_cores} cores)...", file=sys.stderr)
    print(f"Writing raw results to {output_path} as found...", file=sys.stderr)

    # Truncate output file before starting
    open(output_path, 'w').close()

    lock = Lock()
    with Pool(n_cores, initializer=_init_worker, initargs=(lock, output_path)) as pool:
        results = pool.map(search_worker, tasks)

    # Deduplicate across workers by canonical form, sort, rewrite with names
    print(f"\nDeduplicating and sorting...", file=sys.stderr)
    raw_codes = []
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if line: raw_codes.append(line)

    all_seen = set()
    merged   = []
    for code_str in raw_codes:
        digits  = tuple(int(c) for c in code_str)
        T       = np.eye(4)
        for i, d in enumerate(digits): T = T @ JOINTS[i%2][d]
        close_d = _find_close_d(T, len(digits) % 2)
        canon   = canonical_form(digits + (close_d,))
        if canon not in all_seen:
            all_seen.add(canon)
            merged.append(code_str)

    return sorted(merged)


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Find missing primitive Rubik's Snake loops")
    ap.add_argument('N', type=int, nargs='?', default=14,
                    help='Number of segments to search (default: 14)')
    ap.add_argument('-o', '--output', default=None,
                    help='Output file (default: simple_loops_N.txt)')
    args = ap.parse_args()
    N           = args.N
    output_path = args.output or f'simple_loops_{N}.txt'

    print("Computing canonical forms of existing simple_loops...", file=sys.stderr)
    existing = compute_existing_canonical()
    print(f"  {len(existing)} existing entries.\n", file=sys.stderr)

    t0      = time.time()
    missing = search_N(N, existing, output_path)
    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s.", file=sys.stderr)

    # Starting index for new names (continuing from existing entries)
    next_idx = {8: 3, 10: 11, 12: 52, 14: 1}.get(N, 1)

    if missing:
        print(f"\n{len(missing)} primitive loop(s) found for N={N}.", file=sys.stderr)
        with open(output_path, 'w') as f:
            for i, code in enumerate(missing):
                f.write(f"loop{N}_{next_idx + i} {code}\n")
        print(f"Written to {output_path}", file=sys.stderr)
    else:
        open(output_path, 'w').close()
        print(f"\nNo primitive loops for N={N}.", file=sys.stderr)

if __name__ == '__main__':
    main()
