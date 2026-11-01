# twin

# %%== cell 1 ==
# make_motion_dataset -- precompute one cached "motion image" per frame into a NEW parallel folder, leaving
# the dataset untouched. Each file is a registration-aware temporal-RGB: B = aligned previous frame,
# G = current frame, R = aligned next frame. Neighbours are rigidly aligned to the centre by mask centroid
# + body axis first, so the stored motion is internal deformation, not the worm's gross glide.
# SPEED: frames are grouped by worm and each frame is decoded + posed exactly once (was up to 3x), workers
# run on all CPU cores, and PNG is written at low compression. Incremental: existing files are skipped.
import os, re, glob, numpy as np, pandas as pd, cv2
from concurrent.futures import ProcessPoolExecutor
from format_dataset_size import folder_name
cv2.setNumThreads(1)                      # avoid oversubscription: parallelism is across worms, not inside cv2
np.random.seed(42)
IMAGE_SIZE = 384
REGISTER   = True
N_WORKERS  = max(1, (os.cpu_count() or 4))
USE_PROCESSES = True                      # set False to fall back to threads if your runner dislikes fork
FORCE      = False
PNG_COMPRESSION = 1                       # 0..9, lower = faster write, bigger file (1 is a good tradeoff)
DATASET_NAME = os.environ.get('MOTION_DATASET') or os.environ.get('FLATMORPH_PRESET', 'dataset_repaired_wormsplit_relabel_v2_test30_chook_cdense8')   # overridden by predict_images for a 'predict' split, default unchanged
ROOT = os.environ.get('FLATMORPH_DATASET', '') or os.path.join('..', '..', 'Dataset_stereotype_regeneration', 'dataset_training_stereotype')
CSV_DIR = os.path.join(ROOT, 'dataset_images_list'); DDIR = os.path.join(CSV_DIR, DATASET_NAME)
proc_dir = folder_name(IMAGE_SIZE)
PROC = os.path.join(ROOT, proc_dir)
MOTION_DIR = os.path.join(ROOT, proc_dir + '_MOTION')
os.makedirs(MOTION_DIR, exist_ok=True)
FRE = re.compile(r'^(?P<set>.+)_[Ff]rame_(?P<n>\d+)_in_(?P<tot>\d+)$')
def parse(b):
    m = FRE.match(os.path.splitext(os.path.basename(b))[0]); return (m.group('set'), int(m.group('n')), int(m.group('tot'))) if m else None
def worm_key(rel):                         # group by class-folder + set id (one video)
    b = os.path.basename(rel); p = parse(b)
    return (os.path.dirname(rel), p[0], p[2]) if p else (os.path.dirname(rel), b, 0)
print('source:', PROC); print('output:', MOTION_DIR, '| register:', REGISTER, '| workers:', N_WORKERS)

# %%== cell 2 ==
# ---- pose + aligner (pure functions, picklable for multiprocessing) ----
def pose(gray):
    m = cv2.moments((gray > 8).astype(np.uint8), binaryImage=True)
    if m['m00'] < 50: return None
    cx = m['m10'] / m['m00']; cy = m['m01'] / m['m00']
    mu20 = m['mu20'] / m['m00']; mu02 = m['mu02'] / m['m00']; mu11 = m['mu11'] / m['m00']
    theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    ecc = ((mu20 - mu02) ** 2 + 4 * mu11 ** 2) ** 0.5 / (mu20 + mu02 + 1e-9)
    return cx, cy, float(theta), float(ecc)

def align_to(src_gray, src_pose, dst_pose, dst_gray):
    H, W = src_gray.shape
    scx, scy, sth, sec = src_pose; dcx, dcy, dth, dec = dst_pose
    angles = [0.0] if min(sec, dec) < 0.15 else [dth - sth, dth - sth + np.pi]
    cand = []
    for a in angles:
        M = cv2.getRotationMatrix2D((scx, scy), np.degrees(a), 1.0)
        M[0, 2] += (dcx - scx); M[1, 2] += (dcy - scy)
        cand.append(cv2.warpAffine(src_gray, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0))
    if len(cand) == 1: return cand[0]
    dm = dst_gray > 8
    sads = [float(np.abs((c - dst_gray)[dm]).mean()) if dm.any() else np.abs(c).mean() for c in cand]
    return cand[int(np.argmin(sads))]

def encode_motion(R, gt, B, worm):
    out = np.zeros((gt.shape[0], gt.shape[1], 3), np.uint8)
    out[..., 0] = (np.clip(R, 0, 255) * worm).astype(np.uint8)   # B = aligned previous
    out[..., 1] = (np.clip(gt, 0, 255) * worm).astype(np.uint8)  # G = current
    out[..., 2] = (np.clip(B, 0, 255) * worm).astype(np.uint8)   # R = aligned next
    return out

# %%== cell 3 ==
# ---- one task = one worm: decode + pose each frame ONCE, then build all its motion frames ----
def process_worm(args):
    cls_dir, frames = args     # frames: list of (n, src_path, dst_path)
    made = err = 0
    gray, pos = {}, {}
    def g(n, path):
        if n not in gray:
            img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
            gray[n] = None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return gray[n]
    def pz(n, path):
        if n not in pos:
            gg = g(n, path); pos[n] = None if gg is None else pose(gg)
        return pos[n]
    by_n = {n: (s, d) for n, s, d in frames}
    for n, src, dst in frames:
        try:
            if not FORCE and os.path.exists(dst): continue
            gt = g(n, src)
            if gt is None: err += 1; continue
            pt = pz(n, src)
            def prep(nn):
                sp = by_n.get(nn)
                if sp is None: return gt
                gn = g(nn, sp[0])
                if gn is None or pt is None: return gt
                pn = pz(nn, sp[0])
                if pn is None: return gt
                return align_to(gn, pn, pt, gt) if REGISTER else gn
            out = encode_motion(prep(n - 1), gt, prep(n + 1), gt > 8)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            cv2.imencode('.png', out, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])[1].tofile(dst)
            made += 1
        except Exception:
            err += 1
    return made, err

# build worm groups from the split CSVs (paths mirror the dataset exactly)
seen = set(); groups = {}
_MOTION_SPLITS = tuple(s for s in os.environ.get('MOTION_SPLITS', 'train,valid,test').split(',') if s)   # overridden by predict_images (e.g. 'predict'), default unchanged
for sp in _MOTION_SPLITS:
    f = next((x for x in glob.glob(os.path.join(DDIR, '*')) if x.endswith('_%s.csv' % sp)), None)
    if not f: continue
    for ip in pd.read_csv(f, sep=';')['image_path']:
        rel = str(ip)
        if rel in seen: continue
        seen.add(rel)
        src = os.path.join(PROC, rel)
        if not os.path.exists(src): continue
        p = parse(rel)
        if p is None: continue
        groups.setdefault(worm_key(rel), []).append((p[1], src, os.path.join(MOTION_DIR, rel)))
tasks = [(k[0], sorted(v)) for k, v in groups.items()]
n_frames = sum(len(v) for _, v in tasks)
print('worms:', len(tasks), '| frames:', n_frames)

# %%== cell 4 ==
import time
t0 = time.time(); made = err = 0
if USE_PROCESSES:
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for i, (m, e) in enumerate(ex.map(process_worm, tasks), 1):
            made += m; err += e
            if i % 100 == 0: print('  %d/%d worms | made %d ...' % (i, len(tasks), made), flush=True)
else:
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        for i, (m, e) in enumerate(ex.map(process_worm, tasks), 1):
            made += m; err += e
            if i % 100 == 0: print('  %d/%d worms | made %d ...' % (i, len(tasks), made), flush=True)
cached = sum(1 for _, v in tasks for _, _, d in v if os.path.exists(d))
print('\ncomputed: %d | errors: %d | total cached: %d / %d | %.1fs' % (made, err, cached, n_frames, time.time() - t0))
print('cached motion images at:', MOTION_DIR)
print('the temporal trainer loads these via the same image_path, just from the _MOTION folder.')
