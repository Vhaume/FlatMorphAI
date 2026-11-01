# -*- coding: utf-8 -*-

# %% [markdown]

# # New shape and luminosity metrics: bilateral asymmetry and interior bright peak
# 
# 	This notebook computes two new descriptors for train, valid, test and golden, and writes new_metrics_out/shape_lum_extra_<split>.csv, separator ';'. They separate the compact Walnut from the C-shape where global shape fails.
# 
# 	Metric 1, bilateral asymmetry of the width profile. Along the medial axis, the distance to the border is measured on the left and on the right of each point, and their difference is quantified. A curved C-shape is strongly left/right asymmetric (inner versus outer side of the fold), whereas a compact Walnut stays balanced. Columns bilat_asym_mean, bilat_asym_max, bilat_diff_std, bilat_diff_range. Measured AUC for Walnut versus C-shape is about 0.91.
# 
# 	Metric 2, contrast of the interior bright peak. In the deep interior of the body, outside the border band, we measure how much the brightest pixel stands out. A C-shape shows internal bright spots (translucent folds or the seam of the fold), whereas an opaque Walnut stays uniform. Columns bpeak_contrast, bpeak_frac. Measured AUC is about 0.74, and F18 falls on the Walnut side.
# 
# 	Reproducibility, random_state 42. CLAHE, Zuiderveld, Graphics Gems IV, 1994, DOI 10.1016/B978-0-12-336156-1.50061-6. Morphological skeleton, Zhang & Suen, CACM 27(3), 1984, DOI 10.1145/357994.358023. Width profile along the medial axis, Stephens, Johnson-Kerner, Bialek & Ryu, Dimensionality and Dynamics in the Behavior of C. elegans, PLoS Computational Biology 4(4), 2008, DOI 10.1371/journal.pcbi.1000028. Ridge detection, Frangi et al., Multiscale vessel enhancement filtering, MICCAI 1998, DOI 10.1007/BFb0056195.

# %%

# Cell 1: imports, robust paths (Windows and WSL), image and mask indexes
import os, re, glob, sys, numpy as np, pandas as pd
np.random.seed(42)
# Plain imports. This file used to carry an ``_ensure`` helper that ran
# ``pip install`` at import time (cv2 -> opencv-python-headless). That is a
# silent installer: the application loads this module through
# ``lightgbm_bundle._load_funcs_from_source``, which KEEPS every top-level
# assignment, so the call really did run -- with stdout and stderr redirected,
# so nothing would have been seen. On a machine without OpenCV it would have
# installed a DIFFERENT OpenCV from the pinned opencv-python==5.0.0.93, and
# OpenCV is calculation-sensitive: the descriptor values can move. A missing
# dependency must fail loudly instead; the caller already treats this family as
# optional and leaves its columns NaN.
import cv2
from skimage.morphology import skeletonize
from collections import deque
def _find(*cands):
    """First candidate directory that exists; otherwise the first non-empty one."""
    for c in cands:
        if not c: continue
        if os.name!='nt' and re.match(r'^[A-Za-z]:[\\/]',c): continue
        if os.path.isdir(c): return c
    return next((c for c in cands if c), '')
# Roots derived from this file and from the environment only. The paths that
# used to sit here were absolute paths on one developer's machine; none of them
# existed anywhere else, so they only ever served as the value returned when
# nothing was found.
HERE=_find(os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '',
           os.getcwd())
DS=_find(os.environ.get('FLATMORPH_DATASET', ''),
         os.path.join(HERE, 'dataset_training_stereotype'))
O=os.path.join(HERE,'analysis_out'); NMO=os.path.join(HERE,'new_metrics_out'); os.makedirs(NMO,exist_ok=True)
MD=os.path.join(DS,'mask_model_Mask2Former_20260324_0451')
PRESET=os.environ.get('FLATMORPH_PRESET', 'dataset_repaired_wormsplit_relabel_v2_test30_chook_cdense8')
B=os.path.join(DS,'dataset_images_list',PRESET,PRESET+'_')
MASK={os.path.basename(p):p for p in glob.glob(os.path.join(MD,'*.png'))}
IMG={}
for p in glob.glob(os.path.join(DS,'images','**','*.png'),recursive=True): IMG.setdefault(os.path.basename(p),p)
clahe=cv2.createCLAHE(clipLimit=4.0,tileGridSize=(8,8))
print('masks',len(MASK),'| images',len(IMG))

# %%

# Cell 2: metric 1, bilateral asymmetry of the width profile (medial axis + perpendicular rays)
def _longest_path(sk):
    pts=set(map(tuple,np.argwhere(sk)))
    if not pts: return []
    nb=lambda y,x:[(y+dy,x+dx) for dy in(-1,0,1) for dx in(-1,0,1) if (dy or dx) and (y+dy,x+dx) in pts]
    def bfs(s):
        seen={s:None}; dq=deque([s]); c=s
        while dq:
            c=dq.popleft()
            for n in nb(*c):
                if n not in seen: seen[n]=c; dq.append(n)
        return c,seen
    a,_=bfs(next(iter(pts))); b,seen=bfs(a); path=[]; c=b
    while c is not None: path.append(c); c=seen[c]
    return path[::-1]
BIL_COLS=['bilat_asym_mean','bilat_asym_max','bilat_diff_std','bilat_diff_range']
def bilateral(mask):
    b=(mask>0).astype(np.uint8)
    if b.sum()<50: return None
    sk=skeletonize(b>0); path=_longest_path(sk)
    if len(path)<15: return None
    path=np.array(path,float); N=len(path); asy=[]; dl=[]
    for i in range(2,N-2):
        t=path[i+2]-path[i-2]; nr=np.hypot(*t)
        if nr<1e-6: continue
        t/=nr; n=np.array([-t[1],t[0]]); y0,x0=path[i]
        def march(s):
            for r in range(1,60):
                yy=int(round(y0+s*r*n[0])); xx=int(round(x0+s*r*n[1]))
                if yy<0 or xx<0 or yy>=b.shape[0] or xx>=b.shape[1] or b[yy,xx]==0: return r-1
            return 59
        dp=march(1); dm=march(-1)
        if dp+dm<1: continue
        asy.append(abs(dp-dm)/(dp+dm+1e-6)); dl.append(dp-dm)
    if len(asy)<5: return None
    dl=np.array(dl)
    return dict(bilat_asym_mean=float(np.mean(asy)),bilat_asym_max=float(np.max(asy)),
               bilat_diff_std=float(np.std(dl)),bilat_diff_range=float(dl.max()-dl.min()))

# %%

# Cell 3: metric 2, contrast of the bright peak in the deep interior (CLAHE + mask erosion)
BPK_COLS=['bpeak_contrast','bpeak_frac']
def brightpeak(g,mask):
    mm=(mask>0)
    if mm.sum()<50: return None
    gi=clahe.apply(g).astype(float)
    rad=int(max(3,0.15*np.sqrt(mm.sum())))
    ker=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*rad+1,2*rad+1))
    inner=cv2.erode(mm.astype(np.uint8),ker).astype(bool)
    if inner.sum()<30: inner=mm
    vals=gi[inner]; medv=np.median(vals); sd=vals.std()+1e-6
    span=np.percentile(gi[mm],99)-np.percentile(gi[mm],1)+1e-6
    return dict(bpeak_contrast=float((np.percentile(vals,99)-medv)/span),
               bpeak_frac=float((vals>medv+1.5*sd).mean()))
def features(bn):
    mp=MASK.get(bn); ip=IMG.get(bn)
    if mp is None or ip is None: return {k:np.nan for k in BIL_COLS+BPK_COLS}
    m=cv2.imdecode(np.fromfile(mp,np.uint8),0); g=cv2.imdecode(np.fromfile(ip,np.uint8),0)
    if m is None or g is None: return {k:np.nan for k in BIL_COLS+BPK_COLS}
    if g.shape!=m.shape: g=cv2.resize(g,(m.shape[1],m.shape[0]))
    ys,xs=np.where(m>0)
    if len(ys)<50: return {k:np.nan for k in BIL_COLS+BPK_COLS}
    mr=8; y0,y1=max(0,ys.min()-mr),ys.max()+mr; x0,x1=max(0,xs.min()-mr),xs.max()+mr
    mc=m[y0:y1,x0:x1]; gc=g[y0:y1,x0:x1]
    out={k:np.nan for k in BIL_COLS+BPK_COLS}
    rb=bilateral(mc);   out.update(rb or {})
    rp=brightpeak(gc,mc); out.update(rp or {})
    return out

# %%

# Cell 4: computation for all splits, writes new_metrics_out/shape_lum_extra_<split>.csv (sep ';')
def run(split, gold=False):
    src=os.path.join(O,'golden_metrics.csv') if gold else B+split+'_mask3_metrics.csv'
    d=pd.read_csv(src,sep=';'); d['image']=[os.path.basename(str(x)) for x in d['image']]
    rows=[]
    for i,bn in enumerate(d['image']):
        rows.append({'image':bn,**features(bn)})
        if i%400==0: print('  %s %d/%d'%(split,i,len(d)),flush=True)
    out=pd.DataFrame(rows)
    path=os.path.join(NMO,'shape_lum_extra_%s.csv'%split)
    out.to_csv(path,sep=';',index=False)
    print('-> %s (%d rows, coverage %.2f)'%(path,len(out),out['bilat_asym_mean'].notna().mean()),flush=True)
for sp in [s for s in os.environ.get('FLATMORPH_SPLITS', 'train,valid,test').split(',') if s]: run(sp)   # FLATMORPH_SPLITS, default train,valid,test
print('DONE')
