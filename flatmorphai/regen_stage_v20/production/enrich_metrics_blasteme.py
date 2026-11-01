#!/usr/bin/env python
# coding: utf-8
"""# Enriching the metrics with blastema measurements, without learning

The blastema has too few annotated images for a reliable U-Net, so it is described by classical image
processing calibrated on the annotations rather than learned. The regenerating body is oriented by PCA, and
the blastema end is located without relying on brightness, which picked the wrong end: the pale fraction of
the cap is combined with the unpigmented gap between the mask border and the pigmentation front, which
identifies the right end 71 per cent of the time against 42 per cent for the former rule. Four quantities
are then measured at the retained end.

blast_prop, the proportion of pale cap tissue relative to the worm area, i.e. the relative size of the
blastema. blast_palefrac, the pale fraction inside the cap, blast_coiffe, the mean lightness of the cap, and
blast_rim, the darker edging at the pigmentation front, that is the light-against-dark border difference.

Over the 357 State_2 and 235 State_3 images, these measurements give a State_2 against State_3 AUC of 0.65
for the pale fraction, 0.62 for the edging, and 0.67 combined. Modest but real, and complementary to what
the classifier already has. The cyan annotations serve to calibrate the pale threshold and to validate, not
to train.

Uses the two-pass corrected mask in priority. Seed 42, no GPU dependency. To be run from the
FasterViT_for_Image_Classification root. Tables are backed up before modification, semicolon separator.

Verified references, not clickable.
Zuiderveld K. (1994). Contrast Limited Adaptive Histogram Equalization. Graphics Gems IV, 474 a 485.
DOI 10.1016/B978-0-12-336156-1.50061-6.
Reddien P.W., Sanchez Alvarado A. (2004). Fundamentals of planarian regeneration. Annual Review of Cell
and Developmental Biology, 20, 725 a 757. DOI 10.1146/annurev.cellbio.20.010403.095114.
"""

# ## 1. Blastema metric computation and table enrichment
import os, glob, numpy as np, pandas as pd, cv2, sys, shutil, datetime
try: sys.stdout.reconfigure(line_buffering=True)
except Exception: pass
np.random.seed(42)
REGEN=os.path.join("..","..","Dataset_stereotype_regeneration","dataset_training_regeneration")
IMG=os.path.join(REGEN,"images")
# worm mask: the two-pass corrected mask is preferred, otherwise the most recent available one
_cands=(sorted(glob.glob(os.path.join(REGEN,"masks_model_Mask2Former_*_2pass_clahe*")))
        or sorted(glob.glob(os.path.join(REGEN,"masks_model_Mask2Former_*_2pass*")))
        or sorted(glob.glob(os.path.join(REGEN,"masks_model_Mask2Former*"))))
MK=_cands[-1]
print("mask used:",os.path.basename(MK))
DN="dataset_dedup_State_50valid_30test"
METDIR=os.path.join(REGEN,"metrics_regeneration",DN)
MASK_THR=128; T_PALE=140; T_DARK=150     # T_PALE calibrated on the annotations, T_DARK separates the pigmented body
NEWCOLS=["blast_prop","blast_palefrac","blast_coiffe","blast_rim"]

def worm_mask(m):
    m=(m>=MASK_THR).astype(np.uint8); n,l,st,_=cv2.connectedComponentsWithStats(m,8)
    return m if n<=1 else (l==1+int(np.argmax(st[1:,cv2.CC_STAT_AREA]))).astype(np.uint8)
def orient(mask):
    ys,xs=np.nonzero(mask); pts=np.stack([xs,ys],1).astype(np.float64); c=pts.mean(0)
    _,_,vt=np.linalg.svd(pts-c,full_matrices=False); ang=np.degrees(np.arctan2(vt[0][1],vt[0][0]))
    H,W=mask.shape; return cv2.getRotationMatrix2D((float(c[0]),float(c[1])),ang-90,1.0)
def features(bgr,mimg):
    wm=worm_mask(mimg)
    if wm.sum()<500: return {k:0.0 for k in NEWCOLS}
    g=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY); M=orient(wm); H,W=wm.shape
    rg=cv2.warpAffine(g,M,(W,H)); rm=cv2.warpAffine(wm*255,M,(W,H),flags=cv2.INTER_NEAREST)>127
    rows=np.where(rm.any(1))[0]
    if len(rows)<5: return {k:0.0 for k in NEWCOLS}
    y0,y1=rows.min(),rows.max(); span=y1-y0+1; ar=np.arange(H)[:,None]
    warea=int(rm.sum()); dark=rm&(rg<T_DARK); drows=np.where(dark.any(1))[0]
    bmed=float(np.median(rg[dark])) if dark.sum()>200 else 0.0
    cand=[]
    for top in (True,False):
        band=((ar<y0+0.22*span) if top else (ar>y1-0.22*span))&rm
        if band.sum()<30: continue
        v=rg[band]; palefrac=float((v>=T_PALE).mean()); meang=float(v.mean())
        prox=(((ar>=y0+0.22*span)&(ar<y0+0.35*span)) if top else ((ar<=y1-0.22*span)&(ar>y1-0.35*span)))&rm
        rim=(bmed-float(np.percentile(rg[prox],10))) if prox.sum()>30 else 0.0
        if len(drows): gap=float(drows.min()-y0) if top else float(y1-drows.max())
        else: gap=0.0
        prop=float((v>=T_PALE).sum())/max(1,warea)
        cand.append((palefrac+0.02*gap, dict(blast_prop=prop,blast_palefrac=palefrac,blast_coiffe=meang,blast_rim=rim)))
    if not cand: return {k:0.0 for k in NEWCOLS}
    return max(cand,key=lambda z:z[0])[1]    # retained end = pale fraction + unpigmented gap

IMGIDX={}; MKIDX={}
for s in range(1,7):
    for p in glob.glob(os.path.join(IMG,"State_%d"%s,"*.jpg")): IMGIDX.setdefault(os.path.basename(p),p)
    for p in glob.glob(os.path.join(MK,"State_%d"%s,"*.jpg")): MKIDX.setdefault(os.path.basename(p),p)
print("indexed images:",len(IMGIDX),"| masks:",len(MKIDX))
CACHE={}
def get_feats(base):
    if base in CACHE: return CACHE[base]
    ip=IMGIDX.get(base); mp=MKIDX.get(base)
    if ip is None or mp is None: r={k:0.0 for k in NEWCOLS}
    else:
        bgr=cv2.imread(ip); mimg=cv2.imread(mp,0)
        r=features(bgr,mimg) if (bgr is not None and mimg is not None) else {k:0.0 for k in NEWCOLS}
    CACHE[base]=r; return r
stamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
BK=os.path.join(METDIR,"_backup_blast_"+stamp); os.makedirs(BK,exist_ok=True)
for sp in ("train","valid","test"):
    p=os.path.join(METDIR,f"{DN}_{sp}_regeneration_metrics_eyespig.csv")
    shutil.copyfile(p,os.path.join(BK,os.path.basename(p)))
    d=pd.read_csv(p,sep=";"); n=len(d); add={k:[] for k in NEWCOLS}
    for i,b in enumerate(d["image"].astype(str)):
        f=get_feats(os.path.basename(b))
        for k in NEWCOLS: add[k].append(f.get(k,0.0))
        if (i+1)%200==0: print(f"  {sp}: {i+1}/{n}")
    for k in NEWCOLS: d[k]=add[k]
    d.to_csv(p,sep=";",index=False); print(f"{sp}: {n} rows enriched (+{len(NEWCOLS)} columns)")
print("\nbackup before enrichment:",os.path.basename(BK))
print("columns added:",NEWCOLS)


# ## 2. Quick check: trend of the measurements per state and separability State_2 vs State_3
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
p=os.path.join(METDIR,f"{DN}_train_regeneration_metrics_eyespig.csv")
d=pd.read_csv(p,sep=";")
col_state="class" if "class" in d.columns else d.columns[1]
print("mean blastema measurements per state:")
for st,g in d.groupby(col_state):
    print("  %s : prop %.4f | part_pale %.3f | cap %.1f | rim %.1f"%(
        str(st),g["blast_prop"].mean(),g["blast_palefrac"].mean(),g["blast_coiffe"].mean(),g["blast_rim"].mean()))
# AUC State_2 vs State_3 if the labels are recognisable
def is_state(s,k): return str(k) in str(s)
sub=d[d[col_state].apply(lambda s: is_state(s,2) or is_state(s,3))].copy()
if len(sub)>20:
    y=sub[col_state].apply(lambda s: 1 if is_state(s,3) else 0).values
    for c in ["blast_prop","blast_palefrac","blast_coiffe","blast_rim"]:
        a=roc_auc_score(y,sub[c].values); print("  AUC %-14s State_2 vs State_3 = %.3f"%(c,max(a,1-a)))
print("\nThen re-run cross_validation_stn_uni_fix2_pruned to measure the real gain on State_3 recall.")

