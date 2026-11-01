#!/usr/bin/env python
# coding: utf-8
"""# Blastema enrichment with the trained U-Net segmenter

The lightness-threshold proxy failed to detect the blastema, as was shown. The segmenter learned on the
cyan annotations, by contrast, detects it reliably -- Dice 0.857 -- and its proportion separates State_2
from State_3 with an AUC of 0.84 overall and 0.86 to 0.92 within each acquisition year. This notebook loads
that segmenter, blasteme_unet_384_best, and computes three measurements for every classifier image: the
proportion of blastema over the body, the dark edging at the pigmentation front, and a year-normalised
proportion, then adds them to the three metric tables.

Same geometry as the segmenter training: PCA orientation, zoom on both ends, the strongest-response end
retained, four-channel input (RGB plus local z), worm mask dilated to include the unpigmented blastema. The
proportion is the soft blastema area -- the sum of the probabilities -- divided by the worm area.

The normalised proportion corrects a year shift measured on the diagnostics: a State_2 scores 0.024 of
proportion in 2023 but 0.014 in 2024. The proportion is centred and scaled within each year, with the mean
and standard deviation learned on the training table alone and then applied to all three tables, which
avoids any leakage. The variable is defined by blast_seg_prop_yr, equation 1.

blast_seg_prop_yr = (blast_seg_prop - m_year) / s_year     equation 1

where m_year and s_year are the mean and standard deviation of blast_seg_prop within the acquisition year,
estimated on the training table. The year is read from the date prefix of the file name.

The last cell adds the new columns to the keeplist if their Pearson correlation stays below 0.90 against
the metrics already retained. If a segmenter column reproduces an older blastema proxy column above 0.90,
it replaces it, the segmenter being the validated measurement.

To be run from the FasterViT_for_Image_Classification root on GPU, after training the segmenter. Tables and
keeplist are backed up before modification, semicolon separator, seed 42.

Verified references, not clickable.
Ronneberger O., Fischer P., Brox T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.
arXiv:1505.04597. DOI 10.1007/978-3-319-24574-4_28.
Tan M., Le Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. arXiv:1905.11946.
"""

# ## 1. Segmenter loading, blastema measurements and year-normalised proportion
import os, glob, re, sys, shutil, datetime, numpy as np, pandas as pd, cv2, torch, torch.nn as nn
import segmentation_models_pytorch as smp
try: sys.stdout.reconfigure(line_buffering=True)
except Exception: pass
np.random.seed(42); torch.manual_seed(42)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch",torch.__version__,"| CUDA:",torch.cuda.is_available(),"| device:",device)
if device.type!="cuda": print("  WARNING: CPU, inference will be slow.")
REGEN=os.path.join("..","..","Dataset_stereotype_regeneration","dataset_training_regeneration")
IMG=os.path.join(REGEN,"images")
_mk=(sorted(glob.glob(os.path.join(REGEN,"masks_model_Mask2Former_*_2pass_clahe*")))
     or sorted(glob.glob(os.path.join(REGEN,"masks_model_Mask2Former_*_2pass*")))
     or sorted(glob.glob(os.path.join(REGEN,"masks_model_Mask2Former*"))))
MK=_mk[-1]; print("mask used:",os.path.basename(MK))
DN="dataset_dedup_State_50valid_30test"; METDIR=os.path.join(REGEN,"metrics_regeneration",DN)
CK=os.path.join("checkpoints_regeneration","blasteme_unet_384_complet","blasteme_unet_384_best.pth")
S=384; FRAC=0.45; MASK_THR=128
NM=[0.485,0.456,0.406,0.20]; NS=[0.229,0.224,0.225,0.15]
RAWCOLS=["blast_seg_prop","blast_seg_rim"]+["blast_forme_solidite","blast_forme_circularite","blast_forme_etendue","blast_forme_allongement","blast_forme_position","blast_forme_largeur","blast_forme_flou"]
NEWCOLS=RAWCOLS+["blast_seg_prop_yr"]

# ---- geometry identical to the segmenter ----
def worm_mask(m):
    m=(m>=MASK_THR).astype(np.uint8); n,l,st,_=cv2.connectedComponentsWithStats(m,8)
    return m if n<=1 else (l==1+int(np.argmax(st[1:,cv2.CC_STAT_AREA]))).astype(np.uint8)
def orient(bgr,mask):
    ys,xs=np.nonzero(mask); pts=np.stack([xs,ys],1).astype(np.float64); c=pts.mean(0)
    _,_,vt=np.linalg.svd(pts-c,full_matrices=False); ang=np.degrees(np.arctan2(vt[0][1],vt[0][0]))
    H,W=mask.shape; M=cv2.getRotationMatrix2D((float(c[0]),float(c[1])),ang-90,1.0)
    rb=cv2.warpAffine(bgr,M,(W,H)); rm=cv2.warpAffine((mask*255).astype(np.uint8),M,(W,H),flags=cv2.INTER_NEAREST)>127
    return rb,rm
def ends(rm):
    ys,xs=np.nonzero(rm); y0,y1=ys.min(),ys.max(); span=y1-y0+1
    yt=y0+int(0.12*span); rt=rm[yt]; xt=int(np.where(rt)[0].mean()) if rt.any() else rm.shape[1]//2
    yb=y1-int(0.12*span); rb=rm[yb]; xb=int(np.where(rb)[0].mean()) if rb.any() else rm.shape[1]//2
    return (xt,yt),(xb,yb),span
def crop_box(rbgr,rm,cx,cy,span):
    h=int(FRAC*span); H,W=rm.shape
    y0=max(0,int(cy)-h); y1=min(H,int(cy)+h); x0=max(0,int(cx)-h); x1=min(W,int(cx)+h)
    k=max(1,int(0.12*span)); rmz=cv2.dilate(rm.astype(np.uint8),np.ones((k,k),np.uint8)).astype(bool)  # includes the blastema
    g3=rbgr.copy(); g3[~rmz]=0; c=g3[y0:y1,x0:x1]
    if c.size==0: return None
    ch,cw=c.shape[:2]; side=max(ch,cw); sq=np.zeros((side,side,3),np.uint8); oy=(side-ch)//2; ox=(side-cw)//2
    sq[oy:oy+ch,ox:ox+cw]=c
    return cv2.resize(sq,(S,S)),(y0,y1,x0,x1,oy,ox,side)
def build_channels(im):
    g=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY).astype(np.float32)
    lm=cv2.blur(g,(21,21)); var=cv2.blur(g*g,(21,21))-lm*lm; ls=np.sqrt(np.maximum(var,0.0))+1e-3
    z=(lm-g)/ls; zc=np.clip((z+1.0)/6.0,0.0,1.0)
    return torch.from_numpy(np.concatenate([(im.astype(np.float32)/255.0).transpose(2,0,1),zc[None]],0).astype(np.float32))
class Backbone(nn.Module):
    def __init__(self,net):
        super().__init__(); self.net=net
        self.register_buffer("m",torch.tensor(NM).view(1,4,1,1)); self.register_buffer("s",torch.tensor(NS).view(1,4,1,1))
    def forward(self,x): return torch.sigmoid(self.net((x-self.m)/self.s))
net=smp.Unet(encoder_name="timm-efficientnet-b0",encoder_weights=None,in_channels=4,classes=1,activation=None)
model=Backbone(net).to(device)
_ck=torch.load(CK,map_location=device)
_sd={k:v for k,v in _ck["state_dict"].items() if not k.startswith("_enc.")}  # training duplicated the encoder under _enc, an alias to ignore
model.load_state_dict(_sd); model.eval()   # strict by default: the real net.*/m/s keys must all match
print("segmenter loaded:",CK,"| recorded validation Dice:",round(float(_ck.get("val_dice",0.0)),3))

def _predict_end(rbgr,rm,pt,span):
    cb=crop_box(rbgr,rm,pt[0],pt[1],span)
    if cb is None: return None
    hcrop,hbox=cb; side=hbox[-1]
    with torch.no_grad():
        p=model(build_channels(cv2.cvtColor(hcrop,cv2.COLOR_BGR2RGB))[None].to(device)).cpu()[0,0].numpy()
    scale=(side/float(S))**2
    return hcrop,p,float(p.sum())*scale
def darkrim(hcrop,pmask):
    m=pmask.astype(np.uint8)
    if m.sum()<10: return 0.0
    g=cv2.GaussianBlur(cv2.cvtColor(hcrop,cv2.COLOR_BGR2GRAY),(0,0),1.0).astype(np.float32)
    tissue=(hcrop.sum(2)>0); bodytis=tissue&(g<150)
    near=cv2.dilate(m,np.ones((3,3),np.uint8),iterations=4).astype(bool)
    rim=near&(~m.astype(bool))&tissue; far=bodytis&(~cv2.dilate(m,np.ones((3,3),np.uint8),iterations=12).astype(bool))
    if rim.sum()<20 or far.sum()<150: return 0.0
    return float(np.median(g[far]))-float(np.percentile(g[rim],10))
FORMECOLS=["blast_forme_solidite","blast_forme_circularite","blast_forme_etendue",
           "blast_forme_allongement","blast_forme_position","blast_forme_largeur",
           "blast_forme_flou"]
def forme_feats(hcrop,p):
    """Seven blastema shape descriptors, all of them DIMENSIONLESS ratios.

    No length in pixels is reported; each measurement is a ratio of two quantities of the same
    dimension, which makes it invariant to a change of resolution. The mask is
    reduced to its main connected component, any secondary blob being segmentation noise
    and not a second cap.
    """
    vide={k:0.0 for k in FORMECOLS}
    m=(p>=0.5).astype(np.uint8)
    if int(m.sum())<30: return vide
    n,lab,st,cen=cv2.connectedComponentsWithStats(m,8)
    if n<=1: return vide
    k=1+int(np.argmax(st[1:,cv2.CC_STAT_AREA])); mm=(lab==k).astype(np.uint8)
    aire=float(st[k,cv2.CC_STAT_AREA])
    cont,_=cv2.findContours(mm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not cont: return vide
    c=max(cont,key=cv2.contourArea); per=float(cv2.arcLength(c,True))
    # Solidity and circularity compare two quantities derived from the SAME contour. Mixing a
    # pixel count with a polygon area yields values above 1, the polygon area running through the
    # centres of the border pixels and not through their corners.
    aire_c=float(cv2.contourArea(c)); aire_h=float(cv2.contourArea(cv2.convexHull(c)))
    out=dict(vide)
    out["blast_forme_solidite"]=aire_c/aire_h if aire_h>0 else 0.0
    out["blast_forme_circularite"]=4.0*float(np.pi)*aire_c/(per*per) if per>0 else 0.0
    out["blast_forme_etendue"]=aire/float(max(int(st[k,cv2.CC_STAT_WIDTH])*int(st[k,cv2.CC_STAT_HEIGHT]),1))
    if len(c)>=5:
        (_,_),(l1,l2),_=cv2.minAreaRect(c)
        out["blast_forme_allongement"]=float(min(l1,l2))/float(max(l1,l2)) if max(l1,l2)>0 else 0.0
    ys,xs=np.nonzero(mm); cy=float(ys.mean())
    tissu=(hcrop.sum(2)>0); yt=np.nonzero(tissu.any(1))[0]
    tip=float(yt.min()) if len(yt) else 0.0
    out["blast_forme_position"]=(cy-tip)/float(np.sqrt(max(aire,1.0)))
    li=int(round(cy))
    if 0<=li<mm.shape[0]:
        lb=float(mm[li].sum()); lv=float(tissu[li].sum())
        out["blast_forme_largeur"]=lb/lv if lv>0 else 0.0
    env=cv2.dilate(mm,np.ones((3,3),np.uint8),iterations=3).astype(bool)
    out["blast_forme_flou"]=float(((p>0.2)&(p<0.8)&env).sum())/max(aire,1.0)
    return out
def seg_feats(bgr,mimg):
    wm=worm_mask(mimg); ba=int(wm.sum())
    if ba<50: return {k:0.0 for k in RAWCOLS}
    rbgr,rm=orient(bgr,wm); a,b,span=ends(rm); best=None
    for pt in (a,b):
        r=_predict_end(rbgr,rm,pt,span)
        if r is None: continue
        if best is None or r[2]>best[2]: best=r
    if best is None: return {k:0.0 for k in RAWCOLS}
    hcrop,p,soft=best
    _d=dict(blast_seg_prop=soft/ba, blast_seg_rim=darkrim(hcrop,(p>=0.5)))
    _d.update(forme_feats(hcrop,p))
    return _d

IMGIDX={}; MKIDX={}
for s in range(1,7):
    for p in glob.glob(os.path.join(IMG,"State_%d"%s,"*.jpg")): IMGIDX.setdefault(os.path.basename(p),p)
    for p in glob.glob(os.path.join(MK,"State_%d"%s,"*.jpg")): MKIDX.setdefault(os.path.basename(p),p)
print("indexed images:",len(IMGIDX))
CACHE={}
def gf(base):
    if base in CACHE: return CACHE[base]
    ip=IMGIDX.get(base); mp=MKIDX.get(base)
    if ip is None or mp is None: r={k:0.0 for k in RAWCOLS}
    else:
        bgr=cv2.imread(ip); mm=cv2.imread(mp,0); r=seg_feats(bgr,mm) if (bgr is not None and mm is not None) else {k:0.0 for k in RAWCOLS}
    CACHE[base]=r; return r
def year_of(base):
    mo=re.match(r"(\d{4})-",str(base)); return mo.group(1) if mo else "autre"

# ---- pass 1: raw measurements over the 3 tables, kept in memory ----
tabs={}
for sp in ("train","valid","test"):
    d=pd.read_csv(os.path.join(METDIR,f"{DN}_{sp}_regeneration_metrics_eyespig.csv"),sep=";")
    base=d["image"].astype(str).map(lambda s:os.path.basename(s))
    for k in RAWCOLS: d[k]=[gf(b).get(k,0.0) for b in base]
    d["_annee"]=base.map(year_of); tabs[sp]=d; print(f"{sp}: {len(d)} rows measured")

# ---- year statistics learned on the training table (equation 1), without leakage ----
tr=tabs["train"]; g=tr.groupby("_annee")["blast_seg_prop"]
mu=g.mean().to_dict(); sd=g.std(ddof=0).to_dict()
gm=float(tr["blast_seg_prop"].mean()); gs=float(tr["blast_seg_prop"].std(ddof=0)) or 1.0
print("mean proportion per year (train):",{k:round(v,4) for k,v in mu.items()})
def prop_yr(row):
    a=row["_annee"]; m=mu.get(a,gm); s=sd.get(a,gs); s=s if s and s>1e-9 else gs
    return (row["blast_seg_prop"]-m)/s

# ---- pass 2: add the normalised proportion and write, with a backup ----
stamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"); BK=os.path.join(METDIR,"_backup_blastseg_"+stamp); os.makedirs(BK,exist_ok=True)
for sp in ("train","valid","test"):
    d=tabs[sp]; d["blast_seg_prop_yr"]=d.apply(prop_yr,axis=1); d=d.drop(columns=["_annee"])
    p=os.path.join(METDIR,f"{DN}_{sp}_regeneration_metrics_eyespig.csv"); shutil.copyfile(p,os.path.join(BK,os.path.basename(p)))
    d.to_csv(p,sep=";",index=False); tabs[sp]=d; print(f"{sp}: written (+{len(NEWCOLS)} columns)")
print("backup:",os.path.basename(BK),"| columns added:",NEWCOLS)


# ## 2. Check: trend per state and AUC State_2 vs State_3 (raw and year-normalised)
from sklearn.metrics import roc_auc_score
d=tabs["train"]; c=("class" if "class" in d.columns else d.columns[1])
print("means per state (train):")
for st,g in d.groupby(c):
    print("  %s: proportion %.4f | year-normalised proportion %+.3f | rim %.1f"%(
        str(st),g["blast_seg_prop"].mean(),g["blast_seg_prop_yr"].mean(),g["blast_seg_rim"].mean()))
sub=d[d[c].astype(str).str.contains("State_2")|d[c].astype(str).str.contains("State_3")]
if len(sub)>20:
    y=sub[c].astype(str).str.contains("State_3").astype(int)
    for col in NEWCOLS:
        a=roc_auc_score(y,sub[col]); print("  AUC %-20s State_2 vs State_3 = %.3f"%(col,max(a,1-a)))
print("\nIf blast_seg_prop_yr beats blast_seg_prop on AUC, the year normalisation helps, and cell 3 will keep the better one.")


# ## 3. Keeplist update with a redundancy check (Pearson < 0.90)
# The three new segmenter columns form a family added together. Each is judged only against the
# metrics ALREADY retained, not against each other, otherwise the normalised proportion would be
# dropped as a duplicate of the raw proportion, when the point is to compare both in the classifier.
import numpy as np, pandas as pd, os, shutil, datetime
KP="metric_keeplist.txt"
keep=[l.strip() for l in open(KP)] if os.path.exists(KP) else []
tr=tabs["train"]; num=tr.select_dtypes(include=[np.number])
NEWFAM=["blast_seg_prop","blast_seg_prop_yr","blast_seg_rim"]
base_keep=[k for k in keep if k in num.columns and k not in NEWFAM]   # frozen reference: the existing keeplist
def top_corr(col):
    v=num[col].replace([np.inf,-np.inf],np.nan); best=("",0.0)
    for k in base_keep:
        w=num[k].replace([np.inf,-np.inf],np.nan); m=v.notna()&w.notna()
        if m.sum()<20 or v[m].std()<1e-9 or w[m].std()<1e-9: continue
        r=abs(float(np.corrcoef(v[m],w[m])[0,1]))
        if r>best[1]: best=(k,r)
    return best
PROXY={"blast_seg_prop":"blast_prop","blast_seg_rim":"blast_rim"}   # older proxy column that the segmenter version may replace
added=[]; replaced=[]; skipped=[]
for col in NEWFAM:
    if col in keep: skipped.append((col,"already present",0.0)); continue
    if col not in num.columns: skipped.append((col,"missing from the tables",0.0)); continue
    k,r=top_corr(col)
    if r<0.90:
        keep.append(col); added.append((col,k,r))
    else:
        px=PROXY.get(col)
        if px and px in keep and k==px:
            keep=[x for x in keep if x!=px]; keep.append(col); replaced.append((col,px,r))
        else:
            skipped.append((col,"redundant with %s"%k,r))
if added or replaced:
    shutil.copyfile(KP,KP+".backup_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")) if os.path.exists(KP) else None
    open(KP,"w").write("\n".join(keep)+"\n")
print("=== decisions keeplist ===")
for col,k,r in added: print("  ADDED    %-20s (max correlation %.2f with %s)"%(col,r,k))
for col,px,r in replaced: print("  REPLACED %-20s takes the place of proxy %s (correlation %.2f)"%(col,px,r))
for col,why,r in skipped: print("  IGNORE   %-20s %s%s"%(col,why,(" %.2f"%r) if r else ""))
print("\nkeeplist: %d metrics. Then run cross_validation_stn_uni_fix3_pruned again."%len(keep))

