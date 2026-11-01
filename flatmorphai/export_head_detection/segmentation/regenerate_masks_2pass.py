#!/usr/bin/env python
# coding: utf-8
"""# Regeneration mask correction, two-pass Mask2Former segmentation with calibrated enhancement

This notebook rebuilds the masks of the regeneration set with FlatMorphAI's two-pass path,
first pass over the worm box, second pass over a zoomed crop, fusion guided by the skeleton. It loads the
model through load_model_for_segmentation with the checkpoint model_Mask2Former_20260324_0451 and reuses the
internal functions of evaluate_test_images_v2 with the same default parameters.

Target problem. The unpigmented blastema is almost as pale as the dish background, median grey 181
against 169, so Mask2Former stops at the edge of the pigmented body. Without retraining the model, two
inference levers are used. First, local contrast enhancement before inference. Then, and above all,
lowering the threshold of the worm class at post-processing. Instead of the usual argmax, the per-pixel
worm probability is rebuilt from the model logits and its threshold lowered, which includes the pale
pixels Mask2Former already leans towards classifying as worm without their winning. Since the model has
already been trained on blastemas, it gives them a non-zero probability, and this threshold recovers
them. The second cell calibrates both levers on the cyan-annotated images and retains the setting that
maximises blastema inclusion without inflating the mask towards the background.

To be run from the FasterViT_for_Image_Classification root on GPU. Seed 42.

Downstream warning. Several scripts fetch the mask by the glob masks_model_Mask2Former* and then take the
first result. With two folders present that choice becomes ambiguous. After validation, point the
downstream pipelines at the retained folder or replace the old one.

Verified references, not clickable.
Cheng B., Misra I., Schwing A.G., Kirillov A., Girdhar R. (2022). Masked-attention Mask Transformer for
Universal Image Segmentation. CVPR 2022. arXiv:2112.01527.
Liu Z., Lin Y., Cao Y., Hu H., Wei Y., Zhang Z., Lin S., Guo B. (2021). Swin Transformer: Hierarchical
Vision Transformer using Shifted Windows. ICCV 2021. arXiv:2103.14030.
Zuiderveld K. (1994). Contrast Limited Adaptive Histogram Equalization. Graphics Gems IV, 474 a 485.
DOI 10.1016/B978-0-12-336156-1.50061-6.
"""

# ## 1. Model loading, contrast enhancement, two-pass pipeline
import os, sys, time, glob, numpy as np, cv2, torch
from PIL import Image
np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
REPO=os.getcwd()
if REPO not in sys.path: sys.path.insert(0,REPO)
assert os.path.isdir(os.path.join(REPO,"checkpoints_segmentation_model")), \
    "Run from the FasterViT_for_Image_Classification root"
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch",torch.__version__,"| CUDA dispo:",torch.cuda.is_available(),"| device:",device)
if torch.cuda.is_available(): print("  GPU:",torch.cuda.get_device_name(0))
else: print("  WARNING: CPU, segmentation is very slow. Ask for a GPU node.")

import torch.nn.functional as F
from PIL import Image as PILImage
from flatmorphai.notebook_api.segmentation import load_model_for_segmentation
from evaluate_test_images_v2 import (_predict_batch_gpu,_extract_p2_jobs,_paste_p2_result,
                                     _apply_fusion,_keep_largest_component,_ensure_uint8_rgb,
                                     _maybe_suppress_stdout)
MODEL_NAME="model_Mask2Former_20260324_0451"
model,processor=load_model_for_segmentation(MODEL_NAME,device)
mdev=next(model.parameters()).device; print("model on:",mdev); assert mdev.type==device.type

def predict_prob_batch(images, worm_thr=0.5):
    # Semantic post-processing by probability instead of argmax, to lower the worm class threshold.
    # Reproduces the computation of post_process_semantic_segmentation: per-pixel class probability obtained by
    # sum over the queries of softmax(class logits) times sigmoid(mask logits). At worm_thr=0.5 the
    # result is identical to the original argmax. Lowering worm_thr includes the borderline pale pixels.
    if not images: return []
    sizes=[(im.shape[0],im.shape[1]) for im in images]
    inp=processor(images=[PILImage.fromarray(im) for im in images],return_tensors="pt")
    inp={k:v.to(device) for k,v in inp.items()}
    with torch.no_grad(): outputs=model(**inp)
    mql=outputs.masks_queries_logits if hasattr(outputs,"masks_queries_logits") else outputs["masks_queries_logits"]
    cql=outputs.class_queries_logits if hasattr(outputs,"class_queries_logits") else outputs["class_queries_logits"]
    masks=mql.sigmoid(); classes=cql.softmax(-1)[...,:-1]          # (B,Q,C), the no-object class is removed
    seg=torch.einsum("bqc,bqhw->bchw",classes,masks)              # (B,C,h,w)
    out=[]
    for i,(H,W) in enumerate(sizes):
        s=F.interpolate(seg[i:i+1].float(),size=(H,W),mode="bilinear",align_corners=False)[0]  # (C,H,W)
        p=s/(s.sum(0,keepdim=True)+1e-6); fg=p[1:].sum(0)          # probability of being worm, non-background classes
        out.append((fg>=worm_thr).cpu().numpy().astype(np.uint8)*255)
    return out

# two-pass parameters, identical to the defaults of forecast_mask_with_a_mask2former
BATCH=8; MODEL_SIZE=320; PAD_RATIO=0.50; MERGE="skeleton_guided"
MIN_COMP=10; MIN_PAD=32; MIN_ROI=0.0; BND_DIL=0.02; MERGE_DIL_R=None; MERGE_DIL_F=0.05; KEEP_LARGEST=True

def enhance(img, mode):
    # local contrast enhancement to reveal the pale blastema against the uniform background.
    # CLAHE on the L channel (lightness) brings out the texture of the regenerating tissue without shifting the colours.
    # mode: "none" | "clahe_<clip>" | "clahe_<clip>_t<tile>" | append "_sat" to saturate the colours.
    # A large tile (t16) is gentler and amplifies the background less; a small tile is more local.
    if mode=="none": return img
    parts=mode.split("_")
    clip=3.0; tile=8
    for p in parts[1:]:
        if p.startswith("t") and p[1:].isdigit(): tile=int(p[1:])
        elif p.replace('.','',1).isdigit(): clip=float(p)
    lab=cv2.cvtColor(img,cv2.COLOR_RGB2LAB); L,A,B=cv2.split(lab)
    if parts[0]=="clahe":
        L=cv2.createCLAHE(clipLimit=clip,tileGridSize=(tile,tile)).apply(L)
    out=cv2.cvtColor(cv2.merge([L,A,B]),cv2.COLOR_LAB2RGB)
    if mode.endswith("sat"):
        hsv=cv2.cvtColor(out,cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[...,1]=np.clip(hsv[...,1]*1.4,0,255); out=cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2RGB)
    return out

def two_pass_batch(batch_np, mode="none", worm_thr=0.5):
    batch_np=[enhance(im,mode) for im in batch_np]
    n=len(batch_np)
    p1=predict_prob_batch(batch_np, worm_thr)
    all_jobs=[]; jobs_per=[[] for _ in range(n)]
    for i in range(n):
        j=_extract_p2_jobs(batch_np[i],p1[i],i,PAD_RATIO,MODEL_SIZE,MIN_COMP,MIN_PAD,MIN_ROI)
        all_jobs.extend(j); jobs_per[i]=j
    p2res=[]
    if all_jobs:
        crops=[j["crop"] for j in all_jobs]
        for s in range(0,len(crops),BATCH):
            p2res.extend(predict_prob_batch(crops[s:s+BATCH], worm_thr))
    p2full=[np.zeros(batch_np[i].shape[:2],np.uint8) for i in range(n)]
    for job,cm in zip(all_jobs,p2res): _paste_p2_result(cm,p2full[job["image_idx"]],job)
    for i in range(n):
        if p2full[i].sum()==0 and p1[i].sum()>0: p2full[i]=p1[i].copy()
    out=[]
    for i in range(n):
        cs=jobs_per[i][0]["crop"].shape if jobs_per[i] else None
        with _maybe_suppress_stdout(False):
            fused=_apply_fusion(p1[i],p2full[i],MERGE,BND_DIL,cs,False,MERGE_DIL_R,MERGE_DIL_F)
        if KEEP_LARGEST:
            with _maybe_suppress_stdout(False): fused,_=_keep_largest_component(fused,k=255)
        out.append(fused)
    return out

REGEN=os.path.join("..","..","Dataset_stereotype_regeneration","dataset_training_regeneration")
IMG=os.path.join(REGEN,"images"); CB=os.path.join(REGEN,"cyan_blasteme")
def list_imgs(d): return [n for n in sorted(os.listdir(d)) if n.lower().endswith((".jpg",".jpeg",".png"))]
print("ready. Enhancements: none, clahe_2, clahe_3, clahe_4, clahe_2_t16, clahe_3_sat | worm threshold adjustable")


# ## 2. Enhancement calibration on the cyan annotations (State_2 and State_4)
# For each setting, the share of the cyan blastema included in the two-pass mask is measured.
def worm_mask(m):
    m=(m>=128).astype(np.uint8); n,l,st,_=cv2.connectedComponentsWithStats(m,8)
    return m if n<=1 else (l==1+int(np.argmax(st[1:,cv2.CC_STAT_AREA]))).astype(np.uint8)
def cyanmask(im):
    b,g,r=im[...,0].astype(int),im[...,1].astype(int),im[...,2].astype(int)
    return ((b>110)&(g>110)&((b-r)>60)&((g-r)>50)).astype(np.uint8)
MODES=["none","clahe_2","clahe_3","clahe_4","clahe_2_t16","clahe_3_sat"]
THRS=[0.50,0.45,0.40,0.35,0.30]      # worm class threshold, 0.50 = original argmax
# annotated sample (cyan): State_2 (subtle blastema) and State_4 (marked blastema)
samp=[]
for st in (2,4):
    for cp in sorted(glob.glob(os.path.join(CB,"State_%d"%st,"*.jpg"))):
        nm=os.path.basename(cp); ip=os.path.join(IMG,"State_%d"%st,nm)
        if not os.path.isfile(ip): continue
        cym=cyanmask(cv2.imread(cp))
        if cym.sum()<150: continue
        samp.append((st,nm,ip,cym))
print("annotated images for the calibration:",len(samp))
def eval_setting(mode,thr):
    incl={2:[],4:[]}; areas=[]
    for s in range(0,len(samp),BATCH):
        chunk=samp[s:s+BATCH]
        imgs=[_ensure_uint8_rgb(cv2.imdecode(np.fromfile(ip,dtype=np.uint8),cv2.IMREAD_UNCHANGED)) for _,_,ip,_ in chunk]
        masks=two_pass_batch(imgs, mode, thr)
        for (st,nm,ip,cym),mk in zip(chunk,masks):
            wm=worm_mask(mk); tot=int(cym.sum())
            incl[st].append(100.0*int((cym.astype(bool)&wm.astype(bool)).sum())/tot); areas.append(int(wm.sum()))
    return np.mean(incl[2]),np.mean(incl[4]),np.mean(areas)
base=eval_setting("none",0.50); print("reference none, threshold 0.50: State_2 %.1f%% | State_4 %.1f%% | area %d"%base)
# step 1: best enhancement at threshold 0.50
print("\n-- enhancement sweep (threshold 0.50) --")
resC={"none":base}
for mode in MODES:
    if mode=="none": continue
    resC[mode]=eval_setting(mode,0.50)
    print("  %-12s : State_2 %.1f%% | State_4 %.1f%% | aire %d"%(mode,resC[mode][0],resC[mode][1],resC[mode][2]))
candC={m:v for m,v in resC.items() if v[2]<=1.5*base[2]}
ENHANCE_CHOICE=max(candC,key=lambda m:candC[m][0])
print("enhancement retained:",ENHANCE_CHOICE)
# step 2: lowering the worm class threshold, at the retained enhancement
print("\n-- sweep of the worm class threshold (enhancement %s) --"%ENHANCE_CHOICE)
resT={}
for thr in THRS:
    resT[thr]=eval_setting(ENHANCE_CHOICE,thr)
    print("  threshold %.2f: State_2 %.1f%% | State_4 %.1f%% | area %d"%(thr,resT[thr][0],resT[thr][1],resT[thr][2]))
base_area=resT[0.50][2]
candT={t:v for t,v in resT.items() if v[2]<=1.5*base_area}   # discards a threshold that inflates the mask (leak towards the background)
WORM_THR_CHOICE=max(candT,key=lambda t:candT[t][0])
print("\nfinal setting retained: enhancement %s | worm threshold %.2f"%(ENHANCE_CHOICE,WORM_THR_CHOICE))
print("  inclusion State_2 %.1f%% (against %.1f%% argmax without enhancement) | State_4 %.1f%%"%(
    resT[WORM_THR_CHOICE][0],base[0],resT[WORM_THR_CHOICE][1]))


# ## 3. Full pass over the whole regeneration set with the retained setting
ENHANCE=ENHANCE_CHOICE          # or force it by hand, for example "clahe_2_t16"
WORM_THR=WORM_THR_CHOICE        # or force it, for example 0.40
OUT=os.path.join(REGEN,"masks_model_Mask2Former_20260324_0451_2pass_%s_thr%02d"%(ENHANCE,int(WORM_THR*100)))
STATES=["State_%d"%s for s in range(1,7)]
MAX_PER_STATE=None    # None = everything; 8 for a quick trial
OVERWRITE=False
print("enhancement:",ENHANCE,"| worm threshold:",WORM_THR,"| output:",OUT)
t0=time.time(); ntot=0; nempty=0
for st in STATES:
    src=os.path.join(IMG,st); dst=os.path.join(OUT,st); os.makedirs(dst,exist_ok=True)
    names=list_imgs(src)                                   # lists images only, ignores sub-folders (e.g. _to_delete)
    if MAX_PER_STATE: names=names[:MAX_PER_STATE]
    print("### %s : %d images"%(st,len(names)))
    for k in range(0,len(names),BATCH):
        grp=names[k:k+BATCH]
        imgs=[_ensure_uint8_rgb(cv2.imdecode(np.fromfile(os.path.join(src,nm),dtype=np.uint8),cv2.IMREAD_UNCHANGED)) for nm in grp]
        masks=two_pass_batch(imgs, ENHANCE, WORM_THR)
        for nm,mk in zip(grp,masks):
            sp=os.path.join(dst,nm)
            if (not OVERWRITE) and os.path.isfile(sp): ntot+=1; continue
            pr=mk.astype(float)
            if pr.max()==0: nempty+=1
            else: pr*=255.0/pr.max()
            Image.fromarray(pr).convert("L").save(sp); ntot+=1
        if (k//BATCH)%15==0: print("  %s %d/%d  (%.2f s/img)"%(st,min(k+BATCH,len(names)),len(names),(time.time()-t0)/max(1,ntot)))
print("done: %d masks | %d empty | %.0f s | folder %s"%(ntot,nempty,time.time()-t0,OUT))


# ## 4. Check: cyan blastema included, old versus new, and plate
OLD=glob.glob(os.path.join(REGEN,"masks_model_Mask2Former_2024*"))[0]
NEW=OUT
import pandas as pd
rows=[]
for st in (2,4):
    for cp in sorted(glob.glob(os.path.join(CB,"State_%d"%st,"*.jpg"))):
        nm=os.path.basename(cp); cym=cyanmask(cv2.imread(cp))
        if cym.sum()<150: continue
        op=os.path.join(OLD,"State_%d"%st,nm); npth=os.path.join(NEW,"State_%d"%st,nm)
        if not (os.path.isfile(op) and os.path.isfile(npth)): continue
        om=worm_mask(cv2.imread(op,0)); nmk=worm_mask(cv2.imread(npth,0)); tot=int(cym.sum())
        rows.append({"state":st,"old":100.0*int((cym.astype(bool)&om.astype(bool)).sum())/tot,
                     "new":100.0*int((cym.astype(bool)&nmk.astype(bool)).sum())/tot})
D=pd.DataFrame(rows); D.to_csv("blasteme_inclus_ancien_vs_nouveau.csv",sep=";",index=False)
for st in (2,4):
    s=D[D.state==st]
    if len(s): print("State_%d (n=%d): blastema included old %.1f%% -> new %.1f%%"%(st,len(s),s["old"].mean(),s["new"].mean()))
tiles=[]
for cp in sorted(glob.glob(os.path.join(CB,"State_2","*.jpg")))[:8]:
    nm=os.path.basename(cp); ip=os.path.join(IMG,"State_2",nm)
    op=os.path.join(OLD,"State_2",nm); npth=os.path.join(NEW,"State_2",nm)
    if not all(os.path.isfile(x) for x in (ip,op,npth)): continue
    im=cv2.imread(ip); cym=cyanmask(cv2.imread(cp)); om=worm_mask(cv2.imread(op,0)); nmk=worm_mask(cv2.imread(npth,0))
    ov=im.copy()
    for msk,col in ((om,(0,0,255)),(nmk,(0,255,0))):
        c,_=cv2.findContours(msk.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(ov,c,-1,col,3)
    ov[cym>0]=(0.4*ov[cym>0]+0.6*np.array([255,255,0])).astype(np.uint8)
    ys,xs=np.nonzero(nmk|om)
    if len(xs):
        y0,y1,x0,x1=max(0,ys.min()-30),min(ov.shape[0],ys.max()+30),max(0,xs.min()-30),min(ov.shape[1],xs.max()+30); ov=ov[y0:y1,x0:x1]
    tiles.append(cv2.resize(ov,(260,260)))
if len(tiles)>=4:
    mont=np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,len(tiles)-len(tiles)%4,4)])
    cv2.imwrite("masque_ancien_rouge_nouveau_vert_cyan.png",mont)
    print("plate: masque_ancien_rouge_nouveau_vert_cyan.png (red old, green new, cyan blastema)")

