#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# =============================================================================================
# Mask2Former two-pass regeneration masks, worm chosen by its shape (copy of regenerate_masks_2pass)
#
# regenerate_masks_2pass and regenerate_masks_2pass_centre are not modified. The only difference with
# regenerate_masks_2pass: after the two passes are fused, when the mask holds several objects, the largest
# one (KEEP_LARGEST) is no longer kept but the most regular one, of greatest circularity
# C = 4 pi A / P^2 (A area, P perimeter of the outer contour). Enhancement and threshold identical to the
# folder in service masks_model_Mask2Former_20260324_0451_2pass_clahe_2_t16 (CLAHE clip 2, tile 16, threshold 0.50).
#
# Why. The reflection of the well wall is sometimes larger than the worm (6 masks without a worm over 3225
# annotated images), and neither size nor position in the image is enough: the image-centre rule
# (regenerate_masks_2pass_centre_20261011_075308) fixes 5 images, breaks 4 and keeps the reflection on
# 2024-06-14-Djap-J11-R2-Caf+500-Q4. On the 10 images marked by the experimenter (cyan square on the worm), the
# worm has the greatest circularity in all 10 cases (0.48 to 0.84 against 0.22 to 0.73 for the reflection or
# the veil): the worm contour is smooth and closed, the reflection's is a thin or jagged arc. Smallest gap:
# 2024-06-03-Djap-J0-R2-Caf+500-Q4 (0.64 against 0.60, reflection of the worm in the wall). The rule has only
# been seen on 10 images: the pass checks it on the 10 marks (VER_MARQUE) and produces a sheet of ALL the
# images with several candidates, for visual verification before adoption.
#
# Candidates: area >= AIRE_MIN px and >= REL_MIN times the area of the largest object (discards dust).
# Contrast K = median grey of the object minus median grey of a ring around it (more negative = darker than its
# surroundings). Three cases, fixed before the pass:
#   1. a single candidate: kept;
#   2. the most regular object is also the darkest (shape and contrast agree): kept. This is the case of the
#      2023 images on a pink background (black worm K -142 and -127, grey veil or reflection K -22 and -56);
#   3. disagreement: shape decides provisionally (iridescent reflections are darker than the worm but
#      irregular -- Djap-J9, J10, J11-Q1) and the image is ALWAYS flagged a_verifier for a visual check.
#      Circularity measures compactness as much as contour regularity: an elongated worm facing a round,
#      smooth reflection would mislead it, hence no automatic decision without both criteria agreeing.
#   Objects cut by the image frame (bounding box touching the border): their contour is truncated, so their
#   circularity is meaningless. Shape is only compared between whole objects; a cut object is retained only
#   if every candidate is cut (2024-06-06-Djap-J3-R2-Caf-500-Q2: worm cut at the bottom, iridescent
#   reflection cut at the top). Contrast is still compared over every candidate: a worm cut by the frame and
#   darker than the retained object yields a disagreement, hence a verification. Added on 11 October 2026
#   after 2024-06-17-Djap-J14-R1-Ctrl-Q1 (image-corner object, C 0.58, retained instead of the whole worm,
#   C 0.42): over the 38 multi-candidate images of the pass regenerate_masks_2pass_forme_20261011_100049,
#   only that image changes.
#
# Outputs. Folder essai_forme_masks_model_Mask2Former_20260324_0451_2pass_clahe_2_t16 (a name outside the
# pattern masks_model_Mask2Former_* read by the downstream notebooks: it will only be used after validation
# and renaming), log Claude outputs/masques_forme/composantes_masques_forme.csv (for each image: area,
# circularity, relative thickness, contrast and distance to the centre of each candidate), cyan check, sheets.
# Images are read with cv2.IMREAD_COLOR (EXIF orientation applied, like cv2.imread in the downstream
# notebooks; IMREAD_UNCHANGED ignored it and gave a rotated mask for 2023-02-06-T12-J0-R2-LDPE-Q1, EXIF 6).
# Graphics card required, about 40 minutes. Seed 42.
#
# Verified references, not clickable.
# Cheng B., Misra I., Schwing A.G., Kirillov A., Girdhar R. (2022). Masked-attention Mask Transformer for
# Universal Image Segmentation. CVPR 2022. arXiv:2112.01527.
# Zuiderveld K. (1994). Contrast Limited Adaptive Histogram Equalization. Graphics Gems IV, 474-485.
# DOI 10.1016/B978-0-12-336156-1.50061-6.
# Suzuki S., Abe K. (1985). Topological structural analysis of digitized binary images by border following.
# Computer Vision, Graphics, and Image Processing 30(1), 32-46. DOI 10.1016/0734-189X(85)90016-7
# (contours, cv2.findContours; circularity 4 pi A / P^2 computed on that contour).
# =============================================================================================
import os, sys, time, glob, datetime, numpy as np, cv2, torch, pandas as pd
from PIL import Image
np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
HORO = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


class _Double(object):
    def __init__(self, flux, chemin): self.flux = flux; self.fic = open(chemin, "w", encoding="utf-8")
    def write(self, t): self.flux.write(t); self.fic.write(t); self.fic.flush()
    def flush(self): self.flux.flush(); self.fic.flush()


if not isinstance(sys.stdout, _Double): sys.stdout = _Double(sys.stdout, "regenerate_masks_2pass_forme_%s.log" % HORO)
REPO = os.getcwd()
if REPO not in sys.path: sys.path.insert(0, REPO)
assert os.path.isdir(os.path.join(REPO, "checkpoints_segmentation_model")), "Run from the FasterViT_for_Image_Classification root"
if not torch.cuda.is_available(): raise SystemExit("graphics processor unavailable, run aborted.")
device = torch.device("cuda")
print("torch", torch.__version__, "| GPU :", torch.cuda.get_device_name(0))

import torch.nn.functional as F
from PIL import Image as PILImage
from flatmorphai.notebook_api.segmentation import load_model_for_segmentation
from evaluate_test_images_v2 import (_extract_p2_jobs, _paste_p2_result, _apply_fusion, _ensure_uint8_rgb, _maybe_suppress_stdout)
MODEL_NAME = "model_Mask2Former_20260324_0451"
model, processor = load_model_for_segmentation(MODEL_NAME, device)
mdev = next(model.parameters()).device; print("model on:", mdev); assert mdev.type == "cuda", "the model is not on the graphics card"


def predict_prob_batch(images, worm_thr=0.5):
    # identical to regenerate_masks_2pass (at worm_thr = 0.50 the result is the original argmax)
    if not images: return []
    sizes = [(im.shape[0], im.shape[1]) for im in images]
    inp = processor(images=[PILImage.fromarray(im) for im in images], return_tensors="pt")
    inp = {k: v.to(device) for k, v in inp.items()}
    with torch.no_grad(): outputs = model(**inp)
    mql = outputs.masks_queries_logits if hasattr(outputs, "masks_queries_logits") else outputs["masks_queries_logits"]
    cql = outputs.class_queries_logits if hasattr(outputs, "class_queries_logits") else outputs["class_queries_logits"]
    masks = mql.sigmoid(); classes = cql.softmax(-1)[..., :-1]
    seg = torch.einsum("bqc,bqhw->bchw", classes, masks)
    out = []
    for i, (H, W) in enumerate(sizes):
        s = F.interpolate(seg[i:i + 1].float(), size=(H, W), mode="bilinear", align_corners=False)[0]
        p = s / (s.sum(0, keepdim=True) + 1e-6); fg = p[1:].sum(0)
        out.append((fg >= worm_thr).cpu().numpy().astype(np.uint8) * 255)
    return out


# two-pass parameters, identical to regenerate_masks_2pass
BATCH = 8; MODEL_SIZE = 320; PAD_RATIO = 0.50; MERGE = "skeleton_guided"
MIN_COMP = 10; MIN_PAD = 32; MIN_ROI = 0.0; BND_DIL = 0.02; MERGE_DIL_R = None; MERGE_DIL_F = 0.05
# choice of the object
AIRE_MIN = 300      # same minimum worm area as the cascade (MIN_MASK_AREA)
REL_MIN = 0.10      # a candidate must be at least 10 % of the largest component


def enhance(img, mode):
    # identical to regenerate_masks_2pass
    if mode == "none": return img
    parts = mode.split("_"); clip = 3.0; tile = 8
    for p in parts[1:]:
        if p.startswith("t") and p[1:].isdigit(): tile = int(p[1:])
        elif p.replace('.', '', 1).isdigit(): clip = float(p)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB); L, A, B = cv2.split(lab)
    if parts[0] == "clahe": L = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(L)
    out = cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2RGB)
    if mode.endswith("sat"):
        hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * 1.4, 0, 255); out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return out


def choisir_ver(mask, img=None):
    """Object of greatest circularity among the candidates. Returns (mask, info). img (original RGB)
    is only used to log the contrast of each candidate; it plays no part in the choice."""
    b = (mask == 255).astype(np.uint8); H, W = b.shape
    gris = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(float) if img is not None else None
    n, lab, st, cen = cv2.connectedComponentsWithStats(b, 8)
    if n <= 1: return mask, dict(n_comp=0)
    aires = st[1:, cv2.CC_STAT_AREA].astype(float); ig = int(np.argmax(aires))
    cand = [k for k in range(n - 1) if aires[k] >= AIRE_MIN and aires[k] >= REL_MIN * aires[ig]]
    if not cand: cand = [ig]
    circ, epais, contr, bord = {}, {}, {}, {}
    for k in cand:
        comp = (lab == k + 1).astype(np.uint8)
        cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); c = max(cs, key=cv2.contourArea)
        P = cv2.arcLength(c, True); circ[k] = 4.0 * np.pi * aires[k] / max(P * P, 1e-6)
        x0, y0, w0, h0 = st[k + 1, 0], st[k + 1, 1], st[k + 1, 2], st[k + 1, 3]
        bord[k] = int(x0 == 0 or y0 == 0 or x0 + w0 >= W or y0 + h0 >= H)   # object cut by the image frame
        epais[k] = float(cv2.distanceTransform(comp, cv2.DIST_L2, 5).max() / np.sqrt(aires[k]))
        contr[k] = np.nan
        if gris is not None:   # median grey of the object minus median grey of a ring of 0.15 sqrt(area) around it
            r = max(5, int(0.15 * np.sqrt(aires[k]))); anneau = cv2.dilate(comp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) - comp
            contr[k] = float(np.median(gris[comp > 0]) - np.median(gris[anneau > 0]))
    entiers = [k for k in cand if not bord[k]] or cand            # the shape of an object cut by the frame cannot be measured
    ic = max(entiers, key=lambda k: circ[k])                      # decision: shape, among whole objects
    accord, rapport, a_verifier = 1, np.nan, 0
    if gris is not None and len(cand) >= 2:
        ik = min(cand, key=lambda k: contr[k])                   # darkest object relative to its surroundings
        if ik != ic:                                             # shape and contrast disagree
            accord = 0; rapport = circ[ic] / max(circ[ik], 1e-6)
            a_verifier = 1                                       # disagreement: systematic visual check
    d = np.hypot(cen[1:, 0] - (W - 1) / 2.0, cen[1:, 1] - (H - 1) / 2.0) / max(H, W)
    out = np.where(lab == ic + 1, 255, 0).astype(np.uint8)
    f = lambda v: "|".join("%.4f" % v[k] for k in cand)
    return out, dict(n_comp=n - 1, n_cand=len(cand), aire_grande=aires[ig], aire_retenue=aires[ic], circ_retenue=circ[ic], accord=accord, rapport_circ=rapport, a_verifier=a_verifier,
                     change=int(ic != ig), cand_aire=f(aires), cand_circ=f(circ), cand_epais=f(epais), cand_contraste=f(contr), cand_bord="|".join(str(bord[k]) for k in cand), cand_dist=f(d),
                     cand_x="|".join("%.0f" % cen[k + 1][0] for k in cand), cand_y="|".join("%.0f" % cen[k + 1][1] for k in cand))


def two_pass_batch(batch_np, mode="none", worm_thr=0.5):
    # identical to regenerate_masks_2pass up to the merge, then choisir_ver instead of the largest
    origine = batch_np
    batch_np = [enhance(im, mode) for im in batch_np]; n = len(batch_np)
    p1 = predict_prob_batch(batch_np, worm_thr)
    all_jobs = []; jobs_per = [[] for _ in range(n)]
    for i in range(n):
        j = _extract_p2_jobs(batch_np[i], p1[i], i, PAD_RATIO, MODEL_SIZE, MIN_COMP, MIN_PAD, MIN_ROI)
        all_jobs.extend(j); jobs_per[i] = j
    p2res = []
    if all_jobs:
        crops = [j["crop"] for j in all_jobs]
        for s in range(0, len(crops), BATCH): p2res.extend(predict_prob_batch(crops[s:s + BATCH], worm_thr))
    p2full = [np.zeros(batch_np[i].shape[:2], np.uint8) for i in range(n)]
    for job, cm in zip(all_jobs, p2res): _paste_p2_result(cm, p2full[job["image_idx"]], job)
    for i in range(n):
        if p2full[i].sum() == 0 and p1[i].sum() > 0: p2full[i] = p1[i].copy()
    out = []
    for i in range(n):
        cs = jobs_per[i][0]["crop"].shape if jobs_per[i] else None
        with _maybe_suppress_stdout(False):
            fused = _apply_fusion(p1[i], p2full[i], MERGE, BND_DIL, cs, False, MERGE_DIL_R, MERGE_DIL_F)
        out.append(choisir_ver(fused, origine[i]))
    return out


REGEN = os.path.join("..", "..", "Dataset_stereotype_regeneration", "dataset_training_regeneration")
IMG = os.path.join(REGEN, "images")
def list_imgs(d): return [n for n in sorted(os.listdir(d)) if n.lower().endswith((".jpg", ".jpeg", ".png"))]


# In[ ]:


# =============================================================================================
# 2. Full pass (resume: masks already written are skipped)
# =============================================================================================
ENHANCE = "clahe_2_t16"; WORM_THR = 0.50
ANCIEN = os.path.join(REGEN, "masks_model_Mask2Former_20260324_0451_2pass_clahe_2_t16")
OUT = os.path.join(REGEN, "essai_forme_masks_model_Mask2Former_20260324_0451_2pass_clahe_2_t16")
SORTIE = os.path.join("Claude outputs", "masques_forme"); os.makedirs(SORTIE, exist_ok=True)
JOURNAL = os.path.join(SORTIE, "composantes_masques_forme.csv")
STATES = ["State_%d" % s for s in range(1, 7)]
print("enhancement:", ENHANCE, "| worm threshold:", WORM_THR, "| rule: object of greatest circularity (candidates: area >= %d px and >= %.2f x the largest)" % (AIRE_MIN, REL_MIN))
print("output:", OUT)
faits = set(pd.read_csv(JOURNAL, sep=";").image) if os.path.exists(JOURNAL) else set()
t0 = time.time(); ntot = 0; rows = []
for st in STATES:
    src = os.path.join(IMG, st); dst = os.path.join(OUT, st); os.makedirs(dst, exist_ok=True)
    names = [nm for nm in list_imgs(src) if not (nm in faits and os.path.isfile(os.path.join(dst, nm)))]
    print("### %s: %d images to process" % (st, len(names)))
    for k in range(0, len(names), BATCH):
        grp = names[k:k + BATCH]
        imgs = [_ensure_uint8_rgb(cv2.imdecode(np.fromfile(os.path.join(src, nm), dtype=np.uint8), cv2.IMREAD_COLOR)) for nm in grp]
        for nm, (mk, inf) in zip(grp, two_pass_batch(imgs, ENHANCE, WORM_THR)):
            pr = mk.astype(float)
            if pr.max() > 0: pr *= 255.0 / pr.max()
            Image.fromarray(pr).convert("L").save(os.path.join(dst, nm)); ntot += 1
            inf.update(image=nm, classe=st); rows.append(inf)
        if len(rows) >= 200:
            pd.DataFrame(rows).to_csv(JOURNAL, sep=";", index=False, mode="a", header=not os.path.exists(JOURNAL)); rows = []
        if (k // BATCH) % 15 == 0: print("  %s %d/%d  (%.2f s/img)" % (st, min(k + BATCH, len(names)), len(names), (time.time() - t0) / max(1, ntot)))
if rows: pd.DataFrame(rows).to_csv(JOURNAL, sep=";", index=False, mode="a", header=not os.path.exists(JOURNAL))
print("done: %d masks written in this run | %.0f s" % (ntot, time.time() - t0))


# In[ ]:


# =============================================================================================
# 3. Summary: images whose choice changes, check against the cyan annotations, sheet
# =============================================================================================
D = pd.read_csv(JOURNAL, sep=";").drop_duplicates("image", keep="last")
print("images: %d | no component: %d | single component: %d | choice other than the largest: %d"
      % (len(D), int((D.n_comp == 0).sum()), int((D.n_comp == 1).sum()), int(D.change.fillna(0).sum())))
ch = D[D.change == 1].sort_values(["classe", "image"])
ch.to_csv(os.path.join(SORTIE, "images_choix_change.csv"), sep=";", index=False)
print(ch[["image", "classe", "n_cand", "aire_grande", "aire_retenue", "circ_retenue", "cand_circ"]].to_string(index=False))

# check on the 10 marked images (interior point of the worm, pixels of the original image)
VER_MARQUE = {"2023-04-11-J0-R1-E-Q1.jpg": (551, 1018), "2023-04-12-J1-R3-B+E-Q4.jpg": (1395, 893),
              "2024-06-03-Djap-J0-R2-Caf+500-Q4.jpg": (984, 322), "2024-06-05-Djap-J2-R1-Caf+500-Q4.jpg": (1634, 626),
              "2024-06-07-Smed-J4-R3-Caf-500-Q1.jpg": (312, 426), "2024-06-12-Djap-J9-R3-Ctrl-Q3.jpg": (1125, 201),
              "2024-06-11-Smed-J8-R2-Ctrl-Q2.jpg": (266, 625), "2024-06-13-Djap-J10-R2-Caf+500-Q4.jpg": (995, 780),
              "2024-06-14-Djap-J11-R2-Caf+500-Q1.jpg": (477, 785), "2024-06-14-Djap-J11-R2-Caf+500-Q4.jpg": (493, 461)}
ok = 0
for nm, (x, y) in VER_MARQUE.items():
    st = D.loc[D.image == nm, "classe"].values[0]
    m = (cv2.imread(os.path.join(OUT, st, nm), 0) >= 128).astype(np.uint8)
    dd = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)[y, x]
    ok += int(dd <= 30); print("  %-40s marked worm %s in the mask" % (nm, "YES" if dd <= 30 else "NO (%.0f px)" % dd))
print("check of the 10 marked images: %d / 10" % ok)
plus = D[D.n_cand >= 2]
print("images with several candidates: %d | shape and contrast agree: %d | disagree: %d | to check: %d"
      % (len(plus), int((plus.accord == 1).sum()), int((plus.accord == 0).sum()), int(plus.a_verifier.sum())))
av = plus[plus.a_verifier == 1]; av.to_csv(os.path.join(SORTIE, "images_a_verifier.csv"), sep=";", index=False)
print(plus[plus.accord == 0][["image", "classe", "rapport_circ", "a_verifier", "cand_circ", "cand_contraste"]].to_string(index=False))


def cyan(b):
    b_, g, r = cv2.split(b.astype(np.int16))
    m = ((b_ > 120) & (g > 120) & (r < 110) & ((b_ - r) > 60) & ((g - r) > 40)).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


# check: share of the annotated cyan (minus the cyan of the original image) inside the mask, old against new
res = []
for srcn in ("cyan_position_tete", "cyan_eye", "cyan_blasteme"):
    for f in glob.glob(os.path.join(REGEN, srcn, "State_*", "*.jpg")):
        st = os.path.basename(os.path.dirname(f)); nm = os.path.basename(f)
        ip = os.path.join(IMG, st, nm); a_p = os.path.join(ANCIEN, st, nm); n_p = os.path.join(OUT, st, nm)
        if not all(os.path.isfile(x) for x in (ip, a_p, n_p)): continue
        im = cv2.imread(ip); an = cv2.imread(f)
        if an is None or an.shape != im.shape: continue
        c = cyan(an) & (1 - cyan(im)); nc = int(c.sum())
        if nc == 0: continue
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
        fa = [float((c & cv2.dilate((cv2.imread(p, 0) >= 128).astype(np.uint8), ker)).sum()) / nc for p in (a_p, n_p)]
        res.append(dict(source=srcn, image=nm, classe=st, cyan_dans_ancien=fa[0], cyan_dans_nouveau=fa[1]))
R = pd.DataFrame(res); R.to_csv(os.path.join(SORTIE, "controle_cyan_ancien_nouveau.csv"), sep=";", index=False)
print("annotations checked: %d | cyan outside the mask (less than half inside): old %d, new %d"
      % (len(R), int((R.cyan_dans_ancien < 0.5).sum()), int((R.cyan_dans_nouveau < 0.5).sum())))
print("annotations that fall outside the mask under the new rule (to examine):")
print(R[(R.cyan_dans_nouveau < 0.5) & (R.cyan_dans_ancien >= 0.5)].to_string(index=False))

# sheet of ALL the multi-candidate images: retained object in green, other candidates in red,
# circularity C and contrast K (object grey minus ring grey) written next to each object
tiles = []
for _, r in D[D.n_cand >= 2].sort_values(["classe", "image"]).iterrows():
    im = cv2.imread(os.path.join(IMG, r.classe, r.image)); H, W = im.shape[:2]
    keep = (cv2.imread(os.path.join(OUT, r.classe, r.image), 0) >= 128).astype(np.uint8)
    xs = [int(v) for v in str(r.cand_x).split("|")]; ys = [int(v) for v in str(r.cand_y).split("|")]
    cs_ = [float(v) for v in str(r.cand_circ).split("|")]; ks_ = [float(v) for v in str(r.cand_contraste).split("|")]
    for x, y, c, k_ in zip(xs, ys, cs_, ks_):
        col = (0, 255, 0) if keep[min(y, H - 1), min(x, W - 1)] else (0, 0, 255)
        cv2.circle(im, (x, y), 18, col, -1); cv2.putText(im, "C%.2f K%.0f" % (c, k_), (x + 25, y), cv2.FONT_HERSHEY_SIMPLEX, 2.0, col, 5)
    cs, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(im, cs, -1, (0, 255, 0), 6)
    t = cv2.resize(im, (480, 270))
    etiq = r.image[:36] + ("  TO CHECK" if r.a_verifier == 1 else "")
    cv2.putText(t, etiq, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255) if r.accord == 0 else (0, 0, 255), 1); tiles.append(t)
for k in range(0, len(tiles), 12):
    bloc = tiles[k:k + 12]
    while len(bloc) % 3: bloc.append(np.zeros_like(tiles[0]))
    cv2.imwrite(os.path.join(SORTIE, "planche_candidates_%02d.jpg" % (k // 12)), np.vstack([np.hstack(bloc[i:i + 3]) for i in range(0, len(bloc), 3)]))
print("sheets: %s/planche_candidates_*.jpg (%d multi-candidate images; green: retained object, red: discarded, C: circularity, K: contrast)" % (SORTIE, len(tiles)))
print("done.")
