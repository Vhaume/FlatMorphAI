# CHAIN v22 (22 October 2026): copy of train_modele_production_stade_v21.py (unmodified). Cascade ancrage_cascade_v3c14_rayon_max_v5s1neg_s040 (start point:
# largest border radius among the points in contact with the blastema), blastema threshold 0.40, s040 masks unchanged.
# CHAIN v21 (20 October 2026): copy of train_modele_production_stade_v20.py (unmodified). Complete descriptors, without NaN or imputation: eyeun_* at 0
# when there is no eyespot peak, every image of the computed lists, stops if a descriptor or a decimal state is missing.
# CHAIN v20 (20 October 2026): copy of train_modele_production_stade_v19.py (unmodified). Blastema threshold 0.40 (calibrer_seuil_objet_blasteme_20261020_023318,
# maximal IoU of the blastema object) and cascade ancrage_cascade_v3c14_rayon_max_v5s1neg_s040 (v3c12 at threshold 0.40).
# Version v19 (19 October 2026): copy of train_modele_production_stade_v18 (unmodified). Inputs from ancrage_cascade_v3c14_rayon_max_v5s1neg_s040: composites
# processed_composite_384_uni_final_v22 and log ancres_composite_v18.csv. Outputs suffixed v19.
# Version v19 (18 October 2026): copy of train_modele_production_stade_v17 (unmodified). Inputs from ancrage_cascade_v3c14_rayon_max_v5s1neg_s040:
# composites processed_composite_384_uni_final_v22 and log ancres_composite_v18.csv. Outputs suffixed v19.
# Version v19 (18 October 2026): copy of train_modele_production_stade_v16 (unmodified). Only the inputs change, coming from the final cascade
# method (ancrage_cascade_v3c14_rayon_max_v5s1neg_s040): composites processed_composite_384_uni_final_v22 and log
# ancres_composite_v18.csv, and the outputs, suffixed v19. Nothing is overwritten.
# [V16] Copy of train_modele_production_stade_v15 (unmodified), 18 October 2026. Only the descriptor set changes: every descriptor is
# recomputed by enrich_metrics_v22 on the retained pipeline (masks refined by the blastema, head from the cascade chain,
# retained blastema segmenter and eyespot detector), folder metrics_regeneration/<set>_v22, and the list of
# retained descriptors is metriques_finales_v22.csv, rebuilt on the training set. Run enrich_metrics_v22 first.
# [V15] Production model of the state classifier on the v15 composites (17-18 October 2026): skan 2 axis, ouv3_skan2
# masks, cascade. Copy of train_modele_production_stade_v9 (unmodified), 97 descriptors kept (trial
# evaluation_pli_test_stade_v15_familles: the reduced blastema + eyespots set was not retained by the rule). Same architecture, loss,
# hyper-parameters, 30 epochs, seed 42. Graphics-card memory handling: expandable-segment allocation, free-memory check
# before training, explicit release of the model and the optimiser after each training run.
import os
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import gc
import os, re, glob, math, time, datetime, sys, numpy as np, pandas as pd, cv2
import torch, torch.nn as nn, torch.nn.functional as F
import timm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
try: sys.stdout.reconfigure(line_buffering=True)
except Exception: pass
np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, "| cuda :", torch.cuda.is_available())
if torch.cuda.is_available(): print("   carte :", torch.cuda.get_device_name(0))
if device.type != "cuda": raise SystemExit("graphics card not found, run aborted.")
_libre, _total = torch.cuda.mem_get_info()
print("   card memory: %.1f GB free out of %.1f GB" % (_libre / 1e9, _total / 1e9))
if _libre < 12e9:
    raise SystemExit("less than 12 GB free on the card (%.1f GB): close the other notebooks or programs using it, then run again." % (_libre / 1e9))

MODE_PROD = os.environ.get("MODE_PROD", "unique")        # unique ou ensemble
assert MODE_PROD in ("unique", "ensemble")

# ---- constants STRICTLY identical to arm 5 ----
CLASS_TYPE = "regeneration"; BACKBONE = "convnextv2_base.fcmae_ft_in22k_in1k_384"; IMAGE_SIZE = 384
dataset_classes = ["State_%d" % i for i in range(1, 7)]; NCLS = 6
N2I = {c: i for i, c in enumerate(dataset_classes)}; STAGE_MIN, STAGE_MAX = 1.0, 6.0
REGEN_ROOT = os.path.join("..", "..", "Dataset_stereotype_regeneration", "dataset_training_" + CLASS_TYPE)
LIST_ROOT = os.path.join(REGEN_ROOT, "dataset_images_list")
DATASET_NAME = os.environ.get("REGEN_DATASET", "dataset_dedup_State_50valid_30test")
# Composite set. By default the one from the FINAL cascade of 3 October 2026, segmenter
# stratified and MARGE_BL 0.84. Set REGEN_COMPOSITES in the environment to choose another one,
# for example processed_composite_384_uni_cascade2 to reproduce arm 5 exactly.
SOUS_DOSSIER = os.environ.get("REGEN_COMPOSITES", "processed_composite_384_uni_final_v22")   # [V15]
_VA6 = os.path.join("Claude outputs", "essai_ancrage_cascade_v3c14_s040", "verdict.txt")   # [V20] cascade v3c14, check
assert os.path.exists(_VA6), "run test_ancrage_cascade_v3c14_s040 first: " + _VA6
print("check of the v3c14 cascade:", open(_VA6, encoding="utf-8").read().strip().replace("\n", " | "))
for _e in ("essai_ancrage_cascade_skan2_v5s1neg_s070",):
    _VERD = os.path.join("Claude outputs", _e, "verdict.txt")
    if not os.path.exists(_VERD) or not open(_VERD, encoding="utf-8").read().startswith("RETENUE"):
        raise SystemExit("preliminary trial missing or not retained: %s" % _VERD)
    print(_e, ":", open(_VERD, encoding="utf-8").read().strip().replace("\n", " | "))
IMAGE_DIR = os.path.join(REGEN_ROOT, SOUS_DOSSIER)
METRICS_ROOT = os.path.join(REGEN_ROOT, "metrics_regeneration", DATASET_NAME + "_v22")   # [V16] descriptors recomputed by enrich_metrics_v22
assert os.path.isdir(METRICS_ROOT), "v19 metrics missing, run enrich_metrics_v22 first: " + METRICS_ROOT
PROJ_METRIC_HID = 128; LAMBDA_AUX = 0.3; LAMBDA_VIS_AUX = 0.3; LAMBDA_GATE = 1e-3
LR, HEAD_LR_MULT, WD, BATCH = 2e-5, 5.0, 0.08, 12
DROP_PATH = 0.1; HUBER_BETA = 0.5
EPOCHS = 30; WARMUP_EPOCHS = 3; KFOLD = 5
ZSRC = int(os.environ.get("ZSRC", "1"))
CH_MAP = {"entier": (0, 0, 0), "entier+zoom": (0, 1, 1), "entier+superzoom": (0, 2, 2), "complet": (0, 1, 2)}
CURMODE = "entier+superzoom"
SORTIE = os.path.join("checkpoints_regeneration", "stade_production_v22")
# The previous model is never overwritten silently. It is renamed with the date, and a new empty
# folder is created. REPRISE = 1 keeps the folder in place to resume a training run.
REPRISE = int(os.environ.get("REPRISE", "0"))
if not REPRISE and os.path.isdir(SORTIE) and glob.glob(os.path.join(SORTIE, "*.pth")):
    _old = SORTIE + "_avant_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.rename(SORTIE, _old)
    print("previous model set aside:", os.path.basename(_old))
os.makedirs(SORTIE, exist_ok=True)
assert os.path.isdir(IMAGE_DIR), "composites missing, run regenerate_composite_v2 first: " + IMAGE_DIR
print("composites:", SOUS_DOSSIER, "| channels:", CH_MAP[CURMODE], "| mode:", MODE_PROD)

def metric_csv(sp): return os.path.join(METRICS_ROOT, DATASET_NAME + "_" + sp + "_regeneration_metrics_eyespig.csv")
def decimal_csv(sp): return os.path.join(METRICS_ROOT, "decimal_" + sp + ".csv")

_m = pd.read_csv(metric_csv("train"), sep=";")
COLS_TOUS = [c for c in _m.columns if c not in ("image", "class")]
COLS_TOUS = [c for c in COLS_TOUS if not _m[c].isna().all()]
_self = "metriques_finales_v22.csv"
assert os.path.exists(_self), "list of retained descriptors not found, run enrich_metrics_v22: " + _self
COLS_SEL = [c for c in pd.read_csv(_self, sep=";")["descripteur"].tolist() if c in COLS_TOUS]
print("descriptors: full %d | retained %d" % (len(COLS_TOUS), len(COLS_SEL)))

DEC = {}
for sp in ("train", "valid", "test"):
    d = pd.read_csv(decimal_csv(sp), sep=";")
    for im, dv in zip(d["image"], d["stade_decimal"]): DEC[str(im)] = float(dv)
_BRUT = {}
for sp in ("train", "valid", "test"):
    d = pd.read_csv(metric_csv(sp), sep=";")
    for i, b in enumerate(d["image"]): _BRUT[str(b)] = d.loc[i, COLS_TOUS]

METRIC_COLS = list(COLS_SEL); METRIC_DIM = len(METRIC_COLS)
MET = {k: v[METRIC_COLS].to_numpy(np.float32) for k, v in _BRUT.items()}
_allmet = np.stack(list(MET.values())); _med = np.nanmedian(_allmet, axis=0); assert not np.isnan(_allmet).any(), "missing descriptor (NaN)"
_f = np.where(np.isnan(_allmet), _med, _allmet); MEAN = _f.mean(0); STD = _f.std(0); STD[STD == 0] = 1.0
def norm(v): v = np.where(np.isnan(v), _med, v); return ((v - MEAN) / STD).astype(np.float32)


def worm_key(stem):
    mm = re.match(r"(\d{4})-(\d{2})-(\d{2})-.*?J(\d+)-?(.*)", stem)
    if not mm: return stem
    y, mo, dd, j, rest = mm.groups()
    try:
        return (datetime.date(int(y), int(mo), int(dd)) - datetime.timedelta(days=int(j))).isoformat() + "|" + rest
    except Exception:
        return stem


paths, labels, decs, mets, groups = [], [], [], [], []
_replis = 0
for sp in ("train", "valid", "test"):
    f = next((x for x in glob.glob(os.path.join(LIST_ROOT, DATASET_NAME, "*")) if x.endswith("_" + sp + ".csv")), None)
    sdf = pd.read_csv(f, sep=";"); kc = sdf.columns[-2]
    for _, r in sdf.iterrows():
        rel = str(r["image_path"]); base = os.path.basename(rel); stem = os.path.splitext(base)[0]
        lab = N2I.get(str(r[kc]), -1)
        if lab < 0: continue
        ip = os.path.join(IMAGE_DIR, os.path.dirname(rel), stem + ".png")
        if not os.path.exists(ip):
            alt = [os.path.join(IMAGE_DIR, c, stem + ".png") for c in dataset_classes]
            alt = [a for a in alt if os.path.exists(a)]
            if not alt: continue
            ip = alt[0]; _replis += 1
        paths.append(ip); labels.append(lab); decs.append(DEC[base])
        mets.append(norm(MET[base])); groups.append(worm_key(stem))
labels = np.array(labels); decs = np.array(decs, np.float32); mets = np.stack(mets); groups = np.array(groups)
print("images: %d | worms: %d | descriptors: %d | paths resolved elsewhere: %d"
      % (len(paths), len(set(groups)), METRIC_DIM, _replis))
assert len(paths) == 2196, "image set different from that of arm 5, %d instead of 2196" % len(paths)

_ref = timm.create_model(BACKBONE, pretrained=False, num_classes=0)
cfgd = timm.data.resolve_data_config({}, model=_ref); del _ref
mean, std = cfgd.get("mean", (0.485, 0.456, 0.406)), cfgd.get("std", (0.229, 0.224, 0.225))
mean_t = torch.tensor(mean).view(3, 1, 1); std_t = torch.tensor(std).view(3, 1, 1)


def load_comp(p):
    d = np.fromfile(p, dtype=np.uint8); im = cv2.imdecode(d, cv2.IMREAD_COLOR)
    if im is None: im = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)
    if im.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE): im = cv2.resize(im, (IMAGE_SIZE, IMAGE_SIZE))
    i0, i1, i2 = CH_MAP[CURMODE]
    return np.stack([im[:, :, i0], im[:, :, i1], im[:, :, i2]], axis=2)


class DS(Dataset):
    def __init__(self, idx, train): self.idx = idx; self.train = train
    def __len__(self): return len(self.idx)
    def __getitem__(self, k):
        i = self.idx[k]; im = load_comp(paths[i])
        if self.train:
            if np.random.rand() < 0.5: im = im[:, ::-1].copy()
            if np.random.rand() < 0.5: im = im[::-1].copy()
            if np.random.rand() < 0.5: im = np.rot90(im, np.random.randint(1, 4)).copy()
        x = torch.from_numpy(im.transpose(2, 0, 1).astype(np.float32) / 255.0); x = (x - mean_t) / std_t
        return x, torch.from_numpy(mets[i]), float(decs[i]), int(labels[i])


class _STN(nn.Module):
    def __init__(self):
        super().__init__()
        self.loc = nn.Sequential(nn.Conv2d(3, 16, 7, stride=2, padding=3), nn.ReLU(True), nn.MaxPool2d(2),
                                 nn.Conv2d(16, 32, 5, stride=2, padding=2), nn.ReLU(True), nn.AdaptiveAvgPool2d(4))
        self.fc = nn.Sequential(nn.Linear(32 * 4 * 4, 64), nn.ReLU(True), nn.Linear(64, 6))
        self.fc[-1].weight.data.zero_()
        self.fc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32))
    def forward(self, x):
        th = self.fc(self.loc(x).flatten(1)).view(-1, 2, 3)
        g = F.affine_grid(th, x.size(), align_corners=False)
        return F.grid_sample(x, g, align_corners=False)


class Net(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__(); self.stn = _STN()
        self.backbone = timm.create_model(BACKBONE, pretrained=pretrained, num_classes=0,
                                          global_pool="avg", drop_path_rate=DROP_PATH)
        C = int(self.backbone.num_features)
        self.vis_norm = nn.LayerNorm(C); self.metric_gate = nn.Parameter(torch.ones(METRIC_DIM))
        self.metric_mlp = nn.Sequential(
            nn.Linear(METRIC_DIM, PROJ_METRIC_HID), nn.BatchNorm1d(PROJ_METRIC_HID), nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(PROJ_METRIC_HID, PROJ_METRIC_HID), nn.BatchNorm1d(PROJ_METRIC_HID), nn.ReLU(True))
        self.head = nn.Sequential(nn.Linear(C + PROJ_METRIC_HID, 64), nn.ReLU(True), nn.Dropout(0.2), nn.Linear(64, 1))
        self.logvar_head = nn.Sequential(nn.Linear(C + PROJ_METRIC_HID, 64), nn.ReLU(True), nn.Linear(64, 1))
        self.metric_aux = nn.Linear(PROJ_METRIC_HID, 1); self.vis_aux = nn.Linear(C, 1)
    def _sq(self, t): return STAGE_MIN + (STAGE_MAX - STAGE_MIN) * torch.sigmoid(t)
    def forward(self, x, m):
        x = self.stn(x); f = self.vis_norm(self.backbone(x)); g = self.metric_mlp(m * self.metric_gate)
        z = torch.cat([f, g], 1)
        return (self._sq(self.head(z).squeeze(1)), self.logvar_head(z).squeeze(1),
                self._sq(self.metric_aux(g).squeeze(1)), self._sq(self.vis_aux(f).squeeze(1)))


huber = nn.SmoothL1Loss(beta=HUBER_BETA)
def nll(mu, lv, y): lv = torch.clamp(lv, -6, 4); return (0.5 * torch.exp(-lv) * (mu - y) ** 2 + 0.5 * lv).mean()
HEADK = ("stn", "head", "logvar_head", "metric_mlp", "metric_aux", "vis_aux", "metric_gate", "vis_norm")
def make_opt(model):
    hp = [p for n, p in model.named_parameters() if any(k in n for k in HEADK)]
    bp = [p for n, p in model.named_parameters() if not any(k in n for k in HEADK)]
    return torch.optim.AdamW([{"params": bp, "lr": LR}, {"params": hp, "lr": LR * HEAD_LR_MULT}], weight_decay=WD)

def entrainer(idx, nom):
    """Train a model on the given indices and write its checkpoint."""
    ck = os.path.join(SORTIE, nom + ".pth")
    tmp = os.path.join(SORTIE, nom + "_encours.pth")
    dep = 0; model = Net().to(device); opt = make_opt(model)
    lf = lambda ep: ((ep + 1) / max(1, WARMUP_EPOCHS) if ep < WARMUP_EPOCHS
                     else 0.5 * (1 + math.cos(math.pi * min(1, (ep - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)))))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lf)
    if os.path.exists(ck):
        print("   %s already trained, skipped" % nom); return ck
    if os.path.exists(tmp):
        c = torch.load(tmp, map_location=device)
        model.load_state_dict(c["state_dict"]); opt.load_state_dict(c["opt"]); sch.load_state_dict(c["sch"])
        dep = int(c["epoque"]); print("   resuming %s at epoch %d" % (nom, dep))
    tl = DataLoader(DS(idx, True), batch_size=BATCH, shuffle=True, num_workers=3, pin_memory=True, drop_last=True)
    t0 = time.time()
    for ep in range(dep, EPOCHS):
        model.train(); tot = 0.0; nb = 0
        for x, m, yd, yi in tl:
            x, m = x.to(device, non_blocking=True), m.to(device, non_blocking=True)
            yd = yd.float().to(device, non_blocking=True)
            opt.zero_grad(); mu, lv, rm, rv = model(x, m)
            loss = (nll(mu, lv, yd) + LAMBDA_AUX * huber(rm, yd) + LAMBDA_VIS_AUX * huber(rv, yd)
                    + LAMBDA_GATE * model.metric_gate.abs().mean())
            loss.backward(); opt.step(); tot += float(loss.item()); nb += 1
        sch.step()
        print("   %s epoch %2d/%d | loss %.4f | %.0f s" % (nom, ep + 1, EPOCHS, tot / max(nb, 1), time.time() - t0), flush=True)
        if (ep + 1) % 10 == 0 and ep + 1 < EPOCHS:
            torch.save({"state_dict": model.state_dict(), "opt": opt.state_dict(),
                        "sch": sch.state_dict(), "epoque": ep + 1}, tmp)
    torch.save({
        "state_dict": model.state_dict(),
        # everything needed to replay the inference without this notebook
        "backbone": BACKBONE, "image_size": IMAGE_SIZE, "ch_map": CH_MAP[CURMODE], "mode_canaux": CURMODE,
        "stage_min": STAGE_MIN, "stage_max": STAGE_MAX, "proj_metric_hid": PROJ_METRIC_HID,
        "drop_path": DROP_PATH, "metric_cols": METRIC_COLS, "metric_dim": METRIC_DIM,
        "metric_mean": MEAN.astype(np.float32), "metric_std": STD.astype(np.float32),
        "metric_median": _med.astype(np.float32),
        "composites": SOUS_DOSSIER, "dataset": DATASET_NAME,
        "ancrage": {"module": "ancrage_cascade_v3c14_rayon_max_v5s1neg_s040, axe cadrage_squelette_skan2", "ordre": "blastema, eyespots, darkness", "masques": "masks_model_Mask2Former_20260324_0451_2pass_clahe_2_t16_forme_raffine_v5s1neg_s040_annotfoi_lisse_env_p10_ouv3_skan2",
                    "PEAK_ANCRAGE": 0.85, "MARGE_BL": 0.84, "PEAK_THR": 0.61, "detecteur_yeux": "fondu_v5_jeu3E", "metriques": DATASET_NAME + "_v22",
                    "exactitude_ancrage": "see Claude outputs/essai_ancrage_cascade_v3c14_s040/verdict.txt"},
        "cv_bras5": {"exactitude_exacte": 0.7354, "bacc": 0.7053, "w05": 0.817,
                     "gain_bacc_vs_bras1": 0.0322, "ic95": [0.0106, 0.0542]},
        "epoques": EPOCHS, "graine": 42, "n_images_entrainement": len(idx),
        "date": datetime.date.today().isoformat()}, ck)
    if os.path.exists(tmp): os.replace(tmp, tmp + ".fini")
    print("   written %s (%.0f s)" % (ck, time.time() - t0))
    del model, opt, sch, tl; gc.collect(); torch.cuda.empty_cache()   # releasing the card memory
    return ck


tous = np.arange(len(paths))
if MODE_PROD == "unique":
    cks = [entrainer(tous, "stade_production_tout")]
else:
    gkf = GroupKFold(n_splits=KFOLD); cks = []
    for k, (tr, _) in enumerate(gkf.split(tous, labels, groups), 1):
        cks.append(entrainer(tr, "stade_production_pli%d" % k))
print("\nmodels written:", cks)

def charger_modele(chemin):
    # PyTorch 2.6 switched weights_only to True by default, which refuses the numpy arrays this
    # checkpoint carries -- the standardisation means and standard deviations. The file is produced
    # by this notebook, so its origin is trusted and it is loaded explicitly in full mode.
    try:
        c = torch.load(chemin, map_location=device, weights_only=False)
    except TypeError:
        c = torch.load(chemin, map_location=device)
    global METRIC_DIM
    METRIC_DIM = int(c["metric_dim"])
    m = Net(pretrained=False).to(device)
    m.load_state_dict(c["state_dict"]); m.eval()
    return m, c


def predire(models, X, M):
    """Mean of the predictions of one or more models. X already normalised, M already standardised."""
    with torch.no_grad():
        P = []
        for mdl in models:
            mu, lv, _, _ = mdl(X.to(device), M.to(device))
            P.append(mu.cpu().numpy())
        return np.mean(P, axis=0)


mods = []; conf = None
for ck in cks:
    m, c = charger_modele(ck); mods.append(m); conf = c
print("reloaded from the file alone | descriptors expected:", conf["metric_dim"],
      "| channels:", conf["ch_map"], "| composites:", conf["composites"])
assert list(conf["metric_cols"]) == METRIC_COLS

dl = DataLoader(DS(tous, False), batch_size=32, shuffle=False, num_workers=3)
P = np.zeros(len(tous), np.float32); o = 0
for x, m_, yd, yi in dl:
    p = predire(mods, x, m_); P[o:o + len(p)] = p; o += len(p)
cls = np.clip(np.rint(P), 1, NCLS).astype(int); vrai = labels + 1
print("\nchain check, on the TRAINING data, with no performance meaning")
print("   exact-class accuracy %.4f | tolerance +-0.5 %.4f" % ((cls == vrai).mean(), (np.abs(P - decs) <= 0.5).mean()))
pd.DataFrame({"image": [os.path.basename(p_) for p_ in paths], "classe_vraie": vrai,
              "stade_decimal_vrai": decs, "prediction": P, "classe_predite": cls,
              "groupe": groups}).to_csv("predictions_modele_production_v22.csv", sep=";", index=False)
print("written predictions_modele_production_v22.csv")
