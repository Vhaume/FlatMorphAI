"""metrics_3mask.py  -- self-contained 3-mask metric computation for the fusion model.
Dependencies: numpy, opencv-python (cv2), scikit-image.
compute_metric_vector(prev_mask, cur_mask, next_mask, cur_gray) -> dict of all metric names.
Masks are HxW uint8/bool at the ORIGINAL frame resolution. cur_gray is the raw current-frame grayscale.
"""
import numpy as np
import cv2

def _clean(m):
    m=(m>0).astype(np.uint8)
    if m.sum()==0: return m
    n,lab=cv2.connectedComponents(m)
    if n>1:
        sizes=[(lab==i).sum() for i in range(1,n)]; m=(lab==(1+int(np.argmax(sizes)))).astype(np.uint8)
    ff=(m*255).copy(); h,w=ff.shape; fl=ff.copy(); cv2.floodFill(fl,np.zeros((h+2,w+2),np.uint8),(0,0),255)
    return (m|(cv2.bitwise_not(fl)>0)).astype(np.uint8)

# ---- scale-free descriptors: static (one mask) and dynamic (the 3-frame window) ----
from skimage.measure import regionprops, label as _sklabel
from skimage.morphology import skeletonize

def _largest(m):
    n, lab = cv2.connectedComponents((m > 0).astype(np.uint8))
    if n <= 1: return (m > 0)
    s = [(lab == i).sum() for i in range(1, n)]
    return lab == (1 + int(np.argmax(s)))

def shape_desc(m):
    if m is None: return None
    m = (m > 0).astype(np.uint8)
    if m.sum() < 15: return None
    m = _largest(m).astype(np.uint8)
    ys, xs = np.where(m)
    pad = 4; y0, y1 = max(ys.min() - pad, 0), ys.max() + pad + 1; x0, x1 = max(xs.min() - pad, 0), xs.max() + pad + 1
    c = m[y0:y1, x0:x1]
    pr = max(regionprops(_sklabel(c).astype(int)), key=lambda p: p.area)
    area = float(pr.area); per = float(pr.perimeter) or 1.0
    L = float(pr.major_axis_length); W = float(pr.minor_axis_length) or 1e-6
    s = np.sqrt(area) + 1e-6
    hu = cv2.HuMoments(cv2.moments(c * 255)).flatten()
    hulog = [float(-np.sign(h) * np.log10(abs(h) + 1e-30)) for h in hu]
    cnts, _ = cv2.findContours(c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    hull_area = cv2.contourArea(cv2.convexHull(cnt)) + 1e-6
    fold = float((hull_area - area) / hull_area)
    hull_idx = cv2.convexHull(cnt, returnPoints=False)
    ndef, maxdef = 0, 0.0
    if hull_idx is not None and len(hull_idx) > 3 and len(cnt) > 3:
        d = cv2.convexityDefects(cnt, hull_idx)
        if d is not None:
            depths = d.reshape(-1, 4)[:, 3] / 256.0; depths = depths[depths > 0.02 * s]
            ndef = int(len(depths)); maxdef = float(depths.max() / s) if len(depths) else 0.0
    pts = cnt[:, 0, :].astype(np.float64)
    if len(pts) >= 8:
        step = max(1, len(pts) // 64); p = pts[::step]
        v = np.roll(p, -1, 0) - p; ang = np.arctan2(v[:, 1], v[:, 0])
        dang = np.diff(np.unwrap(np.concatenate([ang, ang[:1]])))
        tot_turn = float(np.sum(np.abs(dang)) / (2 * np.pi)); max_turn = float(np.max(np.abs(dang)))
    else:
        tot_turn, max_turn = 1.0, 0.0
    sk = skeletonize(c > 0); skn = int(sk.sum())
    nb = cv2.filter2D(sk.astype(np.uint8), -1, np.ones((3, 3), np.uint8))
    branch = int(((nb >= 4) & sk).sum())
    endpts = np.argwhere((nb == 2) & sk)
    if len(endpts) >= 2:
        dd = np.hypot(endpts[:, 0, None] - endpts[None, :, 0], endpts[:, 1, None] - endpts[None, :, 1]); span = float(dd.max())
    else:
        yy, xx = np.where(sk); span = float(np.hypot(yy.max() - yy.min(), xx.max() - xx.min())) if len(yy) else 1.0
    skel_tort = float(skn / (span + 1e-6))
    return dict(area=area, elong=L / (W + 1e-6), wl=W / (L + 1e-6), ecc=float(pr.eccentricity),
                sol=float(pr.solidity), ext=float(pr.extent), thin=4 * np.pi * area / (per * per + 1e-6),
                circ=per * per / (4 * np.pi * area + 1e-6), fold=fold, n_def=float(ndef), max_def=maxdef,
                tot_turn=tot_turn, max_turn=max_turn, skel_tort=skel_tort, branch=float(branch),
                hu1=hulog[0], hu2=hulog[1], hu3=hulog[2],
                cy=float(pr.centroid[0]) + y0, cx=float(pr.centroid[1]) + x0, ori=float(pr.orientation),
                full=(m > 0))

def _angdiff(a, b):
    d = abs(a - b) % np.pi; return float(min(d, np.pi - d))
def _aligned_symdiff(da, db):
    ma, mb = da['full'], db['full']
    if ma.shape != mb.shape: return np.nan
    dy = int(round(da['cy'] - db['cy'])); dx = int(round(da['cx'] - db['cx']))
    mb2 = np.roll(np.roll(mb, dy, 0), dx, 1)
    inter = np.logical_and(ma, mb2).sum(); union = np.logical_or(ma, mb2).sum()
    return float((union - inter) / (union + 1e-6))

STATIC_COLS = ['elong_T0','wl_T0','ecc_T0','thin_T0','circ_T0','sol_T0','ext_T0','fold_T0','maxdef_T0','ndef_T0',
               'totturn_T0','maxturn_T0','skeltort_T0','branch_T0','hu1_T0','hu2_T0','hu3_T0']
DYN_COLS = ['net_disp','path_len','straightness','dir_cos','ori_tv',
            'elong_chg','wl_chg','ecc_chg','sol_chg','thin_chg','fold_chg','maxdef_chg','totturn_chg','circ_chg',
            'elong_osc','ecc_osc','wl_osc','sol_osc','thin_osc','fold_osc','circ_osc','area_cv',
            'scrunch','symdiff_AB','symdiff_BC']


# ---- candidate twist-specific metrics (single frame): medial-axis thickness peak and self-overlap darkness ----
NEW_COLS = ['width_peak', 'width_cv', 'width_max', 'dark_contrast', 'dark_frac']
def new_metrics(mask, gray):
    m = (mask > 0).astype(np.uint8); out = {k: np.nan for k in NEW_COLS}
    if m.sum() < 15: return out
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5); sk = skeletonize(m > 0)
    dts = dt[sk] if sk.sum() > 0 else dt[m > 0]
    s = np.sqrt(float(m.sum())) + 1e-6; mu = float(dts.mean()) + 1e-6
    out['width_peak'] = float(dts.max() / mu); out['width_cv'] = float(dts.std() / mu); out['width_max'] = float(dts.max() / s)
    if gray is not None and getattr(gray, 'shape', None) == m.shape:
        g = gray[m > 0].astype(np.float32); med = float(np.median(g)); p5 = float(np.percentile(g, 5)); sd = float(g.std()) + 1e-6
        out['dark_contrast'] = float((med - p5) / (med + 1e-6)); out['dark_frac'] = float((g < med - 0.6 * sd).mean())
    return out



# ---- curvature metrics (arc-vs-undulation), mask midline, T0 + evolution over the 3 masks ----
from collections import deque as _deque
def _spath(sk):
    pts=[tuple(p) for p in np.argwhere(sk)]; S=set(pts)
    if len(pts)<6: return None
    def _nb(p):
        y,x=p
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if dy==0 and dx==0: continue
                q=(y+dy,x+dx)
                if q in S: yield q
    deg={p:sum(1 for _ in _nb(p)) for p in pts}; ends=[p for p in pts if deg[p]==1]; start=ends[0] if ends else pts[0]
    def _bfs(s):
        dist={s:0}; par={s:None}; dq=_deque([s])
        while dq:
            u=dq.popleft()
            for v in _nb(u):
                if v not in dist: dist[v]=dist[u]+1; par[v]=u; dq.append(v)
        f=max(dist,key=dist.get); return f,dist,par
    a,_,_=_bfs(start); b,dist,par=_bfs(a); path=[]; cur=b
    while cur is not None: path.append(cur); cur=par[cur]
    return path[::-1]
def _curv_profile(mask,N=40):
    sk=skeletonize((mask>0)); path=_spath(sk)
    if path is None: return None
    P=np.array(path,float); d=np.r_[0,np.cumsum(np.linalg.norm(np.diff(P,axis=0),axis=1))]
    if d[-1]<1e-6: return None
    u=np.linspace(0,d[-1],N); Px=np.interp(u,d,P[:,1]); Py=np.interp(u,d,P[:,0])
    ang=np.unwrap(np.arctan2(np.gradient(Py),np.gradient(Px))); return np.convolve(np.gradient(ang),np.ones(5)/5,mode='same')
def _curv_align(a,b):
    best=b; bc=-1
    for c in (b,b[::-1],-b,-b[::-1]):
        cc=abs(np.corrcoef(a,c)[0,1]) if np.std(a)>1e-9 and np.std(c)>1e-9 else 0
        if cc>bc: bc=cc; best=c
    return best
CURV_T0_COLS=['curv_mean_abs','curv_max_abs','curv_bending_energy','curv_net_to_total','curv_infl_count','curv_std']
def curv_metrics_T0(mask):
    out={c:np.nan for c in CURV_T0_COLS}; k=_curv_profile(mask)
    if k is None: return out
    ak=np.abs(k); total=ak.sum()+1e-6; net=abs(k.sum()); mx=ak.max()+1e-12
    ks=k[ak>0.05*mx]; sg=np.sign(ks)
    out['curv_mean_abs']=float(ak.mean()); out['curv_max_abs']=float(mx); out['curv_bending_energy']=float((k*k).mean())
    out['curv_net_to_total']=float(net/total); out['curv_infl_count']=float(np.sum(sg[1:]!=sg[:-1])) if len(sg)>1 else 0.0
    out['curv_std']=float(np.std(k))
    return out
CURV_DYN_COLS=['curv_undulation','curv_mean_chg','curv_bending_chg','curv_std_osc']
def curv_dynamic(mA,mB,mC):
    out={c:np.nan for c in CURV_DYN_COLS}; kA,kB,kC=_curv_profile(mA),_curv_profile(mB),_curv_profile(mC)
    if kB is None: return out
    chg=[]
    for kk in (kA,kC):
        if kk is None: continue
        chg.append(float(np.mean(np.abs(kB-_curv_align(kB,kk)))))
    out['curv_undulation']=float(np.mean(chg)) if chg else np.nan
    mets=[curv_metrics_T0(m) for m in (mA,mB,mC)]
    def seq(key): return [mm[key] for mm in mets if mm.get(key)==mm.get(key)]
    def chgk(key): v=seq(key); return float(np.mean(np.abs(np.diff(v)))) if len(v)>1 else np.nan
    def osck(key): v=seq(key); return float(np.std(v)) if len(v)>1 else np.nan
    out['curv_mean_chg']=chgk('curv_mean_abs'); out['curv_bending_chg']=chgk('curv_bending_energy'); out['curv_std_osc']=osck('curv_std')
    return out



# ---- circular-arc metrics (circling Wildtype keeps a constant-radius arc) ----
CIRC_COLS=['circ_fit_resid','circ_radius_norm','arc_span','circ_resid_chg','circ_radius_chg','rigid_translation','moved_kept_arc']
def _circle_fit(mask):
    sk=skeletonize(mask>0); path=_spath(sk)
    if path is None: return None
    P=np.array(path,float); y=P[:,0]; x=P[:,1]
    A=np.c_[x,y,np.ones(len(x))]; bb=-(x*x+y*y)
    try: sol,_,_,_=np.linalg.lstsq(A,bb,rcond=None)
    except Exception: return None
    D,E,F=sol; cx,cy=-D/2,-E/2; R=np.sqrt(max((D*D+E*E)/4-F,1e-6))
    r=np.sqrt((x-cx)**2+(y-cy)**2); s=np.sqrt(float(mask.sum())); ang=np.unwrap(np.arctan2(y-cy,x-cx))
    return float(np.std(r-R)/(R+1e-6)), float(R/s), float(abs(ang[-1]-ang[0]))
def circ_metrics_T0(mask):
    out={'circ_fit_resid':np.nan,'circ_radius_norm':np.nan,'arc_span':np.nan}; c=_circle_fit(mask)
    if c is not None: out['circ_fit_resid'],out['circ_radius_norm'],out['arc_span']=c
    return out
def circ_dynamic(mA,mB,mC):
    out={'circ_resid_chg':np.nan,'circ_radius_chg':np.nan}
    cs=[_circle_fit(m) for m in (mA,mB,mC)]; rr=[c[0] for c in cs if c is not None]; rad=[c[1] for c in cs if c is not None]
    if len(rr)>1: out['circ_resid_chg']=float(np.mean(np.abs(np.diff(rr))))
    if len(rad)>1: out['circ_radius_chg']=float(np.mean(np.abs(np.diff(rad))))
    return out

METRIC_COLS = STATIC_COLS + DYN_COLS + NEW_COLS + CURV_T0_COLS + CURV_DYN_COLS + CIRC_COLS

def static_features(B):
    return dict(elong_T0=B['elong'], wl_T0=B['wl'], ecc_T0=B['ecc'], thin_T0=B['thin'], circ_T0=B['circ'],
                sol_T0=B['sol'], ext_T0=B['ext'], fold_T0=B['fold'], maxdef_T0=B['max_def'], ndef_T0=B['n_def'],
                totturn_T0=B['tot_turn'], maxturn_T0=B['max_turn'], skeltort_T0=B['skel_tort'], branch_T0=B['branch'],
                hu1_T0=B['hu1'], hu2_T0=B['hu2'], hu3_T0=B['hu3'])

def dyn_features(d3, Bn):
    # d3 = three real, temporally ordered frame descriptors; Bn = the true current frame (size normalizer)
    A, B, C = d3
    s = np.sqrt(Bn['area']) + 1e-6
    cs = np.array([[A['cx'], A['cy']], [B['cx'], B['cy']], [C['cx'], C['cy']]], float)
    steps = np.linalg.norm(np.diff(cs, axis=0), axis=1) / s
    net = float(np.linalg.norm(cs[2] - cs[0]) / s); path = float(steps.sum())
    straight = float(net / (path + 1e-6))
    vAB = cs[1] - cs[0]; vBC = cs[2] - cs[1]; nv = np.linalg.norm(vAB) * np.linalg.norm(vBC)
    dir_cos = float(vAB @ vBC / nv) if nv > 1e-6 else 0.0
    ori_tv = _angdiff(A['ori'], B['ori']) + _angdiff(B['ori'], C['ori'])
    def colk(k): return np.array([A[k], B[k], C[k]], float)
    def osc(k): return float(colk(k).std())
    def chg(k): return float(np.mean(np.abs(np.diff(colk(k)))))
    return dict(net_disp=net, path_len=path, straightness=straight, dir_cos=dir_cos, ori_tv=float(ori_tv),
                elong_chg=chg('elong'), wl_chg=chg('wl'), ecc_chg=chg('ecc'), sol_chg=chg('sol'), thin_chg=chg('thin'),
                fold_chg=chg('fold'), maxdef_chg=chg('max_def'), totturn_chg=chg('tot_turn'), circ_chg=chg('circ'),
                elong_osc=osc('elong'), ecc_osc=osc('ecc'), wl_osc=osc('wl'), sol_osc=osc('sol'), thin_osc=osc('thin'),
                fold_osc=osc('fold'), circ_osc=osc('circ'), area_cv=float(colk('area').std() / (colk('area').mean() + 1e-6)),
                scrunch=float((chg('elong') + chg('thin')) / (net + 0.05)),
                symdiff_AB=_aligned_symdiff(A, B), symdiff_BC=_aligned_symdiff(B, C))

def compute_metric_vector(prev_mask, cur_mask, next_mask, cur_gray):
    """Return a dict with every metric in METRIC_COLS for one (prev,cur,next) triplet."""
    feat={k: np.nan for k in METRIC_COLS}
    cm=_clean(cur_mask); Bc=shape_desc(cm)
    if Bc is not None:
        feat.update(static_features(Bc))
        feat.update(new_metrics(cm, cur_gray))
        feat.update(curv_metrics_T0(cm)); feat.update(circ_metrics_T0(cm))
    md3=[_clean(prev_mask), cm, _clean(next_mask)]; d3=[shape_desc(x) for x in md3]
    if all(d is not None for d in d3):
        try: feat.update(dyn_features(d3, Bc if Bc is not None else d3[1]))
        except Exception: pass
    feat.update(curv_dynamic(md3[0], md3[1], md3[2])); feat.update(circ_dynamic(md3[0], md3[1], md3[2]))
    nd=feat.get("net_disp",np.nan); rl=feat.get("rel_len_chg",np.nan); ec=feat.get("ecc_chg",np.nan); cr=feat.get("circ_resid_chg",np.nan)
    if nd==nd: feat["rigid_translation"]=nd/((0 if rl!=rl else rl)+(0 if ec!=ec else ec)+0.05)
    if nd==nd and cr==cr: feat["moved_kept_arc"]=nd/(cr+0.05)
    return feat
