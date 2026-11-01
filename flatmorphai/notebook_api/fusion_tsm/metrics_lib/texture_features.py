"""texture_features.py -- full formatted-image texture battery (shared by exploration, precompute, selection).
Deps: numpy, opencv-python, scikit-image. all_texture_features(gray, bgr, mask) -> dict over TEX_COLS."""
import numpy as np
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
# GPU-accelerated, self-validating drop-in for skimage.filters.gabor (falls
# back to the exact skimage CPU path when no GPU is available or on any
# mismatch, so metric values never change). See gabor_gpu.py.
try:
    from gabor_gpu import gabor  # sibling module on the metrics_lib path
    # Batched variant: one GPU convolution for all (frequency, theta) of a frame
    # instead of one launch each. Self-validating; identical results.
    from gabor_gpu import gabor_many
except Exception:  # noqa: BLE001
    from skimage.filters import gabor

    def gabor_many(image, specs):
        """Fallback: per-call skimage gabor, same return shape as the GPU one."""
        return [gabor(image, frequency=f, theta=t) for f, t in specs]
from skimage.morphology import skeletonize
from skimage.measure import regionprops, label as _lb
from collections import deque

def _spath(sk):
    pts=[tuple(p) for p in np.argwhere(sk)]; S=set(pts)
    if len(pts)<8: return None
    def nb(p):
        y,x=p
        for dy in(-1,0,1):
            for dx in(-1,0,1):
                if dy==0 and dx==0: continue
                q=(y+dy,x+dx)
                if q in S: yield q
    deg={p:sum(1 for _ in nb(p)) for p in pts}; ends=[p for p in pts if deg[p]==1]; start=ends[0] if ends else pts[0]
    def bfs(s):
        dist={s:0}; par={s:None}; dq=deque([s])
        while dq:
            u=dq.popleft()
            for v in nb(u):
                if v not in dist: dist[v]=dist[u]+1; par[v]=u; dq.append(v)
        f=max(dist,key=dist.get); return f,dist,par
    a,_,_=bfs(start); b,dist,par=bfs(a); path=[]; cur=b
    while cur is not None: path.append(cur); cur=par[cur]
    return path[::-1]
TEX_COLS=['ti_std','ti_skew','ti_kurt','ti_dark_contrast','ti_dark_frac','ti_p10p90',
          'te_int_edge_mean','te_int_edge_p99','te_int_to_bnd','te_edge_density','te_trans_edge_frac','te_lap_var',
          'tl_dip','tl_break','tl_abrupt','tl_range','tl_std','tl_n_notches',
          'tx_lbp_uniform','tx_lbp_entropy','tg_contrast','tg_homogeneity','tg_energy','tg_correlation','tb_gabor_mean','tb_gabor_aniso',
          'tk_bhat5_mean','tk_bhat5_max','tk_bhat11_mean','tk_bhat11_max','tk_darkspot_max','tk_darkspot_frac','tk_dark_blobs',
          'ts_neck_count','ts_neck_depth','ts_bulge_neck','ts_width_periodic','ts_dark_at_neck',
          'tc_coh_mean','tc_coh_drop','tf_dir_ratio','tf_spec_entropy','to_darkcore_max','to_darkcore_mean','to_bimodal_frac',
          'tcol_sat_mean','tcol_sat_std','tcol_hue_std']
def texture_metrics(gray,mask):
    out={c:np.nan for c in TEX_COLS}; m=(mask>0).astype(np.uint8)
    if m.sum()<80: return out
    ys,xs=np.where(m); y0,y1=ys.min(),ys.max()+1; x0,x1=xs.min(),xs.max()+1
    g=gray[y0:y1,x0:x1]; mm=m[y0:y1,x0:x1]; gf=g.astype(np.float32)
    body=gf[mm>0]; mu=float(body.mean()); sd=float(max(body.std(),1.0)); med=float(np.median(body)); ptp=float(max(np.ptp(body),1.0))
    out['ti_std']=sd/(mu+1e-6); out['ti_skew']=float(np.mean(((body-mu)/sd)**3)); out['ti_kurt']=float(np.mean(((body-mu)/sd)**4))
    out['ti_dark_contrast']=float((med-np.percentile(body,5))/(med+1e-6)); out['ti_dark_frac']=float((body<med-0.6*sd).mean()); out['ti_p10p90']=float(np.percentile(body,10)/(np.percentile(body,90)+1e-6))
    gb=cv2.GaussianBlur(gf,(0,0),1.0); gx=cv2.Sobel(gb,cv2.CV_32F,1,0,3); gy=cv2.Sobel(gb,cv2.CV_32F,0,1,3); gmag=np.sqrt(gx*gx+gy*gy); ang=np.arctan2(gy,gx)
    interior=cv2.erode(mm,np.ones((7,7),np.uint8)); inb=interior>0; boundary=(mm-cv2.erode(mm,np.ones((3,3),np.uint8)))>0
    ie=gmag[inb]/sd if interior.sum()>20 else gmag[mm>0]/sd; be=gmag[boundary]/sd if boundary.sum()>10 else ie
    out['te_int_edge_mean']=float(ie.mean()); out['te_int_edge_p99']=float(np.percentile(ie,99)); out['te_int_to_bnd']=float(ie.mean()/(be.mean()+1e-6)); out['te_edge_density']=float((ie>np.percentile(ie,90)).mean())
    pr=max(regionprops(_lb(mm).astype(int)),key=lambda p:p.area); ori=float(pr.orientation)
    strong=(inb)&(gmag>np.percentile(gmag[inb],90)) if interior.sum()>20 else (mm>0)
    if strong.sum()>5: out['te_trans_edge_frac']=float((np.abs((ang[strong]-(ori+np.pi/2)+np.pi/2)%np.pi-np.pi/2)<np.deg2rad(30)).mean())
    out['te_lap_var']=float(cv2.Laplacian(gb,cv2.CV_32F)[mm>0].var()/(sd*sd))
    dt=cv2.distanceTransform(mm,cv2.DIST_L2,5); sk=skeletonize(mm>0); path=_spath(sk)
    lm=cv2.blur(gf,(15,15)); loc=(lm-gf)/sd
    if path is not None and len(path)>=12:
        prof=np.convolve(np.array([gf[y,x] for y,x in path],float),np.ones(5)/5,mode='same'); dd=np.abs(np.diff(prof))/sd; d2=np.abs(np.diff(prof,2))/sd
        out['tl_dip']=float((np.median(prof)-prof.min())/(np.median(prof)+1e-6)); out['tl_break']=float(dd.max()); out['tl_abrupt']=float(d2.max())
        out['tl_range']=float((prof.max()-prof.min())/(med+1e-6)); out['tl_std']=float(prof.std()/sd)
        thr=med-0.5*sd; out['tl_n_notches']=float(np.sum((prof[1:-1]<prof[:-2])&(prof[1:-1]<prof[2:])&(prof[1:-1]<thr)))
        w=np.convolve(np.array([dt[y,x] for y,x in path],float),np.ones(3)/3,mode='same'); mnw=w.mean()+1e-6
        out['ts_neck_count']=float(np.sum((w[1:-1]<w[:-2])&(w[1:-1]<w[2:])&(w[1:-1]<0.85*mnw)))
        out['ts_neck_depth']=float((mnw-w.min())/mnw); out['ts_bulge_neck']=float(w.max()/(w.min()+1e-6))
        dw=w-w.mean(); ac=np.correlate(dw,dw,'full')[len(dw)-1:]; ac=ac/(ac[0]+1e-9)
        out['ts_width_periodic']=float(ac[2:max(len(dw)//2,3)].max()) if len(dw)>6 else 0.0
        ni=int(np.argmin(w)); yy,xx=path[ni]; out['ts_dark_at_neck']=float(loc[yy,xx]) if (0<=yy<loc.shape[0] and 0<=xx<loc.shape[1]) else np.nan
    q=np.clip((gf-body.min())/ptp*255,0,255).astype(np.uint8); lbp=local_binary_pattern(q,8,1,method='uniform'); lv=lbp[mm>0]
    h,_=np.histogram(lv,bins=np.arange(11),density=True); out['tx_lbp_uniform']=float(h[:9].sum()); out['tx_lbp_entropy']=float(-(h[h>0]*np.log2(h[h>0])).sum())
    gq=q.copy(); gq[mm==0]=int(np.clip((med-body.min())/ptp*255,0,255)); gq=(gq//16).astype(np.uint8)
    glcm=graycomatrix(gq,[1,2],[0,np.pi/2],levels=16,symmetric=True,normed=True)
    for prop in ['contrast','homogeneity','energy','correlation']: out['tg_'+prop]=float(graycoprops(glcm,prop).mean())
    _resp=gabor_many(gf/255.0,[(0.25,th) for th in (0,np.pi/4,np.pi/2,3*np.pi/4)])
    en=np.array([float(np.sqrt(rr*rr+ii*ii)[mm>0].mean()) for rr,ii in _resp])
    out['tb_gabor_mean']=float(en.mean()); out['tb_gabor_aniso']=float(en.std()/(en.mean()+1e-6))
    # subtle dark texture: morphological black-hat (small dark features) + local dark-spots
    for s in (5,11):
        bh=cv2.morphologyEx(g,cv2.MORPH_BLACKHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(s,s))).astype(np.float32)/sd
        v=bh[inb] if interior.sum()>20 else bh[mm>0]; out['tk_bhat%d_mean'%s]=float(v.mean()); out['tk_bhat%d_max'%s]=float(np.percentile(v,99))
    lvv=loc[inb] if interior.sum()>20 else loc[mm>0]
    out['tk_darkspot_max']=float(np.percentile(lvv,99)); out['tk_darkspot_frac']=float((lvv>1.0).mean())
    bh5=cv2.morphologyEx(g,cv2.MORPH_BLACKHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    nb_,_=cv2.connectedComponents(((bh5>max(8,2*sd)).astype(np.uint8))*(interior.astype(np.uint8))); out['tk_dark_blobs']=float(max(nb_-1,0))
    # --- structure-tensor orientation coherence + drop ---
    Jxx=cv2.GaussianBlur(gx*gx,(0,0),3); Jyy=cv2.GaussianBlur(gy*gy,(0,0),3); Jxy=cv2.GaussianBlur(gx*gy,(0,0),3)
    coh=np.sqrt((Jxx-Jyy)**2+4*Jxy*Jxy)/(Jxx+Jyy+1e-6); cbv=coh[mm>0]
    out['tc_coh_mean']=float(cbv.mean()); out['tc_coh_drop']=float(np.percentile(cbv,90)-np.percentile(cbv,10))
    # --- spectral: directional high-freq ratio + spectral entropy ---
    crop=gf*(mm>0); Fp=np.abs(np.fft.fftshift(np.fft.fft2(crop)))**2; hh,ww=Fp.shape; cyy,cxx=hh//2,ww//2
    YY,XX=np.mgrid[0:hh,0:ww]; rad=np.sqrt((YY-cyy)**2+(XX-cxx)**2); hi=rad>0.25*min(hh,ww)
    eh=Fp[hi&(np.abs(XX-cxx)>np.abs(YY-cyy))].sum(); ev=Fp[hi&(np.abs(YY-cyy)>=np.abs(XX-cxx))].sum()
    out['tf_dir_ratio']=float(min(eh,ev)/(max(eh,ev)+1e-9))
    pp=Fp[hi]/(Fp[hi].sum()+1e-9); out['tf_spec_entropy']=float(-(pp[pp>0]*np.log2(pp[pp>0])).sum()/np.log2(max(int(hi.sum()),2)))
    # --- cross-body self-overlap: transverse profiles, dark-core / bimodal (a fold = double thickness) ---
    if path is not None and len(path)>=15:
        Pp=np.array(path,float); dc=[]; bm=[]; idxs=np.linspace(2,len(Pp)-3,min(20,len(Pp)-4)).astype(int)
        for ii in idxs:
            tv=Pp[ii+1]-Pp[ii-1]; nv=np.array([-tv[1],tv[0]]); nv=nv/(np.linalg.norm(nv)+1e-9); s=[]
            for dd2 in range(-15,16):
                yy2=int(round(Pp[ii,0]+nv[0]*dd2)); xx2=int(round(Pp[ii,1]+nv[1]*dd2))
                if 0<=yy2<gf.shape[0] and 0<=xx2<gf.shape[1] and mm[yy2,xx2]: s.append(gf[yy2,xx2])
            if len(s)>=6:
                s=np.array(s); c=len(s)//2; edge=(s[:2].mean()+s[-2:].mean())/2; center=s[max(c-1,0):c+2].mean()
                dc.append((edge-center)/sd); bm.append(1.0 if (s.max()-s.min())>1.2*sd and s.argmin() not in (0,len(s)-1) else 0.0)
        if dc: out['to_darkcore_max']=float(np.max(dc)); out['to_darkcore_mean']=float(np.mean(dc)); out['to_bimodal_frac']=float(np.mean(bm))
    return out


def color_metrics(bgr,mask):
    out={'tcol_sat_mean':np.nan,'tcol_sat_std':np.nan,'tcol_hue_std':np.nan}; m=(mask>0)
    if m.sum()<80 or bgr is None: return out
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV); s=hsv[:,:,1][m].astype(np.float32); h=hsv[:,:,0][m].astype(np.float32)
    out['tcol_sat_mean']=float(s.mean()/255.0); out['tcol_sat_std']=float(s.std()/255.0); out['tcol_hue_std']=float(h.std())
    return out


# ================== ADDITIONAL FAMILIES (Laws, GLRLM, Gabor bank, fractal, matched twist) ==================
_LAWS=dict(L5=np.array([1,4,6,4,1.]),E5=np.array([-1,-2,0,2,1.]),S5=np.array([-1,0,2,0,-1.]),R5=np.array([1,-4,6,-4,1.]))
LAWS_COLS=['tw_L5E5','tw_L5S5','tw_L5R5','tw_E5S5','tw_E5R5','tw_S5R5','tw_E5E5','tw_S5S5','tw_R5R5']
def laws_features(g,m):
    out={c:np.nan for c in LAWS_COLS}
    if (m>0).sum()<80: return out
    gf=g.astype(np.float32)-cv2.blur(g.astype(np.float32),(15,15)); mp={}
    for a in ['L5','E5','S5','R5']:
        for b in ['L5','E5','S5','R5']: mp[a+b]=np.abs(cv2.filter2D(gf,cv2.CV_32F,np.outer(_LAWS[a],_LAWS[b])))
    en=lambda n: float(mp[n][m>0].mean()); base=en('L5L5')+1e-6
    for p in [('L5E5','E5L5'),('L5S5','S5L5'),('L5R5','R5L5'),('E5S5','S5E5'),('E5R5','R5E5'),('S5R5','R5S5')]:
        out['tw_'+p[0]]=(en(p[0])+en(p[1]))/2/base
    for d in ['E5E5','S5S5','R5R5']: out['tw_'+d]=en(d)/base
    return out
GLRLM_COLS=['tr_SRE','tr_LRE','tr_GLN','tr_RLN','tr_RP']
def glrlm_features(g,m,L=16,maxdim=96):
    out={c:np.nan for c in GLRLM_COLS}
    if (m>0).sum()<80: return out
    ys,xs=np.where(m>0); g2=g[ys.min():ys.max()+1,xs.min():xs.max()+1]; m2=(m[ys.min():ys.max()+1,xs.min():xs.max()+1]>0)
    sc=maxdim/max(g2.shape)
    if sc<1.0:
        g2=cv2.resize(g2,(max(int(g2.shape[1]*sc),8),max(int(g2.shape[0]*sc),8))); m2=cv2.resize(m2.astype(np.uint8),(g2.shape[1],g2.shape[0]))>0
    body=g2[m2]; q=np.clip(((g2.astype(np.float32)-body.min())/(np.ptp(body)+1e-6)*(L-1)),0,L-1).astype(np.int32); q[~m2]=-1
    H,W=q.shape; P=np.zeros((L,max(H,W)+1),float)
    for dy,dx in [(0,1),(1,0),(1,1),(1,-1)]:
        seen=np.zeros_like(q,bool)
        for y in range(H):
            for x in range(W):
                if q[y,x]<0 or seen[y,x]: continue
                py,px=y-dy,x-dx
                if 0<=py<H and 0<=px<W and q[py,px]==q[y,x]: continue
                rl=0; cy,cx=y,x; val=q[y,x]
                while 0<=cy<H and 0<=cx<W and q[cy,cx]==val: seen[cy,cx]=True; rl+=1; cy+=dy; cx+=dx
                P[val,rl]+=1
    nr=P.sum()+1e-9; j=np.arange(P.shape[1])[None,:]
    out['tr_SRE']=float((P/(j**2+1e-9)).sum()/nr); out['tr_LRE']=float((P*j**2).sum()/nr)
    out['tr_GLN']=float((P.sum(1)**2).sum()/nr); out['tr_RLN']=float((P.sum(0)**2).sum()/nr); out['tr_RP']=float(nr/(m2.sum()+1e-9))
    return out
GABORBANK_COLS=['tgb_aniso0','tgb_aniso1','tgb_aniso2','tgb_hl_ratio','tgb_mean']
def gaborbank_features(g,m):
    out={c:np.nan for c in GABORBANK_COLS}
    if (m>0).sum()<80: return out
    gf=g/255.0; pf=[]; thetas=(0,np.pi/4,np.pi/2,3*np.pi/4)
    freqs=(0.1,0.2,0.4)
    # One batched GPU convolution for all 3x4 (frequency, theta) kernels.
    _resp=gabor_many(gf,[(f,t) for f in freqs for t in thetas])
    for fi in range(len(freqs)):
        en=np.array([float(np.sqrt(rr*rr+ii*ii)[m>0].mean())
                     for rr,ii in _resp[fi*len(thetas):(fi+1)*len(thetas)]])
        pf.append(en.mean()); out['tgb_aniso%d'%fi]=float(en.std()/(en.mean()+1e-6))
    pf=np.array(pf); out['tgb_hl_ratio']=float(pf[2]/(pf[0]+1e-6)); out['tgb_mean']=float(pf.mean())
    return out
FRACTAL_COLS=['tfr_dim','tfr_lac']
def fractal_features(g,m):
    out={c:np.nan for c in FRACTAL_COLS}
    mm=(m>0); ys,xs=np.where(mm)
    if len(ys)<40: return out
    e=(cv2.Canny(g,40,120)>0)&mm; sub=e[ys.min():ys.max()+1,xs.min():xs.max()+1]
    sizes=[2,4,8,16]; cnts=[]
    for s in sizes:
        h,w=sub.shape; c=0
        for yy in range(0,h,s):
            for xx in range(0,w,s):
                if sub[yy:yy+s,xx:xx+s].any(): c+=1
        cnts.append(max(c,1))
    out['tfr_dim']=float(-np.polyfit(np.log(sizes),np.log(cnts),1)[0])
    box=8; ms=[]
    for yy in range(0,mm.shape[0]-box,box):
        for xx in range(0,mm.shape[1]-box,box):
            b=mm[yy:yy+box,xx:xx+box]
            if b.any(): ms.append(g[yy:yy+box,xx:xx+box][b].mean())
    ms=np.array(ms); out['tfr_lac']=float(ms.var()/(ms.mean()**2+1e-6)) if len(ms)>3 else np.nan
    return out
MATCH_COLS=['tm_twist_max']
def matched_twist(g,m,ori):
    out={c:np.nan for c in MATCH_COLS}
    if (m>0).sum()<80: return out
    gf=cv2.GaussianBlur(g.astype(np.float32),(0,0),1.0)/255.0; best=0.0
    _resp=gabor_many(gf,[(0.3,ori+np.pi/2+dth) for dth in (-0.4,0.0,0.4)])
    for r,_i in _resp:
        v=(-r)[m>0]; best=max(best,float(np.percentile(v,99)))
    out['tm_twist_max']=best; return out

EXTRA_COLS=LAWS_COLS+GLRLM_COLS+GABORBANK_COLS+FRACTAL_COLS+MATCH_COLS
ALL_TEX_COLS=list(dict.fromkeys(list(TEX_COLS)+['tcol_sat_mean','tcol_sat_std','tcol_hue_std']+EXTRA_COLS))

def all_texture_features(gray, bgr, mask):
    """gray: formatted grayscale; bgr: formatted BGR (or None to skip colour); mask: foreground."""
    out={k: np.nan for k in ALL_TEX_COLS}
    out.update(texture_metrics(gray, mask))
    out.update(color_metrics(bgr, mask) if bgr is not None else {})
    out.update(laws_features(gray, mask)); out.update(glrlm_features(gray, mask)); out.update(gaborbank_features(gray, mask)); out.update(fractal_features(gray, mask))
    m=(mask>0).astype(np.uint8); ori=0.0
    if m.sum()>=80:
        try: ori=float(max(regionprops(_lb(m).astype(int)),key=lambda p:p.area).orientation)
        except Exception: ori=0.0
    out.update(matched_twist(gray, mask, ori))
    return out
