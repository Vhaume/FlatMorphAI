"""fold_twist_metrics.py — validated SOTA fold/twist descriptors + image metrics. Shared module."""
import os, re, numpy as np, cv2
from skimage.filters import frangi, sato as _sato, meijering as _meij
from skimage.feature import hessian_matrix_eigvals
from skimage.morphology import skeletonize

# Hessian / vesselness on CPU (skimage) BY DEFAULT.
#
# A GPU drop-in for skimage.feature.hessian_matrix exists (hessian_gpu.py) and is
# exact to ~1e-6 single-stream, BUT it decomposes Frangi/Sato/Meijering into ~140
# tiny GPU convolutions per frame; called from the descriptor THREAD POOL (7
# workers contending on one GTX 1060), the launch/transfer overhead serialises and
# makes a run far SLOWER than skimage's CPU path, which parallelises well across
# the threads (its C code releases the GIL). skimage CPU is also the exact float64
# reference the golden (74/79) was computed with. So the GPU path is OPT-IN only,
# via FLATMORPH_V4J_GPU_CONV=1, for single-stream use; the default is skimage CPU.
if os.environ.get("FLATMORPH_V4J_GPU_CONV", "0") in ("1", "true", "True"):
    try:
        from hessian_gpu import hessian_matrix  # sibling module on metrics_lib path
        import skimage.filters.ridges as _ridges
        _ridges.hessian_matrix = hessian_matrix
    except Exception:  # noqa: BLE001 - keep pure-skimage behaviour if unavailable
        from skimage.feature import hessian_matrix
else:
    from skimage.feature import hessian_matrix
from scipy.spatial import cKDTree
from collections import deque

CYAN_LO, CYAN_HI = (80,80,80), (100,255,255)

# ---- helpers: masks, cyan extraction, the SOTA dense descriptor bank ----
def body_mask(gray, thr=18):
    m=(gray>thr).astype(np.uint8); n,lab=cv2.connectedComponents(m)
    if n>1: m=(lab==(1+int(np.argmax([(lab==i).sum() for i in range(1,n)])))).astype(np.uint8)
    return m>0

def extract(path):
    bgr=cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_COLOR)
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV); cyan=cv2.inRange(hsv,CYAN_LO,CYAN_HI)>0
    clean=cv2.inpaint(bgr,(cyan*255).astype(np.uint8),3,cv2.INPAINT_TELEA)
    gray=cv2.cvtColor(clean,cv2.COLOR_BGR2GRAY); worm=body_mask(gray)|cyan
    n,lab=cv2.connectedComponents(worm.astype(np.uint8))
    if n>1: worm=lab==(1+int(np.argmax([(lab==i).sum() for i in range(1,n)])))
    return gray, worm, cyan

def _hessian_ridge(g):
    try: H=hessian_matrix(g, sigma=2.0, order='rc', use_gaussian_derivatives=True)
    except TypeError: H=hessian_matrix(g, sigma=2.0, order='rc')
    l1,l2=hessian_matrix_eigvals(H); return np.maximum(l1,0)   # dark line -> positive 2nd deriv

DESC=['frangi_dark','hess_dark','dark_od','blackhat','coh_drop','width_anom','self_prox','twist_cross','pc_energy','pc_sym','offtex','sato_dark','meijering_dark']
def dense_maps(gray, worm):
    g0=gray.astype(np.float32)
    # robust per-worm intensity normalization: cancels the formatter's equalization + pigment (yellow vs brown)
    # global bias, and ignores bright segmentation-leak pixels, so the ridge metrics see STRUCTURE not color/brightness.
    _vm = worm & (g0 < 235)
    if int(_vm.sum()) > 50:
        _lo,_hi = np.percentile(g0[_vm], [5, 95]); g = np.clip((g0-_lo)/(_hi-_lo+1e-6), 0, 1).astype(np.float32)*255.0
    else:
        g = g0
    m={}
    m['frangi_dark']=np.nan_to_num(frangi(g/255.0,sigmas=range(1,6,1),black_ridges=True))
    m['hess_dark']=_hessian_ridge(g)
    bg=cv2.GaussianBlur(g,(0,0),9); loc=cv2.GaussianBlur((g-bg)**2,(0,0),9)**0.5+1e-3
    m['dark_od']=np.clip(bg-g,0,None)/loc
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)); m['blackhat']=cv2.morphologyEx(g,cv2.MORPH_BLACKHAT,k)
    gx=cv2.Sobel(g,cv2.CV_32F,1,0,3); gy=cv2.Sobel(g,cv2.CV_32F,0,1,3)
    Jxx=cv2.GaussianBlur(gx*gx,(0,0),3); Jyy=cv2.GaussianBlur(gy*gy,(0,0),3); Jxy=cv2.GaussianBlur(gx*gy,(0,0),3)
    coh=np.sqrt((Jxx-Jyy)**2+4*Jxy**2)/(Jxx+Jyy+1e-6); m['coh_drop']=1.0-np.clip(coh,0,1)
    ys2,xs2=np.where(worm)
    if len(ys2)>10:
        cyy=ys2-ys2.mean(); cxx=xs2-xs2.mean(); axis=0.5*np.arctan2(2*np.mean(cyy*cxx),(cxx*cxx).mean()-(cyy*cyy).mean())
    else: axis=0.0
    ori=0.5*np.arctan2(2*Jxy,Jxx-Jyy); m['offtex']=np.clip(coh,0,1)*np.sin(ori-axis)**2
    dt=cv2.distanceTransform(worm.astype(np.uint8),cv2.DIST_L2,5); m['width_anom']=np.abs(dt-cv2.GaussianBlur(dt,(0,0),11))
    sp=np.zeros_like(g); S=np.argwhere(skeletonize(worm))
    if len(S)>8:
        d,_=cKDTree(S).query(S,k=min(24,len(S)))
        nn=np.array([row[row>12.0][0] if np.any(row>12.0) else 1e3 for row in d])
        for (yy,xx),v in zip(S,np.exp(-nn/18.0)): sp[yy,xx]=v
        sp=cv2.dilate(sp,np.ones((9,9),np.float32))
    m['self_prox']=sp
    m['twist_cross']=twist_map(gray,worm)
    gg=gray.astype(np.float32)/255.0
    m['sato_dark']=np.nan_to_num(_sato(gg,sigmas=range(1,5),black_ridges=True))
    m['meijering_dark']=np.nan_to_num(_meij(gg,sigmas=range(1,5),black_ridges=True))
    en,sym=pc_energy_map(gray,worm); m['pc_energy']=en; m['pc_sym']=sym
    for kk in m: m[kk]=np.nan_to_num(m[kk]).astype(np.float32)*worm
    return m

def contour_cues(worm):
    cs,_=cv2.findContours(worm.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    o=dict(bnd_concav_max=0.0,bnd_concav_n=0.0)
    if not cs: return o
    c=max(cs,key=cv2.contourArea)[:,0,:].astype(np.float32); n=len(c)
    if n<31: return o
    K=15; a=np.roll(c,K,0); b=c; d=np.roll(c,-K,0); v1=a-b; v2=d-b
    cosv=(v1*v2).sum(1)/(np.linalg.norm(v1,axis=1)*np.linalg.norm(v2,axis=1)+1e-6)
    ang=np.arccos(np.clip(cosv,-1,1)); cross=(a-b)[:,0]*(d-b)[:,1]-(a-b)[:,1]*(d-b)[:,0]
    concav=np.where(cross<0,np.pi-ang,0.0)
    o['bnd_concav_max']=float(concav.max()); o['bnd_concav_n']=float((concav>np.pi*0.5).sum()); return o

def image_metrics(gray, worm):
    mp=dense_maps(gray,worm); area=max(int(worm.sum()),1); out={}; g=gray.astype(np.float32)
    worm_e=cv2.erode(worm.astype(np.uint8),np.ones((9,9),np.uint8))>0      # interior: drops the ~4px boundary ring
    if worm_e.sum()<50: worm_e=worm
    # exclude BRIGHT non-worm pixels (segmentation leak + glare): they bias the intensity-based fold metrics
    BRIGHT=235.0
    valid=worm_e & (g<BRIGHT)
    if valid.sum()<50: valid=worm_e
    ring=worm&(~worm_e); area_v=max(int(valid.sum()),1)
    for kk,v in mp.items():
        on=v[valid]; med=float(np.median(on))                             # peaks on the interior, leak/glare pixels removed
        out[kk+'_max']=float(on.max()); out[kk+'_p99']=float(np.percentile(on,99))
        out[kk+'_prom']=float(np.percentile(on,99)-med)
        hi=(v>(on.mean()+2*on.std()))&valid; out[kk+'_hifrac']=float(hi.sum()/area_v)
        nb,_=cv2.connectedComponents(hi.astype(np.uint8)); out[kk+'_nblob']=float(max(nb-1,0))
    dt=cv2.distanceTransform(worm.astype(np.uint8),cv2.DIST_L2,5); w=dt[skeletonize(worm)]
    out['med_width_cv']=float(w.std()/(w.mean()+1e-6)) if w.size else 0.0
    # ---- segmentation-quality metrics: tell the model when a frame is poorly segmented ----
    out['seg_bright_edge']=float((g[ring]>178).mean()) if ring.sum() else 0.0   # bright pixels in the boundary ring = segmentation halo
    out['glare_frac']=float((g[worm]>235).mean()) if worm.sum() else 0.0        # saturated glare over the worm
    out['bright_leak_frac']=float((g[worm]>220).mean()) if worm.sum() else 0.0   # bright non-worm pixels inside the mask (leak)
    cs,_=cv2.findContours(worm.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if cs:
        per=cv2.arcLength(max(cs,key=cv2.contourArea),True); out['mask_roughness']=float(per*per/(4*np.pi*area+1e-6))
    else: out['mask_roughness']=0.0
    out.update(contour_cues(worm)); out.update(twist_features(gray,worm)); out.update(loop_features(worm)); out.update(axial_twist_features(gray,worm)); return out

def set_id(name):
    b=os.path.splitext(str(name))[0]
    return re.sub(r'_Frame_\d+_in_\d+.*$','',re.sub(r'_Frame\d+.*$','',b))
# ---- TWIST detector: projected self-crossing of a helical screw (direct + indirect) ----
# Skeleton topology after Zhang & Suen thinning (CACM 1984, DOI 10.1145/357994.358023).
# A transverse crossing with a dark overlap is the 2D signature of an out-of-plane twist,
# the projected writhe of a helix (Fuller 1971 DOI 10.1073/pnas.68.4.815; Broekmans et al.
# eLife 2016 DOI 10.7554/eLife.17227). An omega touch crosses tangentially and stays bright,
# so crossing transversality and overlap darkness separate a twist from a mere bend.
def _neighcount(sk):
    return cv2.filter2D(sk.astype(np.uint8),-1,np.array([[1,1,1],[1,0,1],[1,1,1]],np.uint8),borderType=cv2.BORDER_CONSTANT)
def _longest_path_width(sk,dt):
    S=np.argwhere(sk)
    if len(S)<10: return np.array([])
    idx={(y,x):i for i,(y,x) in enumerate(S)}; adj=[[] for _ in S]
    for i,(y,x) in enumerate(S):
        for dy in(-1,0,1):
            for dx in(-1,0,1):
                if dy or dx:
                    j=idx.get((y+dy,x+dx))
                    if j is not None: adj[i].append(j)
    def bfs(s0):
        d=[-1]*len(S); d[s0]=0; q=deque([s0]); par=[-1]*len(S); last=s0
        while q:
            u=q.popleft(); last=u
            for v in adj[u]:
                if d[v]<0: d[v]=d[u]+1; par[v]=u; q.append(v)
        return par,last
    _,a=bfs(0); par,b=bfs(a); path=[]; u=b
    while u!=-1: path.append(u); u=par[u]
    return np.array([dt[S[i][0],S[i][1]] for i in path],np.float32)
def twist_features(gray,worm):
    g=gray.astype(np.float32); sk=skeletonize(worm); dt=cv2.distanceTransform(worm.astype(np.uint8),cv2.DIST_L2,5)
    nc=_neighcount(sk); bpts=sk&(nc>=3); epts=sk&(nc==1)
    nbranch=cv2.connectedComponents(cv2.dilate(bpts.astype(np.uint8),np.ones((3,3),np.uint8))) [0]-1
    nend=cv2.connectedComponents(epts.astype(np.uint8))[0]-1
    bg=cv2.GaussianBlur(g,(0,0),9); loc=cv2.GaussianBlur((g-bg)**2,(0,0),9)**0.5+1e-3; od=np.clip(bg-g,0,None)/loc
    cross_incoh=cross_depth=0.0; Sb=np.argwhere(bpts)
    if len(Sb):
        cy,cx=Sb[len(Sb)//2]; R=18
        sub=sk[max(cy-R,0):cy+R,max(cx-R,0):cx+R]; ys,xs=np.where(sub)
        if len(ys)>6:
            ys=ys-ys.mean(); xs=xs-xs.mean(); J=np.array([[(xs*xs).mean(),(xs*ys).mean()],[(xs*ys).mean(),(ys*ys).mean()]])
            ev=np.linalg.eigvalsh(J); cross_incoh=float(1-(ev[1]-ev[0])/(ev[1]+ev[0]+1e-6))
        cross_depth=float(od[cy,cx])
    w=_longest_path_width(sk,dt); fft_peak=0.0; n_pinch=0
    if len(w)>=24:
        w=cv2.GaussianBlur(w.reshape(-1,1),(0,0),2).ravel(); wn=w-cv2.GaussianBlur(w.reshape(-1,1),(0,0),8).ravel()
        sig=wn/max(w.std(),1e-3); sp=np.abs(np.fft.rfft(sig*np.hanning(len(sig))))[1:]
        if sp.sum()>0: fft_peak=float(sp.max()/sp.sum())
        ss=np.sign(np.where(np.abs(sig)>0.25,sig,0)); ss=ss[ss!=0]; n_pinch=int((np.abs(np.diff(ss))>1).sum()) if len(ss)>1 else 0
    return dict(tw_nbranch=float(nbranch),tw_nend=float(nend),tw_cross_incoh=float(cross_incoh),
                tw_cross_depth=float(cross_depth),tw_width_fft=float(fft_peak),tw_n_pinch=float(n_pinch),
                tw_score=float(cross_incoh*cross_depth*(nbranch>0)))
def twist_map(gray,worm):
    g=gray.astype(np.float32); sk=skeletonize(worm); nc=_neighcount(sk); bpts=(sk&(nc>=3)).astype(np.float32)
    dt=cv2.distanceTransform(worm.astype(np.uint8),cv2.DIST_L2,5); wa=np.abs(dt-cv2.GaussianBlur(dt,(0,0),11))
    bg=cv2.GaussianBlur(g,(0,0),9); loc=cv2.GaussianBlur((g-bg)**2,(0,0),9)**0.5+1e-3; od=np.clip(bg-g,0,None)/loc
    bb=cv2.GaussianBlur(cv2.dilate(bpts,np.ones((9,9),np.float32)),(0,0),5)
    return np.nan_to_num((bb*(1+wa)*(1+od))*worm).astype(np.float32)

# ---- Contrast-invariant phase / local energy (subtle folds), Morrone & Owens, Pattern
# Recognition Letters 1987, DOI 10.1016/0167-8655(87)90013-4. Monogenic signal via the
# Riesz transform, log-Gabor bandpass. Phase symmetry fires on line/ridge features
# independent of contrast, so it catches a faint fold the darkness metric misses. ----
def pc_energy_map(gray, worm, wl=10.0, sigmaOnf=0.55):
    g=gray.astype(np.float32); h,w=g.shape; G=np.fft.fft2(g)
    u=np.fft.fftfreq(w); v=np.fft.fftfreq(h); U,V=np.meshgrid(u,v); R=np.sqrt(U*U+V*V); R[0,0]=1.0
    f0=1.0/wl; lg=np.exp(-(np.log(R/f0)**2)/(2*np.log(sigmaOnf)**2)); lg[0,0]=0.0
    Hk=(1j*U-V)/(R+1e-9)
    even=np.real(np.fft.ifft2(G*lg)); odd=np.fft.ifft2(G*lg*Hk); o1=np.real(odd); o2=np.imag(odd)
    energy=np.sqrt(even*even+o1*o1+o2*o2); sym=np.clip(np.abs(even)-np.sqrt(o1*o1+o2*o2),0,None)
    en=energy/(energy[worm].mean()+1e-6) if worm.any() else energy
    return np.nan_to_num(en*worm).astype(np.float32), np.nan_to_num(sym*worm).astype(np.float32)

# ---- Topological loop persistence (a screw coil encloses a hole, a bend does not).
# Betti-1 of the silhouette with a dilation filtration, after Edelsbrunner, Letscher &
# Zomorodian, Topological Persistence and Simplification, Discrete & Computational
# Geometry 2002, DOI 10.1007/s00454-002-2885-2. Dependency-free via contour hierarchy. ----
def loop_features(worm):
    wu=worm.astype(np.uint8); cs,hi=cv2.findContours(wu,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    area=max(int(worm.sum()),1); holes=[cv2.contourArea(cs[i]) for i in range(len(cs)) if hi is not None and hi[0][i][3]!=-1]
    loop_n=len(holes); loop_area=float(max(holes)/area) if holes else 0.0; persist=0
    if holes:
        for r in range(1,16):
            m=cv2.dilate(wu,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1)))
            _,hh=cv2.findContours(m,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
            if hh is not None and any(h2[3]!=-1 for h2 in hh[0]): persist=r
            else: break
    return dict(loop_n=float(loop_n),loop_area_frac=loop_area,loop_persist=float(persist))

# ---- Texture-only twist (a straight worm can twist with no shape change): the helix or lay
# angle of the surface striations relative to the body axis, exactly the yarn-twist problem.
# Oriented texture field after Rao & Schunck, Computing oriented texture fields, CVGIP
# Graphical Models and Image Processing 1991, DOI 10.1016/1049-9652(91)90059-S, and Kass &
# Witkin, Analyzing oriented patterns, CVGIP 1986, DOI 10.1016/S0734-189X(86)80033-0. Axial
# grey-level banding after Eldessouki et al., Digital Signal Processing 2014,
# DOI 10.1016/j.dsp.2013.12.015. An oblique consistent orientation is a twist, an axial one
# is plain and a transverse one is a ring, so off-axis striation is the screw signature. ----
def _ordered_centerline(sk):
    S=np.argwhere(sk)
    if len(S)<12: return np.empty((0,2),int)
    idx={(y,x):i for i,(y,x) in enumerate(S)}; adj=[[] for _ in S]
    for i,(y,x) in enumerate(S):
        for dy in(-1,0,1):
            for dx in(-1,0,1):
                if dy or dx:
                    j=idx.get((y+dy,x+dx))
                    if j is not None: adj[i].append(j)
    def bfs(s0):
        d=[-1]*len(S); d[s0]=0; q=deque([s0]); par=[-1]*len(S); last=s0
        while q:
            u=q.popleft(); last=u
            for v in adj[u]:
                if d[v]<0: d[v]=d[u]+1; par[v]=u; q.append(v)
        return par,last
    _,a=bfs(0); par,b=bfs(a); path=[]; u=b
    while u!=-1: path.append(u); u=par[u]
    return S[path[::-1]]
def axial_twist_features(gray, worm):
    g=gray.astype(np.float32); sk=skeletonize(worm); P=_ordered_centerline(sk)
    out=dict(ax_helix_absmean=0.0,ax_helix_consist=0.0,ax_offaxis_frac=0.0,ax_band_fft=0.0,ax_coh_mean=0.0)
    if len(P)<20: return out
    P=P[::3]
    ys=cv2.GaussianBlur(P[:,0].astype(np.float32).reshape(-1,1),(0,0),2).ravel()
    xs=cv2.GaussianBlur(P[:,1].astype(np.float32).reshape(-1,1),(0,0),2).ravel()
    tang=np.arctan2(np.gradient(ys),np.gradient(xs))
    gx=cv2.Sobel(g,cv2.CV_32F,1,0,3); gy=cv2.Sobel(g,cv2.CV_32F,0,1,3)
    Jxx=cv2.GaussianBlur(gx*gx,(0,0),3); Jyy=cv2.GaussianBlur(gy*gy,(0,0),3); Jxy=cv2.GaussianBlur(gx*gy,(0,0),3)
    R=7; helix=[]; coh=[]; prof=[]
    for (yy,xx),ta in zip(P,tang):
        y0,y1=max(yy-R,0),yy+R; x0,x1=max(xx-R,0),xx+R
        a=Jxx[y0:y1,x0:x1].mean(); b=Jyy[y0:y1,x0:x1].mean(); c=Jxy[y0:y1,x0:x1].mean()
        ori=0.5*np.arctan2(2*c,a-b); cc=np.sqrt((a-b)**2+4*c*c)/(a+b+1e-6)
        h=np.arctan2(np.sin(ori-ta),np.cos(ori-ta)); h=(h+np.pi/2)%np.pi-np.pi/2
        helix.append(h); coh.append(cc); prof.append(float(g[max(yy-2,0):yy+2,max(xx-2,0):xx+2].mean()))
    helix=np.array(helix); coh=np.array(coh); w=coh/(coh.sum()+1e-6)
    out['ax_helix_absmean']=float(np.degrees(np.sum(np.abs(helix)*w)))
    out['ax_helix_consist']=float(np.abs(np.sum(w*np.exp(2j*helix))))
    hd=np.abs(np.degrees(helix)); out['ax_offaxis_frac']=float(np.sum(w[(hd>50)&(hd<88)]))
    out['ax_coh_mean']=float(coh.mean())
    pr=np.array(prof); pr=pr-cv2.GaussianBlur(pr.reshape(-1,1),(0,0),6).ravel()
    if len(pr)>=16:
        sp=np.abs(np.fft.rfft(pr*np.hanning(len(pr))))[1:]; out['ax_band_fft']=float(sp.max()/(sp.sum()+1e-6))
    # straightness of the body axis (end-to-end over arc length): 1 = straight, low = coiled
    end=float(np.hypot(P[0,0]-P[-1,0],P[0,1]-P[-1,1])); arc=float(np.sum(np.hypot(np.diff(P[:,0]),np.diff(P[:,1])))+1e-6)
    st=end/arc; out['ax_straightness']=st
    # straight-body interaction: the off-axis twist cue counts only when the body is straight
    out['ax_offaxis_x_straight']=out['ax_offaxis_frac']*st
    out['ax_helixcons_x_straight']=out['ax_helix_consist']*st
    return out

# ---- CURATED set promoted to the model after the full study (see metric status notebook) ----
# Topology winners work on all body shapes. The ax_* set covers the straight-bodied screw
# that shape and topology miss, the hard case flagged on the GABA/Amitraze/Haloperidol/Baclofen worms.
CURATED = ['bnd_concav_max','self_prox_nblob','twist_cross_nblob','twist_cross_max','twist_cross_p99',
           'tw_nbranch','tw_nend','med_width_cv','width_anom_max','loop_persist','loop_n',
           'ax_straightness','ax_offaxis_x_straight','ax_helix_consist','ax_helix_absmean',
           # ---- local-peak fold metrics (color-invariant) for the subtle straight-screw folds ----
           'hess_dark_max','hess_dark_p99','frangi_dark_max','frangi_dark_p99',
           'sato_dark_max','sato_dark_p99','meijering_dark_max','meijering_dark_p99',
           'self_prox_max','self_prox_p99','dark_od_max','dark_od_p99',
           'seg_bright_edge','glare_frac','bright_leak_frac','mask_roughness']
def curated_metrics(gray, worm):
    """Return only the curated metrics (same values as image_metrics), for the precompute."""
    im = image_metrics(gray, worm.astype(bool))
    return {k: float(im.get(k, np.nan)) for k in CURATED}

# ---- TEMPORAL screwing signature across T-1, T0, T+1 formatted frames ----
# Screw-like is a behavior, not a shape. A true screw rotates its surface twist along the
# body axis over time, so the oblique striations travel like a barber pole, while a static
# control does not. This is the one cue that separates the GABA screw from the Amitraze
# control, which are identical in any single frame. Oriented texture across frames follows
# Rao & Schunck CVGIP GMIP 1991 DOI 10.1016/1049-9652(91)90059-S; the axial band travel is
# the temporal extension of Eldessouki et al. Digital Signal Process 2014 DOI 10.1016/j.dsp.2013.12.015.
TEMPORAL_CURATED = ['ax_band_shift', 'ax_helix_rot']

def _axis_helix_profile(gray, worm):
    g=gray.astype(np.float32); sk=skeletonize(worm); P=_ordered_centerline(sk)
    if len(P)<20: return None
    P=P[::3]
    ys=cv2.GaussianBlur(P[:,0].astype(np.float32).reshape(-1,1),(0,0),2).ravel()
    xs=cv2.GaussianBlur(P[:,1].astype(np.float32).reshape(-1,1),(0,0),2).ravel()
    tang=np.arctan2(np.gradient(ys),np.gradient(xs))
    gx=cv2.Sobel(g,cv2.CV_32F,1,0,3); gy=cv2.Sobel(g,cv2.CV_32F,0,1,3)
    Jxx=cv2.GaussianBlur(gx*gx,(0,0),3); Jyy=cv2.GaussianBlur(gy*gy,(0,0),3); Jxy=cv2.GaussianBlur(gx*gy,(0,0),3)
    R=7; hel=[]; prof=[]
    for (yy,xx),ta in zip(P,tang):
        a=Jxx[max(yy-R,0):yy+R,max(xx-R,0):xx+R].mean(); b=Jyy[max(yy-R,0):yy+R,max(xx-R,0):xx+R].mean(); c=Jxy[max(yy-R,0):yy+R,max(xx-R,0):xx+R].mean()
        ori=0.5*np.arctan2(2*c,a-b); h=np.arctan2(np.sin(ori-ta),np.cos(ori-ta)); h=(h+np.pi/2)%np.pi-np.pi/2
        hel.append(np.degrees(h)); prof.append(float(g[max(yy-2,0):yy+2,max(xx-2,0):xx+2].mean()))
    return np.array(hel), np.array(prof)

def helix_rotation(frames, worms):
    """Temporal screwing signature from 3 formatted grayscale frames (T-1,T0,T+1) + masks."""
    out=dict(ax_band_shift=0.0, ax_helix_rot=0.0)
    pr=[_axis_helix_profile(g, np.asarray(w).astype(bool)) for g,w in zip(frames,worms)]
    if any(p is None for p in pr): return out
    means=[float(np.average(h)) for h,_ in pr]
    out['ax_helix_rot']=float(np.mean(np.abs(np.diff(means))))
    shifts=[]
    for (h0,p0),(h1,p1) in zip(pr[:-1],pr[1:]):
        n=min(len(p0),len(p1)); a=p0[:n]-p0[:n].mean(); b=p1[:n]-p1[:n].mean()
        if a.std()<1e-3 or b.std()<1e-3: continue
        cc=np.correlate(a/(a.std()+1e-6), b/(b.std()+1e-6), mode='full')/n
        shifts.append(abs(int(np.argmax(cc))-(n-1)))
    out['ax_band_shift']=float(np.mean(shifts)) if shifts else 0.0
    return out

# ---- EXPLORATORY motion family: dense optical-flow vorticity across T-1,T0,T+1 ----
# The curl (vorticity) of the flow field is local rotation, the spin of a screwing worm,
# distinct from the band travel. Vorticity of optic flow: Beusmans, JOSA A 1994,
# DOI 10.1364/josaa.11.000048. Body roll as a behaviour: Bilbao et al., Roll maneuvers are
# essential for active reorientation of C. elegans in 3D media, PNAS 2018,
# DOI 10.1073/pnas.1706754115. These are candidates to TEST, not protected.
def flow_motion(frames, worms):
    out=dict(ax_flow_curl=0.0, ax_flow_div=0.0, ax_flow_mag=0.0, ax_flow_circ=0.0, ax_flow_axial=0.0)
    cu=[]; dv=[]; mg=[]; ci=[]; ax=[]
    for g0,g1,w in zip(frames[:-1],frames[1:],worms[1:]):
        try:
            fl=cv2.calcOpticalFlowFarneback(g0.astype(np.uint8),g1.astype(np.uint8),None,0.5,3,15,3,5,1.2,0)
        except Exception:
            continue
        vx=fl[...,0]; vy=fl[...,1]; m=np.asarray(w).astype(bool)
        if m.sum()<50: continue
        curl=cv2.Sobel(vy,cv2.CV_32F,1,0,3)-cv2.Sobel(vx,cv2.CV_32F,0,1,3)
        div =cv2.Sobel(vx,cv2.CV_32F,1,0,3)+cv2.Sobel(vy,cv2.CV_32F,0,1,3)
        mag =np.sqrt(vx*vx+vy*vy)
        cu.append(float(np.mean(np.abs(curl[m])))); dv.append(float(np.mean(np.abs(div[m])))); mg.append(float(np.mean(mag[m])))
        # decompose flow relative to the body principal axis: circumferential (rotation) vs axial (slide)
        ys,xs=np.where(m); cyy=ys-ys.mean(); cxx=xs-xs.mean()
        th=0.5*np.arctan2(2*np.mean(cyy*cxx),(cxx*cxx).mean()-(cyy*cyy).mean())
        ux,uy=np.cos(th),np.sin(th)                      # body-axis unit vector
        fa=vx[m]*ux+vy[m]*uy                              # axial component
        fc=vx[m]*(-uy)+vy[m]*ux                           # circumferential (perpendicular) component
        ax.append(float(np.mean(np.abs(fa)))); ci.append(float(np.mean(np.abs(fc))))
    if cu:
        out['ax_flow_curl']=float(np.mean(cu)); out['ax_flow_div']=float(np.mean(dv)); out['ax_flow_mag']=float(np.mean(mg))
        out['ax_flow_circ']=float(np.mean(ci)); out['ax_flow_axial']=float(np.mean(ax))
    return out

def texture_decorrelation(frames, worms):
    """How fast the body surface changes frame to frame. A screwing surface decorrelates faster."""
    dec=[]
    for g0,g1,w in zip(frames[:-1],frames[1:],worms[1:]):
        m=np.asarray(w).astype(bool)
        if m.sum()<50: continue
        a=g0[m].astype(np.float32); b=g1[m].astype(np.float32)
        if a.std()<1e-3 or b.std()<1e-3: continue
        c=float(np.mean((a-a.mean())*(b-b.mean()))/(a.std()*b.std()+1e-6))
        dec.append(1.0-max(min(c,1.0),-1.0))
    return dict(ax_tex_decorr=float(np.mean(dec)) if dec else 0.0)

# temporal metrics added to the dataset. PROTECTED: ax_band_shift, ax_helix_rot. The rest are exploratory.
TEMPORAL_CURATED = ['ax_band_shift', 'ax_helix_rot', 'ax_flow_curl', 'ax_flow_div', 'ax_flow_mag',
                    'ax_flow_circ', 'ax_flow_axial', 'ax_tex_decorr']
def temporal_metrics(frames, worms):
    out={}; out.update(helix_rotation(frames,worms)); out.update(flow_motion(frames,worms)); out.update(texture_decorrelation(frames,worms)); return out

# ---- SHARED-ALIGNMENT temporal prep: pose all 3 frames with ONE transform from the middle (T0)
# frame, so the worm's real rotation and surface travel survive. Formatting each frame to its own
# canonical pose (the old way) erases inter-frame motion, which is why raw temporal metrics read at
# chance. Sequence registration to a common reference is standard motion analysis. ----
def _align_ref(mask):
    from skimage.measure import regionprops, label as _lb
    m=(np.asarray(mask)>0).astype(np.uint8)
    if m.sum()<40: return None
    pr=max(regionprops(_lb(m)),key=lambda p:p.area)
    cy,cx=pr.centroid; ang=float(np.degrees(pr.orientation)); L=float(pr.axis_major_length) or 1.0
    return float(cx),float(cy),ang,L

def aligned_triplet(raws, masks, size=256, target_len=180.0):
    ref=_align_ref(masks[1])                       # reference = the middle frame (T0)
    if ref is None: return None,None
    cx,cy,ang,L=ref; s=target_len/max(L,1.0)
    M=cv2.getRotationMatrix2D((cx,cy),ang,s); M[0,2]+=size/2-cx; M[1,2]+=size/2-cy
    grays=[]; worms=[]
    for raw,mk in zip(raws,masks):
        g=cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY) if (raw is not None and raw.ndim==3) else raw
        mm=(np.asarray(mk)>0).astype(np.uint8)
        ga=cv2.warpAffine((g*mm).astype(np.uint8),M,(size,size))
        wa=cv2.warpAffine(mm,M,(size,size))>0
        grays.append(ga); worms.append(wa)
    return grays,worms

def temporal_metrics_aligned(raws, masks):
    grays,worms=aligned_triplet(raws,masks)
    if grays is None: return {c:0.0 for c in TEMPORAL_CURATED}
    return temporal_metrics(grays,worms)

# ---- COLOUR metrics (computed on the colour formatted image): the one thing colour adds is telling a
# real fold (the worm's own darker pigment) from a border shadow / reflection (a foreign, bluer dark).
# Pigment identity + a dark-region hue-shift. Light scalar features for the metric branch, no colour CNN.
COLOR_CURATED = ['col_sat_mean','col_sat_std','col_lab_a_mean','col_lab_b_mean',
                 'col_chroma_std','col_blue_frac','col_darkb_shift','col_dark_blue_frac']
def color_metrics(bgr, worm):
    out = {k: 0.0 for k in COLOR_CURATED}
    w = np.asarray(worm).astype(bool)
    if w.sum() < 50: return out
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    sat = hsv[..., 1][w]; v = hsv[..., 2][w]; a = lab[..., 1][w]; b = lab[..., 2][w]
    chroma = np.sqrt((a - 128) ** 2 + (b - 128) ** 2)
    dark = v < np.percentile(v, 25)
    out['col_sat_mean'] = float(sat.mean()); out['col_sat_std'] = float(sat.std())
    out['col_lab_a_mean'] = float(a.mean() - 128); out['col_lab_b_mean'] = float(b.mean() - 128)   # b>0 yellow
    out['col_chroma_std'] = float(chroma.std())
    out['col_blue_frac'] = float((b < 118).mean())                                                  # bluish (border/reflection)
    out['col_darkb_shift'] = float(np.median(b[dark]) - np.median(b)) if dark.sum() else 0.0          # <0 = dark region bluer than body
    out['col_dark_blue_frac'] = float((b[dark] < 118).mean()) if dark.sum() else 0.0
    return out
