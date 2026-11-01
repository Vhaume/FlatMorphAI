# -*- coding: utf-8 -*-
# Centerline LUMINOSITY features derived from expert comments (a channel not captured by the current metrics).
# Cues: Snake "dark head, light tail" -> cl_axial_dark_grad. Screw "translucent before the twist" + "zone both dark
# AND wide (overlap)" -> cl_lum_step, cl_overlap_dark. Brightness asymmetry of the two ends -> cl_headtail_asym.
# All are PROPORTIONAL (luminosity stretched to 0..1, normalized position). random_state 42.
import numpy as np, cv2
from collections import deque
from skimage.morphology import skeletonize
CLX=['cl_interior_bright','cl_lum_abrupt','cl_lum_step','cl_axial_dark_grad','cl_headtail_asym','cl_overlap_dark','cl_lum_localrange']
def _lp(sk):
    ys,xs=np.where(sk); nodes=set(zip(ys.tolist(),xs.tolist()))
    if len(nodes)<20: return []
    adj={(y,x):[nn for nn in [(y+dy,x+dx) for dy in(-1,0,1) for dx in(-1,0,1) if(dy or dx)] if nn in nodes] for (y,x) in nodes}
    def far(s):
        d={s:0};pr={s:None};q=deque([s]);last=s
        while q:
            u=q.popleft();last=u
            for v in adj[u]:
                if v not in d:d[v]=d[u]+1;pr[v]=u;q.append(v)
        return last,pr
    a,_=far(next(iter(nodes)));b,par=far(a);path=[];c=b
    while c is not None:path.append(c);c=par[c]
    return path
def lum_features(mask,gray):
    nan={c:np.nan for c in CLX}
    b=(np.asarray(mask)>0).astype(np.uint8)
    if b.sum()<300: return nan
    g=np.asarray(gray)
    if g.shape!=b.shape: g=cv2.resize(g,(b.shape[1],b.shape[0]))
    fg=g[b>0].astype(float)
    if fg.size<50: return nan
    lo,hi=np.percentile(fg,2),np.percentile(fg,98); gn=np.clip((g.astype(float)-lo)/(hi-lo+1e-6),0,1)
    dt=cv2.distanceTransform(b,cv2.DIST_L2,5); sk=skeletonize(b>0); path=_lp(sk)
    if len(path)<25: return nan
    L=len(path)
    lum=[]; wid=[]
    for (y,x) in path:
        y0,y1,x0,x1=max(0,y-3),y+4,max(0,x-3),x+4; patch=gn[y0:y1,x0:x1]; m=b[y0:y1,x0:x1]>0
        lum.append(patch[m].mean() if m.any() else np.nan); wid.append(dt[y,x]*2)
    lum=np.array(lum); wid=np.array(wid); wn=wid/(np.median(wid)+1e-6)
    k=max(2,L//25); sm=np.convolve(np.nan_to_num(lum,nan=np.nanmedian(lum)),np.ones(k)/k,'same')
    interior=slice(L//6,5*L//6); med=np.nanmedian(lum)+1e-6
    ibp=float(np.nanmax(lum[interior])-med)
    d1=np.abs(np.diff(sm))*L/100.0; di=d1[L//6:5*L//6-1] if len(d1)>L//6 else d1
    la=float(np.max(di)) if len(di) else 0.0
    win=max(3,L//15); steps=[abs(sm[i+win]-sm[i-win]) for i in range(L//6+win,5*L//6-win)]
    ls=float(np.max(steps)) if steps else 0.0
    # head->tail axial gradient (luminosity slope along the body)
    t=np.linspace(0,1,L); good=~np.isnan(lum)
    grad=float(np.polyfit(t[good],lum[good],1)[0]) if good.sum()>5 else 0.0
    # brightness asymmetry of the two ends (first 15% vs last 15%)
    e=max(3,L//7); a1=np.nanmean(lum[:e]); a2=np.nanmean(lum[-e:]); ht=float(abs(a1-a2)/(a1+a2+1e-6))
    # overlap: fraction of the centerline that is both DARK and WIDE (the body crosses itself -> thicker and darker)
    dark=lum<(med-0.5*np.nanstd(lum)); wide=wn>1.2; ov=float(np.mean(dark&wide))
    # cl_lum_localrange: maximum local luminance amplitude in the interior (max over windows of max-min of the smoothed profile).
    # A SINGLE physical metric that subsumes the interior peak (cl_interior_bright) AND the step (cl_lum_step), since a
    # peak and a step both produce a strong local amplitude. Corr 0.93 with the peak and 0.995 with the step,
    # but <0.89 with every other retained feature -> strictly clean Pearson filter (|r|<0.90 everywhere). Verified with
    # multiple seeds (6 seeds) as equivalent to the pair, with Walnut and Screw preserved. Replaces both in the final model.
    wlr=max(3,L//12); ai,bi=L//6,5*L//6
    lr=[np.max(sm[max(ai,i-wlr):min(bi,i+wlr)])-np.min(sm[max(ai,i-wlr):min(bi,i+wlr)]) for i in range(ai,bi)]
    clr=float(np.max(lr)) if lr else 0.0
    return {'cl_interior_bright':ibp,'cl_lum_abrupt':la,'cl_lum_step':ls,'cl_axial_dark_grad':grad,'cl_headtail_asym':ht,'cl_overlap_dark':ov,'cl_lum_localrange':clr}
