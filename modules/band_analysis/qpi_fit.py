"""QPI peak inference, not a direct measurement of Bloch band structure.

Protocol: Liu et al., Nano Lett. 22, 3245 (2022), Fig.3g-h:
positive Gaussian + linear background on LINEAR Fourier amplitude.
Huang et al., PRB 103,094502 (2021), SI II: Gaussian peak positions,
then parabolic scattering dispersion. No automatic symmetry or q=k assumption.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from scipy.constants import hbar, electron_mass, elementary_charge

FWHM = 2*np.sqrt(2*np.log(2))
MASS_COEFFICIENT = hbar**2*np.pi**2/(2*electron_mass)/(elementary_charge*1e-3)*1e18

def gaussian_line(x, p, reference):
    amplitude, center, sigma, background, slope = p
    return background+slope*(x-reference)+amplitude*np.exp(-.5*((x-center)/sigma)**2)

def evaluate_result(x, result):
    """Full fitted profile, including all joint components, never one peak alone."""
    if 'joint_parameters' not in result:
        return gaussian_line(x,result['parameters'],result['reference'])
    p=result['joint_parameters'];xx=np.asarray(x);out=p[-2]+p[-1]*(xx-result['reference'])
    for j in range((len(p)-2)//3):
        amplitude,center,sigma=p[3*j:3*j+3]
        out=out+amplitude*np.exp(-.5*((xx-center)/sigma)**2)
    return out

def fit_multiple(x,y,seeds,low,high,resolution):
    """Independent positive Gaussians with shared linear background.

    No center/sigma/amplitude symmetry is assumed. Midpoints between user seeds
    bound each center to prevent component-label swapping; outputs follow seed order.
    Full covariance retains cross-component and common-background correlations.
    """
    seeds=np.asarray(seeds,float);x=np.asarray(x,float);y=np.asarray(y,float)
    if seeds.ndim!=1 or len(seeds)<2:raise ValueError('联合拟合至少选择两个分支。')
    if not np.isfinite(np.r_[seeds,low,high,resolution]).all() or resolution<=0 or not low<min(seeds)<=max(seeds)<high:
        raise ValueError('全部峰初值必须位于联合拟合范围内部。')
    order=np.argsort(seeds);ordered=seeds[order]
    if np.min(np.diff(ordered))<resolution:raise ValueError('峰初值相距不足一个 FFT 像素，当前数据不能可靠区分。')
    bounds=np.r_[low,(ordered[:-1]+ordered[1:])/2,high]
    take=(x>=low)&(x<=high);xx=x[take];yy=y[take];k=len(seeds);npar=3*k+2
    if len(xx)<max(9,npar+5):raise ValueError(f'联合拟合需要至少 {npar+5} 个原生间距采样点。')
    if not np.isfinite(yy).all():raise ValueError('联合范围含缺失值，不能跨缺失区拟合。')
    scale=np.ptp(yy)
    if scale<=np.finfo(float).eps*max(1,np.max(abs(yy))):raise ValueError('无可辨认强度变化。')
    yy=yy/scale;ref=(low+high)/2;slope=(yy[-1]-yy[0])/(xx[-1]-xx[0]);base=(yy[-1]+yy[0])/2
    p=[];lower=[];upper=[]
    for j,seed in enumerate(ordered):
        amp=max(float(np.interp(seed,xx,yy)-base-slope*(seed-ref)),.1)
        p.extend([amp,seed,max(resolution/2,(bounds[j+1]-bounds[j])/4)])
        lower.extend([0,bounds[j],resolution/8]);upper.extend([np.inf,bounds[j+1],high-low])
    p.extend([base,slope]);lower.extend([-np.inf,-np.inf]);upper.extend([np.inf,np.inf])
    def model(p):
        value=p[-2]+p[-1]*(xx-ref)
        for j in range(k):value=value+p[3*j]*np.exp(-.5*((xx-p[3*j+1])/p[3*j+2])**2)
        return value
    opt=least_squares(lambda p:model(p)-yy,p,bounds=(lower,upper),x_scale='jac',max_nfev=6000)
    if not opt.success or np.linalg.matrix_rank(opt.jac)<npar:raise ValueError('联合拟合未收敛或参数不可辨识。')
    dof=len(xx)-npar;rss=float(opt.fun@opt.fun)
    cov=np.linalg.inv(opt.jac.T@opt.jac)*(rss/dof)
    factors=np.ones(npar);factors[np.arange(k)*3]=scale;factors[-2:]=scale
    cov=cov*factors[:,None]*factors[None,:];p=opt.x*factors
    se=np.sqrt(np.maximum(np.diag(cov),0))
    if not np.isfinite(np.r_[p,se,cov.ravel()]).all():raise ValueError('联合拟合误差无效。')
    result=[None]*k
    for j,original in enumerate(order):
        ids=[3*j,3*j+1,3*j+2,npar-2,npar-1];a,mu,sigma,b,m=p[ids];err=se[ids];flags=[]
        if min(mu-low,high-mu)<max(resolution,2*sigma):flags.append('峰尾未充分包含/靠近总窗口边界')
        if min(mu-bounds[j],bounds[j+1]-mu)<resolution/2:flags.append('峰位触及分支约束边界')
        if FWHM*sigma<2*resolution:flags.append('峰宽不足两个 FFT 像素')
        if err[0]<=0 or a<3*err[0]:flags.append('峰振幅显著性不足 3σ')
        if err[1]>sigma:flags.append('峰位不确定度大于峰标准宽度')
        for other in range(k):
            if other==j:continue
            denom=se[3*j+1]*se[3*other+1]
            if denom>0 and abs(cov[3*j+1,3*other+1]/denom)>.95:
                flags.append('与其他峰峰位强相关，分峰不可靠');break
        result[original]=dict(center=float(mu),center_se=float(err[1]),sigma=float(sigma),fwhm=float(FWHM*sigma),
             fwhm_se=float(FWHM*err[2]),amplitude=float(a),amplitude_se=float(err[0]),background=float(b),slope=float(m),
             reference=ref,parameters=p[ids].tolist(),covariance=cov[np.ix_(ids,ids)].tolist(),
             joint_parameters=p.tolist(),joint_covariance=cov.tolist(),joint_component=int(j),joint_seed_order=order.tolist(),
             n=len(xx),dof=dof,residual_rms=float(np.sqrt(rss/len(xx))*scale),low=float(low),high=float(high),
             seed=float(seeds[original]),center_bounds=[float(bounds[j]),float(bounds[j+1])],flags=flags,accepted=not flags,
             uncertainty='conditional joint OLS covariance 1 sigma including cross-component covariance; systematic/FFT correlations excluded')
    return result

def fit_peak(x, y, seed, low, high, resolution):
    """OLS covariance: conditional 1 sigma, not an instrumental/systematic error.
    Input sampled no more densely than native FFT bins. Missing points are not filled.
    """
    x,y=np.asarray(x,float),np.asarray(y,float)
    if not np.isfinite([seed,low,high,resolution]).all() or not low<seed<high or resolution<=0:
        raise ValueError('峰初值必须位于拟合范围内部；FFT 像素间距必须为正。')
    take=(x>=low)&(x<=high)
    xx,yy=x[take],y[take]
    if len(xx)<9:raise ValueError('范围内不足 9 个原生间距采样点；扩大范围。')
    if not np.isfinite(yy).all():raise ValueError('拟合范围含缺失值；请缩小范围，不能跨缺失区拟合。')
    span=high-low;reference=(low+high)/2
    scale=np.ptp(yy)
    if scale<=np.finfo(float).eps*max(1,np.max(abs(yy))):raise ValueError('无可辨认强度变化。')
    yy=yy/scale
    slope=(yy[-1]-yy[0])/(xx[-1]-xx[0]);base=(yy[-1]+yy[0])/2
    amp=max(float(np.interp(seed,xx,yy)-(base+slope*(seed-reference))),.1)
    p0=[amp,seed,max(resolution/2,min(span/8,span/3)),base,slope]
    lower=[0,low,resolution/8,-np.inf,-np.inf];upper=[np.inf,high,span,np.inf,np.inf]
    fit=least_squares(lambda p:gaussian_line(xx,p,reference)-yy,p0,bounds=(lower,upper),x_scale='jac',max_nfev=3000)
    if not fit.success:raise ValueError('峰拟合未收敛：'+fit.message)
    dof=len(xx)-5;rss=float(fit.fun@fit.fun)
    if np.linalg.matrix_rank(fit.jac)<5:raise ValueError('拟合不可辨识：参数协方差奇异。')
    cov=np.linalg.inv(fit.jac.T@fit.jac)*(rss/dof)
    transform=np.diag([scale,1,1,scale,scale]);cov=transform@cov@transform
    p=fit.x.copy();p[[0,3,4]]*=scale
    se=np.sqrt(np.maximum(np.diag(cov),0));a,mu,sigma,b,m=p
    reasons=[]
    if min(mu-low,high-mu)<max(resolution,2*sigma):reasons.append('峰尾未充分包含/靠近边界')
    if FWHM*sigma<2*resolution:reasons.append('峰宽不足两个 FFT 像素')
    if se[0]<=0 or a<3*se[0]:reasons.append('峰振幅显著性不足 3σ')
    if se[1]>sigma:reasons.append('峰位不确定度大于峰标准宽度')
    if np.linalg.cond(fit.jac)>1e10:reasons.append('参数病态相关')
    return dict(center=float(mu),center_se=float(se[1]),sigma=float(sigma),fwhm=float(FWHM*sigma),
                fwhm_se=float(FWHM*se[2]),amplitude=float(a),amplitude_se=float(se[0]),
                background=float(b),slope=float(m),reference=reference,parameters=p.tolist(),
                covariance=cov.tolist(),n=len(xx),dof=dof,residual_rms=float(np.sqrt(rss/len(xx))*scale),
                low=float(low),high=float(high),seed=float(seed),flags=reasons,
                accepted=not reasons,uncertainty='conditional OLS covariance 1 sigma; FFT/window correlations and systematics excluded')

def fit_parabolic_scattering(energy, q, q_se, carrier, branch, q_center=0.):
    """Fit q(E) with q errors (NOT unweighted E(q)).
    E=E0+a*(q-q_center)^2; assumed intrapocket diameter Q=2k.
    q in cycles/nm; a in meV/(cycles/nm)^2; m/me=376.03/abs(a).
    E0 constrained outside fitted energies; branch fixed, center fixed.
    """
    e,q,se=map(lambda a:np.asarray(a,float),(energy,q,q_se))
    if len(e)<5 or len(np.unique(e))!=len(e):raise ValueError('至少需要 5 个不同能量的已接受峰。')
    if not np.isfinite(np.r_[e,q,se,q_center]).all() or np.any(se<=0):raise ValueError('峰位及其非零误差必须有效。')
    if np.ptp(q)<=3*np.median(se):raise ValueError('峰位变化不足以分辨色散，不能可靠反推有效质量。')
    if carrier not in ('electron','hole') or branch not in (-1,1):raise ValueError('指定电子/空穴型和单侧分支。')
    if np.any(branch*(q-q_center)<=0):raise ValueError('所选峰必须位于散射中心的同一指定侧。')
    sign=1 if carrier=='electron' else -1
    edge=e.min() if sign==1 else e.max();erange=max(float(np.ptp(e)),1e-3)
    # delta>0 puts E0 below/above the fitted energy window; never crosses a gap implicitly.
    def prediction(p):
        delta,loga=p;E0=edge-sign*delta;a=sign*np.exp(loga)
        return q_center+branch*np.sqrt((e-E0)/a)
    initial_a=erange/max(float(np.ptp((q-q_center)**2)),1e-8)
    opt=least_squares(lambda p:(prediction(p)-q)/se,[erange/2,np.log(initial_a)],
                      bounds=([1e-9,-30],[np.inf,30]),x_scale='jac',max_nfev=5000)
    if not opt.success or np.linalg.matrix_rank(opt.jac)<2:raise ValueError('抛物线散射模型不收敛或不可辨识。')
    cov=np.linalg.inv(opt.jac.T@opt.jac) # supplied q errors treated as absolute
    if not np.isfinite(cov).all():raise ValueError('模型协方差无效；请检查能区和散射分支。')
    delta,loga=opt.x;E0=edge-sign*delta;a=sign*np.exp(loga)
    J=np.diag([-sign,a]);physical_cov=J@cov@J.T
    err=np.sqrt(np.diag(physical_cov));mass=MASS_COEFFICIENT/abs(a)
    crossing=-E0/a>0
    result=dict(E0_meV=float(E0),E0_se_meV=float(err[0]),a_meV_per_cycles_nm_squared=float(a),a_se=float(err[1]),
                mass_over_me=float(mass),mass_se=float(mass*err[1]/abs(a)),covariance_E0_a=physical_cov.tolist(),
                carrier=carrier,branch=branch,q_center_cycles_nm=float(q_center),n=len(e),dof=len(e)-2,
                reduced_chi_squared=float(opt.fun@opt.fun/(len(e)-2)),energy_meV=e.tolist(),q_observed=q.tolist(),
                q_fit=prediction(opt.x).tolist(),q_residual=(q-prediction(opt.x)).tolist(),
                k_relative_nm_inverse=(np.pi*(q-q_center)).tolist(),k_se_nm_inverse=(np.pi*se).tolist(),
                assumption='single intrapocket diameter Q=2k; fixed calibrated scattering center; parabolic normal-state branch; E=eV',
                uncertainty='conditional 1 sigma from supplied q errors; ignores energy uncertainty, calibration, window correlations and model error',
                fermi_crossing=bool(crossing))
    if crossing:
        kf=np.pi*np.sqrt(-E0/a);grad=np.array([kf/(2*E0),-kf/(2*a)])
        result.update(kF_relative_nm_inverse=float(kf),kF_se_nm_inverse=float(np.sqrt(max(0,grad@physical_cov@grad))))
    return result
