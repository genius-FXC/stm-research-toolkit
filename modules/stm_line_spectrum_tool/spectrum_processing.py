"""Spectral processing matching plot_multiple_spectra2(1).py's executed code.

Every operation uses a copy; measured pixel spectra remain unchanged.
"""
from __future__ import annotations
import numpy as np

METHODS={'none':'不归一化','minmax':'[0,1] 归一化','current':'首个采集点电流比例校正','first':'除以首个保留值','abs_mean':'旧版：外侧窗口均值'}

def smooth_spectra(bias,signal,method,order=3,sigma=1.):
    x=np.asarray(bias,float).copy();y=np.asarray(signal,float).copy()
    if method=='none': return x,y
    if method not in ('adjacent','gaussian'): raise ValueError('未知谱线平滑方法。')
    if int(order)!=order or order<1: raise ValueError('平滑 order 必须为正整数；窗口点数=2×order+1。')
    order=int(order)
    if len(x)<=2*order+1: raise ValueError('平滑窗口过大：裁掉两端后至少需保留两个偏压点。')
    if method=='adjacent':
        # Intentional sequential/in-place updating, exactly as supplied script.
        # Earlier smoothed values participate in later windows.
        for i in range(order,len(x)-order):
            total=y[:,i].copy()
            for j in range(1,order+1): total+=y[:,i+j]+y[:,i-j]
            y[:,i]=total/(2*order+1)
        y=y[:,order:-order]
    else:
        if not np.isfinite(sigma) or sigma<=0: raise ValueError('高斯 sigma 必须大于 0，单位为偏压采样点。')
        d=np.arange(-order,order+1,dtype=float)
        kernel=np.exp(-d**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi));kernel/=kernel.sum()
        y=np.column_stack([(y[:,i-order:i+order+1]*kernel).sum(axis=1) for i in range(order,len(x)-order)])
    return x[order:-order],y

def process_spectra(bias,signal,cfg,current_first_pA=None,current_first_bias_mV=None):
    method=cfg.normalization_mode or ('abs_mean' if cfg.normalize else 'none')
    if method not in METHODS: raise ValueError('未知归一化方法。')
    numeric=[cfg.x_offset_mV,cfg.x_factor,cfg.y_factor,cfg.smoothing_sigma]
    if not np.isfinite(numeric).all() or cfg.x_factor==0: raise ValueError('偏压/信号缩放参数必须有限，偏压倍数不能为 0。')
    x,y=smooth_spectra(bias,signal,cfg.smoothing_method,cfg.smoothing_order,cfg.smoothing_sigma)
    x=(x+cfg.x_offset_mV)*cfg.x_factor
    crop_lo=float(x.min()) if cfg.spectrum_min_mV is None else float(cfg.spectrum_min_mV)
    crop_hi=float(x.max()) if cfg.spectrum_max_mV is None else float(cfg.spectrum_max_mV)
    if not np.isfinite([crop_lo,crop_hi]).all() or crop_lo>=crop_hi: raise ValueError('谱线裁切下限必须小于上限。')
    keep=(x>=crop_lo-1e-8)&(x<=crop_hi+1e-8)
    if keep.sum()<2: raise ValueError('平滑和裁切后至少需要两个实测偏压点，请调整窗口。')
    x=x[keep];y=y[:,keep]
    denominator=None;factor=None;used_current=None;reference=None
    if method=='none': y*=cfg.y_factor
    elif method=='minmax':
        minimum=y.min(axis=1);denominator=np.ptp(y,axis=1)
        if np.any(denominator==0): raise ValueError('[0,1] 归一化遇到常数谱，分母为 0；请换方法。')
        y=(y-minimum[:,None])/denominator[:,None]
    elif method=='first':
        denominator=y[:,0].copy()
        if np.any(denominator==0): raise ValueError('首个保留谱值为 0，无法相除；请调整裁切范围或方法。')
        y/=denominator[:,None]
    elif method=='current':
        if current_first_pA is None: raise ValueError('此 3DS 未保存 Current (A) 点谱，无法按首个采集点电流校正。')
        used_current=np.asarray(current_first_pA,float)
        reference=int(cfg.current_reference_sample)
        if cfg.current_reference_sample!=reference or not 0<=reference<len(y): raise ValueError(f'参考谱序号需为 1–{len(y)}。')
        if not np.isfinite(used_current).all() or np.any(used_current==0): raise ValueError('首个采集点电流含 0 或无效值，无法比例校正。')
        factor=used_current/used_current[reference]
        y=cfg.y_factor*y/factor[:,None]
    else:
        norm=(abs(x)>=cfg.norm_lo_mV-1e-8)&(abs(x)<=cfg.norm_hi_mV+1e-8)
        if cfg.norm_lo_mV<0 or cfg.norm_hi_mV<=cfg.norm_lo_mV or norm.sum()<2: raise ValueError('外侧归一化窗口至少需要两个偏压点。')
        denominator=y[:,norm].mean(axis=1)
        if np.any(denominator<=0): raise ValueError('外侧归一化窗口均值含非正值，请修改窗口或方法。')
        y/=denominator[:,None]
    if not np.isfinite(y).all(): raise ValueError('处理后出现非有限数值，请检查归一化参数。')
    unit='pA' if method in ('none','current') else 'a.u.'
    meta=dict(method=method,label=METHODS[method],unit=unit,reference_program='plot_multiple_spectra2(1).py',
              order_of_operations='smooth full energy spectra -> x=(bias+x_offset)*x_factor -> inclusive crop -> normalize/scale -> optional zero anchor -> waterfall offsets',
              smoothing=dict(method=cfg.smoothing_method,order=int(cfg.smoothing_order),window_points=2*int(cfg.smoothing_order)+1,sigma_points=cfg.smoothing_sigma,edge_handling='remove order points at each end when smoothing',adjacent_update='sequential in-place on a private copy, same as supplied script'),
              x_offset_mV=cfg.x_offset_mV,x_factor=cfg.x_factor,y_factor=cfg.y_factor,crop_mV=[crop_lo,crop_hi],
              y_factor_applied=method in ('none','current'),current_reference_sample_0based=reference,
              current_first_bias_mV=current_first_bias_mV,current_correction='g_i * y_factor * I_reference(first recorded bias) / I_i(first recorded bias)',
              first_normalization='first value after smoothing, x transform and crop, retaining the original acquisition order',
              selected_pixel_count=len(signal),processed_bias_count=len(x),raw_bias_count=len(bias),raw_measured_spectra_unchanged=True)
    return dict(bias_mV=x,values=y,denominator=denominator,current_factor=factor,current_first_pA=used_current,metadata=meta)
