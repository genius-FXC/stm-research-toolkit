"""One-sided amplitude FFT of a selected spatial profile, with physical axes."""
from __future__ import annotations
import numpy as np

def profile_fft(distance_nm, values, window='hann'):
    if window not in ('hann','none'):
        raise ValueError('FFT 窗函数应为 Hann 或无窗。')
    s=np.asarray(distance_nm,dtype=float);y=np.asarray(values,dtype=float)
    if s.ndim!=1 or y.shape!=s.shape or not np.isfinite(s).all() or not np.isfinite(y).all():
        raise ValueError('FFT 需要有限的一维距离与剖面数据。')
    if len(s)<4:
        return dict(available=False,reason='FFT 至少需要 4 个沿线像素；请拉长线段。',window=window)
    steps=np.diff(s)
    if np.any(steps<=0):raise ValueError('FFT 沿线距离必须递增。')
    n=len(s);ds=float((s[-1]-s[0])/(n-1))
    uniform=bool(np.allclose(steps,ds,rtol=1e-6,atol=1e-10))
    target=np.linspace(s[0],s[-1],n)
    # This is an FFT-only copy. Measured pixel spectra and the profile stay intact.
    used=y.copy() if uniform else np.interp(target,s,y)
    mean=float(used.mean());centered=used-mean
    weights=np.hanning(n) if window=='hann' else np.ones(n)
    transformed=np.fft.rfft(centered*weights)
    frequencies=np.fft.rfftfreq(n,d=ds)
    amplitude=np.abs(transformed)/weights.sum()
    amplitude[1:]*=2
    if n%2==0:amplitude[-1]/=2  # Even-N Nyquist has no distinct negative partner.
    peak_index=int(1+np.argmax(amplitude[1:]))
    threshold=64*np.finfo(float).eps*max(1.,float(np.max(abs(used))))
    peak=None if amplitude[peak_index]<=threshold else dict(index=peak_index,frequency_cycles_per_nm=float(frequencies[peak_index]),period_nm=float(1/frequencies[peak_index]),amplitude=float(amplitude[peak_index]))
    return dict(available=True,window=window,mean_removed=True,removed_mean=mean,
        sample_count=n,resampled=not uniform,sampling='direct_uniform' if uniform else 'linear_resample_for_fft_only',
        mean_step_nm=ds,min_original_step_nm=float(steps.min()),max_original_step_nm=float(steps.max()),
        measured_length_nm=float(s[-1]-s[0]),fft_span_nm=float(n*ds),
        frequency_spacing_cycles_per_nm=float(1/(n*ds)),nyquist_cycles_per_nm=float(1/(2*ds)),
        coherent_gain=float(weights.mean()),zero_padding=0,
        uniform_distance_nm=target.tolist(),uniform_profile=used.tolist(),demeaned_profile=centered.tolist(),window_weights=weights.tolist(),
        frequency_cycles_per_nm=frequencies.tolist(),period_nm=[None if f==0 else float(1/f) for f in frequencies],
        amplitude=amplitude.tolist(),coefficient_real=transformed.real.tolist(),coefficient_imag=transformed.imag.tolist(),peak=peak,
        amplitude_definition='abs(rfft((uniform_profile-mean)*window))/sum(window); double positive bins except even-N Nyquist',
        frequency_definition='cycles/nm; period_nm=1/f; angular wavevector=2*pi*f',
        input_definition='current selected-channel profile after its Raw/Subtract average/Linear fit processing',
        resampling_note='Nonuniform digital pixel paths are linearly resampled to N equally spaced projected distances for this FFT only; interpolation can attenuate high-frequency amplitude.')
