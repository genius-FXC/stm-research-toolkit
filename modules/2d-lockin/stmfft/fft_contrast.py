"""ImageJ-style display window/level controls, independent of pixel ranks."""
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

class FFTContrast(tk.Toplevel):
    def __init__(self, app):
        super().__init__(app);self.app=app;self.title('FFT 亮度 / 对比度');self.geometry('600x590');self.minsize(570,560)
        self.transient(app);self.busy=False;self.signature=None;self.domain=(0.,1.);self.controls=[]
        box=ttk.Frame(self,padding=16);box.pack(fill='both',expand=True)
        ttk.Label(box,text='FFT 亮度 / 对比度',font=('',14,'bold')).pack(anchor='w')
        ttk.Label(box,text='按实际色阶连续调节，不再按像素百分位拖动。').pack(anchor='w',pady=6)
        self.hist=tk.Canvas(box,height=115,bg='#faf9f6',highlightthickness=1,highlightbackground='#ddd');self.hist.pack(fill='x')
        self.info=tk.StringVar();ttk.Label(box,textvariable=self.info,wraplength=550).pack(anchor='w',pady=6)
        self.vars={}
        for key,title in [('black','黑位'),('white','白位'),('brightness','亮度'),('contrast','对比度')]:
            row=ttk.Frame(box);row.pack(fill='x',pady=6);ttk.Label(row,text=title,width=9).pack(side='left')
            v=tk.DoubleVar();self.vars[key]=v
            scale=ttk.Scale(row,from_=0,to=1000,variable=v,command=lambda value,k=key:self.slide(k));scale.pack(side='left',fill='x',expand=True);self.controls.append(scale)
            for text,delta in [('−',-1),('+',1)]:
                b=ttk.Button(row,text=text,width=3,command=lambda k=key,d=delta:self.nudge(k,d));b.pack(side='left',padx=2);self.controls.append(b)
        row=ttk.Frame(box);row.pack(fill='x',pady=8);self.lower=tk.StringVar();self.upper=tk.StringVar()
        ttk.Label(row,text='色阶范围').pack(side='left')
        for v in (self.lower,self.upper):
            e=ttk.Entry(row,textvariable=v,width=18);e.pack(side='left',padx=5);e.bind('<Return>',lambda e:self.manual());self.controls.append(e)
        b=ttk.Button(row,text='应用',command=self.manual);b.pack(side='left');self.controls.append(b)
        row=ttk.Frame(box);row.pack(fill='x',pady=8)
        for name,fn in [('自动',lambda:self.preset(1,99)),('全范围',lambda:self.preset(0,100)),('放大当前调节范围',self.focus_range)]:
            b=ttk.Button(row,text=name,command=fn);b.pack(side='left',padx=3);self.controls.append(b)
        ttk.Label(box,text='亮度：平移色阶；对比度：围绕中心缩放色阶宽度。\n± 每次微调一步；强峰跨度太大时可放大当前调节范围。\n推荐「对数振幅（dB）」：0 dB 为本图最大振幅，−20 dB 为其 1/10。\n仅调整显示；相位固定 −π 至 π。',wraplength=550).pack(anchor='w',pady=8)
        self.after(50,self.watch)
    def watch(self):
        if not self.winfo_exists():return
        signature=(id(self.app.fft_result),self.app.fft_view_var.get())
        if signature!=self.signature:
            self.signature=signature;self.sync_data()
        self.after(200,self.watch)
    def sync_data(self):
        if self.app.fft_result is None:
            self.info.set('请先打开 SXM')
            for c in self.controls:c.configure(state='disabled')
            return
        a,_,fixed=self.app._fft_base_display_array();self.finite=a[np.isfinite(a)]
        if not self.finite.size:return
        lo,hi=float(self.finite.min()),float(self.finite.max())
        if hi<=lo:hi=lo+max(abs(lo)*1e-6,1e-30)
        self.domain=(lo,hi)
        for c in self.controls:c.configure(state='disabled' if fixed else 'normal')
        self.refresh()
    def refresh(self):
        if self.app.fft_result is None:return
        lo,hi=self.app._fft_display_array()[2];a,b=self.domain;span=b-a
        self.busy=True
        self.vars['black'].set(np.clip((lo-a)/span*1000,0,1000));self.vars['white'].set(np.clip((hi-a)/span*1000,0,1000))
        self.vars['brightness'].set(np.clip((b-(lo+hi)/2)/span*1000,0,1000))
        self.vars['contrast'].set(np.clip(-np.log10(max((hi-lo)/span,1e-6))/6*1000,0,1000));self.busy=False
        self.lower.set(f'{lo:.10g}');self.upper.set(f'{hi:.10g}')
        self.info.set(f'{self.app.fft_view_var.get()} · 黑位 {lo:.5g} / 白位 {hi:.5g}')
        counts,_=np.histogram(self.finite,bins=100,range=self.domain);h=np.log1p(counts);h/=max(float(h.max()),1)
        c=self.hist;c.delete('all');w=max(c.winfo_width(),530)
        for i,y in enumerate(h):c.create_rectangle(i*w/100,105-y*95,(i+1)*w/100,105,fill='#b58a72',outline='')
        x1=np.clip((lo-a)/span,0,1)*w;x2=np.clip((hi-a)/span,0,1)*w
        c.create_line(x1,105,x2,5,fill='#356f9b',width=2)
        for x,col in [(x1,'#222'),(x2,'#b34a32')]:c.create_line(x,0,x,110,fill=col,width=2)
    def apply(self,lo,hi):
        if not (np.isfinite(lo) and np.isfinite(hi) and lo<hi):return
        self.app.fft_levels['limits']=[float(lo),float(hi)];self.app.update_fft_contrast();self.refresh()
    def slide(self,key):
        if self.busy or self.app.fft_result is None:return
        a,b=self.domain;span=b-a;v=self.vars[key].get()/1000;lo,hi=self.app._fft_display_array()[2];gap=span*1e-6
        if key=='black':lo=min(a+span*v,hi-gap)
        elif key=='white':hi=max(a+span*v,lo+gap)
        elif key=='brightness':center=b-span*v;width=hi-lo;lo,hi=center-width/2,center+width/2
        else:center=(lo+hi)/2;width=span*10**(-6*v);lo,hi=center-width/2,center+width/2
        self.apply(lo,hi)
    def nudge(self,key,delta):self.vars[key].set(np.clip(self.vars[key].get()+delta,0,1000));self.slide(key)
    def focus_range(self):
        self.domain=tuple(self.app._fft_display_array()[2]);self.refresh()
    def preset(self,lo,hi):
        self.sync_data();lower,upper=np.percentile(self.finite,[lo,hi])
        if upper<=lower:lower,upper=self.domain
        self.apply(lower,upper)
    def manual(self):
        try:
            lo,hi=float(self.lower.get()),float(self.upper.get())
            if not(np.isfinite(lo) and np.isfinite(hi) and lo<hi):raise ValueError('请输入有限数值，且黑位小于白位。')
            self.apply(lo,hi)
        except ValueError as e:messagebox.showwarning('色阶范围',str(e),parent=self)
