"""Snapshot-based QPI inference workspace. No desktop automation."""
from __future__ import annotations
import csv, json, hashlib, copy
from pathlib import Path
from datetime import datetime
import numpy as np
from dataclasses import asdict
from volume import Contrast
from matplotlib import colormaps
from PySide6 import QtCore, QtWidgets as W
from PySide6.QtCore import Qt
import pyqtgraph as pg
from qpi_fit import fit_peak, gaussian_line, fit_parabolic_scattering, evaluate_result

def field(value, lo=-1e6, hi=1e6):
    w=W.QDoubleSpinBox();w.setDecimals(7);w.setRange(lo,hi);w.setValue(value);w.setKeyboardTracking(False);return w

def snapshot(cut, volume, contrast):
    if cut.params['mode'] not in ('line','path'):raise ValueError('请先建立直线或折线 q–E 截面。')
    # Recompute LINEAR amplitude, irrespective of the main display's power/log choice.
    c=volume.path(cut.params['vertices'],cut.params['full_width'],'amplitude',cut.params['interpolation'])
    shift=cut.x[0];c.x+=shift
    resolution=max(float(np.diff(volume.qx)[0]),float(np.diff(volume.qy)[0]))
    n=int(np.floor(np.ptp(c.x)/resolution))+1
    if n<9:raise ValueError('路径不足 9 个原生 FFT 间距；请加长路径。')
    x=np.linspace(c.x[0],c.x[-1],n)
    values=np.array([np.interp(x,c.x,row) for row in c.values])
    display=copy.deepcopy(contrast)
    if cut.params.get('metric')!='amplitude':display.fit(volume.amplitude)
    return dict(x=x,energy=c.y.copy(),values=values,resolution=resolution,
                metadata=copy.deepcopy(dict(source=volume.meta,cut=cut.params,
                display_contrast=asdict(display),
                axis='signed q from scattering origin' if cut.params['mode']=='line' else 'cumulative q-path distance',
                sampling='amplitude resampled at spacing >= max native FFT bin; correlated errors may remain',
                intensity='linear FFT amplitude, pA DFT sum',references=['10.1021/acs.nanolett.1c04863 Fig.3g-h','10.1103/PhysRevB.103.094502 SI II','10.1038/nphys3450 Fig.3'])))

def labeled(text, widget):
    group=W.QWidget(); row=W.QHBoxLayout(group); row.setContentsMargins(0,0,0,0); row.setSpacing(5)
    label=W.QLabel(text);label.setBuddy(widget);label.setSizePolicy(W.QSizePolicy.Maximum,W.QSizePolicy.Preferred);row.addWidget(label);row.addWidget(widget,1)
    return group

class Batch(QtCore.QThread):
    finished_rows=QtCore.Signal(object)
    progress=QtCore.Signal(int)
    def __init__(self, data, configs, parent=None):super().__init__(parent);self.data=data;self.configs=copy.deepcopy(configs)
    def run(self):
        out=[]
        for i,c in enumerate(self.configs):
            out.append(calculate(self.data,i,c));self.progress.emit(i+1)
        self.finished_rows.emit(out)

def calculate(data,i,c):
    try:
        if not c.get('enabled',True):return {'error':'该能量已停用','accepted':False}
        corners=data['metadata']['cut'].get('corner_distances',[])[1:-1]
        if data['metadata']['cut']['mode']=='path' and any(c['low']<z<c['high'] for z in corners):
            raise ValueError('拟合范围跨折线拐角；请选择单一直线段。')
        return fit_peak(data['x'],data['values'][i],c['seed'],c['low'],c['high'],data['resolution'])
    except (ValueError,RuntimeError,np.linalg.LinAlgError) as e:return {'error':str(e),'accepted':False}

class BaseQPIDialog(W.QDialog):
    def __init__(self,data,parent=None):
        super().__init__(parent);self.setWindowTitle('QPI peak fitting · q–E dispersion');self.resize(1400,900)
        self.data=data;self.index=0;self.busy=False;self.band=None;self.worker=None
        x=data['x'];center=(x[0]+x[-1])/2;span=(x[-1]-x[0])*.25
        self.configs=[dict(seed=float(center+span),low=float(center),high=float(x[-1])) for _ in data['energy']]
        self.results=[None]*len(data['energy']);self.updating=False
        layout=W.QVBoxLayout(self)
        note=W.QLabel('QPI 散射色散，不是直接的 k–E 能带。点击 q–E 图选能量/峰初值；拖动线切图中的范围和峰初值。拟合线性 |FFT|，不拟合 log 色值。');note.setWordWrap(True);layout.addWidget(note)
        self.actions=W.QWidget();action_layout=W.QVBoxLayout(self.actions);form=W.QHBoxLayout();action_layout.addLayout(form);layout.addWidget(self.actions)
        self.energy=W.QComboBox();self.energy.addItems([f'{e:.6g} mV' for e in data['energy']]);form.addWidget(self.energy)
        self.low=field(self.configs[0]['low']);self.high=field(self.configs[0]['high']);self.seed=field(self.configs[0]['seed'])
        for text,w in [('下限',self.low),('上限',self.high),('峰初值',self.seed)]:form.addWidget(labeled(text,w));w.valueChanged.connect(self.edit_config)
        commands=W.QHBoxLayout();action_layout.addLayout(commands)
        for text,fn in [('拟合当前能量',self.fit_current),('当前范围/初值复制到全部',self.copy_config),('逐能量拟合全部',self.fit_all),('导出/保存分析',self.export),('重开分析',self.reopen)]:
            b=W.QPushButton(text);b.clicked.connect(fn);commands.addWidget(b)
        self.energy.currentIndexChanged.connect(self.select_energy)
        self.plot_split=W.QSplitter(Qt.Vertical);layout.addWidget(self.plot_split,1)
        split=W.QSplitter();self.plot_split.addWidget(split)
        self.map=pg.PlotWidget();self.profile=pg.PlotWidget();split.addWidget(self.map);split.addWidget(self.profile)
        self.map.setLabel('bottom','q (cycles/nm)' if data['metadata']['cut']['mode']=='line' else 'q-path distance (cycles/nm)');self.map.setLabel('left','Bias / E=eV (mV / meV)');self.map.setTitle('q–E 散射截面 · 全能量统一衬度；拟合使用线性振幅')
        self.profile.setLabel('bottom','q / q-path distance (cycles/nm)');self.profile.setLabel('left','|FFT| (pA DFT sum)')
        def edges(a):return np.r_[a[0]-(a[1]-a[0])/2,(a[:-1]+a[1:])/2,a[-1]+(a[-1]-a[-2])/2]
        xx,ee=np.meshgrid(edges(x),edges(data['energy']));display=Contrast(**data['metadata']['display_contrast'])
        lut=colormaps[display.cmap](np.linspace(0,1,256),bytes=True)
        self.mesh=pg.PColorMeshItem(xx,ee,display.normalized(data['values']),colorMap=pg.ColorMap(np.linspace(0,1,256),lut),levels=(0,1),enableAutoLevels=False);self.map.addItem(self.mesh)
        self.map.scene().sigMouseClicked.connect(self.map_click)
        self.mark=pg.InfiniteLine(angle=0,pen=pg.mkPen('#ef9d32'));self.map.addItem(self.mark)
        self.points=pg.ScatterPlotItem(size=7,brush='#f45e48');self.map.addItem(self.points)
        self.dispersion_line=self.map.plot(pen=pg.mkPen('#f45e48',width=1),connect='finite')
        self.qerrors=pg.ErrorBarItem(pen=pg.mkPen('#f45e48'));self.map.addItem(self.qerrors)
        self.model_curve=self.map.plot(pen=pg.mkPen('#12beee',width=2))
        self.observed=self.profile.plot(pen='#444444',symbol='o',symbolSize=3)
        self.model=self.profile.plot(pen=pg.mkPen('#db4c29',width=2));self.background=self.profile.plot(pen=pg.mkPen('#888888',style=Qt.DashLine))
        self.region=pg.LinearRegionItem([self.low.value(),self.high.value()],bounds=[x[0],x[-1]]);self.profile.addItem(self.region)
        self.peak=pg.InfiniteLine(self.seed.value(),movable=True,pen=pg.mkPen('#bf6200'));self.profile.addItem(self.peak)
        self.region.sigRegionChangeFinished.connect(self.region_changed);self.peak.sigPositionChangeFinished.connect(self.peak_changed)
        self.message=W.QLabel();self.message.setWordWrap(True);layout.addWidget(self.message)
        bottom=W.QSplitter();self.plot_split.addWidget(bottom);self.plot_split.setSizes([440,300]);self.plot_split.setChildrenCollapsible(False)
        self.table=W.QTableWidget(len(data['energy']),9);self.table.setHorizontalHeaderLabels(['接受','E (meV)','q / s','峰位 ±1σ','FWHM','宽度 ±1σ','残差 RMS','范围','状态']);self.table.horizontalHeader().setSectionResizeMode(W.QHeaderView.ResizeToContents);bottom.addWidget(self.table)
        self.table.cellClicked.connect(lambda row,col:self.energy.setCurrentIndex(row));self.table.itemChanged.connect(self.accept_changed)
        self.diagnostic_tabs=W.QTabWidget();bottom.addWidget(self.diagnostic_tabs)
        self.width_plot=pg.PlotWidget();self.width_plot.setLabel('bottom','E (meV)');self.width_plot.setLabel('left','FWHM (cycles/nm)');self.diagnostic_tabs.addTab(self.width_plot,'峰宽随能量')
        self.residual_plot=pg.PlotWidget();self.residual_plot.setLabel('bottom','q / s (cycles/nm)');self.residual_plot.setLabel('left','data - fit (pA)');self.diagnostic_tabs.addTab(self.residual_plot,'当前线切残差')
        self.band_plot=pg.PlotWidget();self.band_plot.setTitle('模型推断 k–E：尚未指定模型');self.band_plot.setLabel('bottom','k relative to pocket center (nm⁻¹)');self.band_plot.setLabel('left','E (meV)');self.diagnostic_tabs.addTab(self.band_plot,'模型能带（高级）');self.diagnostic_tabs.setTabVisible(2,False)
        bottom.setSizes([650,750]);bottom.setChildrenCollapsible(False)
        bandbox=W.QGroupBox('模型推断：单口袋直径散射 Q=2k（需确认，不能用于任意口袋间散射）');bl=W.QVBoxLayout(bandbox);layout.addWidget(bandbox)
        br=W.QHBoxLayout();bl.addLayout(br)
        self.assumption=W.QCheckBox('我确认所选分支满足 Q=2k、E=eV，所选能区可用正常态抛物线近似');bl.insertWidget(0,self.assumption)
        self.carrier=W.QComboBox();self.carrier.addItems(['电子型','空穴型']);br.addWidget(self.carrier)
        self.branch=W.QComboBox();self.branch.addItems(['散射中心右侧','散射中心左侧']);br.addWidget(self.branch)
        self.center=field(0);br.addWidget(labeled('固定中心 q₀ / s₀',self.center))
        self.emin=field(float(data['energy'].min()));self.emax=field(float(data['energy'].max()))
        for name,w in [('E 下限',self.emin),('E 上限',self.emax)]:br.addWidget(labeled(name,w))
        go=W.QPushButton('拟合所选能区的已接受峰');self.band_go=go;go.clicked.connect(self.fit_band);br.addWidget(go)
        self.band_text=W.QLabel('先核实散射通道。模型只适用于过 q=0 的直线截面；折线路径长度不能直接转成 k。');self.band_text.setWordWrap(True);bl.addWidget(self.band_text)
        self.assumption.toggled.connect(self.invalidate_band)
        for w in (self.center,self.emin,self.emax):w.valueChanged.connect(self.invalidate_band)
        for w in (self.carrier,self.branch):w.currentIndexChanged.connect(self.invalidate_band)
        if data['metadata']['cut']['mode']!='line':go.setEnabled(False);self.assumption.setEnabled(False)
        self.progress=W.QProgressBar();self.progress.setRange(0,len(data['energy']));self.progress.hide();layout.addWidget(self.progress)
        self.refresh_table();self.select_energy(0)

    def invalidate_band(self,*args):
        self.band=None;self.model_curve.setData([],[]);self.band_plot.clear()
        self.band_text.setText('模型未拟合或已失效。仅用同一散射分支、选定能区的已接受峰；避开不适用的超导能隙/拐折区。')

    def select_energy(self,i):
        if i<0:return
        self.index=i;self.updating=True;c=self.configs[i]
        for key in ('low','high','seed'):getattr(self,key).setValue(c[key])
        self.region.setRegion((c['low'],c['high']));self.peak.setValue(c['seed']);self.updating=False
        self.show_profile()

    def edit_config(self,*args):
        if self.updating:return
        self.configs[self.index]={k:getattr(self,k).value() for k in ('low','high','seed')}
        self.results[self.index]=None;self.invalidate_band();self.select_energy(self.index);self.refresh_table()

    def region_changed(self):
        if self.updating:return
        lo,hi=self.region.getRegion();self.updating=True;self.low.setValue(lo);self.high.setValue(hi);self.updating=False;self.edit_config()

    def peak_changed(self):
        if not self.updating:self.seed.setValue(self.peak.value())

    def map_click(self,event):
        if self.busy or event.button()!=Qt.LeftButton or not self.map.plotItem.vb.sceneBoundingRect().contains(event.scenePos()):return
        p=self.map.plotItem.vb.mapSceneToView(event.scenePos());i=int(np.argmin(abs(self.data['energy']-p.y())))
        self.energy.setCurrentIndex(i);self.seed.setValue(p.x())

    def show_profile(self):
        x=self.data['x'];y=self.data['values'][self.index];self.observed.setData(x,y);self.mark.setValue(self.data['energy'][self.index])
        self.model.setData([],[]);self.background.setData([],[]);self.residual_plot.clear();r=self.results[self.index]
        self.profile.setTitle(f'实测能量 {self.data["energy"][self.index]:.6g} meV · 单峰高斯 + 线性背景')
        if r and 'error' not in r:
            keep=(x>=r['low'])&(x<=r['high']);xx=x[keep];fit=evaluate_result(xx,r)
            self.model.setData(xx,fit);self.background.setData(xx,r['background']+r['slope']*(xx-r['reference']))
            self.residual_plot.plot(xx,y[keep]-fit,pen='#bb542a',symbol='o',symbolSize=3)
            self.message.setText(f"q={r['center']:.6g} ± {r['center_se']:.3g}; FWHM={r['fwhm']:.6g} ± {r['fwhm_se']:.3g} cycles/nm。"+'；'.join(r['flags'])+'  误差为条件拟合 1σ，不含漂移/标定/窗函数与相关噪声；FWHM 不是寿命。')
        else:self.message.setText(r['error'] if r else '选择峰初值和拟合范围。每个能量可独立调整；窗口应包含峰两侧背景。')

    def fit_current(self):
        self.results[self.index]=calculate(self.data,self.index,self.configs[self.index]);self.invalidate_band();self.refresh_table();self.show_profile()

    def copy_config(self):
        self.configs=[dict(self.configs[self.index]) for _ in self.configs];self.results=[None]*len(self.results);self.invalidate_band();self.refresh_table();self.show_profile()

    def fit_all(self):
        self.busy=True;self.actions.setEnabled(False);self.table.setEnabled(False);self.region.setMovable(False);self.peak.setMovable(False);self.progress.show();self.progress.setValue(0)
        self.worker=Batch(self.data,self.configs,self);self.worker.progress.connect(self.progress.setValue);self.worker.finished_rows.connect(self.batch_done);self.worker.start()

    def batch_done(self,results):
        self.results=results;self.busy=False;self.actions.setEnabled(True);self.table.setEnabled(True);self.region.setMovable(True);self.peak.setMovable(True);self.progress.hide();self.invalidate_band();self.refresh_table();self.show_profile()

    def accept_changed(self,item):
        if item.column()!=0:return
        r=self.results[item.row()]
        if r and 'error' not in r:r['accepted']=item.checkState()==Qt.Checked;self.invalidate_band();self.refresh_plots()

    def refresh_table(self):
        self.table.blockSignals(True)
        for i,e in enumerate(self.data['energy']):
            r=self.results[i];valid=r is not None and 'error' not in r
            check=W.QTableWidgetItem();check.setFlags(Qt.ItemIsEnabled|Qt.ItemIsSelectable|(Qt.ItemIsUserCheckable if valid else Qt.NoItemFlags));check.setCheckState(Qt.Checked if valid and r['accepted'] else Qt.Unchecked);self.table.setItem(i,0,check)
            c=self.configs[i];values=[f'{e:.6g}',*[f'{r[k]:.6g}' if valid else '' for k in ('center','center_se','fwhm','fwhm_se','residual_rms')],f"{c['low']:.5g} .. {c['high']:.5g}",r.get('error','；'.join(r.get('flags',[])) or '拟合通过') if r else '未拟合']
            for j,s in enumerate(values,1):
                item=W.QTableWidgetItem(s);item.setFlags(Qt.ItemIsEnabled|Qt.ItemIsSelectable);self.table.setItem(i,j,item)
        self.table.blockSignals(False);self.refresh_plots()

    def refresh_plots(self):
        positions=[r['center'] if r and 'error' not in r and r['accepted'] else np.nan for r in self.results]
        self.dispersion_line.setData(positions,self.data['energy'],connect='finite')
        rows=[(e,r) for e,r in zip(self.data['energy'],self.results) if r and 'error' not in r and r['accepted']]
        self.width_plot.clear()
        if not rows:self.points.setData([],[]);self.qerrors.setData(x=np.array([]),y=np.array([]),width=np.array([]));return
        e=np.array([t[0] for t in rows]);q=np.array([t[1]['center'] for t in rows]);se=np.array([t[1]['center_se'] for t in rows])
        self.points.setData(q,e);self.qerrors.setData(x=q,y=e,width=2*se)
        widths=np.array([t[1]['fwhm'] for t in rows]);errors=np.array([t[1]['fwhm_se'] for t in rows])
        self.width_plot.plot(e,widths,pen=None,symbol='o',symbolSize=5);self.width_plot.addItem(pg.ErrorBarItem(x=e,y=widths,height=2*errors,pen='#3887a7'))

    def fit_band(self):
        try:
            if self.busy:raise ValueError('请等待逐能量拟合完成。')
            if self.data['metadata']['cut']['mode']!='line' or not self.assumption.isChecked():raise ValueError('先确认单口袋直径散射和正常态抛物线模型假设。')
            if self.emin.value()>=self.emax.value():raise ValueError('能区下限必须小于上限。')
            rows=[(e,r) for e,r in zip(self.data['energy'],self.results) if r and 'error' not in r and r['accepted'] and self.emin.value()<=e<=self.emax.value()]
            self.band=fit_parabolic_scattering([e for e,r in rows],[r['center'] for e,r in rows],[r['center_se'] for e,r in rows], 'electron' if self.carrier.currentIndex()==0 else 'hole',1 if self.branch.currentIndex()==0 else -1,self.center.value())
            self.show_band()
        except ValueError as e:self.band=None;self.model_curve.setData([],[]);self.band_text.setText(str(e))

    def show_band(self):
        b=self.band
        self.diagnostic_tabs.setTabVisible(2,True);self.diagnostic_tabs.setCurrentIndex(2)
        self.model_curve.setData(b['q_fit'],b['energy_meV'])
        self.band_plot.clear();self.band_plot.setTitle('模型推断 k–E（Q=2k；非直接能带测量）')
        k=np.asarray(b['k_relative_nm_inverse']);e=np.asarray(b['energy_meV'])
        self.band_plot.plot(k,e,pen=None,symbol='o',symbolSize=5)
        self.band_plot.addItem(pg.ErrorBarItem(x=k,y=e,width=2*np.asarray(b['k_se_nm_inverse']),pen='#2277aa'))
        self.band_plot.plot(np.pi*(np.asarray(b['q_fit'])-b['q_center_cycles_nm']),e,pen='#dd4422')
        self.band_text.setText(f"模型推断（非直接测量）：E₀={b['E0_meV']:.6g} ± {b['E0_se_meV']:.3g} meV；|m*|/mₑ={b['mass_over_me']:.5g} ± {b['mass_se']:.3g}；χ²/dof={b['reduced_chi_squared']:.4g}。"+(f" kF（相对口袋中心）={b['kF_relative_nm_inverse']:.5g} ± {b['kF_se_nm_inverse']:.3g} nm⁻¹。" if b['fermi_crossing'] else ' 模型未在 E=0 穿越。')+' 使用 Q−Q₀=2π(q−q₀)、k=(Q−Q₀)/2；误差不含能量分辨率、模型误差与坐标标定误差。χ² 大或残差有结构时不可据此宣称真实能带。')

    def export_to(self,root):
        root=Path(root);root.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(root/'snapshot.npz',x=self.data['x'],energy=self.data['energy'],values=self.data['values'])
        state=dict(schema=1,metadata=self.data['metadata'],resolution=self.data['resolution'],configs=self.configs,results=self.results,band=self.band,
                   implementation_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ('qpi_fit.py','qpi_dialog.py','volume.py')},
                   snapshot_sha256=hashlib.sha256((root/'snapshot.npz').read_bytes()).hexdigest(),
                   band_controls=dict(confirmed=self.assumption.isChecked(),carrier=self.carrier.currentIndex(),branch=self.branch.currentIndex(),center=self.center.value(),emin=self.emin.value(),emax=self.emax.value()))
        (root/'analysis.qpi.json').write_text(json.dumps(state,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        with (root/'peaks.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f);writer.writerow(['E_meV','accepted','q_or_s_cycles_nm','q_se_1sigma','FWHM_cycles_nm','FWHM_se_1sigma','residual_rms','seed','low','high','status'])
            for e,c,r in zip(self.data['energy'],self.configs,self.results):
                r=r or {};writer.writerow([e,r.get('accepted',False),*[r.get(k,'') for k in ('center','center_se','fwhm','fwhm_se','residual_rms')],c['seed'],c['low'],c['high'],r.get('error',';'.join(r.get('flags',[])))])
        with (root/'profiles_and_residuals.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f);writer.writerow(['E_meV','q_or_s_cycles_nm','linear_FFT_amplitude_pA','fit_in_window','residual'])
            for e,y,r in zip(self.data['energy'],self.data['values'],self.results):
                for q,value in zip(self.data['x'],y):
                    model=float(evaluate_result(q,r)) if r and 'error' not in r and r['low']<=q<=r['high'] else None
                    writer.writerow([e,q,value if np.isfinite(value) else '',model if model is not None else '',value-model if model is not None else ''])
        if self.band:
            b=self.band
            with (root/'model_inferred_band.csv').open('w',encoding='utf-8-sig',newline='') as f:
                w=csv.writer(f);w.writerow(['E_meV','q_observed_cycles_nm','q_model_cycles_nm','q_residual','k_relative_nm_inverse_ASSUMED_Q_EQUALS_2K','k_se_1sigma']);w.writerows(zip(*[b[k] for k in ('energy_meV','q_observed','q_fit','q_residual','k_relative_nm_inverse','k_se_nm_inverse')]))
        self.figure(root/'qpi_results.png')
        return root

    def figure(self,path):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig=Figure(figsize=(12,8),dpi=160,layout='constrained');FigureCanvasAgg(fig);a,b,c,d=fig.subplots(2,2).ravel()
        im=a.pcolormesh(self.data['x'],self.data['energy'],self.data['values'],shading='nearest',cmap='viridis');fig.colorbar(im,ax=a,label='linear |FFT| (pA DFT sum)');a.set(xlabel='q / path distance (cycles/nm)',ylabel='E=eV (meV)',title='QPI q-E section (not a direct band)')
        rows=[(e,r) for e,r in zip(self.data['energy'],self.results) if r and 'error' not in r and r['accepted']]
        for e,r in rows:a.errorbar(r['center'],e,xerr=r['center_se'],fmt='r.',capsize=2);b.errorbar(e,r['fwhm'],yerr=r['fwhm_se'],fmt='k.',capsize=2)
        b.set(xlabel='E (meV)',ylabel='FWHM (cycles/nm)',title='Gaussian width; conditional 1 sigma')
        if self.band:
            z=self.band;a.plot(z['q_fit'],z['energy_meV'],'c-');c.errorbar(z['k_relative_nm_inverse'],z['energy_meV'],xerr=z['k_se_nm_inverse'],fmt='k.')
            k=np.pi*(np.asarray(z['q_fit'])-z['q_center_cycles_nm']);c.plot(k,z['energy_meV'],'r-');d.plot(z['energy_meV'],z['q_residual'],'o-')
            c.set_title('Model-inferred band ONLY: Q=2k');c.set(xlabel='k relative to pocket center (1/nm)',ylabel='E (meV)');d.set(xlabel='E (meV)',ylabel='q residual (cycles/nm)',title=f"E0={z['E0_meV']:.4g} meV; |m*|/me={z['mass_over_me']:.4g}")
        else:
            i=self.index;x=self.data['x'];y=self.data['values'][i];c.plot(x,y,'k.-');r=self.results[i]
            if r and 'error' not in r:
                keep=(x>=r['low'])&(x<=r['high']);fit=evaluate_result(x[keep],r);c.plot(x[keep],fit,'r-');d.plot(x[keep],y[keep]-fit,'k.-')
            c.set(xlabel='q / s (cycles/nm)',ylabel='linear |FFT|',title=f"Profile E={self.data['energy'][i]:.5g} meV");d.set(xlabel='q / s (cycles/nm)',ylabel='data - fit')
        fig.savefig(path)

    def export(self):
        p=W.QFileDialog.getExistingDirectory(self,'选择输出目录（新建独立分析文件夹）')
        if p:
            try:self.message.setText('已保存：'+str(self.export_to(Path(p)/datetime.now().strftime('QPI_%Y%m%d_%H%M%S_%f'))))
            except Exception as e:self.message.setText('保存失败：'+str(e))

    @classmethod
    def load_analysis(cls,path,parent=None):
        path=Path(path);s=json.loads(path.read_text(encoding='utf-8'));npz=path.parent/'snapshot.npz'
        if hashlib.sha256(npz.read_bytes()).hexdigest()!=s['snapshot_sha256']:raise ValueError('数据快照校验失败。')
        with np.load(npz,allow_pickle=False) as a:data={k:a[k].copy() for k in ('x','energy','values')}
        data.update(resolution=s['resolution'],metadata=s['metadata']);dlg=cls(data,parent)
        dlg.configs=s['configs'];dlg.results=s['results'];bc=s['band_controls']
        dlg.assumption.setChecked(bc['confirmed']);dlg.carrier.setCurrentIndex(bc['carrier']);dlg.branch.setCurrentIndex(bc['branch'])
        for k in ('center','emin','emax'):getattr(dlg,k).setValue(bc[k])
        dlg.band=s['band'];dlg.refresh_table();dlg.select_energy(0)
        if dlg.band:dlg.show_band()
        return dlg

    def reopen(self):
        p,_=W.QFileDialog.getOpenFileName(self,'重开 QPI 分析','','QPI analysis (*.qpi.json)')
        if p:
            try:self.reopened=self.load_analysis(p,self);self.reopened.show()
            except Exception as e:self.message.setText(str(e))

    def closeEvent(self,event):
        if any(t.isRunning() for t in self.findChildren(QtCore.QThread)):event.ignore();self.message.setText('拟合进行中，请等待完成。')
        else:event.accept()

# Public workspace supports multiple independent branches; schema-1 sessions migrate.
from qpi_pair import PairQPIDialog as QPIDialog
