"""Paired QPI peaks: measured midpoint/half-separation before optional band model."""
import json,csv,hashlib
import numpy as np
from PySide6 import QtWidgets as W
import pyqtgraph as pg
from qpi_dialog import labeled,field
from qpi_multi import MultiQPIDialog
from qpi_fit import fit_parabolic_scattering


def paired_rows(energy, first, second, low, high):
    rows=[]
    for e,a,b in zip(energy,first,second):
        if not low<=e<=high or not all(r and 'error' not in r and r.get('accepted') for r in (a,b)):continue
        va=a['center_se']**2;vb=b['center_se']**2;cov=0.;source='cross-peak covariance unavailable; assumed zero'
        if a.get('joint_group') and a.get('joint_group')==b.get('joint_group'):
            matrix=np.asarray(a['joint_covariance']);cov=float(matrix[3*a['joint_component']+1,3*b['joint_component']+1]);source='full joint peak covariance'
        cm=(va+vb+2*cov)/4;rv=(va+vb-2*cov)/4
        if not np.isfinite([cm,rv]).all() or min(cm,rv)<=0:continue
        ql,qh=sorted([a['center'],b['center']]);radius=(qh-ql)/2
        if radius<=0:continue
        rows.append(dict(energy=float(e),left=ql,right=qh,midpoint=(qh+ql)/2,midpoint_se=float(np.sqrt(cm)),radius=radius,radius_se=float(np.sqrt(rv)),covariance_note=source))
    if not rows:raise ValueError('选定能区没有两个分支同时接受且误差有效的峰。')
    return rows

class PairQPIDialog(MultiQPIDialog):
    def __init__(self,data,parent=None):
        super().__init__(data,parent)
        self.pair_result=None;self.pair_signature=None
        old=self.assumption.parentWidget();idx=self.layout().indexOf(old);self.layout().removeWidget(old)
        self.analysis_tabs=W.QTabWidget();self.layout().insertWidget(idx,self.analysis_tabs)
        box=W.QWidget();bl=W.QVBoxLayout(box);bl.setContentsMargins(6,4,6,4)
        self.analysis_tabs.addTab(box,'双峰配对（默认）');self.analysis_tabs.addTab(old,'单侧峰模型（高级）')
        row=W.QHBoxLayout();bl.addLayout(row)
        self.pair_a=W.QComboBox();self.pair_b=W.QComboBox()
        self.pair_low=field(float(min(data['energy'])));self.pair_high=field(float(max(data['energy'])))
        for title,w in [('峰分支 A',self.pair_a),('峰分支 B',self.pair_b),('E 下限 (meV)',self.pair_low),('E 上限 (meV)',self.pair_high)]:row.addWidget(labeled(title,w))
        go=W.QPushButton('计算双峰中心与间距');row.addWidget(go);go.clicked.connect(self.calculate_pair)
        row=W.QHBoxLayout();bl.addLayout(row)
        self.pair_confirm=W.QCheckBox('两峰是同一散射环的过中心截面；该通道满足直径散射 Q=2k')
        row.addWidget(self.pair_confirm);self.pair_carrier=W.QComboBox();self.pair_carrier.addItems(['电子型（带底）','空穴型（带顶）']);row.addWidget(self.pair_carrier)
        model=W.QPushButton('由双峰间距拟合模型能带');row.addWidget(model);model.clicked.connect(self.fit_pair_band)
        self.pair_text=W.QLabel('先选择两个峰分支：自动取中点与半间距，无需指定左右或固定中心。幅度和峰宽不要求相同。');self.pair_text.setWordWrap(True);bl.addWidget(self.pair_text)
        self.pair_plot=pg.PlotWidget();self.pair_plot.setLabel('bottom','q / s (cycles/nm)');self.pair_plot.setLabel('left','E (meV)')
        self.pair_tab=self.diagnostic_tabs.addTab(self.pair_plot,'双峰中心与半间距')
        self.pair_band_plot=pg.PlotWidget();self.pair_band_plot.setLabel('bottom','k relative to pocket center (nm⁻¹)');self.pair_band_plot.setLabel('left','E (meV)')
        self.pair_band_tab=self.diagnostic_tabs.addTab(self.pair_band_plot,'双峰模型能带');self.diagnostic_tabs.setTabVisible(self.pair_band_tab,False)
        for w in (self.pair_a,self.pair_b,self.pair_carrier):w.currentIndexChanged.connect(self.clear_pair)
        for w in (self.pair_low,self.pair_high):w.valueChanged.connect(self.clear_pair)
        self.pair_confirm.toggled.connect(self.clear_pair)
        self.sync_pairs();self.all_table.setMaximumHeight(90);self.plot_split.setSizes([320,300]);self.resize(1500,1080)
        for plot in (self.map,self.profile,self.width_plot,self.residual_plot,self.band_plot,self.pair_plot,self.pair_band_plot):
            for axis in ('bottom','left'):plot.getAxis(axis).enableAutoSIPrefix(False)
        self.analysis_tabs.setToolTip('双峰测量不需要 Q=2k 假设；转换模型能带才需要确认。')

    def sync_pairs(self):
        if not hasattr(self,'pair_a'):return
        for n,w in enumerate((self.pair_a,self.pair_b)):
            old=w.currentData();w.blockSignals(True);w.clear()
            for t in self.tracks:w.addItem(t['name'],t['id'])
            i=w.findData(old)
            if n==1 and len(self.tracks)>1 and old==self.pair_a.currentData():i=1 if self.pair_a.currentIndex()==0 else 0
            w.setCurrentIndex(i if i>=0 else min(n,len(self.tracks)-1));w.blockSignals(False)

    def rebuild_track_box(self):
        super().rebuild_track_box();self.sync_pairs()

    def signature(self):
        ids=(self.pair_a.currentData(),self.pair_b.currentData())
        return hashlib.sha256(json.dumps([[(t['id'],t['results']) for t in self.tracks if t['id'] in ids],ids,self.pair_low.value(),self.pair_high.value(),self.pair_confirm.isChecked(),self.pair_carrier.currentIndex()],sort_keys=True).encode()).hexdigest()

    def clear_pair(self,*args):
        if not hasattr(self,'pair_plot'):return
        self.pair_result=None;self.pair_signature=None;self.pair_plot.clear();self.pair_band_plot.clear();self.diagnostic_tabs.setTabVisible(self.pair_band_tab,False)
        self.pair_text.setText('配对结果尚未计算或已失效；选择两个分支后重新计算。')

    def refresh_plots(self):
        super().refresh_plots()
        if getattr(self,'pair_signature',None) and self.signature()!=self.pair_signature:self.clear_pair()

    def calculate_pair(self):
        try:
            if self.busy:raise ValueError('请等待峰拟合完成。')
            self.stash();self.clear_pair()
            if not self.straight_model:raise ValueError('请使用单一直线截面；折线路径间距不能作为散射环半径。')
            ids=[self.pair_a.currentData(),self.pair_b.currentData()]
            if ids[0]==ids[1]:raise ValueError('请选择两个不同的峰分支。')
            ts=[next(t for t in self.tracks if t['id']==k) for k in ids]
            rows=paired_rows(self.data['energy'],ts[0]['results'],ts[1]['results'],self.pair_low.value(),self.pair_high.value())
            self.pair_result=dict(track_ids=ids,rows=rows,model=None,definition='midpoint=(qL+qR)/2; half separation=(qR-qL)/2; radius only for a central cut')
            self.pair_signature=self.signature();self.draw_pair()
            self.pair_text.setText(f'已配对 {len(rows)} 个能量；中点随能量逐点显示，不强制固定中心。半间距只有在过环中心截面上才是半径。缺失或未接受的峰不补齐。')
            return True
        except (ValueError,StopIteration) as e:self.pair_text.setText(str(e));return False

    def draw_pair(self):
        self.pair_plot.clear();rows=self.pair_result['rows'];e=np.array([r['energy'] for r in rows])
        self.pair_plot.addLegend();self.pair_plot.setTitle('双峰中点（橙）与半间距（蓝）；不是直接能带')
        for key,color in [('midpoint','#d87524'),('radius','#1688c5')]:
            x=np.array([r[key] for r in rows]);err=np.array([r[key+'_se'] for r in rows]);self.pair_plot.plot(x,e,pen=None,symbol='o',symbolBrush=color,name=key);self.pair_plot.addItem(pg.ErrorBarItem(x=x,y=e,width=2*err,pen=color))
        self.diagnostic_tabs.setCurrentIndex(self.pair_tab)
        if self.pair_result.get('model'):self.draw_pair_band()

    def fit_pair_band(self):
        if not self.calculate_pair():return
        try:
            if not self.pair_confirm.isChecked():raise ValueError('中心和间距已算出。转换能带前，请确认过环中心截面和 Q=2k 散射对应关系。')
            rows=self.pair_result['rows']
            model=fit_parabolic_scattering([r['energy'] for r in rows],[r['radius'] for r in rows],[r['radius_se'] for r in rows],'electron' if self.pair_carrier.currentIndex()==0 else 'hole',1,0)
            model['assumption']='Paired peaks on a central cut of the SAME diameter-scattering feature; radius in cycles/nm, k=pi*radius. Not general interpocket inversion.'
            self.pair_result['model']=model;self.draw_pair_band()
        except (ValueError,np.linalg.LinAlgError) as e:self.pair_text.setText(str(e))

    def draw_pair_band(self):
        b=self.pair_result['model'];self.pair_band_plot.clear();self.pair_band_plot.setTitle('双峰间距推断的模型 k–E：需满足直径散射')
        e=np.array(b['energy_meV']);k=np.array(b['k_relative_nm_inverse']);se=np.array(b['k_se_nm_inverse'])
        for sign in (-1,1):
            self.pair_band_plot.plot(sign*k,e,pen=None,symbol='o',symbolSize=5)
            self.pair_band_plot.addItem(pg.ErrorBarItem(x=sign*k,y=e,width=2*se,pen='#1688c5'))
            self.pair_band_plot.plot(sign*np.pi*np.array(b['q_fit']),e,pen='#db7524')
        self.diagnostic_tabs.setTabVisible(self.pair_band_tab,True);self.diagnostic_tabs.setCurrentIndex(self.pair_band_tab)
        self.pair_text.setText(f"双峰模型：E₀={b['E0_meV']:.5g} ± {b['E0_se_meV']:.2g} meV；|m*|/mₑ={b['mass_over_me']:.5g} ± {b['mass_se']:.2g}；χ²/dof={b['reduced_chi_squared']:.4g}。k=π×半间距（q 单位 cycles/nm）。左右 k 为模型镜像，不是两次独立测量；误差为条件 1σ。请检查中点是否随能量漂移。")

    def export_to(self,root):
        self.stash()
        if self.pair_signature and self.signature()!=self.pair_signature:self.clear_pair()
        root=super().export_to(root);p=root/'analysis.qpi.json';s=json.loads(p.read_text(encoding='utf-8'))
        s['pair_analysis']=self.pair_result;s['pair_controls']=dict(ids=[self.pair_a.currentData(),self.pair_b.currentData()],low=self.pair_low.value(),high=self.pair_high.value(),confirmed=self.pair_confirm.isChecked(),carrier=self.pair_carrier.currentIndex())
        s['implementation_sha256']['qpi_pair.py']=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest();p.write_text(json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        if self.pair_result:
            rows=self.pair_result['rows']
            with (root/'paired_peaks.csv').open('w',encoding='utf-8-sig',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            b=self.pair_result.get('model')
            if b:
                with (root/'paired_model_band.csv').open('w',encoding='utf-8-sig',newline='') as f:
                    w=csv.writer(f);w.writerow(['E_meV','k_abs_nm_inverse','k_se','model_k_abs_nm_inverse']);w.writerows(zip(b['energy_meV'],b['k_relative_nm_inverse'],b['k_se_nm_inverse'],np.pi*np.array(b['q_fit'])))
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            fig=Figure(figsize=(10,4),dpi=160,layout='constrained');FigureCanvasAgg(fig);a,c=fig.subplots(1,2)
            e=[r['energy'] for r in rows]
            for key in ('midpoint','radius'):a.errorbar([r[key] for r in rows],e,xerr=[r[key+'_se'] for r in rows],fmt='.',label=key)
            a.set(xlabel='q / s (cycles/nm)',ylabel='E (meV)',title='Measured midpoint and half-separation');a.legend()
            if b:
                for sign in (-1,1):
                    c.errorbar(sign*np.array(b['k_relative_nm_inverse']),e,xerr=b['k_se_nm_inverse'],fmt='.')
                    c.plot(sign*np.pi*np.array(b['q_fit']),e)
                c.set(xlabel='relative k (1/nm)',ylabel='E (meV)',title='Model inferred ONLY: Q=2k')
            else:c.text(.5,.5,'No band model fitted',ha='center');c.axis('off')
            fig.savefig(root/'paired_analysis.png')
        return root

    @classmethod
    def load_analysis(cls,path,parent=None):
        dlg=super().load_analysis(path,parent);s=json.loads(__import__('pathlib').Path(path).read_text(encoding='utf-8'));dlg.sync_pairs();c=s.get('pair_controls')
        if c:
            for w,k in zip((dlg.pair_a,dlg.pair_b),c['ids']):w.setCurrentIndex(w.findData(k))
            dlg.pair_low.setValue(c['low']);dlg.pair_high.setValue(c['high']);dlg.pair_confirm.setChecked(c['confirmed']);dlg.pair_carrier.setCurrentIndex(c['carrier'])
            dlg.pair_result=s.get('pair_analysis')
            if dlg.pair_result:dlg.pair_signature=dlg.signature();dlg.draw_pair()
        return dlg
