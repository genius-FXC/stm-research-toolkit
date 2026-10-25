"""Multiple independent q-E tracks and joint overlapping-peak inference."""
import copy,csv,json,hashlib,uuid
from pathlib import Path
import numpy as np
from PySide6 import QtCore,QtWidgets as W
from PySide6.QtCore import Qt
import pyqtgraph as pg
from qpi_dialog import BaseQPIDialog,calculate
from qpi_fit import fit_multiple,evaluate_result,fit_parabolic_scattering

COLORS=['#da5537','#147db3','#8f55b4','#26976a','#c18e13','#d147a5']

def joint_row(data,tracks,i):
    chosen=[t for t in tracks if t.get('joint',True) and t['configs'][i].get('enabled',True)]
    ids=[t['id'] for t in chosen]
    if len(ids)<2:return {k:{'error':'该能量至少需要两个已启用的联合分支','accepted':False} for k in ids}
    configs=[t['configs'][i] for t in chosen];lo=min(c['low'] for c in configs);hi=max(c['high'] for c in configs)
    try:
        if data['metadata']['cut']['mode']=='path' and any(lo<z<hi for z in data['metadata']['cut'].get('corner_distances',[])[1:-1]):
            raise ValueError('联合范围跨折线拐角，请选同一直线段。')
        fits=fit_multiple(data['x'],data['values'][i],[c['seed'] for c in configs],lo,hi,data['resolution'])
        group=str(uuid.uuid4())
        for r in fits:r.update(joint_group=group,joint_track_ids=ids)
        return dict(zip(ids,fits))
    except (ValueError,RuntimeError,np.linalg.LinAlgError) as e:return {k:{'error':str(e),'accepted':False} for k in ids}

class JointBatch(QtCore.QThread):
    ready=QtCore.Signal(object);progress=QtCore.Signal(int)
    def __init__(self,data,tracks,parent):super().__init__(parent);self.data=data;self.tracks=copy.deepcopy(tracks)
    def run(self):
        rows=[]
        for i in range(len(self.data['energy'])):rows.append(joint_row(self.data,self.tracks,i));self.progress.emit(i+1)
        self.ready.emit(rows)

class MultiQPIDialog(BaseQPIDialog):
    def __init__(self,data,parent=None):
        super().__init__(data,parent)
        self.active=0;self.tracks=[];self.overlays=[];self.components=[];self.switching=False
        self.tracks=[self.make_track('P1')]
        self.track_bar=W.QWidget();bar=W.QHBoxLayout(self.track_bar)
        self.track_box=W.QComboBox();bar.addWidget(W.QLabel('当前散射分支'));bar.addWidget(self.track_box)
        self.track_name=W.QLineEdit('P1');self.track_name.setMaximumWidth(140);bar.addWidget(self.track_name)
        for label,method in [('新增峰/分支',self.add_track),('删除当前分支',self.remove_track)]:
            b=W.QPushButton(label);b.clicked.connect(method);bar.addWidget(b)
        self.joint_check=W.QCheckBox('参与联合拟合');self.joint_check.setChecked(True);bar.addWidget(self.joint_check)
        self.enabled_check=W.QCheckBox('当前能量启用');self.enabled_check.setChecked(True);bar.addWidget(self.enabled_check)
        self.global_range=W.QCheckBox('拟合范围应用于当前分支的全部偏压');self.global_range.setChecked(True);bar.addWidget(self.global_range)
        bar.addStretch();self.layout().insertWidget(1,self.track_bar)
        jointbar=W.QHBoxLayout()
        for label,method in [('当前能量联合多峰拟合',self.fit_joint_current),('全部能量联合多峰拟合',self.fit_joint_all)]:
            b=W.QPushButton(label);b.clicked.connect(method);jointbar.addWidget(b)
        jointbar.addWidget(W.QLabel('联合范围＝参与分支窗口的最小下限至最大上限；共用线性背景，各峰独立。'))
        self.layout().insertLayout(3,jointbar)
        self.joint_buttons=[jointbar.itemAt(i).widget() for i in range(2)]
        self.all_table=W.QTableWidget(0,7);self.all_table.setMaximumHeight(120);self.all_table.setHorizontalHeaderLabels(['分支','E (meV)','q / s','峰位 1σ','FWHM','接受','状态']);self.all_table.horizontalHeader().setSectionResizeMode(W.QHeaderView.Stretch)
        self.layout().insertWidget(4,self.all_table);self.all_table.cellClicked.connect(self.all_row_click)
        self.track_box.currentIndexChanged.connect(self.switch_track);self.track_name.editingFinished.connect(self.rename_track)
        self.joint_check.toggled.connect(self.joint_selection);self.enabled_check.toggled.connect(self.enable_current)
        self.rebuild_track_box()
        self.straight_model=data['metadata']['cut']['mode']=='line' or len(data['metadata']['cut'].get('vertices',[]))==2
        self.band_go.setEnabled(self.straight_model);self.assumption.setEnabled(self.straight_model)
        self.assumption.setText('我确认本分支相对指定中心满足 |Q−Q₀|=2|k|、E=eV，所选能区可用正常态抛物线近似')
        self.band_text.setText('各分支中心独立且可非零；两点直线用沿线坐标 s₀。必须确认该直线通过对应散射中心；不自动指定 M 点或强制左右对称。')
        self.setWindowTitle('QPI multi-peak / multi-branch · q–E');self.resize(1500,1080);self.refresh_plots();self.show_profile()

    def controls(self):
        return dict(confirmed=self.assumption.isChecked(),carrier=self.carrier.currentIndex(),branch=self.branch.currentIndex(),center=self.center.value(),emin=self.emin.value(),emax=self.emax.value())

    def make_track(self,name):
        return dict(id=str(uuid.uuid4()),name=name,joint=True,configs=copy.deepcopy(self.configs),results=copy.deepcopy(self.results),band=copy.deepcopy(self.band),controls=self.controls())

    def stash(self):
        if not hasattr(self,'tracks') or not self.tracks:return
        self.tracks[self.active].update(configs=self.configs,results=self.results,band=self.band,controls=self.controls())

    def restore_controls(self,c):
        for widget,key in [(self.assumption,'confirmed'),(self.carrier,'carrier'),(self.branch,'branch'),(self.center,'center'),(self.emin,'emin'),(self.emax,'emax')]:
            widget.blockSignals(True)
            if key=='confirmed':widget.setChecked(c[key])
            elif key in ('carrier','branch'):widget.setCurrentIndex(c[key])
            else:widget.setValue(c[key])
            widget.blockSignals(False)

    def rebuild_track_box(self):
        self.track_box.blockSignals(True);self.track_box.clear()
        for t in self.tracks:self.track_box.addItem(t['name'])
        self.track_box.setCurrentIndex(self.active);self.track_box.blockSignals(False)

    def switch_track(self,index):
        if index<0 or self.busy:return
        self.stash();self.active=index;t=self.tracks[index];self.switching=True
        self.configs=t['configs'];self.results=t['results'];self.band=t['band'];self.restore_controls(t['controls'])
        self.track_name.setText(t['name']);self.joint_check.blockSignals(True);self.joint_check.setChecked(t['joint']);self.joint_check.blockSignals(False)
        self.model_curve.setData([],[]);self.band_plot.clear()
        if self.band:self.show_band()
        else:self.band_text.setText('当前分支尚无模型。中心 q₀/s₀ 可非零；未知散射对应关系时只提取 q–E，不转换 k–E。')
        self.select_energy(self.index);self.switching=False;self.refresh_table()

    def add_track(self):
        if self.busy:return
        self.stash();t=self.make_track(f'P{len(self.tracks)+1}');t['results']=[None]*len(self.results);t['band']=None;t['controls']['confirmed']=False
        self.tracks.append(t);self.rebuild_track_box();self.track_box.setCurrentIndex(len(self.tracks)-1)
        self.message.setText('新分支已建立；点击该分支的峰设置初值，并调整范围。其他分支保留。')

    def invalidate_joint(self,indices):
        self.stash()
        for i in indices:
            r=self.results[i];group=r.get('joint_group') if r else None
            if not group:continue
            for t in self.tracks:
                rr=t['results'][i]
                if rr and rr.get('joint_group')==group:t['results'][i]=None;t['band']=None
        self.results=self.tracks[self.active]['results'];self.band=self.tracks[self.active]['band']

    def remove_track(self):
        if self.busy or len(self.tracks)==1:return
        self.invalidate_joint(range(len(self.results)));self.tracks.pop(self.active);self.active=0
        t=self.tracks[0];self.configs=t['configs'];self.results=t['results'];self.band=t['band'];self.restore_controls(t['controls']);self.rebuild_track_box();self.switch_track(0)

    def rename_track(self):
        name=self.track_name.text().strip()
        if name:self.tracks[self.active]['name']=name;self.rebuild_track_box();self.refresh_plots()

    def joint_selection(self,value):
        if not self.busy:self.tracks[self.active]['joint']=value

    def enable_current(self,value):
        if self.switching or self.updating or self.busy:return
        self.invalidate_joint([self.index]);self.configs[self.index]['enabled']=value;self.results[self.index]=None;self.invalidate_band();self.refresh_table();self.show_profile()

    def select_energy(self,i):
        super().select_energy(i)
        if hasattr(self,'enabled_check'):
            self.enabled_check.blockSignals(True);self.enabled_check.setChecked(self.configs[self.index].get('enabled',True));self.enabled_check.blockSignals(False)
        if hasattr(self,'all_table'):self.refresh_current_peaks()

    def edit_config(self,*a):
        if self.updating:return
        enabled=self.configs[self.index].get('enabled',True)
        old=self.configs[self.index]
        all_range=hasattr(self,'global_range') and self.global_range.isChecked() and (old['low']!=self.low.value() or old['high']!=self.high.value())
        if hasattr(self,'tracks'):self.invalidate_joint(range(len(self.results)) if all_range else [self.index])
        if all_range:
            for i,c in enumerate(self.configs):c['low']=self.low.value();c['high']=self.high.value();self.results[i]=None
        super().edit_config(*a);self.configs[self.index]['enabled']=enabled

    def fit_current(self):
        self.invalidate_joint([self.index]);super().fit_current();self.stash()

    def copy_config(self):
        self.invalidate_joint(range(len(self.results)));super().copy_config();self.stash()

    def set_busy(self,value):
        self.track_bar.setEnabled(not value)
        for b in self.joint_buttons:b.setEnabled(not value)
        self.band_go.setEnabled(not value and self.straight_model)

    def fit_all(self):
        self.invalidate_joint(range(len(self.results)));self.set_busy(True);super().fit_all()

    def batch_done(self,rows):
        super().batch_done(rows);self.set_busy(False);self.stash();self.refresh_plots()

    def invalidate_band(self,*a):
        super().invalidate_band(*a)
        if hasattr(self,'tracks') and not getattr(self,'switching',False):self.stash()

    def apply_joint(self,rows):
        self.stash()
        for i,changes in rows:
            # Invalidate all siblings of a previous joint fit before replacing a subset.
            oldgroups={r['joint_group'] for t in self.tracks if t['id'] in changes and (r:=t['results'][i]) and 'joint_group' in r}
            for t in self.tracks:
                r=t['results'][i]
                if r and r.get('joint_group') in oldgroups:t['results'][i]=None;t['band']=None
                if t['id'] in changes:t['results'][i]=changes[t['id']];t['band']=None
        t=self.tracks[self.active];self.results=t['results'];self.band=t['band'];self.model_curve.setData([],[]);self.band_plot.clear();self.band_text.setText('峰拟合已更新；需要重新拟合对应分支的能带模型。');self.refresh_table();self.show_profile()

    def fit_joint_current(self):
        if self.busy:return
        self.stash();self.apply_joint([(self.index,joint_row(self.data,self.tracks,self.index))])

    def fit_joint_all(self):
        if self.busy:return
        self.stash();self.busy=True;self.set_busy(True);self.actions.setEnabled(False);self.table.setEnabled(False);self.region.setMovable(False);self.peak.setMovable(False);self.progress.show();self.progress.setValue(0)
        self.worker=JointBatch(self.data,self.tracks,self);self.worker.progress.connect(self.progress.setValue);self.worker.ready.connect(self.joint_done);self.worker.start()

    def joint_done(self,rows):
        self.busy=False;self.set_busy(False);self.actions.setEnabled(True);self.table.setEnabled(True);self.region.setMovable(True);self.peak.setMovable(True);self.progress.hide();self.apply_joint(list(enumerate(rows)))

    def refresh_plots(self):
        if not hasattr(self,'tracks'):return super().refresh_plots()
        self.stash();super().refresh_plots()
        active_color=COLORS[self.active%len(COLORS)];self.points.setBrush(active_color);self.dispersion_line.setPen(pg.mkPen(active_color,width=2));self.qerrors.setData(pen=active_color)
        for item in self.width_plot.listDataItems():item.setSymbolBrush(active_color)
        for item in self.overlays:self.map.removeItem(item)
        self.overlays=[]
        for j,t in enumerate(self.tracks):
            if j==self.active:continue
            color=COLORS[j%len(COLORS)];rs=t['results'];positions=[r['center'] if r and 'error' not in r and r['accepted'] else np.nan for r in rs]
            if not np.isfinite(positions).any():continue
            line=self.map.plot(positions,self.data['energy'],pen=pg.mkPen(color,width=1),symbol='o',symbolSize=5,symbolBrush=color,connect='finite');self.overlays.append(line)
            rows=[(e,r) for e,r in zip(self.data['energy'],rs) if r and 'error' not in r and r['accepted']]
            if rows:
                e=np.array([e for e,r in rows]);q=np.array([r['center'] for e,r in rows]);err=pg.ErrorBarItem(x=q,y=e,width=2*np.array([r['center_se'] for e,r in rows]),pen=color);self.map.addItem(err);self.overlays.append(err)
                self.width_plot.plot(e,[r['fwhm'] for e,r in rows],pen=None,symbol='o',symbolBrush=color,symbolSize=4)
                self.width_plot.addItem(pg.ErrorBarItem(x=e,y=np.array([r['fwhm'] for e,r in rows]),height=2*np.array([r['fwhm_se'] for e,r in rows]),pen=color))
        if hasattr(self,'all_table'):self.refresh_current_peaks()

    def refresh_current_peaks(self):
        self.stash();self.all_table.setRowCount(len(self.tracks))
        for j,t in enumerate(self.tracks):
            r=t['results'][self.index] or {};values=[t['name'],f"{self.data['energy'][self.index]:.6g}",*[f'{r[k]:.6g}' if k in r else '' for k in ('center','center_se','fwhm')],'是' if r.get('accepted') else '否',r.get('error','；'.join(r.get('flags',[])) or ('已拟合' if r else '未拟合'))]
            for k,value in enumerate(values):
                item=W.QTableWidgetItem(value);item.setFlags(Qt.ItemIsSelectable|Qt.ItemIsEnabled);item.setForeground(pg.mkColor(COLORS[j%len(COLORS)]));self.all_table.setItem(j,k,item)

    def all_row_click(self,row,col):
        if not self.busy:self.track_box.setCurrentIndex(row)

    def show_profile(self):
        super().show_profile()
        if not hasattr(self,'tracks'):return
        self.stash()
        for item in self.components:self.profile.removeItem(item)
        self.components=[];x=self.data['x'];r=self.results[self.index]
        self.profile.setTitle(f"E={self.data['energy'][self.index]:.6g} meV · 当前 {self.tracks[self.active]['name']} · 独立多分支")
        for j,t in enumerate(self.tracks):
            rr=t['results'][self.index]
            if not rr or 'error' in rr:continue
            use=(x>=rr['low'])&(x<=rr['high']);xx=x[use]
            if r and 'joint_group' in r and rr.get('joint_group')==r['joint_group']:
                # Individual peak + shared background; orange main curve is their total.
                yy=rr['background']+rr['slope']*(xx-rr['reference'])+rr['amplitude']*np.exp(-.5*((xx-rr['center'])/rr['sigma'])**2)
            elif j==self.active:continue
            else:yy=evaluate_result(xx,rr)
            item=self.profile.plot(xx,yy,pen=pg.mkPen(COLORS[j%len(COLORS)],style=Qt.DashLine));self.components.append(item)
        if r and 'joint_parameters' in r:self.message.setText(self.message.text()+'  实线为联合总模型，彩色虚线为各峰+共同背景；完整联合协方差已保存。')

    def fit_band(self):
        try:
            if self.busy:raise ValueError('请等待峰拟合完成。')
            if not self.straight_model or not self.assumption.isChecked():raise ValueError('需确认直线通过对应散射中心及 |Q−Q₀|=2|k| 假设。折线累计距离不能直接转换 k。')
            if self.emin.value()>=self.emax.value():raise ValueError('能区上下限无效。')
            rows=[(e,r) for e,r in zip(self.data['energy'],self.results) if r and 'error' not in r and r['accepted'] and self.emin.value()<=e<=self.emax.value()]
            b=fit_parabolic_scattering([e for e,r in rows],[r['center'] for e,r in rows],[r['center_se'] for e,r in rows], 'electron' if self.carrier.currentIndex()==0 else 'hole',1 if self.branch.currentIndex()==0 else -1,self.center.value())
            b['assumption']='This independent branch: |Q-Q0|=2|k|, line passes through fixed scattering center; NOT general interpocket inversion.'
            b['track_id']=self.tracks[self.active]['id'];b['track_name']=self.tracks[self.active]['name']
            b['scattering_center_q_vector_cycles_nm']=self.position_q(self.center.value())
            self.band=b;self.show_band();self.stash()
        except (ValueError,np.linalg.LinAlgError) as e:self.band=None;self.model_curve.setData([],[]);self.band_plot.clear();self.band_text.setText(str(e));self.stash()

    def show_band(self):
        super().show_band()
        if not hasattr(self,'tracks'):return
        point=self.position_q(self.center.value())
        self.band_text.setText(self.band_text.text()+f"  当前分支中心坐标={self.center.value():.6g}；对应 q 向量={point} cycles/nm。只将相对中心的位移转换为 k。")

    def position_q(self,s):
        vertices=np.asarray(self.data['metadata']['cut'].get('vertices',[]),float)
        if len(vertices)<2:return None
        lengths=np.linalg.norm(np.diff(vertices,axis=0),axis=1);distance=np.r_[0,np.cumsum(lengths)]
        ss=float(s)-float(self.data['x'][0]);j=min(len(lengths)-1,max(0,int(np.searchsorted(distance,ss,side='right')-1)))
        return (vertices[j]+(ss-distance[j])/lengths[j]*(vertices[j+1]-vertices[j])).tolist()

    def export_to(self,root):
        self.stash();root=super().export_to(root)
        state=json.loads((root/'analysis.qpi.json').read_text(encoding='utf-8'));state.update(schema=2,tracks=self.tracks,active_track=self.active,global_range=self.global_range.isChecked())
        state['implementation_sha256']['qpi_multi.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        (root/'analysis.qpi.json').write_text(json.dumps(state,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        with (root/'peaks.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f);w.writerow(['track_id','track_name','E_meV','accepted','q_or_s_cycles_nm','q_se_1sigma','FWHM','FWHM_se_1sigma','q1_cycles_nm','q2_cycles_nm','seed','window_low','window_high','joint_group','status'])
            for t in self.tracks:
                for e,c,r in zip(self.data['energy'],t['configs'],t['results']):
                    r=r or {};q=self.position_q(r['center']) if 'center' in r else None
                    w.writerow([t['id'],t['name'],e,r.get('accepted',False),*[r.get(k,'') for k in ('center','center_se','fwhm','fwhm_se')],*(q or ['','']),c['seed'],r.get('low',c['low']),r.get('high',c['high']),r.get('joint_group',''),r.get('error',';'.join(r.get('flags',[])))])
        with (root/'profiles_and_residuals.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f);w.writerow(['track_id','E_meV','q_or_s','observed','full_model_in_window','residual','joint_group'])
            for t in self.tracks:
                for e,y,r in zip(self.data['energy'],self.data['values'],t['results']):
                    if not r or 'error' in r:continue
                    for q,value in zip(self.data['x'],y):
                        if not r['low']<=q<=r['high']:continue
                        model=float(evaluate_result(q,r));w.writerow([t['id'],e,q,value,model,value-model,r.get('joint_group','')])
        with (root/'model_inferred_band.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f);w.writerow(['track_id','track_name','fixed_q0_or_s0','E_meV','q_observed','q_model','q_residual','k_relative_nm_inverse_ASSUMED_DIAMETER','k_se'])
            for t in self.tracks:
                b=t['band']
                if b:
                    for row in zip(*[b[k] for k in ('energy_meV','q_observed','q_fit','q_residual','k_relative_nm_inverse','k_se_nm_inverse')]):w.writerow([t['id'],t['name'],b['q_center_cycles_nm'],*row])
        return root

    def figure(self,path):
        if not hasattr(self,'tracks'):return super().figure(path)
        self.stash()
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig=Figure(figsize=(12,8),dpi=160,layout='constrained');FigureCanvasAgg(fig);a,b,c,d=fig.subplots(2,2).ravel()
        a.pcolormesh(self.data['x'],self.data['energy'],self.data['values'],shading='nearest',cmap='viridis')
        c.plot(self.data['x'],self.data['values'][self.index],color='gray',lw=1,label='data')
        for j,t in enumerate(self.tracks):
            color=COLORS[j%len(COLORS)];rows=[(e,r) for e,r in zip(self.data['energy'],t['results']) if r and 'error' not in r and r['accepted']]
            if rows:
                ee=[e for e,r in rows];a.errorbar([r['center'] for e,r in rows],ee,xerr=[r['center_se'] for e,r in rows],fmt='.',color=color,label=t['name']);b.errorbar(ee,[r['fwhm'] for e,r in rows],yerr=[r['fwhm_se'] for e,r in rows],fmt='.',color=color,label=t['name'])
            r=t['results'][self.index]
            if r and 'error' not in r:
                keep=(self.data['x']>=r['low'])&(self.data['x']<=r['high']);x=self.data['x'][keep];yy=evaluate_result(x,r);c.plot(x,yy,color=color,label=t['name']+' full model');d.plot(x,self.data['values'][self.index][keep]-yy,color=color)
            if t['band']:a.plot(t['band']['q_fit'],t['band']['energy_meV'],color=color)
        a.set(title='Independent QPI branches (not direct bands)',xlabel='q / s (cycles/nm)',ylabel='E (meV)');b.set(title='Widths and conditional 1 sigma',xlabel='E (meV)',ylabel='FWHM (cycles/nm)');c.set(title=f"Profiles E={self.data['energy'][self.index]:g} meV",xlabel='q / s (cycles/nm)',ylabel='linear |FFT|');d.set(title='Full-model residuals',xlabel='q / s (cycles/nm)',ylabel='data - model')
        for ax in (a,b,c):ax.legend(fontsize=8)
        fig.savefig(path)

    @classmethod
    def load_analysis(cls,path,parent=None):
        dlg=super().load_analysis(path,parent);s=json.loads(Path(path).read_text(encoding='utf-8'))
        if s.get('schema',1)==1:
            dlg.stash();dlg.rebuild_track_box();dlg.select_energy(0);dlg.refresh_table();return dlg
        if s.get('schema')!=2:raise ValueError('不支持的 QPI 分析版本。')
        dlg.tracks=s['tracks'];dlg.active=s.get('active_track',0);dlg.global_range.setChecked(s.get('global_range',True))
        t=dlg.tracks[dlg.active];dlg.configs=t['configs'];dlg.results=t['results'];dlg.band=t['band'];dlg.restore_controls(t['controls'])
        dlg.rebuild_track_box();dlg.switch_track(dlg.active);return dlg
