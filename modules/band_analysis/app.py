"""Native Band Analysis window. Numerical FFT core is imported unchanged."""
from __future__ import annotations
import os, sys, json, argparse, traceback
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'vendor'))
os.environ.setdefault('PYQTGRAPH_QT_LIB','PySide6')
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets as W
from PySide6.QtCore import Qt, Signal
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from OpenGL import GL
from matplotlib import colormaps
from interaction import EditorMixin, corner_angles
from volume import (Volume, Cut, Contrast, groups, load_manifest, load_3ds,
                    load_volume, save_volume, export_cut, render_cut_png)
pg.setConfigOptions(antialias=False,background='#ffffff',foreground='#384854',imageAxisOrder='row-major')

def edges(a):
    a=np.asarray(a);return np.r_[a[0]-(a[1]-a[0])/2,(a[:-1]+a[1:])/2,a[-1]+(a[-1]-a[-2])/2]

def spin(value=0,lo=-1e6,hi=1e6,step=.1,dec=4):
    b=W.QDoubleSpinBox();b.setRange(lo,hi);b.setDecimals(dec);b.setSingleStep(step);b.setValue(value);b.setKeyboardTracking(False);return b

def combo(items):
    c=W.QComboBox()
    for label,value in items:c.addItem(label,value)
    return c

def button(text,fn):
    b=W.QPushButton(text);b.clicked.connect(fn);return b

def box(title):
    b=W.QGroupBox(title);l=W.QFormLayout(b);l.setFieldGrowthPolicy(W.QFormLayout.AllNonFixedFieldsGrow);return b,l

class Loader(QtCore.QThread):
    ready=Signal(object);failed=Signal(str);progress=Signal(int)
    def __init__(self,source,pre,window):super().__init__();self.source=source;self.pre=pre;self.window=window
    def run(self):
        try:
            s=self.source
            if s['type']=='3ds':v=load_3ds(s['path'],self.pre,self.window,self.progress.emit)
            elif s['type']=='manifest':v=load_manifest(s['path'],s['group'],self.pre,self.window,self.progress.emit)
            else:v=load_volume(s['path'])
            self.ready.emit(v)
        except Exception:self.failed.emit(traceback.format_exc())

class SlicePlot(pg.PlotWidget):
    def __init__(self,aspect=False):
        super().__init__();self.setMinimumSize(280,200);self.showGrid(x=True,y=True,alpha=.12)
        self.mesh=pg.PColorMeshItem(levels=(0,1),enableAutoLevels=False);self.addItem(self.mesh)
        self.image=pg.ImageItem(axisOrder='row-major');self.addItem(self.image);self._lut_key=None
        self.setMenuEnabled(False)
        if aspect:self.setAspectLocked(True,ratio=1)
    def clear_cut(self):self.mesh.hide();self.image.hide()
    def show_cut(self,values,x,y,xlabel,ylabel,contrast,reset=False):
        regular=all(np.allclose(np.diff(a),np.diff(a)[0],rtol=1e-10,atol=1e-12) for a in (x,y))
        if regular:
            self.mesh.hide();self.image.show();self.image.setImage(contrast.rgba(values),autoLevels=False)
            ex,ey=edges(x),edges(y);self.image.setRect(QtCore.QRectF(ex[0],ey[0],ex[-1]-ex[0],ey[-1]-ey[0]))
        else:
            self.image.hide();self.mesh.show()
            if self._lut_key!=contrast.cmap:
                lut=colormaps[contrast.cmap](np.linspace(0,1,256),bytes=True)
                self.mesh.setColorMap(pg.ColorMap(np.linspace(0,1,256),lut));self._lut_key=contrast.cmap
            xx,yy=np.meshgrid(edges(x),edges(y));self.mesh.setData(xx,yy,contrast.normalized(values),autoLevels=False)
        self.setLabel('bottom',xlabel);self.setLabel('left',ylabel)
        if reset:self.autoRange()
    def wheelEvent(self,event):
        if event.modifiers()&Qt.ControlModifier:
            q=self.plotItem.vb.mapSceneToView(self.mapToScene(event.position().toPoint()))
            f=1.0015**(-event.angleDelta().y());self.plotItem.vb.scaleBy((f,f),center=q);event.accept()
        else:event.ignore()

class EditorPlot(EditorMixin,SlicePlot):
    def __init__(self,owner):super().__init__(True);self.setup_editor(owner)

class StackView(gl.GLViewWidget):
    def __init__(self):
        super().__init__();self.setBackgroundColor('#17242f');self.planes=[];self.lines=[];self.tex=[]
        self.v=None;self.setMinimumSize(350,280)
    def cameraPosition(self):return super().cameraPosition()
    def mouseMoveEvent(self,event):super().mouseMoveEvent(event);self.sort_planes()
    def sort_planes(self):
        sign=1 if np.sin(np.deg2rad(self.opts['elevation']))>=0 else -1
        for i,p in enumerate(self.planes):p.setDepthValue(sign*i)
    def zvalue(self,e):return (np.asarray(e)-(self.v.energy[0]+self.v.energy[-1])/2)*self.zscale
    def reset_camera(self):
        if self.v is None:return
        extent=max(np.ptp(self.v.qx),np.ptp(self.v.qy))
        self.setCameraPosition(pos=QtGui.QVector3D(0,0,0),distance=extent*2.1,elevation=22,azimuth=-58)
        self.sort_planes()
    def rebuild(self,v,contrast,metric,opacity,stride,stretch,reset=False):
        self.v=v;extent=max(np.ptp(v.qx),np.ptp(v.qy));self.zscale=extent*stretch/np.ptp(v.energy)
        # GLImageItem does not free its texture on removal; release explicitly.
        self.makeCurrent()
        try:
            for item in self.planes:
                if item.texture is not None:GL.glDeleteTextures([item.texture]);item.texture=None
                if item.m_vbo_position.isCreated():item.m_vbo_position.destroy()
        finally:self.doneCurrent()
        for item in self.planes+self.lines+self.tex:self.removeItem(item)
        self.planes=[];self.lines=[];self.tex=[]
        dq1=np.diff(v.qx)[0];dq2=np.diff(v.qy)[0];data=v.scalar(metric)
        indices=list(range(0,len(v.energy),stride))
        if indices[-1]!=len(v.energy)-1:indices.append(len(v.energy)-1)
        for i in indices:
            rgba=contrast.rgba(data[i]);rgba[...,3]=int(opacity*255)
            item=gl.GLImageItem(rgba.transpose(1,0,2),smooth=False)
            item.setGLOptions({GL.GL_DEPTH_TEST:False,GL.GL_BLEND:True,'glBlendFunc':(GL.GL_SRC_ALPHA,GL.GL_ONE_MINUS_SRC_ALPHA)})
            item.scale(dq1,dq2,1);item.translate(v.qx[0]-dq1/2,v.qy[0]-dq2/2,float(self.zvalue(v.energy[i])))
            self.addItem(item);self.planes.append(item)
        x0,x1=v.qx[[0,-1]];y0,y1=v.qy[[0,-1]];z0,z1=self.zvalue(v.energy[[0,-1]])
        for z in [z0,z1]:self.line([[x0,y0,z],[x1,y0,z],[x1,y1,z],[x0,y1,z],[x0,y0,z]],'#627d8f',1)
        for x,y in [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]:self.line([[x,y,z0],[x,y,z1]],'#627d8f',1)
        labels=[((x1,y0,z0),'q (cycles/nm)'),((x0,y1,z0),'q (cycles/nm)'),((x0,y0,z1),'Bias (mV)')]
        labels += [((x0,y0,float(self.zvalue(e))),f'{0 if abs(e)<1e-8 else e:g}') for e in np.linspace(v.energy[0],v.energy[-1],5)]
        for pos,text in labels:
            t=gl.GLTextItem(pos=pos,text=text,color='#d7e6ef',font=QtGui.QFont('Segoe UI',10));self.addItem(t);self.tex.append(t)
        self.base_lines=len(self.lines);self.sort_planes()
        if reset:self.reset_camera()
    def line(self,points,color='#ffb467',width=2):
        item=gl.GLLinePlotItem(pos=np.asarray(points,float),color=pg.mkColor(color).getRgbF(),width=width,antialias=True)
        item.setDepthValue(10000);self.addItem(item);self.lines.append(item)
    def mark(self,ei,cut):
        if self.v is None:return
        for item in self.lines[self.base_lines:]:self.removeItem(item)
        self.lines=self.lines[:self.base_lines];v=self.v;z=float(self.zvalue(v.energy[ei]));x0,x1=v.qx[[0,-1]];y0,y1=v.qy[[0,-1]]
        self.line([[x0,y0,z],[x1,y0,z],[x1,y1,z],[x0,y1,z],[x0,y0,z]],'#56d5e4',2);self.energy_marker=self.lines[-1]
        if cut is None:return
        if cut.params['mode']=='tilted':
            for points in [(v.qx,np.full_like(v.qx,y0)),(v.qx,np.full_like(v.qx,y1)),(np.full_like(v.qy,x0),v.qy),(np.full_like(v.qy,x1),v.qy)]:
                xx,yy=points;ee=cut.params['E0_mV']+cut.params['dE_dqx']*xx+cut.params['dE_dqy']*yy
                valid=(ee>=v.energy[0])&(ee<=v.energy[-1]);xyz=np.c_[xx,yy,self.zvalue(ee)];xyz[~valid]=np.nan;self.line(xyz)
        elif 'path_qx' in cut.extra:
            xx,yy=cut.extra['path_qx'],cut.extra['path_qy'];z0,z1=self.zvalue(v.energy[[0,-1]])
            for zz in (z0,z1):self.line(np.c_[xx,yy,np.full_like(xx,zz)])
            for idx in [0,-1]:self.line([[xx[idx],yy[idx],z0],[xx[idx],yy[idx],z1]])

    def mark_energy(self,ei):
        if self.v is None or not hasattr(self,'energy_marker'):return
        v=self.v;z=float(self.zvalue(v.energy[ei]));x0,x1=v.qx[[0,-1]];y0,y1=v.qy[[0,-1]]
        self.energy_marker.setData(pos=np.array([[x0,y0,z],[x1,y0,z],[x1,y1,z],[x0,y1,z],[x0,y0,z]]))

class MainWindow(W.QMainWindow):
    def __init__(self,args):
        super().__init__();self.setWindowTitle('Band Analysis');self.resize(1580,980)
        self.args=args;self.v=None;self.source=None;self.cut=None;self.loading=False;self.loader=None;self.vertices=[];self.contrast=Contrast();self.reset_views=True
        self.settings=QtCore.QSettings('ResearchTools','BandAnalysis');self.reset_section=True;self.selected=None;self.line_tip=None
        self.frame_timer=QtCore.QTimer(self);self.frame_timer.setSingleShot(True);self.frame_timer.setInterval(16);self.frame_timer.timeout.connect(self.update_energy_frame)
        self.export_root=Path(args.output or str(Path.home()/'Documents/Codex/Band Analysis exports'))
        self.timer=QtCore.QTimer(self);self.timer.setSingleShot(True);self.timer.setInterval(32);self.timer.timeout.connect(self.refresh)
        shell=W.QWidget();self.setCentralWidget(shell);outer=W.QVBoxLayout(shell);outer.setContentsMargins(14,10,14,8)
        bar=W.QHBoxLayout();title=W.QLabel('Band Analysis');title.setStyleSheet('font-size:24px;font-weight:600');bar.addWidget(title)
        bar.addWidget(W.QLabel('FFT volume · q–E sections'));bar.addStretch()
        self.toolbar_buttons=[]
        for text,fn in [('打开 3DS',self.open3ds),('SXM 分组索引',self.open_manifest),('打开数据体',self.open_volume),('打开会话',self.open_project),('保存会话',self.save_project),('导出当前截面',self.export),('逐能量 QPI 拟合',self.open_qpi),('重开 QPI 分析',self.open_qpi_saved)]:
            b=button(text,fn);bar.addWidget(b);self.toolbar_buttons.append(b)
        outer.addLayout(bar);split=W.QSplitter();outer.addWidget(split,1)
        scroll=W.QScrollArea();scroll.setWidgetResizable(True);scroll.setMinimumWidth(300);scroll.setMaximumWidth(370);controls=W.QWidget();self.ctrl=W.QVBoxLayout(controls);self.ctrl.setContentsMargins(4,4,10,4);scroll.setWidget(controls);split.addWidget(scroll)
        data,form=box('01  数据与 FFT');self.ctrl.addWidget(data)
        self.info=W.QLabel('打开一个 3DS，或加载同一 3DS 导出的 SXM 组。');self.info.setWordWrap(True);self.info.setTextInteractionFlags(Qt.TextSelectableByMouse);form.addRow(self.info)
        self.group=W.QComboBox();self.group.setToolTip('每次只堆叠同一个 3DS 的 LI X 能量图，不混入 Z。');form.addRow('3DS 组',self.group);self.group.currentIndexChanged.connect(self.change_group)
        self.pre=combo([('去平面','plane'),('去均值','mean'),('逐行中值','row_median'),('逐行线性','row_linear'),('无预处理','none')]);form.addRow('预处理',self.pre)
        self.win=combo([('Hann','hann'),('Tukey','tukey'),('无窗','none')]);form.addRow('窗函数',self.win)
        form.addRow(button('应用 FFT 设置',self.recompute));self.metric_box=combo([('振幅 |FFT| (pA)','amplitude'),('强度 |FFT|² (pA²)','power')]);form.addRow('显示量',self.metric_box);self.metric_box.currentIndexChanged.connect(self.metric_change)
        self.progress=W.QProgressBar();self.progress.setVisible(False);form.addRow(self.progress)
        energy,form=box('02  等能量图');self.ctrl.addWidget(energy)
        self.energy=W.QSlider(Qt.Horizontal);self.energy.setRange(0,0);self.energy.valueChanged.connect(self.energy_change);form.addRow(self.energy)
        self.energy_list=W.QComboBox();self.energy_list.currentIndexChanged.connect(lambda i:self.energy.setValue(max(0,i)));form.addRow('实测偏压',self.energy_list)
        form.addRow(W.QLabel('拖动偏压滑块可连续浏览。'))
        contrast,form=box('03  所有视图共用衬度');self.ctrl.addWidget(contrast)
        self.log=W.QCheckBox('log₁₀ 显示');self.log.setChecked(True);self.log.toggled.connect(self.fit_contrast);form.addRow(self.log)
        self.cmap=combo([('反灰度','gray_r'),('灰度','gray'),('Magma','magma'),('Viridis','viridis'),('Inferno','inferno')]);self.cmap.currentIndexChanged.connect(self.contrast_change);form.addRow('色表',self.cmap)
        self.low=spin(0,-1e15,1e15,.1,7);self.high=spin(1,-1e15,1e15,.1,7);self.gamma=spin(1,.05,10,.1,2)
        for label,w in [('下限',self.low),('上限',self.high),('Gamma',self.gamma)]:form.addRow(label,w);w.valueChanged.connect(self.contrast_change)
        self.p_low=spin(1,0,99,.5,2);self.p_high=spin(99.5,1,100,.1,2)
        row=W.QHBoxLayout();row.addWidget(self.p_low);row.addWidget(W.QLabel('至'));row.addWidget(self.p_high);form.addRow('全体百分位',row)
        form.addRow(button('按整个数据体设定色标',self.fit_contrast))
        self.legend=W.QLabel();self.legend.setMinimumHeight(20);form.addRow(self.legend)
        self.legend_text=W.QLabel();self.legend_text.setWordWrap(True);form.addRow(self.legend_text)
        note=W.QLabel('切换偏压不会重设色标。上下限以当前线性 / log₁₀ 单位输入。');note.setWordWrap(True);form.addRow(note)
        cutbox,form=box('04  截面');self.ctrl.addWidget(cutbox)
        self.mode=combo([('等能面','energy'),('直线（过散射原点 q=0）','line'),('任意折线','path')]);form.addRow('类型',self.mode);self.mode.currentIndexChanged.connect(self.mode_change)
        self.cut_widgets={}
        def field(name,label,widget,modes):
            form.addRow(label,widget);self.cut_widgets[name]=(widget,form.labelForField(widget),modes);return widget
        self.angle=field('angle','角度 (°)',spin(0,-360,360,1,4),['line'])
        self.length=field('length','总长度 (cycles/nm)',spin(5,.000001,10000,.1,6),['line'])
        self.width=field('width','全宽 Δq (cycles/nm)',spin(0,0,100,.01),['line','path'])
        self.interp=combo([('线性（数值插值）','linear'),('最近邻','nearest')]);form.addRow('截面采样',self.interp)
        for w in (self.angle,self.length):w.valueChanged.connect(self.line_settings_changed)
        self.width.valueChanged.connect(self.geometry_changed)
        self.interp.currentIndexChanged.connect(self.queue)
        self.path_table=W.QTableWidget(0,2);self.path_table.setHorizontalHeaderLabels(['屏幕水平 q','屏幕竖直 q']);self.path_table.horizontalHeader().setSectionResizeMode(W.QHeaderView.Stretch);self.path_table.setMaximumHeight(165);self.path_table.cellChanged.connect(self.table_changed);form.addRow(self.path_table)
        pathrow=W.QHBoxLayout();pathrow.addWidget(button('添加点',self.add_point));pathrow.addWidget(button('撤销点',self.undo_point));pathrow.addWidget(button('清空',self.clear_path));form.addRow(pathrow)
        self.path_note=W.QLabel('折线：点击添加 / 选中点，拖动移动，右键删除；方向键移动 1 个 FFT 像素。两点折线可点击中部选中整条线。角度相对屏幕水平，晶向需自行标定。Ctrl + 滚轮等比例缩放。');self.path_note.setWordWrap(True);form.addRow(self.path_note)
        stack,form=box('05  三维堆叠显示');self.ctrl.addWidget(stack)
        self.opacity=spin(.16,.01,1,.02,2);self.stride=W.QSpinBox();self.stride.setRange(1,100);self.stride.setValue(1);self.stretch=spin(1.1,.1,8,.1,1)
        for label,w in [('每层不透明度',self.opacity),('每隔 N 张显示',self.stride),('能量轴拉伸',self.stretch)]:form.addRow(label,w);w.valueChanged.connect(self.queue_stack)
        form.addRow(button('重置三维视角',lambda:self.stack.reset_camera()));form.addRow(button('重置二维范围',self.reset_ranges));form.addRow(button('保存 FFT 数据体',self.export_volume))
        self.ctrl.addStretch()
        content=W.QWidget();cl=W.QVBoxLayout(content);cl.setContentsMargins(6,0,0,0);split.addWidget(content);split.setSizes([320,1200])
        top=W.QSplitter();cl.addWidget(top,3)
        left=W.QWidget();ll=W.QVBoxLayout(left);ll.setContentsMargins(0,0,0,0);ll.addWidget(W.QLabel('FFT STACK   ·   拖动旋转 / 滚轮缩放 / 中键平移'))
        self.stack=StackView();ll.addWidget(self.stack,1);top.addWidget(left)
        right=W.QWidget();rl=W.QVBoxLayout(right);rl.setContentsMargins(0,0,0,0);self.iso_title=W.QLabel('CONSTANT ENERGY');isorow=W.QHBoxLayout();isorow.addWidget(self.iso_title);isorow.addStretch();self.gamma_button=button('添加散射原点 q=0',self.add_gamma);isorow.addWidget(self.gamma_button);rl.addLayout(isorow)
        self.iso=EditorPlot(self);rl.addWidget(self.iso,1);top.addWidget(right);top.setSizes([600,520])
        self.overlay=pg.PlotDataItem(pen=pg.mkPen('#ed8b23',width=2),symbol='o',symbolBrush='#ed8b23',symbolSize=7);self.overlay.setZValue(50);self.iso.addItem(self.overlay);self.point_labels=[]
        self.section_title=W.QLabel('SECTION');cl.addWidget(self.section_title);self.section=SlicePlot();cl.addWidget(self.section,3);self.corner_lines=[]
        self.detail=W.QLabel('');self.detail.setWordWrap(True);cl.addWidget(self.detail)
        self.statusBar().showMessage('就绪');self.mode_change()
        self.stack_dirty=True
        if args.manifest:
            self.set_manifest(args.manifest,args.group)
        elif args.file: self.load_file(args.file)
        elif not getattr(args,'no_autoload',False):
            last=self.settings.value('last_source','')
            if last:
                try:self.load_source(json.loads(last))
                except Exception:pass

    def open_qpi(self):
        if self.v is None:return
        try:
            from qpi_dialog import QPIDialog, snapshot
            cut=self.compute_cut()
            dialog=QPIDialog(snapshot(cut,self.v,self.contrast),self)
            if not hasattr(self,'qpi_windows'):self.qpi_windows=[]
            self.qpi_windows.append(dialog);dialog.show()
        except Exception as e:self.error(e)
    def open_qpi_saved(self):
        p,_=W.QFileDialog.getOpenFileName(self,'重开 QPI 分析','','QPI analysis (*.qpi.json)')
        if p:
            try:
                from qpi_dialog import QPIDialog
                dialog=QPIDialog.load_analysis(p,self)
                if not hasattr(self,'qpi_windows'):self.qpi_windows=[]
                self.qpi_windows.append(dialog);dialog.show()
            except Exception as e:self.error(e)

    def error(self,text):
        self.statusBar().showMessage(str(text).splitlines()[-1]);W.QMessageBox.warning(self,'Band Analysis',str(text))
    def queue(self,*a):
        if not self.timer.isActive():self.timer.start()
    def queue_stack(self,*a):self.stack_dirty=True;self.queue()
    def set_manifest(self,path,group=None):
        try:
            names=groups(path);self.group.blockSignals(True);self.group.clear();self.group.addItems(names)
            if group in names:self.group.setCurrentText(group)
            self.group.blockSignals(False);self.manifest=str(Path(path).resolve());self.load_source({'type':'manifest','path':self.manifest,'group':self.group.currentText()})
        except Exception as e:self.error(e)
    def change_group(self):
        if hasattr(self,'manifest') and self.group.currentText():self.load_source({'type':'manifest','path':self.manifest,'group':self.group.currentText()})
    def open_manifest(self):
        p,_=W.QFileDialog.getOpenFileName(self,'选择含 3DS 分组与偏压的索引','','Index (index_and_bias.csv);;CSV (*.csv)')
        if p:self.set_manifest(p)
    def open3ds(self):
        p,_=W.QFileDialog.getOpenFileName(self,'打开 Nanonis 网格光谱','','Nanonis 3DS (*.3ds)')
        if p:self.load_file(p)
    def open_volume(self):
        p,_=W.QFileDialog.getOpenFileName(self,'打开保存的数据体','','FFT Volume (*.npz)')
        if p:self.load_file(p)
    def load_file(self,p):
        self.group.blockSignals(True);self.group.clear();self.group.blockSignals(False)
        self.load_source({'type':'3ds' if str(p).lower().endswith('.3ds') else 'volume','path':str(Path(p).resolve())})
    def load_source(self,s):
        if self.loading:return
        if not Path(s['path']).is_file():raise ValueError('找不到数据：'+s['path'])
        self.pending_source=dict(s);self.loading=True;self.progress.setVisible(True);self.progress.setValue(0);self.statusBar().showMessage('读取数据，使用原 FFT 核心计算…')
        for b in self.toolbar_buttons:b.setEnabled(False)
        self.group.setEnabled(False);self.loader=Loader(s,self.pre.currentData(),self.win.currentData());self.loader.ready.connect(self.loaded);self.loader.failed.connect(self.load_failed);self.loader.progress.connect(self.progress.setValue);self.loader.start()
    def load_failed(self,text):
        self.loading=False;self.group.setEnabled(True);self.progress.setVisible(False)
        if self.v is not None:
            self.pre.setCurrentIndex(self.pre.findData(self.v.meta.get('preprocess','plane')))
            self.win.setCurrentIndex(self.win.findData(self.v.meta.get('window','hann')))
            if self.source and self.source['type']=='manifest':
                self.group.blockSignals(True);self.group.setCurrentText(self.source['group']);self.group.blockSignals(False)
        if hasattr(self,'pending_project'):del self.pending_project
        for b in self.toolbar_buttons:b.setEnabled(True)
        self.error(text)
    def loaded(self,v):
        self.source=self.pending_source;self.v=v;self.loading=False;self.group.setEnabled(True);self.progress.setVisible(False);self.settings.setValue('last_source',json.dumps(self.source))
        for b in self.toolbar_buttons:b.setEnabled(True)
        self.energy.blockSignals(True);self.energy.setRange(0,len(v.energy)-1);self.energy.setValue(int(np.argmin(abs(v.energy))));self.energy.blockSignals(False)
        self.energy_list.blockSignals(True);self.energy_list.clear();self.energy_list.addItems([f'{0 if abs(e)<1e-8 else e:.6g} mV  ·  {i+1}/{len(v.energy)}' for i,e in enumerate(v.energy)]);self.energy_list.setCurrentIndex(self.energy.value());self.energy_list.blockSignals(False)
        self.info.setText(f"{v.meta['title']}\n{len(v.energy)} 张 FFT · {len(v.qx)} × {len(v.qy)}\n{v.energy[0]:g} 至 {v.energy[-1]:g} mV\nFFT 像素间距：{np.diff(v.qx)[0]:.5g} / {np.diff(v.qy)[0]:.5g} cycles/nm")
        self.pre.setEnabled(self.source['type']!='volume');self.win.setEnabled(self.source['type']!='volume')
        self.pre.setCurrentIndex(max(0,self.pre.findData(v.meta.get('preprocess','plane'))));self.win.setCurrentIndex(max(0,self.win.findData(v.meta.get('window','hann'))))
        self.reset_views=True;self.stack_dirty=True;self.fit_contrast()
        if hasattr(self,'pending_project'):
            p=self.pending_project;del self.pending_project;self.restore(p)
        self.statusBar().showMessage('已载入。所有偏压与截面共用色标。')
    def recompute(self):
        if self.source:
            if self.source['type']=='volume':self.error('已保存的数据体只含 FFT。更改预处理需要重新打开原始 3DS 或 SXM 索引。');return
            self.load_source(self.source)
    def metric_change(self):
        if self.v:self.fit_contrast()
    def energy_change(self,i):
        self.energy_list.blockSignals(True);self.energy_list.setCurrentIndex(i);self.energy_list.blockSignals(False)
        if not self.frame_timer.isActive():self.frame_timer.start()
    def fit_contrast(self,*a):
        if self.v is None:return
        if self.p_high.value()<=self.p_low.value():self.error('百分位上限必须大于下限。');return
        self.contrast.logarithmic=self.log.isChecked();self.contrast.fit(self.v.scalar(self.metric_box.currentData()),self.p_low.value(),self.p_high.value())
        for w,value in [(self.low,self.contrast.low),(self.high,self.contrast.high)]:w.blockSignals(True);w.setValue(value);w.blockSignals(False)
        self.contrast_change()
    def contrast_change(self,*a):
        if self.v is None:return
        self.contrast.low=self.low.value();self.contrast.high=self.high.value();self.contrast.gamma=self.gamma.value();self.contrast.cmap=self.cmap.currentData()
        if self.contrast.high<=self.contrast.low:self.statusBar().showMessage('色标上限必须大于下限。');return
        self.contrast.logarithmic=self.log.isChecked();self.stack_dirty=True;self.queue()
    def mode_change(self,*a):
        mode=self.mode.currentData()
        for w,label,modes in self.cut_widgets.values():w.setVisible(mode in modes);label.setVisible(mode in modes)
        self.selected=None;self.gamma_button.setEnabled(mode!='line');self.path_table.setEnabled(mode=='path')
        self.section.setAspectLocked(mode=='energy',ratio=1)
        self.reset_section=True;self.queue()
    def geometry_changed(self,*a):self.reset_section=True;self.update_overlay();self.queue()
    def line_settings_changed(self,*a):self.line_tip=None;self.geometry_changed()
    def editor_points(self):return self.line_points() if self.mode.currentData()=='line' and self.v is not None else self.vertices
    def pixel_step(self):return np.array([np.diff(self.v.qx)[0],np.diff(self.v.qy)[0]])
    def in_bounds(self,p):
        p=np.asarray(p);return bool(np.all(p>=[self.v.qx[0],self.v.qy[0]]) and np.all(p<=[self.v.qx[-1],self.v.qy[-1]]))
    def snap(self,q):
        return np.array([a[int(np.argmin(abs(a-q[i])))] for i,a in enumerate((self.v.qx,self.v.qy))])
    def line_points(self):
        t=np.array([np.cos(np.deg2rad(self.angle.value())),np.sin(np.deg2rad(self.angle.value()))])
        half=self.length.value()/2
        if self.line_tip is not None:
            half=np.linalg.norm(self.line_tip);t=np.asarray(self.line_tip)/half
        bounds=np.minimum(-np.array([self.v.qx[0],self.v.qy[0]]),[self.v.qx[-1],self.v.qy[-1]])
        maximum=min(bounds[k]/abs(t[k]) for k in range(2) if abs(t[k])>1e-12)
        half=min(half,maximum)
        if self.length.value()>2*maximum:
            self.length.blockSignals(True);self.length.setValue(2*maximum);self.length.blockSignals(False)
        if self.line_tip is not None:self.line_tip=(half*t).tolist()
        return np.array([-half*t,half*t]).tolist()
    def set_line_endpoint(self,q,index=1):
        q=np.asarray(q,float)*(1 if index==1 else -1)
        if np.linalg.norm(q)<1e-10:return
        for w,value in [(self.angle,np.degrees(np.arctan2(q[1],q[0]))),(self.length,2*np.linalg.norm(q))]:
            w.blockSignals(True);w.setValue(float(value));w.blockSignals(False)
        self.line_tip=q.tolist()
        self.geometry_changed()
    def select_target(self,target):
        self.selected=target;self.update_overlay()
        self.statusBar().showMessage('已选中整条线；方向键整体移动 1 个 FFT 像素。' if target=='line' else f'已选中 P{target}；方向键移动 1 个 FFT 像素。')
    def nudge(self,dx,dy):
        if self.v is None or self.selected is None:return
        points=np.asarray(self.editor_points(),float);delta=self.pixel_step()*[dx,dy]
        if self.mode.currentData()=='line':
            index=self.selected if isinstance(self.selected,int) else 1
            self.set_line_endpoint(points[index]+delta,index);return
        if self.mode.currentData()!='path':return
        if self.selected=='line' and len(points)==2:points+=delta
        elif isinstance(self.selected,int) and self.selected<len(points):points[self.selected]+=delta
        else:return
        if self.in_bounds(points):self.vertices=points.tolist();self.path_changed()
        else:self.statusBar().showMessage('已到图像边界，保持原位。')
    def delete_point(self,index):
        if 0<=index<len(self.vertices):self.vertices.pop(index);self.selected=None;self.path_changed()
    def add_gamma(self):
        if self.mode.currentData()!='path':self.mode.setCurrentIndex(self.mode.findData('path'))
        if not self.vertices or not np.allclose(self.vertices[-1],[0,0],atol=1e-12):self.vertices.append([0.,0.])
        self.selected=len(self.vertices)-1;self.path_changed();self.iso.setFocus()
    def add_point(self):self.add_gamma()
    def undo_point(self):
        if self.vertices:self.delete_point(len(self.vertices)-1)
    def clear_path(self):self.vertices=[];self.selected=None;self.path_changed()
    def path_changed(self):self.sync_table();self.update_overlay();self.queue()
    def sync_table(self):
        self.path_table.blockSignals(True);self.path_table.setRowCount(len(self.vertices))
        for i,p in enumerate(self.vertices):
            for j,value in enumerate(p):self.path_table.setItem(i,j,W.QTableWidgetItem(f'{value:.9g}'))
        self.path_table.blockSignals(False);self.reset_section=True
    def table_changed(self,*a):
        try:
            points=[[float(self.path_table.item(i,j).text()) for j in range(2)] for i in range(self.path_table.rowCount())]
            if not np.all(np.isfinite(points)):raise ValueError()
            self.vertices=points;self.reset_section=True;self.update_overlay();self.queue()
        except (ValueError,AttributeError):self.statusBar().showMessage('路径坐标需要是有效数值。')
    def update_overlay(self):
        if not hasattr(self,'overlay'):return
        points=self.editor_points() if self.mode.currentData()!='energy' else []
        if len(points):
            p=np.asarray(points);brushes=['#27b8dd' if self.selected==i or self.selected=='line' else '#ed8b23' for i in range(len(points))]
            self.overlay.setData(p[:,0],p[:,1],symbolBrush=brushes,symbolSize=[11 if self.selected==i else 8 for i in range(len(p))])
        else:self.overlay.setData([],[])
        for label in self.point_labels:self.iso.removeItem(label)
        self.point_labels=[];angles=corner_angles(points)
        for i,p in enumerate(points):
            name='q=0' if np.linalg.norm(p)<1e-10 else f'P{i}'
            if i in angles:name+=f'  ∠{angles[i]:.1f}°'
            label=pg.TextItem(name,color='#a95a00',anchor=(0,1),fill=pg.mkBrush(255,255,255,210));label.setPos(*p);label.setZValue(51);self.iso.addItem(label);self.point_labels.append(label)
        if self.mode.currentData()=='line':
            label=pg.TextItem('q=0',color='#169dbc',anchor=(.5,1));label.setPos(0,0);self.iso.addItem(label);self.point_labels.append(label)
    def compute_cut(self):
        v=self.v;m=self.metric_box.currentData();inter=self.interp.currentData();mode=self.mode.currentData()
        if mode=='energy':
            i=self.energy.value();xx,yy=np.meshgrid(v.qx,v.qy)
            return Cut(v.scalar(m)[i],v.qx,v.qy,'q (cycles/nm)','q (cycles/nm)',{'mode':'energy','energy_mV':float(v.energy[i]),'energy_index':i,'metric':m,'interpolation':'none, measured energy'}, {'qx':xx,'qy':yy})
        points=self.line_points() if mode=='line' else self.vertices
        cut=v.path(points,self.width.value(),m,inter)
        cut.params['corner_angles_deg']=corner_angles(points)
        if mode=='line':
            full_length=float(cut.x[-1]);cut.x-=full_length/2;cut.xlabel='q from scattering origin (cycles/nm)';cut.params.update(mode='line',angle_screen_deg=float(np.degrees(np.arctan2(points[1][1],points[1][0]))),length=full_length,gamma=[0,0])
        return cut
    def update_energy_frame(self):
        if self.v is None or self.loading or self.contrast.high<=self.contrast.low:return
        v=self.v;i=self.energy.value();c=self.contrast;m=self.metric_box.currentData()
        self.iso.show_cut(v.scalar(m)[i],v.qx,v.qy,'q (cycles/nm)','q (cycles/nm)',c)
        self.iso_title.setText(f"CONSTANT ENERGY   ·   {0 if abs(v.energy[i])<1e-8 else v.energy[i]:.6g} mV")
        if self.mode.currentData()=='energy':
            self.cut=self.compute_cut();self.section.setAspectLocked(True,ratio=1)
            self.section.show_cut(self.cut.values,self.cut.x,self.cut.y,self.cut.xlabel,self.cut.ylabel,c)
        self.stack.mark_energy(i)
    def refresh(self):
        if self.v is None or self.loading or self.contrast.high<=self.contrast.low:return
        v=self.v;i=self.energy.value();c=self.contrast;m=self.metric_box.currentData();reset=self.reset_views;self.reset_views=False
        self.iso.show_cut(v.scalar(m)[i],v.qx,v.qy,'q (cycles/nm)','q (cycles/nm)',c,reset)
        self.iso_title.setText(f"CONSTANT ENERGY   ·   {0 if abs(v.energy[i])<1e-8 else v.energy[i]:.6g} mV")
        self.update_overlay()
        try:
            self.cut=self.compute_cut();cut=self.cut
            self.section.setAspectLocked(cut.params['mode']=='energy',ratio=1)
            self.section.show_cut(cut.values,cut.x,cut.y,cut.xlabel,cut.ylabel,c,reset or self.reset_section);self.reset_section=False
            for item in self.corner_lines:self.section.removeItem(item)
            self.corner_lines=[]
            if cut.params['mode']=='path':
                for j,pos in enumerate(cut.params['corner_distances']):
                    line=pg.InfiniteLine(pos=pos,angle=90,pen=pg.mkPen('#ed8b23',style=Qt.DashLine),label=f'P{j}');self.section.addItem(line);self.corner_lines.append(line)
            self.section_title.setText(('等能量 q–q 图   ·   ' if cut.params['mode']=='energy' else 'q–E 散射截面（非直接能带）   ·   ')+self.mode.currentText())
            self.detail.setText(f"{cut.values.shape[1]} × {cut.values.shape[0]} 个采样 · 超出数据体：{np.mean(~np.isfinite(cut.values))*100:.1f}%（留空） · q 单位 cycles/nm · 偏压单位 mV。插值仅用于截面显示，不提高原始分辨率。")
        except ValueError as e:
            self.cut=None;self.section.clear_cut();self.section_title.setText('SECTION   ·   '+str(e));self.detail.setText(str(e))
        if self.stack_dirty:
            self.stack.rebuild(v,c,m,self.opacity.value(),self.stride.value(),self.stretch.value(),reset);self.stack_dirty=False
        self.stack.mark(i,self.cut)
        # Legend has the exact same LUT and gamma as every 2D/3D view.
        rgba=colormaps[c.cmap](np.linspace(0,1,256)**(1/c.gamma),bytes=True)[None,:,:].copy()
        image=QtGui.QImage(rgba.data,256,1,rgba.strides[0],QtGui.QImage.Format_RGBA8888).copy();self.legend.setPixmap(QtGui.QPixmap.fromImage(image).scaled(260,16))
        unit='pA' if m=='amplitude' else 'pA²'
        self.legend_text.setText(f"{c.low:.6g}  →  {c.high:.6g} {'log₁₀('+unit+')' if c.logarithmic else unit}\n全局色标 · Gamma {c.gamma:g} · 3D 透明度仅影响叠层合成")
    def reset_ranges(self):self.reset_views=True;self.queue()
    def project(self):
        return {'schema':2,'source':self.source,'preprocess':self.v.meta.get('preprocess','plane'),'window':self.v.meta.get('window','hann'),
                'metric':self.metric_box.currentData(),'contrast':asdict(self.contrast),'energy_index':self.energy.value(),'mode':self.mode.currentData(),
                'vertices':self.vertices,'angle':self.angle.value(),'length':self.length.value(),'line_tip':self.line_tip,'width':self.width.value(),
                'interpolation':self.interp.currentData(),'opacity':self.opacity.value(),'stride':self.stride.value(),'stretch':self.stretch.value()}
    def restore(self,p):
        p=dict(p)
        if p['mode'] in ('qx','qy','plane'):
            angle=90 if p['mode']=='qx' else 0 if p['mode']=='qy' else p['angle']
            offset=-p['q'] if p['mode']=='qx' else p['q'] if p['mode']=='qy' else p['offset']
            old=self.v.plane(angle,offset);p['vertices']=old.params['vertices'];p['mode']='path'
        if p['mode']=='tilted':raise ValueError('当前版本保留三种截面模式；旧倾斜面会话请使用上一版本打开。')
        for widget,key in [(self.metric_box,'metric'),(self.mode,'mode'),(self.interp,'interpolation')]:widget.blockSignals(True);widget.setCurrentIndex(max(0,widget.findData(p[key])));widget.blockSignals(False)
        self.vertices=p['vertices'];self.sync_table();self.energy.setValue(p['energy_index'])
        for key in ['angle','length','width','opacity','stride','stretch']:getattr(self,key).setValue(p.get(key,getattr(self,key).value()))
        self.line_tip=p.get('line_tip')
        self.contrast=Contrast(**p['contrast']);c=self.contrast
        for widget,value in [(self.low,c.low),(self.high,c.high),(self.gamma,c.gamma)]:widget.blockSignals(True);widget.setValue(value);widget.blockSignals(False)
        self.log.blockSignals(True);self.log.setChecked(c.logarithmic);self.log.blockSignals(False);self.cmap.blockSignals(True);self.cmap.setCurrentIndex(self.cmap.findData(c.cmap));self.cmap.blockSignals(False)
        self.stack_dirty=True;self.reset_views=True;self.mode_change()
    def save_project(self):
        if self.v is None:return
        p,_=W.QFileDialog.getSaveFileName(self,'保存可重开会话',str(self.export_root/'session.band.json'),'Band session (*.band.json)')
        if p:Path(p).write_text(json.dumps(self.project(),ensure_ascii=False,indent=2),encoding='utf-8');self.statusBar().showMessage('会话已保存：'+p)
    def open_project(self):
        p,_=W.QFileDialog.getOpenFileName(self,'重开会话','','Band session (*.json)')
        if not p:return
        try:
            data=json.loads(Path(p).read_text(encoding='utf-8'));self.pending_project=data
            self.pre.setCurrentIndex(self.pre.findData(data['preprocess']));self.win.setCurrentIndex(self.win.findData(data['window']))
            if data['source']['type']=='manifest':self.set_manifest(data['source']['path'],data['source']['group'])
            else:self.load_source(data['source'])
        except Exception as e:self.error(e)
    def do_export(self,root=None):
        self.refresh()
        if self.cut is None:raise ValueError('先创建有效截面。')
        root=Path(root or self.export_root);folder=root/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+self.mode.currentData())
        export_cut(folder,self.cut,self.v,self.contrast,self.project())
        render_cut_png(folder/'section.png',self.cut,self.v,self.contrast)
        if not self.stack.grabFramebuffer().save(str(folder/'stack.png')):raise OSError('截面数值已导出，但三维截图保存失败。')
        self.statusBar().showMessage('已导出：'+str(folder));self.last_export=folder;return folder
    def export(self):
        if self.v is None:return
        p=W.QFileDialog.getExistingDirectory(self,'选择导出位置（自动新建独立结果文件夹）',str(self.export_root if self.export_root.exists() else Path.home()/'Documents/Codex'))
        if p:
            try:self.export_root=Path(p);folder=self.do_export();W.QMessageBox.information(self,'已导出',str(folder)+'\n\n数值 CSV / NPZ、带色标截面 PNG、三维截图、全部参数与源文件校验值。')
            except Exception as e:self.error(e)
    def export_volume(self):
        if self.v is None:return
        p,_=W.QFileDialog.getSaveFileName(self,'保存原强度 FFT 数据体',str(self.export_root/'fft_volume.npz'),'FFT volume (*.npz)')
        if p:
            try:save_volume(p,self.v);self.statusBar().showMessage('已保存数据体：'+p)
            except Exception as e:self.error(e)
    def closeEvent(self,event):
        if any(t.isRunning() for t in self.findChildren(QtCore.QThread)):
            self.error('QPI 拟合进行中，请等待完成后关闭。');event.ignore();return
        if self.loading:self.error('正在读取数据，请等待完成后关闭。');event.ignore();return
        event.accept()

STYLE='''QWidget{font-family:"Microsoft YaHei";font-size:12px;color:#344653;background:#f6f8fa}
QGroupBox{font-weight:600;border:1px solid #dce4e9;border-radius:7px;margin-top:12px;padding-top:12px} QGroupBox::title{subcontrol-origin:margin;left:10px}
QPushButton{padding:6px 10px;background:white;border:1px solid #d4dfe6;border-radius:5px} QPushButton:hover{background:#e8f4fc;border-color:#89b5d1}
QComboBox,QDoubleSpinBox,QSpinBox,QTableWidget{background:white;padding:3px;border:1px solid #d4dfe6;border-radius:3px}
QLabel{background:transparent} QSplitter::handle{background:#e5ebef} QStatusBar{background:#e9f1f6}'''

def main():
    p=argparse.ArgumentParser();p.add_argument('file',nargs='?');p.add_argument('--manifest');p.add_argument('--group');p.add_argument('--output');p.add_argument('--report');p.add_argument('--no-autoload',action='store_true');args=p.parse_args()
    app=W.QApplication(sys.argv);app.setStyleSheet(STYLE);app.setApplicationName('Band Analysis');w=MainWindow(args);w.showMaximized()
    if args.report:
        QtCore.QTimer.singleShot(800,lambda:Path(args.report).write_text(json.dumps({'hwnd':int(w.winId()),'pid':os.getpid()}),encoding='utf-8'))
    sys.exit(app.exec())

if __name__=='__main__':main()
