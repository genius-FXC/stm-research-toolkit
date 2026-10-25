from pathlib import Path
import sys,os,json,uuid,datetime,ctypes,urllib.request,socket
from PySide6.QtCore import Qt,QTimer,QProcess,QUrl,QSettings
from PySide6.QtGui import QWindow,QFont
from PySide6.QtWidgets import (QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QListWidget,QTabWidget,QSplitter,QPlainTextEdit,QLineEdit,QComboBox,QDoubleSpinBox,QCheckBox,QFileDialog,QMessageBox,QFormLayout,QScrollArea,QGridLayout,QFrame)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEnginePage
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'adapters'))
from agent_bridge import SessionStore

def button(text,fn):
    b=QPushButton(text);b.clicked.connect(fn);return b
def label(text):
    w=QLabel(text);w.setWordWrap(True);return w
def read_registry():return json.loads((ROOT/'registry/tools.json').read_text(encoding='utf-8'))
def python():return str(ROOT/'.venv/Scripts/python.exe')

class LocalPage(QWebEnginePage):
    def acceptNavigationRequest(self,url,nav_type,is_main):
        return url.scheme() in ('about','blob','data') or (url.host() in ('localhost','127.0.0.1') and url.port()==self.local_port)

class NativePanel(QWidget):
    def __init__(self,tool,host):
        super().__init__();self.tool=tool;self.host=host;self.proc=None;self.foreign=None;self.hwnd=0;self.timer=QTimer(self);self.timer.timeout.connect(self.poll);self.counter=0
        self.layout=QVBoxLayout(self);self.state=label('原有程序将在独立窗口打开。');self.layout.addWidget(self.state);self.startbtn=button('加载原有界面',self.start);self.layout.addWidget(self.startbtn);self.layout.addStretch()
    def start(self):
        if self.proc:return
        self.token='ToolHub_'+self.tool['id'].replace('-','_')+'_'+uuid.uuid4().hex[:8];self.report=ROOT/'logs'/(self.token+'.json');folder=ROOT/self.tool['folder']
        self.proc=QProcess(self);self.proc.setWorkingDirectory(str(folder));self.proc.setProcessChannelMode(QProcess.MergedChannels);self.proc.readyReadStandardOutput.connect(self.output)
        self.proc.errorOccurred.connect(lambda e:self.state.setText('进程无法启动：'+self.proc.errorString()))
        if self.tool['id']=='2d-lockin':args=[str(ROOT/'adapters/native_adapter.py'),str(folder/'app.py'),str(self.report)];exe=python()
        elif self.tool['id']=='band-analysis':args=[str(folder/self.tool['entry']),'--report',str(self.report)];exe=python()
        else:
            config=json.loads((ROOT/'registry/machine.json').read_text(encoding='utf-8'));exe=config['matlab']
            if not Path(exe).is_file():self.state.setText('未找到 MATLAB，请修改 registry/machine.json');self.proc=None;return
            name='kondoFitGui' if self.tool['id']=='kondo' else 'extractOriginGui';title='Kondo resonance fit' if name=='kondoFitGui' else 'Origin curve extractor'
            script=ROOT/'jobs'/(self.token+'.m');escaped=str(folder).replace("'","''")
            script.write_text(f"try\naddpath('{escaped}'); cd('{escaped}'); {name};\nf=findall(groot,'Type','figure','Name','{title}'); set(f,'Name','{self.token}'); waitfor(f); exit;\ncatch e\ndisp(getReport(e)); exit(1);\nend\n",encoding='utf-8')
            code="run('"+str(script).replace("'","''")+"')"
            self.proc.setNativeArguments('-wait -nosplash -nodesktop -logfile "'+str(ROOT/'logs'/(self.token+'_matlab.log'))+'" -r "'+code+'"');args=[]
        self.proc.start(exe,args);self.startbtn.setEnabled(False);self.state.setText('正在加载原有界面… MATLAB 首次启动可能较慢。');self.timer.start(500)
    def output(self):
        text=bytes(self.proc.readAllStandardOutput()).decode('utf-8',errors='replace')
        with (ROOT/'logs'/(self.token+'.log')).open('a',encoding='utf-8') as f:f.write(text)
    def find_hwnd(self):
        if self.tool['id'] in ('2d-lockin','band-analysis'):
            if self.report.exists():return json.loads(self.report.read_text())['hwnd']
            return 0
        hits=[];u=ctypes.windll.user32
        CB=ctypes.WINFUNCTYPE(ctypes.c_bool,ctypes.c_void_p,ctypes.c_void_p)
        def visit(hwnd,param):
            b=ctypes.create_unicode_buffer(1024);u.GetWindowTextW(hwnd,b,1024)
            if self.token in b.value and u.IsWindowVisible(hwnd):hits.append(hwnd)
            return True
        cb=CB(visit);u.EnumWindows(cb,0);return hits[0] if hits else 0
    def poll(self):
        self.counter+=1
        if self.proc.state()==QProcess.NotRunning:
            self.state.setText('程序已退出；请查看 logs 中对应日志。关闭标签后可重新加载。');self.timer.stop();return
        hwnd=self.find_hwnd()
        if hwnd:
            self.timer.stop();self.hwnd=int(hwnd)
            u=ctypes.windll.user32
            u.SetWindowTextW(ctypes.c_void_p(self.hwnd),self.tool['name'])
            u.ShowWindow(ctypes.c_void_p(self.hwnd),3)
            u.SetForegroundWindow(ctypes.c_void_p(self.hwnd))
            self.host.statusBar().showMessage(self.tool['name']+' 已在独立窗口打开')
        elif self.counter>=600:self.timer.stop();self.state.setText('启动超时，请检查日志与许可证。');self.host.statusBar().showMessage(self.state.text())
    def activate(self):
        u=ctypes.windll.user32
        if self.hwnd and u.IsWindow(ctypes.c_void_p(self.hwnd)):
            if u.IsIconic(ctypes.c_void_p(self.hwnd)):u.ShowWindow(ctypes.c_void_p(self.hwnd),9)
            u.SetForegroundWindow(ctypes.c_void_p(self.hwnd));return True
        if self.proc and self.proc.state()!=QProcess.NotRunning and not self.hwnd:
            self.host.statusBar().showMessage(self.tool['name']+' 正在启动，请稍候…');return True
        return False
    def stop(self):
        self.timer.stop()
        if self.proc and self.proc.state()!=QProcess.NotRunning:
            if self.hwnd:
                ctypes.windll.user32.PostMessageW(ctypes.c_void_p(self.hwnd),0x10,0,0)
                self.proc.waitForFinished(1500)
            if self.proc.state()!=QProcess.NotRunning:self.proc.kill();self.proc.waitForFinished(2000)

class WebPanel(QWidget):
    def __init__(self,tool,host):
        super().__init__();self.proc=None;self.ready=False;self.layout=QVBoxLayout(self);self.state=label('正在启动独立线谱服务；页面显示在独立桌面窗口，不打开浏览器。');self.layout.addWidget(self.state)
        self.view=QWebEngineView();self.layout.addWidget(self.view,1)
        with socket.socket() as s:s.bind(('127.0.0.1',0));self.port=s.getsockname()[1]
        page=LocalPage(self.view);page.local_port=self.port;self.view.setPage(page);page.profile().downloadRequested.connect(self.download)
        folder=ROOT/tool['folder'];self.proc=QProcess(self);self.proc.setWorkingDirectory(str(folder));self.proc.setProcessChannelMode(QProcess.MergedChannels);self.proc.readyReadStandardOutput.connect(self.log)
        self.proc.start(python(),[str(folder/'server.py'),'--port',str(self.port)])
        self.timer=QTimer(self);self.timer.timeout.connect(self.poll);self.timer.start(500);self.attempt=0
    def log(self):
        with (ROOT/'logs/line-server.log').open('a',encoding='utf-8') as f:f.write(bytes(self.proc.readAllStandardOutput()).decode('utf-8',errors='replace'))
    def poll(self):
        self.attempt+=1
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{self.port}/api/state',timeout=.15) as r:state=json.load(r)
            if state.get('app')!='stm-line-spectrum':raise ValueError('服务身份不符')
            self.timer.stop();self.view.load(QUrl(f'http://127.0.0.1:{self.port}/'));self.ready=True;self.state.setText('原线谱界面 · 所有原功能保留')
        except Exception:
            if self.proc.state()==QProcess.NotRunning or self.attempt>=120:self.timer.stop();self.state.setText('线谱服务未就绪，请查看 logs/line-server.log。')
    def download(self,item):
        path,_=QFileDialog.getSaveFileName(self,'保存导出文件',item.downloadFileName())
        if path:item.setDownloadDirectory(str(Path(path).parent));item.setDownloadFileName(Path(path).name);item.accept()
        else:item.cancel()
    def stop(self):
        self.timer.stop()
        if self.proc and self.proc.state()!=QProcess.NotRunning:self.proc.terminate();self.proc.waitForFinished(1500)

class BatchPanel(QWidget):
    def __init__(self,tool,host):
        super().__init__();self.tool=tool;self.proc=None;self.layout=QVBoxLayout(self);self.layout.addWidget(label(tool['description']))
        form=QFormLayout();self.input=QLineEdit();self.output=QLineEdit();self.layout.addLayout(form)
        for title,edit,is_output in [('输入文件 / 目录',self.input,False),('输出空目录',self.output,True)]:
            row=QHBoxLayout();row.addWidget(edit);row.addWidget(button('选目录',lambda checked=False,e=edit:self.pick_dir(e)))
            if not is_output:row.addWidget(button('选文件',lambda:self.pick_file()))
            form.addRow(title,row)
        self.order=QComboBox();self.order.addItems(['偏压升序','原采集顺序']);self.radius=QDoubleSpinBox();self.radius.setRange(.001,100);self.radius.setDecimals(3);self.radius.setValue(.25);self.centers=QLineEdit('2.5,0;0,2.5');self.formats=[]
        if tool['id']=='3ds-sxm':form.addRow('偏压排序',self.order);form.addRow('输出格式',label('SXM + 数值索引；Z 与 LI Demod 1 X，保持原单位'))
        elif tool['id']=='fft-stp':form.addRow('输出格式',label('WSxM STP（double）'));form.addRow('算法',label('原 plane_and_fft：扣平面、矩形窗、模值、原补零约定'))
        else:
            form.addRow('Q / cycles·nm⁻¹',self.centers);form.addRow('Mask 半径',self.radius);row=QHBoxLayout()
            for fmt in ['stp','npz','csv']:
                c=QCheckBox(fmt);c.setChecked(fmt!='csv');self.formats.append(c);row.addWidget(c)
            form.addRow('输出格式',row);form.addRow('算法',label('原成对共轭 cosine mask；保留复相位，不对模值 STP 反变换'))
        self.layout.addWidget(label('FFT/iFFT 延续原核心输入限制：单通道 forward、SCAN_DIR=up；其他 SXM 不自动猜测通道。3DS 不完整网格会停止并报告。'))
        self.go=button('检查参数并开始',self.start);self.layout.addWidget(self.go);self.logbox=QPlainTextEdit();self.logbox.setReadOnly(True);self.layout.addWidget(self.logbox,1)
    def pick_dir(self,e):
        p=QFileDialog.getExistingDirectory(self,'选择目录')
        if p:e.setText(p)
    def pick_file(self):
        ext='*.3ds' if self.tool['id']=='3ds-sxm' else '*.sxm';p,_=QFileDialog.getOpenFileName(self,'选择输入','','数据 ('+ext+')')
        if p:self.input.setText(p)
    def build_job(self):
        if not self.input.text().strip() or not self.output.text().strip():raise ValueError('请选择输入和输出')
        j={'tool_id':self.tool['id'],'input':self.input.text().strip(),'output':self.output.text().strip(),'order':'ascending' if self.order.currentIndex()==0 else 'acquisition'}
        if j['tool_id']=='batch-ifft':
            j['centers']=[[float(x) for x in p.split(',')] for p in self.centers.text().split(';') if p.strip()];j['radius']=self.radius.value();j['formats']=[c.text() for c in self.formats if c.isChecked()]
            if not j['formats']:raise ValueError('至少选择一个输出格式')
        return j
    def start(self):
        try:
            j=self.build_job();from batch_runner import prepare
            files,out=prepare(j)
            if QMessageBox.question(self,'确认运行',f'处理 {len(files)} 个输入。输出到：\n{out}\n原数据只读，输出必须为空。')!=QMessageBox.Yes:return
            job=ROOT/'jobs'/(uuid.uuid4().hex+'.json');job.write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf-8')
            self.proc=QProcess(self);self.proc.setProcessChannelMode(QProcess.MergedChannels);self.proc.readyReadStandardOutput.connect(lambda:self.logbox.appendPlainText(bytes(self.proc.readAllStandardOutput()).decode('utf-8',errors='replace')));self.proc.finished.connect(lambda code,status:(self.go.setEnabled(True),self.logbox.appendPlainText('完成' if code==0 else '失败：请查看错误和输出目录的 job_report.json')));self.proc.errorOccurred.connect(lambda e:(self.go.setEnabled(True),self.logbox.appendPlainText(self.proc.errorString())))
            self.go.setEnabled(False);self.proc.start(python(),['-X','utf8',str(ROOT/'adapters/batch_runner.py'),str(job)])
        except Exception as e:QMessageBox.warning(self,'参数检查',str(e))
    def stop(self):
        if self.proc and self.proc.state()!=QProcess.NotRunning:self.proc.kill();self.proc.waitForFinished(1000)

class AgentPanel(QWidget):
    def __init__(self,tool):
        super().__init__();self.tool=tool;self.store=SessionStore(ROOT/'sessions');self.session=self.store.create(tool['id']);lay=QVBoxLayout(self)
        lay.addWidget(label('AI 协作 / 预留接口'));lay.addWidget(label('后端尚未连接。可保存或加载本地会话，并复制上下文到 Codex；不会假装执行拟合。'))
        row=QHBoxLayout();row.addWidget(button('新会话',self.new));row.addWidget(button('加载会话',self.load));lay.addLayout(row)
        self.history=QPlainTextEdit();self.history.setReadOnly(True);lay.addWidget(self.history,1);self.input=QPlainTextEdit();self.input.setPlaceholderText('记录本次数据路径、拟合范围和要求…');self.input.setMaximumHeight(120);lay.addWidget(self.input)
        lay.addWidget(button('保存消息（未连接 AI）',self.save_message));lay.addWidget(button('复制工具与会话上下文',self.copy));self.refresh()
    def new(self):self.session=self.store.create(self.tool['id']);self.refresh()
    def refresh(self):self.history.setPlainText('\n\n'.join(m['role']+': '+m['content'] for m in self.session['messages']))
    def save_message(self):
        text=self.input.toPlainText().strip()
        if text:self.session['messages'].append({'role':'user','content':text});self.store.save(self.session);self.input.clear();self.refresh()
    def load(self):
        p,_=QFileDialog.getOpenFileName(self,'加载 ToolHub 会话',str(ROOT/'sessions'),'JSON (*.json)')
        if p:
            try:
                s=self.store.load(p)
                if s['tool_id']!=self.tool['id']:raise ValueError('该会话属于其他工具，请切换对应标签。')
                self.session=s;self.refresh()
            except Exception as e:QMessageBox.warning(self,'无法加载',str(e))
    def copy(self):
        t=self.tool;text=f"科研工具根目录：{ROOT}\n请先读 {ROOT/'AGENTS.md'} 和 {ROOT/'registry/tools.json'}。\n工具：{t['name']} ({t['id']})\n目录：{ROOT/t['folder']}\n入口：{t['entry']}\n保持原有算法；先检查输入、输出、依赖与适用条件。\n"+self.history.toPlainText();QApplication.clipboard().setText(text)

class ToolWindow(QMainWindow):
    def __init__(self,tool,hub):
        super().__init__();self.tool=tool;self.hub=hub;self.setWindowTitle(tool['name']);self.resize(1280,850)
        shell=QWidget();lay=QVBoxLayout(shell);self.setCentralWidget(shell)
        head=QHBoxLayout();title=label(tool['name']);title.setObjectName('windowTitle');head.addWidget(title);head.addStretch();head.addWidget(button('AI 协作',self.toggle_agent));lay.addLayout(head)
        self.split=QSplitter();lay.addWidget(self.split,1)
        if tool['mode']=='web':self.worker=WebPanel(tool,hub)
        elif tool['mode']=='batch':self.worker=BatchPanel(tool,hub)
        else:
            self.worker=QWidget();info=QVBoxLayout(self.worker);h=label('准备这次分析');h.setObjectName('hero');info.addWidget(h);info.addWidget(label(tool['description']));info.addWidget(label('在右侧记录数据、参数和分析目标，再复制上下文到聊天中继续。'));info.addStretch();info.addWidget(label('工具目录：'+str(ROOT/tool['folder'])));info.addWidget(label('入口：'+tool['entry']))
        self.split.addWidget(self.worker);self.agent=AgentPanel(tool);self.split.addWidget(self.agent);self.split.setSizes([850,380]);self.agent.setVisible(tool['mode']=='ai')
    def toggle_agent(self):self.agent.setVisible(not self.agent.isVisible())
    def closeEvent(self,event):
        proc=getattr(self.worker,'proc',None)
        if proc and proc.state()!=QProcess.NotRunning:
            if QMessageBox.question(self,'关闭工具','请先保存结果。关闭将结束这个工具的运行，是否继续？')!=QMessageBox.Yes:event.ignore();return
        if hasattr(self.worker,'stop'):self.worker.stop()
        self.hub.pages.pop(self.tool['id'],None);event.accept();self.deleteLater()

DESCRIPTIONS={
'band-analysis':'将 FFT 按偏压堆叠，以统一衬度查看平面和折线路径截面。',
'2d-lockin':'打开 SXM，进行选峰、局域振幅与相位分析。',
'line-spectrum':'在图像上画线，查看并导出对应位置的谱线。',
'3ds-sxm':'将网格光谱按偏压导出为 SXM 图像。',
'fft-stp':'对 SXM 做傅里叶变换，导出 STP。',
'batch-ifft':'设置 Q 点和掩膜，批量重建实空间图像。',
'origin-extract':'从 Origin 项目中选择并提取曲线。',
'kondo':'打开原拟合界面，设置模型参数并导出结果。',
'ternes':'准备自旋激发拟合的数据、参数与会话。',
'origin-project':'整理数据与绘图要求，生成 Origin 项目。',
'curve-export':'整理曲线来源、输出格式与导出要求。',
'peaks':'定位谱峰，分析峰位的空间变化。',
'derivative':'计算谱线导数，检查平滑与特征位置。',
'gap':'定义能隙并分析其空间分布。',
'periodicity':'比较实空间特征与频率成分。'}

class Hub(QMainWindow):
    def __init__(self):
        super().__init__();self.setWindowTitle('科研工具中心');self.resize(1380,920);self.setMinimumSize(900,650)
        self.db=read_registry();self.pages={};self.native={};self.settings=QSettings('ResearchTools','Hub');self.category='全部工具';self.columns=0
        shell=QWidget();self.setCentralWidget(shell);outer=QHBoxLayout(shell);outer.setContentsMargins(0,0,0,0);outer.setSpacing(0)
        self.sidebar=QFrame();self.sidebar.setObjectName('sidebar');self.sidebar.setFixedWidth(210);nav=QVBoxLayout(self.sidebar);nav.setContentsMargins(22,30,22,22);nav.setSpacing(12)
        brand=label('◈  Research');brand.setObjectName('brand');nav.addWidget(brand);nav.addWidget(label('科研工具中心'));nav.addSpacing(38)
        self.nav_buttons=[]
        for name in ['全部工具','图像分析','谱学分析','拟合','转换与导出']:
            b=button(name,lambda checked=False,n=name:self.select_category(n));b.setCheckable(True);b.setObjectName('nav');nav.addWidget(b);self.nav_buttons.append(b)
        nav.addStretch();nav.addWidget(button('复制工具索引',self.index));nav.addWidget(button('环境与说明',self.environment));nav.addSpacing(14);foot=label('本地工具 · 独立运行');foot.setObjectName('muted');nav.addWidget(foot);outer.addWidget(self.sidebar)
        content=QWidget();layout=QVBoxLayout(content);layout.setContentsMargins(36,25,36,18);layout.setSpacing(20);outer.addWidget(content,1)
        bar=QHBoxLayout();self.menu=button('☰',self.toggle_sidebar);self.menu.setToolTip('显示或隐藏导航');self.menu.setFixedWidth(42);bar.addWidget(self.menu);bread=label('工作台');bread.setObjectName('muted');bar.addWidget(bread);bar.addStretch();self.search=QLineEdit();self.search.setPlaceholderText('搜索工具或用途…');self.search.setFixedWidth(290);self.search.textChanged.connect(self.render);bar.addWidget(self.search);layout.addLayout(bar)
        self.heading=label('今天，从哪项分析开始？');self.heading.setObjectName('hero');layout.addWidget(self.heading)
        sub=label('熟悉的工具，集中在这里。选择一个，在独立窗口中继续。');sub.setObjectName('muted');layout.addWidget(sub)
        self.recent_box=QWidget();self.recent_layout=QHBoxLayout(self.recent_box);self.recent_layout.setContentsMargins(0,0,0,0);layout.addWidget(self.recent_box);self.render_recent()
        section=QHBoxLayout();self.section_label=label('全部工具');self.section_label.setObjectName('section');section.addWidget(self.section_label);section.addStretch();self.count=label('');self.count.setObjectName('muted');section.addWidget(self.count);layout.addLayout(section)
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setFrameShape(QFrame.NoFrame);self.cards=QWidget();self.grid=QGridLayout(self.cards);self.grid.setContentsMargins(0,0,8,10);self.grid.setSpacing(16);scroll.setWidget(self.cards);layout.addWidget(scroll,1)
        footer=label('AI 协作接口已预留，尚未连接模型。');footer.setObjectName('muted');layout.addWidget(footer)
        self.sidebar.setVisible(self.settings.value('sidebar',True,type=bool));self.select_category('全部工具');self.statusBar().showMessage('就绪')
    def toggle_sidebar(self):
        self.sidebar.setVisible(not self.sidebar.isVisible());self.settings.setValue('sidebar',self.sidebar.isVisible());QTimer.singleShot(0,self.render)
    def select_category(self,name):
        self.category=name
        for b in self.nav_buttons:b.setChecked(b.text()==name)
        self.render()
    def render_recent(self):
        while self.recent_layout.count():
            item=self.recent_layout.takeAt(0)
            if item.widget():item.widget().hide();item.widget().deleteLater()
        ids=self.settings.value('recent',[],type=list);known={t['id']:t for t in self.db['tools']};ids=[i for i in ids if i in known][:3]
        self.recent_box.setVisible(bool(ids))
        if ids:self.recent_layout.addWidget(label('最近使用'))
        for id in ids:self.recent_layout.addWidget(button(known[id]['name'],lambda checked=False,i=id:self.open_tool(i)))
        self.recent_layout.addStretch()
    def render(self):
        if not hasattr(self,'grid'):return
        while self.grid.count():
            item=self.grid.takeAt(0)
            if item.widget():item.widget().hide();item.widget().deleteLater()
        q=self.search.text().strip().lower();items=[t for t in self.db['tools'] if (self.category=='全部工具' or t['category']==self.category) and q in (t['name']+t['category']+DESCRIPTIONS[t['id']]).lower()]
        self.section_label.setText(self.category);self.count.setText(str(len(items))+' 个工具');self.columns=3 if self.width()-(210 if self.sidebar.isVisible() else 0)>1050 else 2
        for old in range(3):self.grid.setColumnStretch(old,1 if old<self.columns else 0)
        for n,t in enumerate(items):
            card=QFrame();card.setObjectName('card');card.setMinimumHeight(192);l=QVBoxLayout(card);l.setContentsMargins(22,20,22,18);l.setSpacing(12)
            meta=label(t['category']);meta.setObjectName('eyebrow');l.addWidget(meta);title=label(t['name']);title.setObjectName('cardTitle');l.addWidget(title);desc=label(DESCRIPTIONS[t['id']]);desc.setObjectName('muted');l.addWidget(desc);l.addStretch();row=QHBoxLayout();mode=label({'native':'独立应用','web':'独立应用','batch':'批量处理','ai':'AI 辅助'}[t['mode']]);mode.setObjectName('muted');row.addWidget(mode);row.addStretch();b=button('打开  ↗',lambda checked=False,i=t['id']:self.open_tool(i));b.setObjectName('launch');row.addWidget(b);l.addLayout(row);self.grid.addWidget(card,n//self.columns,n%self.columns)
        if not items:self.grid.addWidget(label('没有找到匹配的工具，试试其他关键词。'),0,0)
        self.grid.setAlignment(Qt.AlignTop)
    def resizeEvent(self,event):
        super().resizeEvent(event)
        cols=3 if self.width()-(210 if hasattr(self,'sidebar') and self.sidebar.isVisible() else 0)>1050 else 2
        if cols!=self.columns:self.render()
    def open_tool(self,id):
        t=next(t for t in self.db['tools'] if t['id']==id);recent=self.settings.value('recent',[],type=list);self.settings.setValue('recent',[id]+[i for i in recent if i!=id][:5]);self.render_recent()
        if t['mode']=='native':
            if id in self.native and self.native[id].activate():return
            old=self.native.pop(id,None)
            if old:old.deleteLater()
            worker=NativePanel(t,self);worker.setParent(self);worker.hide();self.native[id]=worker;worker.start();self.statusBar().showMessage(t['name']+' 正在启动…');return
        if id in self.pages:w=self.pages[id];w.showNormal() if w.isMinimized() else w.show();w.raise_();w.activateWindow();return
        w=ToolWindow(t,self);self.pages[id]=w;w.showMaximized()
    def index(self):
        QApplication.clipboard().setText(f'科研工具统一目录：{ROOT}\n先读 {ROOT/"AGENTS.md"} 和 {ROOT/"registry/tools.json"}。');self.statusBar().showMessage('工具索引已复制',4000)
    def environment(self):
        w=QMainWindow(self);w.setWindowFlag(Qt.Window,True);w.setWindowTitle('环境与迁移说明');w.resize(900,700);text=QPlainTextEdit();text.setReadOnly(True);text.setPlainText((ROOT/'docs/MIGRATION.md').read_text(encoding='utf-8'));w.setCentralWidget(text);w.show()
    def closeEvent(self,event):
        active=bool(self.pages) or any(w.proc and w.proc.state()!=QProcess.NotRunning for w in self.native.values())
        if active and QMessageBox.question(self,'退出工具中心','仍有工具窗口打开。请先保存结果。退出将关闭这些工具，是否继续？')!=QMessageBox.Yes:event.ignore();return
        for w in list(self.pages.values()):
            if hasattr(w.worker,'stop'):w.worker.stop()
            w.hide();w.deleteLater()
        for w in self.native.values():w.stop()
        event.accept()

STYLE='''QWidget{background:#faf9f6;color:#302f2c;font-family:"Microsoft YaHei";font-size:13px}
QFrame#sidebar{background:#f0eee8;border-right:1px solid #e4e0d8} #sidebar QLabel{background:transparent}
QLabel#brand{font-size:22px;font-weight:600} QLabel#hero{font-size:30px;font-weight:600;padding-top:12px} QLabel#windowTitle{font-size:20px;font-weight:600}
QLabel#muted{color:#79766f;font-size:12px} QLabel#section{font-size:16px;font-weight:600} QLabel#eyebrow{color:#9b705a;font-size:11px} QLabel#cardTitle{font-size:17px;font-weight:600}
QFrame#card{background:#fffefa;border:1px solid #e6e2da;border-radius:12px} #card QLabel{background:transparent}
QPushButton{background:#f4f1eb;border:1px solid #e5dfd5;border-radius:7px;padding:9px 13px} QPushButton:hover{background:#eae4da;border-color:#cfc3b5} QPushButton:pressed{background:#ded5c8} QPushButton:disabled{color:#aaa59d}
QPushButton#nav{background:transparent;border:none;text-align:left;padding:12px} QPushButton#nav:checked{background:#e5dfd4;color:#574537} QPushButton#nav:hover{background:#eae5dc}
QPushButton#launch{background:#a65f46;color:white;border:none;padding:8px 17px} QPushButton#launch:hover{background:#8f4e39}
QLineEdit,QPlainTextEdit,QComboBox,QDoubleSpinBox{background:#fffefa;border:1px solid #ded9cf;border-radius:7px;padding:9px;selection-background-color:#dcc6b9;selection-color:#302f2c}
QLineEdit:focus,QPlainTextEdit:focus{border-color:#b58b73} QScrollArea{background:transparent;border:none} QScrollBar:vertical{background:transparent;width:9px;margin:0} QScrollBar::handle:vertical{background:#d4cfc5;border-radius:4px;min-height:35px} QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0} QSplitter::handle{background:#e6e1d7;width:1px} QStatusBar{background:#f0eee8;color:#80796f;font-size:11px}
'''
if __name__=='__main__':
    sys.path.insert(0,str(ROOT/'scripts'))
    from configure import configure
    configure()
    app=QApplication(sys.argv);app.setStyle('Fusion');app.setStyleSheet(STYLE);win=Hub();win.show()
    if '--open' in sys.argv:win.open_tool(sys.argv[sys.argv.index('--open')+1])
    if '--capture' in sys.argv:
        path=Path(sys.argv[sys.argv.index('--capture')+1]);QTimer.singleShot(2500,lambda:win.grab().save(str(path)))
    sys.exit(app.exec())
