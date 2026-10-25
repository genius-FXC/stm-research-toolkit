"""Local-only interactive line tool; no cloud service or AI dependency."""
from __future__ import annotations
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse,parse_qs,unquote
import argparse,datetime,json,mimetypes,os,re,secrets,subprocess,sys,threading,traceback,webbrowser
from core import ROOT,GridDataset,LineSettings,extract_line,preview_payload
from plotting import save_numeric_bundle,export_images

EXPORTS=ROOT/'exports';UPLOADS=ROOT/'uploads';RUNTIME=ROOT/'runtime'
for folder in [EXPORTS,UPLOADS,RUNTIME]:folder.mkdir(exist_ok=True)
PLOT_LOCK=threading.Lock();ORIGIN_LOCK=threading.Lock()

def native(obj):
    if hasattr(obj,'tolist'):return obj.tolist()
    if isinstance(obj,Path):return str(obj)
    raise TypeError(type(obj).__name__)

class App:
    def __init__(self):
        self.data=None;self.version=0;self.lines={};self.counter=0;self.jobs={};self.lock=threading.RLock();self.token=secrets.token_urlsafe(24)
    def load(self,path):
        path=Path(path).expanduser().resolve()
        if path.suffix.lower()!='.3ds':raise ValueError('请选择 Nanonis .3ds 网格文件。')
        if not path.is_file():raise ValueError('文件不存在，请确认完整路径。')
        d=GridDataset(path)
        with self.lock:
            self.data=d;self.version+=1;self.lines={};self.counter=0
        return self.state()
    def state(self):
        with self.lock:
            return dict(app='stm-line-spectrum',version='1.1.0',data_version=self.version,dataset=self.data.info() if self.data else None,lines=[dict(line_id=k,length_nm=v['length_nm'],start=v['config']['start'],end=v['config']['end']) for k,v in self.lines.items()],exports_root=str(EXPORTS),default_file=CONFIG.get('default_file',''))
    def line(self,body):
        with self.lock:
            if self.data is None:raise ValueError('请先打开一个 3DS。')
            if body.get('data_version',self.version)!=self.version:raise ValueError('数据文件已改变，请在当前图像上重新画线。')
            lid=body.get('line_id')
            if not lid:
                self.counter+=1;lid=f'L{self.counter:03d}'
            elif lid not in self.lines:raise ValueError('这条线不属于当前文件，请新建线。')
            result=extract_line(self.data,LineSettings.from_dict(body['settings']),lid)
            self.lines[lid]=result
            return preview_payload(result)
    def export(self,body):
        with self.lock:
            lid=body['line_id']
            if lid not in self.lines:raise ValueError('请先画线并生成预览。')
            data=self.data;result=self.lines[lid]
            kind=body.get('kind','png')
            if kind not in ['png','svg','pdf','origin','data']:raise ValueError('未知导出类型。')
            jobid=secrets.token_hex(6);stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            stem=re.sub(r'[^a-zA-Z0-9_.-]+','_',data.path.stem).strip('_')[:75] or 'grid'
            folder=EXPORTS/stem/f'{lid}_{stamp}_{jobid[:4]}'
            job=dict(job_id=jobid,status='running',kind=kind,line_id=lid,folder=str(folder),message='正在保存数值与位置记录…',files=[])
            self.jobs[jobid]=job
        def worker():
            try:
                save_numeric_bundle(data,result,folder)
                if kind not in ['origin','data']:
                    job['message']='正在生成高分辨率图片…'
                    with PLOT_LOCK:export_images(data,result,folder,(kind,))
                elif kind=='origin':
                    job['message']='正在生成 Origin 原生曲线与矩阵，请稍候…'
                    with ORIGIN_LOCK:
                        log=folder/'origin_export.log'
                        with log.open('w',encoding='utf-8') as f:
                            cp=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'origin_export.py'),str(folder)],stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),timeout=180)
                        if cp.returncode:raise RuntimeError('Origin 导出失败：'+log.read_text(encoding='utf-8')[-1800:])
                    job['message']='Origin 工程已保存，正在完成图片预览…'
                    with PLOT_LOCK:export_images(data,result,folder,('png',))
                job['files']=[dict(name=p.name,url='/exports/'+p.relative_to(EXPORTS).as_posix()) for p in sorted(folder.iterdir()) if p.is_file()]
                job['status']='complete';job['message']='已保存。数值 CSV 与坐标记录一同导出。'
            except Exception as ex:
                job['status']='error';job['message']=str(ex)
                folder.mkdir(parents=True,exist_ok=True);(folder/'export_error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        threading.Thread(target=worker,daemon=True).start()
        return job.copy()

CONFIG=json.loads((ROOT/('config.json' if (ROOT/'config.json').exists() else 'config.example.json')).read_text(encoding='utf-8'));APP=App()

class Handler(BaseHTTPRequestHandler):
    def log_message(self,fmt,*args):
        print(datetime.datetime.now().strftime('%H:%M:%S'),fmt%args,flush=True)
    def write(self,body,status=200,mime='application/json; charset=utf-8',download=None):
        if not isinstance(body,bytes):body=json.dumps(body,ensure_ascii=False,default=native,allow_nan=False).encode('utf-8')
        self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
        if download:
            from urllib.parse import quote
            self.send_header('Content-Disposition',"attachment; filename*=UTF-8''"+quote(download))
        self.end_headers()
        try:self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError):pass
    def valid_host(self):
        return self.headers.get('Host') in [f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}']
    def do_GET(self):
        if not self.valid_host():return self.write({'error':'本工具仅接受本机地址。'},403)
        req=urlparse(self.path);q=parse_qs(req.query);path=unquote(req.path)
        try:
            if path=='/':
                html=(ROOT/'web/index.html').read_text(encoding='utf-8').replace('__TOKEN__',APP.token)
                return self.write(html.encode('utf-8'),mime='text/html; charset=utf-8')
            if path in ['/app.js','/style.css']:
                return self.write((ROOT/'web'/path[1:]).read_bytes(),mime='text/javascript; charset=utf-8' if path.endswith('.js') else 'text/css; charset=utf-8')
            if path=='/api/state':return self.write(APP.state())
            if path=='/api/map':
                with APP.lock:
                    if APP.data is None:raise ValueError('尚未载入 3DS。')
                    image,_=APP.data.image_png(q.get('view',['topo'])[0],float(q.get('bias',['0'])[0]),q.get('plane',['true'])[0]=='true')
                return self.write(image,mime='image/png')
            if path=='/api/line':
                with APP.lock:r=APP.lines[q['id'][0]]
                return self.write(preview_payload(r))
            if path=='/api/job':return self.write(APP.jobs[q['id'][0]].copy())
            if path.startswith('/exports/'):
                target=(EXPORTS/path[len('/exports/'):]).resolve()
                if not target.is_relative_to(EXPORTS.resolve()) or not target.is_file():return self.write({'error':'文件不存在。'},404)
                return self.write(target.read_bytes(),mime=mimetypes.guess_type(target)[0] or 'application/octet-stream',download=target.name)
            return self.write({'error':'地址不存在。'},404)
        except (ValueError,KeyError) as ex:return self.write({'error':str(ex)},400)
        except Exception as ex:
            traceback.print_exc();return self.write({'error':str(ex)},500)
    def do_POST(self):
        if not self.valid_host() or self.headers.get('X-STM-Token')!=APP.token:return self.write({'error':'请求来自无效页面，请从启动地址重新打开。'},403)
        origin=self.headers.get('Origin')
        if origin and origin not in [f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}']:return self.write({'error':'仅允许本机页面操作。'},403)
        try:
            n=int(self.headers.get('Content-Length','0'))
            if n>512*1024**2:raise ValueError('文件超过512MiB，请使用完整路径载入。')
            raw=self.rfile.read(n);path=urlparse(self.path).path
            if path=='/api/upload':
                name=Path(unquote(self.headers.get('X-Filename','grid.3ds'))).name
                if Path(name).suffix.lower()!='.3ds':raise ValueError('请选择 .3ds。')
                folder=UPLOADS/secrets.token_hex(6);folder.mkdir();p=folder/name;p.write_bytes(raw)
                return self.write(APP.load(p))
            body=json.loads(raw or '{}')
            if path=='/api/load':return self.write(APP.load(body['path'].strip().strip('"')))
            if path=='/api/line':return self.write(APP.line(body))
            if path=='/api/export':return self.write(APP.export(body))
            if path=='/api/restore':
                file=Path(body['path'].strip().strip('"'));meta=json.loads(file.read_text(encoding='utf-8'))
                if meta.get('format')!='stm-line-spectrum/v1':raise ValueError('不是本工具的线段记录 JSON。')
                with APP.lock:
                    if APP.data is None or APP.data.hash!=meta['source_sha256']:raise ValueError('请先打开这份记录对应的原始 3DS。')
                return self.write(APP.line(dict(settings=meta['settings'],data_version=APP.version)))
            if path=='/api/shutdown':
                self.write({'message':'服务已停止，可以关闭页面。'});threading.Thread(target=self.server.shutdown,daemon=True).start();return
            return self.write({'error':'地址不存在。'},404)
        except (ValueError,KeyError,FileNotFoundError) as ex:return self.write({'error':str(ex)},400)
        except Exception as ex:
            traceback.print_exc();return self.write({'error':str(ex)},500)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=CONFIG.get('port',8863));ap.add_argument('--file');ap.add_argument('--open',action='store_true');args=ap.parse_args()
    path=args.file or CONFIG.get('default_file')
    if path and Path(path).is_file():
        try:APP.load(path)
        except Exception:traceback.print_exc()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    url=f'http://127.0.0.1:{server.server_port}/'
    (RUNTIME/'server.json').write_text(json.dumps(dict(pid=os.getpid(),url=url,port=server.server_port,started=datetime.datetime.now().isoformat()),indent=2),encoding='utf-8')
    print('STM Line Spectrum ready:',url,flush=True)
    if args.open:webbrowser.open(url)
    try:server.serve_forever(poll_interval=.2)
    finally:server.server_close()

if __name__=='__main__':main()
