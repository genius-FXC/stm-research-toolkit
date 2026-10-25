"""Provider-neutral extension contract. No API/OAuth credentials or network calls."""
from typing import Protocol, Iterable
from dataclasses import dataclass
from pathlib import Path
import json,uuid,datetime

@dataclass
class AgentEvent:
    type: str  # text_delta, tool_proposal, error, done
    data: dict

class AgentProvider(Protocol):
    def stream(self, messages:list[dict], context:dict, tools:list[dict]) -> Iterable[AgentEvent]: ...
    def cancel(self) -> None: ...

class AuthProvider(Protocol):
    def begin_login(self) -> dict: ...
    def finish_login(self, callback:dict) -> str: ... # credential reference only
    def logout(self) -> None: ...

class SessionStore:
    def __init__(self,folder):self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
    def create(self,tool_id):
        obj={'schema':'toolhub-session/v1','id':str(uuid.uuid4()),'tool_id':tool_id,'created':datetime.datetime.now().isoformat(),'messages':[]};self.save(obj);return obj
    def save(self,obj):
        sid=str(uuid.UUID(obj['id']));p=self.folder/(sid+'.json');temp=p.with_suffix('.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(p)
    def load(self,path):
        obj=json.loads(Path(path).read_text(encoding='utf-8'))
        if obj.get('schema')!='toolhub-session/v1' or not isinstance(obj.get('messages'),list):raise ValueError('只支持 ToolHub 会话格式；未接入 Codex/OAuth 历史导入。')
        uuid.UUID(obj['id']);return obj

def tool_schemas():
    return [{'name':'prepare_analysis_job','description':'准备既有分析工具任务；必须由用户在界面确认后执行。','parameters':{'type':'object','properties':{'tool_id':{'enum':['3ds-sxm','fft-stp','batch-ifft']},'input':{'type':'string'},'output':{'type':'string'},'options':{'type':'object'}},'required':['tool_id','input','output']}}]
