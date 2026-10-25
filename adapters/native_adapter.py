"""Run unchanged Tkinter entry and announce its top-level HWND to the host."""
from pathlib import Path
import os,sys,runpy,json,ctypes
entry=Path(sys.argv[1]).resolve();report=Path(sys.argv[2]);sys.path.insert(0,str(entry.parent));os.chdir(entry.parent)
import tkinter as tk
original=tk.Tk
class HostedTk(original):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        def ready():
            self.title('2D-lockin');self.update_idletasks()
            user=ctypes.windll.user32;user.GetParent.argtypes=[ctypes.c_void_p];user.GetParent.restype=ctypes.c_void_p
            hwnd=user.GetParent(self.winfo_id()) or self.winfo_id()
            report.write_text(json.dumps({'hwnd':hwnd,'pid':os.getpid()}),encoding='utf-8')
            test_input=os.environ.get('TOOLHUB_VERIFY_SXM')
            if test_input and hasattr(self,'open_sxm'):self.open_sxm(test_input)
        self.after(500,ready)
tk.Tk=HostedTk
runpy.run_path(str(entry),run_name='__main__')
