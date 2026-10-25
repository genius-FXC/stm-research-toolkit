"""FFT path editing; screen-distance hit testing and measured-grid nudges."""
import numpy as np
from PySide6 import QtCore, QtWidgets as W
from PySide6.QtCore import Qt

def corner_angles(points):
    p=np.asarray(points,float);result={}
    for i in range(1,len(p)-1):
        a,b=p[i-1]-p[i],p[i+1]-p[i];den=np.linalg.norm(a)*np.linalg.norm(b)
        if den>1e-15:result[i]=float(np.degrees(np.arccos(np.clip(np.dot(a,b)/den,-1,1))))
    return result

def segment_hit(pointer,a,b,tolerance=9):
    p,a,b=map(lambda v:np.asarray(v,float),(pointer,a,b));d=b-a;den=d@d
    if den<=0:return False
    t=np.clip((p-a)@d/den,0,1)
    return np.linalg.norm(p-a-t*d)<=tolerance

class EditorMixin:
    def setup_editor(self,owner):
        self.owner=owner;self.drag_target=None;self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True);self.setMenuEnabled(False)
    def view_point(self,pos):
        p=self.plotItem.vb.mapSceneToView(self.mapToScene(pos.toPoint()))
        return np.array([p.x(),p.y()])
    def screen_point(self,q):
        p=self.mapFromScene(self.plotItem.vb.mapViewToScene(QtCore.QPointF(*q)))
        return np.array([p.x(),p.y()],float)
    def hit(self,pos):
        points=self.owner.editor_points();p=np.array([pos.x(),pos.y()]);screen=[self.screen_point(q) for q in points]
        if screen:
            dist=np.linalg.norm(np.asarray(screen)-p,axis=1);i=int(np.argmin(dist))
            if dist[i]<=13:return i
        if len(screen)==2 and segment_hit(p,*screen):return 'line'
        return None
    def inside(self,pos):return self.plotItem.vb.sceneBoundingRect().contains(self.mapToScene(pos.toPoint()))
    def mousePressEvent(self,event):
        o=self.owner
        if o.v is None or not self.inside(event.position()):return super().mousePressEvent(event)
        hit=self.hit(event.position());mode=o.mode.currentData()
        if event.button()==Qt.RightButton and mode=='path' and isinstance(hit,int):
            o.select_target(hit);menu=W.QMenu(self);a=menu.addAction(f'删除 P{hit}');a.triggered.connect(lambda:o.delete_point(hit))
            self.context_menu=menu;menu.popup(event.globalPosition().toPoint());event.accept();return
        if event.button()!=Qt.LeftButton or mode=='energy':return super().mousePressEvent(event)
        self.setFocus();q=self.view_point(event.position())
        if hit is None:
            if not o.in_bounds(q):event.accept();return
            if mode=='path':o.vertices.append(o.snap(q).tolist());hit=len(o.vertices)-1;o.path_changed()
            else:o.set_line_endpoint(q,1);hit=1
        if mode=='line' and hit=='line':hit=1
        o.select_target(hit);self.drag_target=hit;self.drag_start=q;self.drag_points=np.asarray(o.editor_points()).copy()
        event.accept()
    def mouseMoveEvent(self,event):
        o=self.owner
        if self.drag_target is not None and event.buttons()&Qt.LeftButton:
            q=self.view_point(event.position());hit=self.drag_target
            if o.mode.currentData()=='line':
                if isinstance(hit,int):o.set_line_endpoint(q,hit)
                else:o.set_line_endpoint(q,1)
            elif hit=='line':
                delta=np.rint((q-self.drag_start)/o.pixel_step())*o.pixel_step()
                proposed=self.drag_points+delta
                if o.in_bounds(proposed):o.vertices=proposed.tolist();o.path_changed()
            else:
                if o.in_bounds(q):o.vertices[hit]=o.snap(q).tolist();o.path_changed()
            event.accept();return
        hit=self.hit(event.position()) if o.v is not None else None
        self.viewport().setCursor(Qt.SizeAllCursor if hit=='line' else Qt.OpenHandCursor if hit is not None else Qt.CrossCursor)
        super().mouseMoveEvent(event)
    def mouseReleaseEvent(self,event):
        if self.drag_target is not None:self.drag_target=None;self.owner.queue();event.accept();return
        super().mouseReleaseEvent(event)
    def wheelEvent(self,event):
        if event.modifiers()&Qt.ControlModifier:
            factor=1.0015**(-event.angleDelta().y());q=self.view_point(event.position())
            self.plotItem.vb.scaleBy((factor,factor),center=QtCore.QPointF(*q));event.accept();return
        # Plain wheel pans through the enclosing controls; zoom is explicit Ctrl+wheel.
        event.ignore()
    def keyPressEvent(self,event):
        move={Qt.Key_Left:(-1,0),Qt.Key_Right:(1,0),Qt.Key_Up:(0,1),Qt.Key_Down:(0,-1)}.get(event.key())
        if move and event.modifiers()==Qt.NoModifier:
            self.owner.nudge(*move);event.accept();return
        if event.key()==Qt.Key_Delete and isinstance(self.owner.selected,int) and self.owner.mode.currentData()=='path':
            self.owner.delete_point(self.owner.selected);event.accept();return
        super().keyPressEvent(event)
