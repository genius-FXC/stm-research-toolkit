from __future__ import annotations

from dataclasses import asdict
from io import BytesIO
import json
import math
from pathlib import Path
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from types import SimpleNamespace
from typing import Callable, Optional
import webbrowser

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

from .ai import (
    AIConfig,
    list_openrouter_vision_models,
    load_config as load_ai_config,
    request_openrouter_analysis,
    resolve_openrouter_key,
    save_config as save_ai_config,
    save_openrouter_key,
    build_openrouter_payload,
)
from .analysis import (
    AnalysisState,
    MaskSpec,
    c4x4_allowed_points,
    combine_local_amplitudes,
    compute_fft,
    fractional_coordinates,
    local_complex_component,
    make_frequency_mask,
    phase_histogram,
    preprocess_image,
    reconstruct_from_mask,
)
from .palette import colorize, make_lut, robust_limits
from .sxm import SxmFile, read_sxm


PREPROCESS_OPTIONS = {
    "不处理": "none",
    "减均值": "mean",
    "稳健平面调平": "plane",
    "平面 + 逐行中值": "row_median",
    "平面 + 逐行线性": "row_linear",
}
WINDOW_OPTIONS = {"无窗": "none", "Hann": "hann", "Tukey": "tukey"}
FFT_VIEW_OPTIONS = {
    "对数振幅（dB）": "amplitude_db",
    "对数振幅": "log_amplitude",
    "振幅": "amplitude",
    "功率": "power",
    "相位": "phase",
    "实部": "real",
    "虚部": "imag",
}
RESULT_OPTIONS = {
    "成对 iFFT": "reconstruction",
    "局域振幅": "amplitude",
    "局域相位": "phase",
    "频率 mask": "mask",
    "mask 后 FFT": "masked_fft",
}
COMPOSITE_METHODS = {
    "共同振幅 · 调和平均": "harmonic",
    "共同振幅 · 几何均值": "geometric",
    "共同振幅 · min": "minimum",
    "总体振幅 · 平方平均（RMS）": "rms",
}


def _cjk_font(size: int = 16):
    for candidate in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def _application_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def _nice_scale(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    fraction = value / (10**exponent)
    if fraction >= 5:
        nice = 5
    elif fraction >= 2:
        nice = 2
    else:
        nice = 1
    return nice * (10**exponent)


def _format_number(value: float) -> str:
    if not np.isfinite(value):
        return "—"
    magnitude = abs(value)
    if magnitude != 0 and (magnitude >= 1e4 or magnitude < 1e-3):
        return f"{value:.3e}"
    return f"{value:.4g}"


def _orient_array(array: np.ndarray, flip_x: bool = False, flip_y: bool = False, rotation: int = 0) -> np.ndarray:
    shown = np.asarray(array)
    if flip_x:
        shown = np.fliplr(shown)
    if flip_y:
        shown = np.flipud(shown)
    return np.rot90(shown, -rotation)


def _rgb_on_background(image: Image.Image, color: str = "white") -> Image.Image:
    if image.mode not in {"RGBA", "LA"} and "transparency" not in image.info:
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, color)
    background.alpha_composite(rgba)
    return background.convert("RGB")


class ImagePanel(ttk.Frame):
    def __init__(self, master: tk.Misc, title: str, cursor_callback: Optional[Callable] = None):
        super().__init__(master)
        self.title_var = tk.StringVar(value=title)
        ttk.Label(self, textvariable=self.title_var, font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=4, pady=(2, 0))
        self.canvas = tk.Canvas(self, background="#111111", highlightthickness=0, cursor="crosshair", takefocus=True)
        self.canvas.pack(fill="both", expand=True, padx=3, pady=3)
        self.canvas.bind("<Configure>", lambda _event: self.redraw())
        if cursor_callback:
            self.canvas.bind("<Motion>", cursor_callback)
        self.array: Optional[np.ndarray] = None
        self.palette = "gray"
        self.low = 1.0
        self.high = 99.0
        self.limits: Optional[tuple[float, float]] = None
        self.rotation = 0
        self.flip_x = False
        self.flip_y = False
        self.photo: Optional[ImageTk.PhotoImage] = None
        self.colorbar_photo: Optional[ImageTk.PhotoImage] = None
        self.image_box = (0.0, 0.0, 1.0, 1.0)
        self.overlays: list[dict] = []
        self.overlay_font_size = 13
        self.zoom = 1.0
        self.view_center: Optional[tuple[float, float]] = None
        self._pan_start: Optional[tuple[float, float, float, float]] = None
        self.scale_bar_visible = False
        self.scale_bar_span = 1.0
        self.scale_bar_unit = "nm"
        self.scale_bar_style = "blackwhite"
        self.colorbar_visible = False
        self.colorbar_label = ""
        self.colorbar_ticks: tuple[str, str, str] = ("", "", "")
        self.canvas.bind("<Control-MouseWheel>", self._control_wheel)
        self.canvas.bind("<ButtonPress-3>", self._pan_press)
        self.canvas.bind("<B3-Motion>", self._pan_drag)
        self.canvas.bind("<ButtonRelease-3>", lambda _event: setattr(self, "_pan_start", None))

    def set_data(
        self,
        array: np.ndarray,
        palette: str = "gray",
        low: float = 1.0,
        high: float = 99.0,
        limits: Optional[tuple[float, float]] = None,
        flip_x: bool = False,
        flip_y: bool = False,
        title: Optional[str] = None,
    ) -> None:
        old_shape = self.array.shape if self.array is not None else None
        self.array = np.asarray(array)
        self.palette = palette
        self.low = low
        self.high = high
        self.limits = limits
        self.flip_x = flip_x
        self.flip_y = flip_y
        if title:
            self.title_var.set(title)
        if old_shape != self.array.shape or self.view_center is None:
            self.reset_view(redraw=False)
        self.redraw()

    def clear_data(self, title: str) -> None:
        self.array = None
        self.photo = None
        self.title_var.set(title)
        self.canvas.delete("all")

    def set_overlays(self, overlays: list[dict]) -> None:
        self.overlays = overlays
        self.canvas.delete("overlay")
        if self.array is not None:
            for overlay in self.overlays:
                self._draw_overlay(overlay)

    def set_overlay_font_size(self, size: int) -> None:
        self.overlay_font_size = int(np.clip(size, 7, 72))
        self.canvas.delete("overlay")
        if self.array is not None:
            for overlay in self.overlays:
                self._draw_overlay(overlay)

    def set_scale_bar(self, visible: bool, span: float = 1.0, unit: str = "nm", style: str = "blackwhite") -> None:
        self.scale_bar_visible = bool(visible)
        self.scale_bar_span = float(span)
        self.scale_bar_unit = unit
        self.scale_bar_style = style
        self.canvas.delete("scale_bar")
        self._draw_scale_bar()

    def set_colorbar(self, visible: bool, label: str = "", ticks: tuple[str, str, str] = ("", "", "")) -> None:
        layout_changed = bool(visible) != self.colorbar_visible
        self.colorbar_visible = bool(visible)
        self.colorbar_label = label
        self.colorbar_ticks = ticks
        if layout_changed:
            self.redraw()
        else:
            self.canvas.delete("colorbar")
            self._draw_colorbar()

    def redraw(self) -> None:
        self.canvas.delete("all")
        if self.array is None or self.array.size == 0:
            return
        canvas_w = max(self.canvas.winfo_width(), 10)
        canvas_h = max(self.canvas.winfo_height(), 10)
        colorbar_space = 104 if self.colorbar_visible else 0
        plot_w = max(canvas_w - colorbar_space, 10)
        ny, nx = self.shown_shape()
        base_scale = min(plot_w / nx, canvas_h / ny)
        scale = base_scale * self.zoom
        width = max(1, int(nx * scale))
        height = max(1, int(ny * scale))
        center_col, center_row = self.view_center or ((nx - 1) / 2, (ny - 1) / 2)
        shown_center_col, shown_center_row = self.to_shown(center_col,center_row)
        x0 = plot_w / 2 - (shown_center_col + 0.5) * scale
        y0 = canvas_h / 2 - (shown_center_row + 0.5) * scale
        shown = _orient_array(self.array, self.flip_x, self.flip_y, self.rotation)
        pil = colorize(shown, self.palette, self.low, self.high, self.limits)
        pil = pil.resize((width, height), Image.Resampling.NEAREST)
        self.photo = ImageTk.PhotoImage(pil)
        self.canvas.create_image(x0, y0, image=self.photo, anchor="nw", tags="base")
        self.image_box = (x0, y0, x0 + width, y0 + height)
        for overlay in self.overlays:
            self._draw_overlay(overlay)
        self._draw_scale_bar()
        self._draw_colorbar()

    def _draw_overlay(self, overlay: dict) -> None:
        x, y = self.data_to_canvas(overlay["x"], overlay["y"])
        radius_data = float(overlay.get("radius", 3.0))
        x0, y0, x1, y1 = self.image_box
        ny, nx = self.shown_shape() if self.array is not None else (1, 1)
        radius = radius_data * 0.5 * ((x1 - x0) / nx + (y1 - y0) / ny)
        color = overlay.get("color", "#44ff88")
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, outline=color, width=2, tags="overlay")
        if overlay.get("text"):
            self.canvas.create_text(
                x + radius + 3,
                y - radius - 3,
                text=overlay["text"],
                fill=color,
                anchor="sw",
                font=("Microsoft YaHei UI", -self.overlay_font_size),
                tags="overlay",
            )

    def _visible_image_box(self) -> tuple[float, float, float, float]:
        canvas_w = max(self.canvas.winfo_width(), 10)
        canvas_h = max(self.canvas.winfo_height(), 10)
        plot_w = max(canvas_w - (104 if self.colorbar_visible else 0), 10)
        x0, y0, x1, y1 = self.image_box
        return max(0.0, x0), max(0.0, y0), min(float(plot_w), x1), min(float(canvas_h), y1)

    def _draw_scale_bar(self) -> None:
        if not self.scale_bar_visible or self.array is None or self.scale_bar_span <= 0:
            return
        x0, y0, x1, y1 = self.image_box
        vx0, vy0, vx1, vy1 = self._visible_image_box()
        if vx1 - vx0 < 80 or vy1 - vy0 < 50:
            return
        viewport_span = self.scale_bar_span * (vx1 - vx0) / max(x1 - x0, 1e-12)
        value = _nice_scale(0.22 * viewport_span)
        bar_pixels = value / self.scale_bar_span * (x1 - x0)
        right = vx1 - 16
        left = right - bar_pixels
        y = vy1 - 19
        label = f"{value:g} {self.scale_bar_unit}"
        style = self.scale_bar_style
        if style == "blackwhite":
            self.canvas.create_line(left, y, right, y, fill="white", width=7, tags="scale_bar")
            self.canvas.create_line(left, y, right, y, fill="black", width=3, tags="scale_bar")
            for ox, oy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                self.canvas.create_text((left + right) / 2 + ox, y - 7 + oy, text=label, fill="white", anchor="s", tags="scale_bar")
            self.canvas.create_text((left + right) / 2, y - 7, text=label, fill="black", anchor="s", tags="scale_bar")
        else:
            color = "white" if style == "white" else "black"
            self.canvas.create_line(left, y, right, y, fill=color, width=4, tags="scale_bar")
            self.canvas.create_text((left + right) / 2, y - 7, text=label, fill=color, anchor="s", tags="scale_bar")

    def _draw_colorbar(self) -> None:
        if not self.colorbar_visible or self.array is None:
            return
        canvas_w = max(self.canvas.winfo_width(), 10)
        canvas_h = max(self.canvas.winfo_height(), 10)
        plot_w = max(canvas_w - 104, 10)
        height = int(min(250, max(100, 0.55 * canvas_h)))
        if height < 60 or canvas_w - plot_w < 70:
            return
        width = 18
        gradient = make_lut(self.palette)[::-1][:, None, :]
        image = Image.fromarray(gradient.astype(np.uint8), mode="RGB").resize((width, height), Image.Resampling.BILINEAR)
        self.colorbar_photo = ImageTk.PhotoImage(image)
        x = plot_w + 12
        y = (canvas_h - height) / 2
        self.canvas.create_image(x, y, image=self.colorbar_photo, anchor="nw", tags="colorbar")
        top, middle, bottom = self.colorbar_ticks
        text_x = x + width + 6
        self.canvas.create_text(text_x, y, text=top, fill="white", anchor="nw", tags="colorbar")
        self.canvas.create_text(text_x, y + height / 2, text=middle, fill="white", anchor="w", tags="colorbar")
        self.canvas.create_text(text_x, y + height, text=bottom, fill="white", anchor="sw", tags="colorbar")
        if self.colorbar_label:
            self.canvas.create_text(x, y - 7, text=self.colorbar_label, fill="white", anchor="sw", tags="colorbar")

    def shown_shape(self):
        ny,nx=self.array.shape
        return (nx,ny) if self.rotation%2 else (ny,nx)

    def to_shown(self,col,row):
        ny,nx=self.array.shape
        x=nx-1-col if self.flip_x else col
        y=ny-1-row if self.flip_y else row
        return [(x,y),(ny-1-y,x),(nx-1-x,ny-1-y),(y,nx-1-x)][self.rotation%4]

    def from_shown(self,x,y):
        ny,nx=self.array.shape
        col,row=[(x,y),(y,ny-1-x),(nx-1-x,ny-1-y),(nx-1-y,x)][self.rotation%4]
        return (nx-1-col if self.flip_x else col,ny-1-row if self.flip_y else row)

    def canvas_to_data(self, x: float, y: float) -> Optional[tuple[float, float]]:
        if self.array is None:
            return None
        x0, y0, x1, y1 = self.image_box
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            return None
        ny, nx = self.shown_shape()
        shown_col = (x - x0) / max(x1 - x0, 1) * nx - 0.5
        shown_row = (y - y0) / max(y1 - y0, 1) * ny - 0.5
        col,row=self.from_shown(shown_col,shown_row)
        ny,nx=self.array.shape
        return float(np.clip(col, 0, nx - 1)), float(np.clip(row, 0, ny - 1))

    def data_to_canvas(self, col: float, row: float) -> tuple[float, float]:
        x0, y0, x1, y1 = self.image_box
        ny, nx = self.shown_shape() if self.array is not None else (1, 1)
        shown_col,shown_row=self.to_shown(col,row)
        x = x0 + (shown_col + 0.5) / nx * (x1 - x0)
        y = y0 + (shown_row + 0.5) / ny * (y1 - y0)
        return x, y

    def reset_view(self, redraw: bool = True) -> None:
        self.zoom = 1.0
        if self.array is not None:
            ny, nx = self.array.shape
            self.view_center = ((nx - 1) / 2, (ny - 1) / 2)
        else:
            self.view_center = None
        if redraw:
            self.redraw()

    def _clamp_center(self) -> None:
        if self.array is None or self.view_center is None:
            return
        ny, nx = self.shown_shape()
        col, row = self.to_shown(*self.view_center)
        canvas_w = max(self.canvas.winfo_width() - (104 if self.colorbar_visible else 0), 10)
        canvas_h = max(self.canvas.winfo_height(), 10)
        scale = min(canvas_w / nx, canvas_h / ny) * self.zoom
        half_cols = canvas_w / (2 * scale) - 0.5
        half_rows = canvas_h / (2 * scale) - 0.5
        if 2 * half_cols >= nx - 1:
            col = (nx - 1) / 2
        else:
            col = float(np.clip(col, half_cols, nx - 1 - half_cols))
        if 2 * half_rows >= ny - 1:
            row = (ny - 1) / 2
        else:
            row = float(np.clip(row, half_rows, ny - 1 - half_rows))
        self.view_center = self.from_shown(col,row)

    def zoom_at(self, canvas_x: float, canvas_y: float, factor: float) -> None:
        if self.array is None:
            return
        point = self.canvas_to_data(canvas_x, canvas_y)
        if point is None:
            return
        old_zoom = self.zoom
        self.zoom = float(np.clip(self.zoom * factor, 1.0, 8.0))
        if abs(self.zoom - old_zoom) < 1e-12:
            return
        if self.zoom <= 1.0 + 1e-12:
            self.reset_view()
            return
        ny, nx = self.shown_shape()
        canvas_w = max(self.canvas.winfo_width() - (104 if self.colorbar_visible else 0), 10)
        canvas_h = max(self.canvas.winfo_height(), 10)
        scale = min(canvas_w / nx, canvas_h / ny) * self.zoom
        col, row = point
        shown_col,shown_row=self.to_shown(col,row)
        desired_x0 = canvas_x - (shown_col + 0.5) * scale
        desired_y0 = canvas_y - (shown_row + 0.5) * scale
        shown_center_col = (canvas_w / 2 - desired_x0) / scale - 0.5
        shown_center_row = (canvas_h / 2 - desired_y0) / scale - 0.5
        center_col,center_row=self.from_shown(shown_center_col,shown_center_row)
        self.view_center = (center_col, center_row)
        self._clamp_center()
        self.redraw()

    def _control_wheel(self, event: tk.Event) -> str:
        factor = 1.25 if event.delta > 0 else 1 / 1.25
        self.zoom_at(event.x, event.y, factor)
        return "break"

    def _pan_press(self, event: tk.Event) -> None:
        if self.array is None or self.view_center is None:
            return
        self._pan_start = (event.x, event.y, self.view_center[0], self.view_center[1])

    def _pan_drag(self, event: tk.Event) -> None:
        if self.array is None or self._pan_start is None:
            return
        start_x, start_y, center_col, center_row = self._pan_start
        x0, y0, x1, y1 = self.image_box
        ny, nx = self.shown_shape()
        center_col,center_row=self.to_shown(center_col,center_row)
        scale_x = max((x1 - x0) / nx, 1e-12)
        scale_y = max((y1 - y0) / ny, 1e-12)
        col_delta = (event.x - start_x) / scale_x
        new_col = center_col - col_delta
        row_delta = (event.y - start_y) / scale_y
        new_row = center_row - row_delta
        self.view_center = self.from_shown(new_col,new_row)
        self._clamp_center()
        self.redraw()


class STMFFTApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("STM Fourier Analyzer — SXM / complex FFT / iFFT")
        self.geometry("1580x920")
        self.minsize(1200, 720)

        self.rotation_quarters = 0
        self.rotation_label = tk.StringVar(value="旋转 0°")
        self.sxm: Optional[SxmFile] = None
        self.state = AnalysisState()
        self.processed: Optional[np.ndarray] = None
        self.fft_result = None
        self.reconstruction_fft_result = None
        self.mask: Optional[np.ndarray] = None
        self.reconstruction: Optional[np.ndarray] = None
        self.complex_component: Optional[np.ndarray] = None
        self.current_scan_nm = (1.0, 1.0)
        self.pending_crop: Optional[tuple[int, int, int, int]] = None
        self.click_action = "inspect"
        self._crop_start: Optional[tuple[float, float]] = None
        self._fft_drag_target: Optional[tuple[str, int, int]] = None
        self._keyboard_fft_target: Optional[tuple[str, int, int]] = None
        self.app_dir = _application_dir()
        self.ai_config_path = self.app_dir / "ai_settings.json"
        self.ai_env_path = self.app_dir / ".env.local"
        self.ai_config = load_ai_config(self.ai_config_path)
        self._session_api_key: Optional[str] = None
        self.ai_window: Optional[tk.Toplevel] = None
        self.ai_result_text: Optional[tk.Text] = None
        self.ai_question_text: Optional[tk.Text] = None
        self.ai_run_button: Optional[ttk.Button] = None

        self.channel_var = tk.StringVar()
        self.preprocess_var = tk.StringVar(value="稳健平面调平")
        self.window_var = tk.StringVar(value="Hann")
        self.real_view_var = tk.StringVar(value="线谱工具显示")
        self.real_palette_var = tk.StringVar(value="nanox")
        self.fft_view_var = tk.StringVar(value="对数振幅（dB）")
        self.result_var = tk.StringVar(value="成对 iFFT")
        self.fft_levels = {"low": 1.0, "high": 99.0, "limits": None}
        self._fft_levels_mode = None
        self.fft_contrast_window = None
        self.low_var = tk.DoubleVar(value=1.0)
        self.high_var = tk.DoubleVar(value=99.0)
        self.mask_radius_var = tk.DoubleVar(value=0.15)
        self.mask_shape_var = tk.StringVar(value="gaussian")
        self.mask_pair_var = tk.BooleanVar(value=True)
        self.phase_floor_var = tk.DoubleVar(value=15.0)
        self.c4_overlay_var = tk.BooleanVar(value=False)
        self.annotation_font_size_var = tk.IntVar(value=13)
        self.spatial_use_window_var = tk.BooleanVar(value=False)
        self.batch_amplitude_var = tk.BooleanVar(value=True)
        self.batch_phase_var = tk.BooleanVar(value=True)
        self.batch_reconstruction_var = tk.BooleanVar(value=False)
        self.batch_masked_fft_var = tk.BooleanVar(value=False)
        self.batch_mask_var = tk.BooleanVar(value=False)
        self.batch_enabled_only_var = tk.BooleanVar(value=True)
        self.composite_first_var = tk.StringVar()
        self.composite_second_var = tk.StringVar()
        self.composite_method_var = tk.StringVar(value="共同振幅 · 调和平均")
        self.composite_normalize_var = tk.BooleanVar(value=True)
        self.composite_caution_var = tk.StringVar(value="请先在 FFT 工作区添加两个 Q 点。")
        self.composite_data = None
        self._composite_choices: list[tuple[int, str]] = []
        self.last_export_dir = self.app_dir
        self.scale_visible_var = tk.BooleanVar(value=True)
        self.scale_style_var = tk.StringVar(value="blackwhite")
        self.status_var = tk.StringVar(value="打开一个 SXM 文件开始。")
        self.meta_var = tk.StringVar(value="")
        self.bragg_var = tk.StringVar(value="G1、G2 尚未标定")

        self._build_ui()

    def _build_ui(self) -> None:
        self._build_menu()
        toolbar = ttk.Frame(self, padding=(7, 5))
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="打开 SXM", command=self.open_sxm).pack(side="left", padx=2)
        ttk.Button(toolbar, text="保存项目", command=self.save_project).pack(side="left", padx=2)
        ttk.Button(toolbar, text="载入项目", command=self.load_project).pack(side="left", padx=2)
        ttk.Button(toolbar, text="导出分析", command=self.export_analysis).pack(side="left", padx=2)
        ttk.Button(toolbar, text="重置视图", command=self.reset_all_views).pack(side="left", padx=2)
        self.scale_eye_button = ttk.Button(toolbar, text="👁 标尺", command=self.toggle_scale_bar)
        self.scale_eye_button.pack(side="left", padx=2)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Button(toolbar, text="AI 分析", command=self.run_ai_analysis).pack(side="left", padx=2)
        ttk.Button(toolbar, text="AI 设置", command=self.open_ai_settings).pack(side="left", padx=2)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Label(toolbar, textvariable=self.meta_var).pack(side="left", fill="x", expand=True)

        body = ttk.Panedwindow(self, orient="horizontal")
        body.pack(fill="both", expand=True)
        controls = ttk.Frame(body, width=295, padding=6)
        displays = ttk.Frame(body)
        body.add(controls, weight=0)
        body.add(displays, weight=1)

        notebook = ttk.Notebook(controls)
        self.controls_notebook = notebook
        notebook.pack(fill="both", expand=True)
        data_tab = ttk.Frame(notebook, padding=10)
        fft_tab = ttk.Frame(notebook, padding=10)
        mask_tab = ttk.Frame(notebook, padding=10)
        composite_tab = ttk.Frame(notebook, padding=10)
        notebook.add(data_tab, text="数据")
        notebook.add(fft_tab, text="FFT / Bragg")
        notebook.add(mask_tab, text="Mask / iFFT")
        notebook.add(composite_tab, text="合成振幅")
        self._build_data_controls(data_tab)
        self._build_fft_controls(fft_tab)
        self._build_mask_controls(mask_tab)
        self._build_composite_controls(composite_tab)

        self.workspace_notebook = ttk.Notebook(displays)
        self.workspace_notebook.pack(fill="both", expand=True)
        analysis_workspace = ttk.Frame(self.workspace_notebook)
        composite_workspace = ttk.Frame(self.workspace_notebook)
        self.workspace_notebook.add(analysis_workspace, text="FFT 分析")
        self.workspace_notebook.add(composite_workspace, text="合成分析")
        for workspace in (analysis_workspace, composite_workspace):
            workspace.columnconfigure((0, 1, 2), weight=1, uniform="display")
            workspace.rowconfigure(0, weight=1)
        self.real_panel = ImagePanel(analysis_workspace, "实空间", self._real_motion)
        self.fft_panel = ImagePanel(analysis_workspace, "复数 FFT", self._fft_motion)
        self.result_panel = ImagePanel(analysis_workspace, "重建 / 局域量", self._result_motion)
        rotations=ttk.Frame(analysis_workspace)
        rotations.grid(row=1,column=2,sticky="ew")
        ttk.Button(rotations,text="↶ 90°",command=lambda:self.rotate_views(-1)).pack(side="left")
        ttk.Button(rotations,text="↷ 90°",command=lambda:self.rotate_views(1)).pack(side="left")
        ttk.Label(rotations,textvariable=self.rotation_label).pack(side="left",padx=5)
        self.real_panel.grid(row=0, column=0, sticky="nsew")
        ttk.Button(analysis_workspace, text="使用线谱工具的显示 / 衬度", command=self.use_line_display).grid(row=1, column=0, sticky="ew")
        self.fft_panel.grid(row=0, column=1, sticky="nsew")
        ttk.Button(analysis_workspace, text="FFT 衬度 / 色阶…", command=self.open_fft_contrast).grid(row=1, column=1, sticky="ew")
        self.result_panel.grid(row=0, column=2, sticky="nsew")
        self.composite_first_panel = ImagePanel(composite_workspace, "Q₁ 局域振幅")
        self.composite_second_panel = ImagePanel(composite_workspace, "Q₂ 局域振幅")
        self.composite_result_panel = ImagePanel(composite_workspace, "合成振幅")
        self.composite_first_panel.grid(row=0, column=0, sticky="nsew")
        self.composite_second_panel.grid(row=0, column=1, sticky="nsew")
        self.composite_result_panel.grid(row=0, column=2, sticky="nsew")
        self.fft_panel.canvas.bind("<ButtonPress-1>", self._fft_press)
        self.fft_panel.canvas.bind("<B1-Motion>", self._fft_drag)
        self.fft_panel.canvas.bind("<ButtonRelease-1>", self._fft_release)
        for key in ("Left", "Right", "Up", "Down"):
            self.fft_panel.canvas.bind(f"<KeyPress-{key}>", self._nudge_selected_mask)
        self.real_panel.canvas.bind("<ButtonPress-1>", self._crop_press)
        self.real_panel.canvas.bind("<B1-Motion>", self._crop_drag)
        self.real_panel.canvas.bind("<ButtonRelease-1>", self._crop_release)
        notebook.bind(
            "<<NotebookTabChanged>>",
            lambda _event: self.workspace_notebook.select(
                composite_workspace if notebook.select() == str(composite_tab) else analysis_workspace
            ),
        )

        status = ttk.Label(self, textvariable=self.status_var, relief="sunken", anchor="w", padding=(6, 3))
        status.pack(fill="x")

    def _build_menu(self) -> None:
        menu = tk.Menu(self)
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="导入 SXM…", command=self.open_sxm, accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="导出图片…", command=lambda: self.open_export_image_dialog(False))
        file_menu.add_command(label="导出带圆圈和标注的图片…", command=lambda: self.open_export_image_dialog(True))
        file_menu.add_command(label="一键导出全部 Q 结果…", command=self.batch_export_all_q)
        file_menu.add_command(label="导出合成分析图片…", command=lambda: self.open_export_image_dialog(composite=True))
        file_menu.add_command(label="导出完整分析包…", command=self.export_analysis)
        file_menu.add_separator()
        file_menu.add_command(label="保存项目…", command=self.save_project)
        file_menu.add_command(label="载入项目…", command=self.load_project)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.destroy)
        menu.add_cascade(label="文件", menu=file_menu)

        view_menu = tk.Menu(menu, tearoff=False)
        view_menu.add_checkbutton(label="显示实空间标尺", variable=self.scale_visible_var, command=self.update_panel_decorations)
        style_menu = tk.Menu(view_menu, tearoff=False)
        for label, value in (("黑白描边", "blackwhite"), ("白色", "white"), ("黑色", "black")):
            style_menu.add_radiobutton(label=label, value=value, variable=self.scale_style_var, command=self.update_panel_decorations)
        view_menu.add_cascade(label="标尺颜色", menu=style_menu)
        view_menu.add_command(label="重置缩放和平移", command=self.reset_all_views)
        menu.add_cascade(label="视图", menu=view_menu)

        ai_menu = tk.Menu(menu, tearoff=False)
        ai_menu.add_command(label="分析当前视图", command=self.run_ai_analysis)
        ai_menu.add_command(label="设置", command=self.open_ai_settings)
        menu.add_cascade(label="AI", menu=ai_menu)
        self.config(menu=menu)
        self.bind_all("<Control-o>", lambda _event: self.open_sxm())

    def _labeled_combo(self, parent: tk.Misc, label: str, variable: tk.StringVar, values: list[str], row: int) -> ttk.Combobox:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(7, 2))
        combo = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly")
        combo.grid(row=row + 1, column=0, sticky="ew")
        return combo

    def _build_data_controls(self, tab: ttk.Frame) -> None:
        tab.columnconfigure(0, weight=1)
        self.channel_combo = self._labeled_combo(tab, "通道", self.channel_var, [], 0)
        self.channel_combo.bind("<<ComboboxSelected>>", lambda _e: self._channel_changed())
        combo = self._labeled_combo(tab, "预处理", self.preprocess_var, list(PREPROCESS_OPTIONS), 2)
        combo.bind("<<ComboboxSelected>>", lambda _e: self.update_analysis())
        view = self._labeled_combo(tab, "实空间显示", self.real_view_var, ["线谱工具显示", "原始", "处理后"], 4)
        view.bind("<<ComboboxSelected>>", lambda _e: self.update_views())
        palette = self._labeled_combo(tab, "实空间色表", self.real_palette_var, ["nanox", "copper", "gray", "viridis", "inferno"], 6)
        palette.bind("<<ComboboxSelected>>", lambda _e: self.update_views())

        contrast = ttk.LabelFrame(tab, text="显示衬度（百分位）", padding=7)
        contrast.grid(row=8, column=0, sticky="ew", pady=10)
        contrast.columnconfigure((0, 1), weight=1)
        ttk.Label(contrast, text="低").grid(row=0, column=0)
        ttk.Label(contrast, text="高").grid(row=0, column=1)
        ttk.Spinbox(contrast, from_=0, to=49, increment=0.5, textvariable=self.low_var, width=8, command=self.update_views).grid(row=1, column=0)
        ttk.Spinbox(contrast, from_=51, to=100, increment=0.5, textvariable=self.high_var, width=8, command=self.update_views).grid(row=1, column=1)
        ttk.Button(contrast, text="应用显示衬度", command=self.update_views).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        crop = ttk.LabelFrame(tab, text="裁剪", padding=7)
        crop.grid(row=9, column=0, sticky="ew", pady=6)
        crop.columnconfigure((0, 1), weight=1)
        ttk.Button(crop, text="在实空间框选", command=lambda: self._set_action("crop")).grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Button(crop, text="应用框选", command=self.apply_crop).grid(row=1, column=0, sticky="ew", pady=4)
        ttk.Button(crop, text="恢复整图", command=self.reset_crop).grid(row=1, column=1, sticky="ew", pady=4)
        ttk.Label(crop, text="裁剪会改变 q 分辨率，物理扫描尺寸按像素比例更新。", wraplength=235).grid(row=2, column=0, columnspan=2, sticky="w")

        scale = ttk.LabelFrame(tab, text="长度标尺", padding=7)
        scale.grid(row=10, column=0, sticky="ew", pady=6)
        scale.columnconfigure(1, weight=1)
        ttk.Button(scale, text="👁", width=4, command=self.toggle_scale_bar).grid(row=0, column=0, padx=(0, 5))
        ttk.Combobox(scale, textvariable=self.scale_style_var, values=["blackwhite", "white", "black"], state="readonly").grid(row=0, column=1, sticky="ew")
        ttk.Button(scale, text="应用", command=self.update_panel_decorations).grid(row=0, column=2, padx=(5, 0))

    def _build_fft_controls(self, tab: ttk.Frame) -> None:
        tab.columnconfigure(0, weight=1)
        combo = self._labeled_combo(tab, "窗函数", self.window_var, list(WINDOW_OPTIONS), 0)
        combo.bind("<<ComboboxSelected>>", lambda _e: self.update_analysis())
        fft_view = self._labeled_combo(tab, "FFT 显示", self.fft_view_var, list(FFT_VIEW_OPTIONS), 2)
        fft_view.bind("<<ComboboxSelected>>", lambda _e: self.update_views())

        ttk.Button(tab, text="FFT 衬度 / 黑白位…", command=self.open_fft_contrast).grid(row=6, column=0, sticky="ew", pady=8)

        bragg = ttk.LabelFrame(tab, text="Bragg 标定", padding=7)
        bragg.grid(row=4, column=0, sticky="ew", pady=10)
        bragg.columnconfigure((0, 1), weight=1)
        ttk.Button(bragg, text="下次点击设 G1", command=lambda: self._set_action("g1")).grid(row=0, column=0, sticky="ew")
        ttk.Button(bragg, text="下次点击设 G2", command=lambda: self._set_action("g2")).grid(row=0, column=1, sticky="ew")
        ttk.Button(bragg, text="清除标定", command=self.clear_bragg).grid(row=1, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Checkbutton(bragg, text="叠加 c(4×4) 允许点", variable=self.c4_overlay_var, command=self.update_fft_overlays).grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Label(bragg, textvariable=self.bragg_var, wraplength=235).grid(row=3, column=0, columnspan=2, sticky="w", pady=(5, 0))

        ttk.Label(
            tab,
            text="FFT 光标会报告 qx、qy、|q|、周期；完成 G1/G2 标定后同时报告分数坐标 (h,k)。",
            wraplength=245,
        ).grid(row=5, column=0, sticky="w", pady=8)

        annotation = ttk.LabelFrame(tab, text="倒空间标注", padding=7)
        annotation.grid(row=6, column=0, sticky="ew", pady=(4, 0))
        annotation.columnconfigure(1, weight=1)
        ttk.Label(annotation, text="固定字号 (px)").grid(row=0, column=0, sticky="w", padx=(0, 6))
        font_spin = ttk.Spinbox(
            annotation,
            from_=7,
            to=72,
            increment=1,
            textvariable=self.annotation_font_size_var,
            width=7,
            command=self.update_annotation_style,
        )
        font_spin.grid(row=0, column=1, sticky="ew")
        font_spin.bind("<Return>", lambda _event: self.update_annotation_style())
        font_spin.bind("<FocusOut>", lambda _event: self.update_annotation_style())
        ttk.Label(annotation, text="屏幕和带标注导出图使用同一固定值，不随缩放改变。", wraplength=225).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(5, 0)
        )

    def _build_mask_controls(self, tab: ttk.Frame) -> None:
        tab.columnconfigure(0, weight=1)
        settings = ttk.LabelFrame(tab, text="新 mask", padding=7)
        settings.grid(row=0, column=0, sticky="ew")
        settings.columnconfigure((0, 1), weight=1)
        ttk.Label(settings, text="半径 (1/nm)").grid(row=0, column=0, sticky="w")
        ttk.Entry(settings, textvariable=self.mask_radius_var, width=9).grid(row=1, column=0, sticky="ew")
        ttk.Label(settings, text="边缘").grid(row=0, column=1, sticky="w")
        ttk.Combobox(settings, textvariable=self.mask_shape_var, values=["gaussian", "cosine", "hard"], state="readonly", width=10).grid(row=1, column=1, sticky="ew")
        ttk.Checkbutton(settings, text="自动加入 −Q 共轭点", variable=self.mask_pair_var).grid(row=2, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Button(settings, text="点选 FFT 添加 mask", command=lambda: self._set_action("mask")).grid(row=3, column=0, columnspan=2, sticky="ew")

        ttk.Label(tab, text="已选峰").grid(row=1, column=0, sticky="w", pady=(10, 2))
        self.mask_list = tk.Listbox(tab, height=10, exportselection=False)
        self.mask_list.grid(row=2, column=0, sticky="nsew")
        self.mask_list.bind("<<ListboxSelect>>", self._mask_selection_changed)
        tab.rowconfigure(2, weight=1)
        buttons = ttk.Frame(tab)
        buttons.grid(row=3, column=0, sticky="ew", pady=4)
        buttons.columnconfigure((0, 1, 2), weight=1)
        ttk.Button(buttons, text="启用/停用", command=self.toggle_mask).grid(row=0, column=0, sticky="ew")
        ttk.Button(buttons, text="删除", command=self.delete_mask).grid(row=0, column=1, sticky="ew")
        ttk.Button(buttons, text="清空", command=self.clear_masks).grid(row=0, column=2, sticky="ew")

        result = self._labeled_combo(tab, "结果显示", self.result_var, list(RESULT_OPTIONS), 4)
        result.bind("<<ComboboxSelected>>", lambda _e: self.update_result())
        ttk.Label(tab, text="相位图屏蔽低振幅区域（百分位）").grid(row=6, column=0, sticky="w", pady=(8, 2))
        ttk.Scale(tab, from_=0, to=80, variable=self.phase_floor_var, orient="horizontal", command=lambda _v: self.update_result()).grid(row=7, column=0, sticky="ew")
        ttk.Checkbutton(
            tab,
            text="iFFT/振幅也使用窗函数（会压低边缘）",
            variable=self.spatial_use_window_var,
            command=self.update_result,
        ).grid(row=8, column=0, sticky="w", pady=(9, 0))
        ttk.Button(tab, text="统计当前 Q 的相位分布…", command=self.open_phase_histogram).grid(row=9, column=0, sticky="ew", pady=(9, 0))
        ttk.Label(tab, text="点击 FFT 中的圆圈后，可用方向键按 1 个 FFT 像素微调。", wraplength=245).grid(row=10, column=0, sticky="w", pady=(5, 0))

        batch = ttk.LabelFrame(tab, text="一键导出全部 Q", padding=7)
        batch.grid(row=11, column=0, sticky="ew", pady=(9, 0))
        batch.columnconfigure((0, 1), weight=1)
        ttk.Checkbutton(batch, text="局域振幅", variable=self.batch_amplitude_var).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(batch, text="局域相位", variable=self.batch_phase_var).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(batch, text="成对 iFFT", variable=self.batch_reconstruction_var).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(batch, text="mask 后 FFT", variable=self.batch_masked_fft_var).grid(row=1, column=1, sticky="w")
        ttk.Checkbutton(batch, text="频率 mask", variable=self.batch_mask_var).grid(row=2, column=0, sticky="w")
        ttk.Checkbutton(batch, text="仅启用点", variable=self.batch_enabled_only_var).grid(row=2, column=1, sticky="w")
        ttk.Button(batch, text="选择文件夹并全部导出…", command=self.batch_export_all_q).grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(7, 0)
        )

    def _build_composite_controls(self, tab: ttk.Frame) -> None:
        tab.columnconfigure(0, weight=1)
        ttk.Label(tab, text="选择两个已启用的 Q 点；右侧分别显示它们的局域振幅和合成结果。", wraplength=245).grid(
            row=0, column=0, sticky="w"
        )
        self.composite_first_combo = self._labeled_combo(tab, "第一个 Q", self.composite_first_var, [], 1)
        self.composite_second_combo = self._labeled_combo(tab, "第二个 Q", self.composite_second_var, [], 3)
        for combo in (self.composite_first_combo, self.composite_second_combo):
            combo.bind("<<ComboboxSelected>>", lambda _event: self.update_composite_workspace())
        method = self._labeled_combo(tab, "合成方式", self.composite_method_var, list(COMPOSITE_METHODS), 5)
        method.bind("<<ComboboxSelected>>", lambda _event: self.update_composite_workspace())
        ttk.Checkbutton(
            tab,
            text="各自除以振幅第 90 百分位（推荐）",
            variable=self.composite_normalize_var,
            command=self.update_composite_workspace,
        ).grid(row=7, column=0, sticky="w", pady=(12, 2))
        ttk.Label(
            tab,
            text="调和平均 2A₁A₂/(A₁+A₂)、几何均值 √(A₁A₂) 与 min(A₁,A₂) 都要求两点有信号；"
            "平方平均（RMS）√[(A₁²+A₂²)/2] 更偏向强峰。"
            "这里先分别取 |复包络|，不对不同 Q 的复振幅或相位直接相加。",
            wraplength=245,
        ).grid(row=8, column=0, sticky="w", pady=(4, 10))
        ttk.Button(tab, text="导出合成工作区图片…", command=lambda: self.open_export_image_dialog(composite=True)).grid(
            row=9, column=0, sticky="ew", pady=3
        )
        ttk.Button(tab, text="导出三个数值图 NPZ…", command=self.export_composite_data).grid(
            row=10, column=0, sticky="ew", pady=3
        )
        ttk.Label(
            tab,
            text="合成振幅只展示两个傅里叶分量的空间共现；不能单凭此图证明结构对称性、PDW 或超导序参量。",
            wraplength=245,
        ).grid(row=11, column=0, sticky="w", pady=(10, 0))
        ttk.Label(tab, textvariable=self.composite_caution_var, wraplength=245, foreground="#9a5300").grid(
            row=12, column=0, sticky="w", pady=(8, 0)
        )

    def open_sxm(self, path: Optional[str] = None) -> None:
        if path is None:
            path = filedialog.askopenfilename(filetypes=[("Nanonis SXM", "*.sxm *.SXM"), ("所有文件", "*.*")])
        if not path:
            return
        try:
            sxm = read_sxm(path)
        except Exception as exc:
            messagebox.showerror("无法读取 SXM", str(exc))
            return
        self.sxm = sxm
        self.state = AnalysisState(source_path=str(Path(path).resolve()))
        labels = [channel.label for channel in sxm.channels]
        self.channel_combo.configure(values=labels)
        self.channel_var.set(labels[0])
        self.composite_first_var.set("")
        self.composite_second_var.set("")
        self.preprocess_var.set("稳健平面调平")
        self.window_var.set("Hann")
        scan = sxm.scan_range_nm or (float(sxm.scan_pixels[0]), float(sxm.scan_pixels[1]))
        self.meta_var.set(f"{Path(path).name}  |  {sxm.scan_pixels[0]}×{sxm.scan_pixels[1]} px  |  {scan[0]:.3g}×{scan[1]:.3g} nm")
        self.refresh_mask_list()
        self.update_analysis()

    def _channel_changed(self) -> None:
        if not self.sxm:
            return
        self.state.channel_index = max(0, self.channel_combo.current())
        self.state.crop = None
        self.pending_crop = None
        self.update_analysis()

    def _current_raw(self) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        if not self.sxm:
            raise ValueError("尚未打开 SXM。")
        image = self.sxm.channels[self.state.channel_index].data
        ny, nx = image.shape
        crop = self.state.crop or (0, 0, nx, ny)
        x0, y0, x1, y1 = crop
        return image[y0:y1, x0:x1], crop

    def update_analysis(self) -> None:
        if not self.sxm:
            return
        try:
            raw, crop = self._current_raw()
            method = PREPROCESS_OPTIONS[self.preprocess_var.get()]
            window = WINDOW_OPTIONS[self.window_var.get()]
            self.state.preprocess = method
            self.state.window = window
            processed, finite_fraction = preprocess_image(raw, method)
            full_range = self.sxm.scan_range_nm or (float(self.sxm.scan_pixels[0]), float(self.sxm.scan_pixels[1]))
            full_nx, full_ny = self.sxm.scan_pixels
            width_nm = full_range[0] * processed.shape[1] / full_nx
            height_nm = full_range[1] * processed.shape[0] / full_ny
            self.current_scan_nm = (width_nm, height_nm)
            self.processed = processed
            self.fft_result = compute_fft(processed, width_nm, height_nm, window)
            # Use the selected window to make the FFT easier to inspect, but
            # keep an unwindowed spectrum for spatial reconstruction. A Hann
            # or Tukey window is strongest in the center and would otherwise
            # create an artificial central amplitude maximum in iFFT maps.
            self.reconstruction_fft_result = compute_fft(processed, width_nm, height_nm, "none")
            self.update_views()
            finite_text = f"；有限像素 {finite_fraction * 100:.1f}%" if finite_fraction < 0.999 else ""
            self.status_var.set(f"分析区域 {processed.shape[1]}×{processed.shape[0]} px，{width_nm:.3g}×{height_nm:.3g} nm{finite_text}")
        except Exception as exc:
            messagebox.showerror("分析失败", str(exc))

    def rotate_views(self,step):
        self.rotation_quarters=(self.rotation_quarters+step)%4
        self.sync_rotation()
        self.status_var.set("图像与 FFT 同步旋转；Q 数值保持原数据坐标系。")

    def sync_rotation(self):
        self.rotation_label.set(f"旋转 {self.rotation_quarters*90}°")
        for panel in (self.real_panel,self.fft_panel,self.result_panel,self.composite_first_panel,self.composite_second_panel,self.composite_result_panel):
            panel.rotation=self.rotation_quarters
            panel.reset_view()
        self.update_panel_decorations()

    def use_line_display(self):
        self.real_view_var.set("线谱工具显示");self.real_palette_var.set("nanox")
        self.low_var.set(1);self.high_var.set(99);self.update_views()

    def _spatial_flip_y(self) -> bool:
        # Keep stored data and crop/mask coordinates intact; change display only.
        return bool(self.sxm and self.sxm.header_fields.get("SCAN_DIR", [""])[0].strip().lower()=="up")

    def _real_display_array(self):
        raw,_=self._current_raw()
        if self.real_view_var.get()=="原始":return raw
        if self.real_view_var.get()!="线谱工具显示":return self.processed
        channel=self.sxm.channels[self.state.channel_index]
        if channel.unit.strip().lower()!='m':return raw
        # Match the line-spectrum viewer's ordinary plane subtraction, display only.
        ny,nx=raw.shape;y,x=np.indices(raw.shape);finite=np.isfinite(raw)
        A=np.column_stack([np.ones(raw.size),x.ravel(),y.ravel()])
        if finite.sum()<3:return raw
        fit=np.linalg.lstsq(A[finite.ravel()],raw[finite],rcond=None)[0]
        return raw-(A@fit).reshape(raw.shape)

    def update_views(self) -> None:
        if not self.sxm or self.processed is None or self.fft_result is None:
            return
        raw, _crop = self._current_raw()
        real = self._real_display_array()
        low, high = self._contrast()
        channel = self.sxm.channels[self.state.channel_index]
        self.real_panel.set_data(
            real,
            palette=self.real_palette_var.get(),
            flip_y=self._spatial_flip_y(),
            low=low,
            high=high,
            title=f"实空间：{channel.name} {channel.direction}",
        )
        fft_image, palette, limits = self._fft_display_array()
        self.fft_panel.set_data(
            fft_image,
            palette=palette,
            low=low,
            high=high,
            limits=limits,
            flip_x=False,
            flip_y=True,
            title=f"FFT：{self.fft_view_var.get()}（q 单位 1/nm）",
        )
        self.update_fft_overlays()
        self.update_result()

    def toggle_scale_bar(self) -> None:
        self.scale_visible_var.set(not bool(self.scale_visible_var.get()))
        self.update_panel_decorations()

    def _annotation_font_size(self) -> int:
        try:
            size = int(self.annotation_font_size_var.get())
        except (tk.TclError, ValueError, TypeError):
            size = 13
        size = int(np.clip(size, 7, 72))
        self.annotation_font_size_var.set(size)
        return size

    def update_annotation_style(self) -> None:
        if not hasattr(self, "fft_panel"):
            return
        self.fft_panel.set_overlay_font_size(self._annotation_font_size())

    def update_panel_decorations(self) -> None:
        if not hasattr(self, "real_panel"):
            return
        visible = bool(self.scale_visible_var.get())
        style = self.scale_style_var.get()
        self.scale_eye_button.configure(text=("👁 标尺" if visible else "◌ 标尺"))
        self.real_panel.set_scale_bar(visible, self.current_scan_nm[self.rotation_quarters%2], "nm", style)
        self.fft_panel.set_scale_bar(False)

        mode = RESULT_OPTIONS.get(self.result_var.get(), "reconstruction")
        spatial = mode in {"reconstruction", "amplitude", "phase"}
        self.result_panel.set_scale_bar(visible and spatial, self.current_scan_nm[self.rotation_quarters%2], "nm", style)

        self.real_panel.set_colorbar(False)
        self.fft_panel.set_colorbar(False)
        result_ticks = self._panel_ticks(self.result_panel, phase=(mode == "phase"))
        result_label = {
            "reconstruction": "iFFT",
            "amplitude": "振幅",
            "phase": "相位 (rad)",
            "mask": "mask",
            "masked_fft": "log|FFT|",
        }[mode]
        self.result_panel.set_colorbar(True, result_label, result_ticks)
        unit = "相对振幅" if self.composite_normalize_var.get() else "原始振幅"
        for panel in (self.composite_first_panel, self.composite_second_panel, self.composite_result_panel):
            panel.set_scale_bar(visible, self.current_scan_nm[self.rotation_quarters%2], "nm", style)
            panel.set_colorbar(panel.array is not None, unit, self._panel_ticks(panel))

    def _panel_ticks(self, panel: ImagePanel, phase: bool = False) -> tuple[str, str, str]:
        if phase:
            return ("+π", "0", "−π")
        if panel.array is None:
            return ("", "", "")
        lower, upper = panel.limits or robust_limits(panel.array, panel.low, panel.high)
        return (_format_number(upper), _format_number((upper + lower) / 2), _format_number(lower))

    def _contrast(self) -> tuple[float, float]:
        low = float(self.low_var.get())
        high = float(self.high_var.get())
        if high <= low:
            high = min(100.0, low + 1.0)
        return low, high

    def open_fft_contrast(self):
        from .fft_contrast import FFTContrast
        if self.fft_contrast_window is not None and self.fft_contrast_window.winfo_exists():
            self.fft_contrast_window.lift();self.fft_contrast_window.refresh();return
        self.fft_contrast_window = FFTContrast(self)

    def update_fft_contrast(self):
        if self.fft_result is None:return
        array,palette,limits=self._fft_display_array()
        self.fft_panel.set_data(array,palette=palette,limits=limits,flip_x=False,flip_y=True,
                                title=f"FFT：{self.fft_view_var.get()}（q 单位 1/nm）")
        self.update_fft_overlays()

    def _fft_display_array(self) -> tuple[np.ndarray, str, Optional[tuple[float, float]]]:
        array,palette,fixed=self._fft_base_display_array()
        mode=self.fft_view_var.get()
        if self._fft_levels_mode is not None and self._fft_levels_mode!=mode:
            self.fft_levels={"low":1.0,"high":99.0,"limits":None}
        self._fft_levels_mode=mode
        limits=fixed or self.fft_levels.get("limits")
        if limits is None:limits=robust_limits(array,self.fft_levels["low"],self.fft_levels["high"])
        return array,palette,tuple(limits)

    def _fft_base_display_array(self) -> tuple[np.ndarray, str, Optional[tuple[float, float]]]:
        spectrum = self.fft_result.spectrum
        mode = FFT_VIEW_OPTIONS[self.fft_view_var.get()]
        if mode == "amplitude_db":
            amplitude = np.abs(spectrum)
            finite = amplitude[np.isfinite(amplitude)]
            peak = float(finite.max()) if finite.size else 0.0
            if peak <= 0:
                return np.zeros_like(amplitude), "inferno", None
            # Display only: relative amplitude dB, floored at -240 dB.
            return 20 * np.log10(np.maximum(amplitude / peak, 1e-12)), "inferno", None
        if mode == "log_amplitude":
            return np.log1p(np.abs(spectrum)), "inferno", None
        if mode == "amplitude":
            return np.abs(spectrum), "inferno", None
        if mode == "power":
            return np.abs(spectrum) ** 2, "inferno", None
        if mode == "phase":
            return np.angle(spectrum), "phase", (-np.pi, np.pi)
        if mode == "real":
            return np.real(spectrum), "viridis", None
        return np.imag(spectrum), "viridis", None

    def _set_action(self, action: str) -> None:
        self.click_action = action
        messages = {
            "crop": "在左侧实空间图拖动鼠标框选裁剪区域。",
            "mask": "在中间 FFT 图点击一个峰添加 mask；可连续点击。",
            "g1": "在中间 FFT 图点击第一组 Bragg 点 G1。",
            "g2": "在中间 FFT 图点击第二组 Bragg 点 G2。",
        }
        self.status_var.set(messages.get(action, ""))

    def _real_motion(self, event: tk.Event) -> None:
        point = self.real_panel.canvas_to_data(event.x, event.y)
        if point is None or self.processed is None:
            return
        col, row = int(round(point[0])), int(round(point[1]))
        col = min(col, self.processed.shape[1] - 1)
        row = min(row, self.processed.shape[0] - 1)
        x_nm = col / self.processed.shape[1] * self.current_scan_nm[0]
        y_nm = row / self.processed.shape[0] * self.current_scan_nm[1]
        self.status_var.set(f"实空间 x={x_nm:.4g} nm, y={y_nm:.4g} nm, value={self.processed[row, col]:.5g}")

    def _fft_point(self, event: tk.Event) -> Optional[tuple[int, int, float, float]]:
        if self.fft_result is None:
            return None
        point = self.fft_panel.canvas_to_data(event.x, event.y)
        if point is None:
            return None
        col = int(np.clip(round(point[0]), 0, len(self.fft_result.qx) - 1))
        row = int(np.clip(round(point[1]), 0, len(self.fft_result.qy) - 1))
        return col, row, float(self.fft_result.qx[col]), float(self.fft_result.qy[row])

    def _fft_motion(self, event: tk.Event) -> None:
        point = self._fft_point(event)
        if point is None:
            return
        _col, _row, qx, qy = point
        magnitude = math.hypot(qx, qy)
        period = 1.0 / magnitude if magnitude > 0 else math.inf
        text = f"q=({qx:.4g}, {qy:.4g}) 1/nm，|q|={magnitude:.4g}，1/|q|={period:.4g} nm"
        if self.state.bragg_g1 and self.state.bragg_g2:
            try:
                h, k = fractional_coordinates((qx, qy), self.state.bragg_g1, self.state.bragg_g2)
                text += f"，(h,k)=({h:.4f}, {k:.4f})"
            except ValueError:
                pass
        self.status_var.set(text)

    def _fft_click(self, event: tk.Event) -> None:
        point = self._fft_point(event)
        if point is None:
            return
        _col, _row, qx, qy = point
        if self.click_action == "g1":
            self.state.bragg_g1 = (qx, qy)
            self._keyboard_fft_target = ("g1", 0, 1)
            self.click_action = "inspect"
            self._update_bragg_label()
            self.update_fft_overlays()
        elif self.click_action == "g2":
            self.state.bragg_g2 = (qx, qy)
            self._keyboard_fft_target = ("g2", 0, 1)
            self.click_action = "inspect"
            self._update_bragg_label()
            self.update_fft_overlays()
        elif self.click_action == "mask":
            try:
                radius = float(self.mask_radius_var.get())
                if radius <= 0:
                    raise ValueError
            except Exception:
                messagebox.showerror("Mask 半径", "请输入正的 mask 半径，单位为 1/nm。")
                return
            label = f"Q{len(self.state.masks) + 1}"
            self.state.masks.append(
                MaskSpec(
                    qx=qx,
                    qy=qy,
                    radius=radius,
                    shape=self.mask_shape_var.get(),
                    include_conjugate=bool(self.mask_pair_var.get()),
                    label=label,
                )
            )
            new_index = len(self.state.masks) - 1
            self._keyboard_fft_target = ("mask", new_index, 1)
            self.refresh_mask_list(select_index=new_index)
            self.update_fft_overlays()
            self.update_result()
        else:
            self._fft_motion(event)

    def _fft_press(self, event: tk.Event) -> None:
        self.fft_panel.canvas.focus_set()
        target = self._find_fft_drag_target(event.x, event.y)
        if target is not None:
            self._fft_drag_target = target
            self._keyboard_fft_target = target
            kind, index, sign = target
            if kind == "mask":
                self.mask_list.selection_clear(0, "end")
                self.mask_list.selection_set(index)
                self.mask_list.see(index)
                suffix = "−Q" if sign < 0 else "+Q"
                self.status_var.set(f"拖动 {self.state.masks[index].label} 的 {suffix}；松开后重新计算。")
            else:
                self.status_var.set(f"拖动 {kind.upper()}；松开后更新标定。")
            return
        self._fft_click(event)

    def _nudge_selected_mask(self, event: tk.Event) -> str:
        if self.fft_result is None:
            return "break"
        index = self._selected_mask_index()
        target = self._keyboard_fft_target
        if target is None or (target[0] == "mask" and (index is None or target[1] != index)):
            if index is None or index >= len(self.state.masks):
                self.status_var.set("请先在列表或 FFT 图中选中一个点。")
                return "break"
            target = ("mask", index, 1)
            self._keyboard_fft_target = target
        kind, target_index, sign = target
        if kind == "mask":
            if target_index >= len(self.state.masks):
                return "break"
            spec = self.state.masks[target_index]
            current_qx, current_qy = sign * spec.qx, sign * spec.qy
            label = ("−" if sign < 0 else "+") + spec.label
        elif kind == "g1" and self.state.bragg_g1 is not None:
            current_qx, current_qy = self.state.bragg_g1
            label = "G1"
        elif kind == "g2" and self.state.bragg_g2 is not None:
            current_qx, current_qy = self.state.bragg_g2
            label = "G2"
        else:
            return "break"
        col = int(np.argmin(np.abs(self.fft_result.qx - current_qx)))
        row = int(np.argmin(np.abs(self.fft_result.qy - current_qy)))
        visual_dx = {"Left": -1, "Right": 1}.get(event.keysym, 0)
        visual_dy = {"Up": -1, "Down": 1}.get(event.keysym, 0)
        origin=self.fft_panel.from_shown(0,0)
        target=self.fft_panel.from_shown(visual_dx,visual_dy)
        data_dx,data_dy=target[0]-origin[0],target[1]-origin[1]
        new_col = int(np.clip(col + data_dx, 0, len(self.fft_result.qx) - 1))
        new_row = int(np.clip(row + data_dy, 0, len(self.fft_result.qy) - 1))
        new_qx = float(self.fft_result.qx[new_col])
        new_qy = float(self.fft_result.qy[new_row])
        if kind == "mask":
            spec.qx = sign * new_qx
            spec.qy = sign * new_qy
            self.refresh_mask_list(select_index=target_index)
        elif kind == "g1":
            self.state.bragg_g1 = (new_qx, new_qy)
            self._update_bragg_label()
        else:
            self.state.bragg_g2 = (new_qx, new_qy)
            self._update_bragg_label()
        self.update_fft_overlays()
        if kind == "mask":
            self.update_result()
        self.fft_panel.canvas.focus_set()
        self.status_var.set(
            f"{label} 已沿 {event.keysym} 方向移动 1 个 FFT 像素：q=({new_qx:.5g}, {new_qy:.5g}) 1/nm"
        )
        return "break"

    def _find_fft_drag_target(self, canvas_x: float, canvas_y: float) -> Optional[tuple[str, int, int]]:
        if self.fft_result is None:
            return None
        candidates: list[tuple[float, tuple[str, int, int]]] = []

        def distance_to(qx_value: float, qy_value: float) -> float:
            col = int(np.argmin(np.abs(self.fft_result.qx - qx_value)))
            row = int(np.argmin(np.abs(self.fft_result.qy - qy_value)))
            x, y = self.fft_panel.data_to_canvas(col, row)
            return math.hypot(canvas_x - x, canvas_y - y)

        for index, spec in enumerate(self.state.masks):
            candidates.append((distance_to(spec.qx, spec.qy), ("mask", index, 1)))
            if spec.include_conjugate:
                candidates.append((distance_to(-spec.qx, -spec.qy), ("mask", index, -1)))
        if self.state.bragg_g1:
            candidates.append((distance_to(*self.state.bragg_g1), ("g1", 0, 1)))
        if self.state.bragg_g2:
            candidates.append((distance_to(*self.state.bragg_g2), ("g2", 0, 1)))
        if not candidates:
            return None
        distance, target = min(candidates, key=lambda item: item[0])
        return target if distance <= 18.0 else None

    def _fft_drag(self, event: tk.Event) -> None:
        if self._fft_drag_target is None:
            return
        point = self._fft_point(event)
        if point is None:
            return
        _col, _row, qx, qy = point
        kind, index, sign = self._fft_drag_target
        if kind == "mask":
            self.state.masks[index].qx = sign * qx
            self.state.masks[index].qy = sign * qy
        elif kind == "g1":
            self.state.bragg_g1 = (qx, qy)
            self._update_bragg_label()
        elif kind == "g2":
            self.state.bragg_g2 = (qx, qy)
            self._update_bragg_label()
        self.update_fft_overlays()

    def _fft_release(self, _event: tk.Event) -> None:
        if self._fft_drag_target is None:
            return
        kind, index, _sign = self._fft_drag_target
        self._fft_drag_target = None
        if kind == "mask":
            self.refresh_mask_list(select_index=index)
            self.update_result()
        self.update_fft_overlays()

    def update_fft_overlays(self) -> None:
        if self.fft_result is None:
            return
        qx, qy = self.fft_result.qx, self.fft_result.qy
        dqx = abs(qx[1] - qx[0]) if len(qx) > 1 else 1.0
        dqy = abs(qy[1] - qy[0]) if len(qy) > 1 else 1.0
        overlays: list[dict] = []

        def add_q(point_qx: float, point_qy: float, radius_q: float, color: str, text: str = "") -> None:
            if point_qx < qx[0] or point_qx > qx[-1] or point_qy < qy[0] or point_qy > qy[-1]:
                return
            col = int(np.argmin(np.abs(qx - point_qx)))
            row = int(np.argmin(np.abs(qy - point_qy)))
            radius_px = radius_q / max(0.5 * (dqx + dqy), 1e-12)
            overlays.append({"x": col, "y": row, "radius": max(radius_px, 2.5), "color": color, "text": text})

        for spec in self.state.masks:
            color = "#55ff88" if spec.enabled else "#777777"
            add_q(spec.qx, spec.qy, spec.radius, color, spec.label)
            if spec.include_conjugate:
                add_q(-spec.qx, -spec.qy, spec.radius, color, f"−{spec.label}")
        if self.state.bragg_g1:
            add_q(*self.state.bragg_g1, 2.5 * max(dqx, dqy), "#33ddff", "G1")
        if self.state.bragg_g2:
            add_q(*self.state.bragg_g2, 2.5 * max(dqx, dqy), "#33ddff", "G2")
        if self.c4_overlay_var.get() and self.state.bragg_g1 and self.state.bragg_g2:
            g1, g2 = self.state.bragg_g1, self.state.bragg_g2
            for h, k, hi, ki in c4x4_allowed_points(4):
                px = h * g1[0] + k * g2[0]
                py = h * g1[1] + k * g2[1]
                add_q(px, py, 1.5 * max(dqx, dqy), "#ffd84a", f"{hi},{ki}")
        self.fft_panel.overlay_font_size = self._annotation_font_size()
        self.fft_panel.set_overlays(overlays)

    def _update_bragg_label(self) -> None:
        def fmt(value: Optional[tuple[float, float]]) -> str:
            return "未设" if value is None else f"({value[0]:.4g}, {value[1]:.4g})"
        self.bragg_var.set(f"G1={fmt(self.state.bragg_g1)}\nG2={fmt(self.state.bragg_g2)}  1/nm")

    def clear_bragg(self) -> None:
        self.state.bragg_g1 = None
        self.state.bragg_g2 = None
        if self._keyboard_fft_target and self._keyboard_fft_target[0] in {"g1", "g2"}:
            self._keyboard_fft_target = None
        self._update_bragg_label()
        self.update_fft_overlays()

    def refresh_mask_list(self, select_index: Optional[int] = None) -> None:
        self.mask_list.delete(0, "end")
        for spec in self.state.masks:
            enabled = "●" if spec.enabled else "○"
            pair = "±Q" if spec.include_conjugate else "+Q"
            self.mask_list.insert("end", f"{enabled} {spec.label}  ({spec.qx:.3g},{spec.qy:.3g})  r={spec.radius:.3g}  {pair}")
        if self.state.masks:
            index = 0 if select_index is None else int(np.clip(select_index, 0, len(self.state.masks) - 1))
            self.mask_list.selection_set(index)
            self.mask_list.see(index)
        self.refresh_composite_choices()

    def _composite_choice_index(self, value: str) -> Optional[int]:
        return next((index for index, label in self._composite_choices if label == value), None)

    def refresh_composite_choices(self) -> None:
        first_index = self._composite_choice_index(self.composite_first_var.get())
        second_index = self._composite_choice_index(self.composite_second_var.get())
        self._composite_choices = [
            (index, f"{spec.label}  ({spec.qx:.3g}, {spec.qy:.3g}) 1/nm")
            for index, spec in enumerate(self.state.masks)
            if spec.enabled
        ]
        values = [label for _index, label in self._composite_choices]
        self.composite_first_combo.configure(values=values)
        self.composite_second_combo.configure(values=values)
        available = [index for index, _label in self._composite_choices]
        if first_index not in available:
            first_index = available[0] if available else None
        if second_index not in available or second_index == first_index:
            second_index = next((index for index in available if index != first_index), None)
        labels = dict(self._composite_choices)
        self.composite_first_var.set(labels.get(first_index, ""))
        self.composite_second_var.set(labels.get(second_index, ""))

    def update_composite_workspace(self) -> None:
        first_index = self._composite_choice_index(self.composite_first_var.get())
        second_index = self._composite_choice_index(self.composite_second_var.get())
        spatial_fft = self.fft_result if self.spatial_use_window_var.get() else self.reconstruction_fft_result
        if spatial_fft is None or first_index is None or second_index is None or first_index == second_index:
            self.composite_data = None
            self.composite_caution_var.set("请选择两个不同且已启用的 Q 点。")
            for panel in (self.composite_first_panel, self.composite_second_panel, self.composite_result_panel):
                panel.clear_data("请选择两个不同且已启用的 Q 点")
            self.update_panel_decorations()
            return
        first_spec = self.state.masks[first_index]
        second_spec = self.state.masks[second_index]
        separation = float(np.hypot(first_spec.qx - second_spec.qx, first_spec.qy - second_spec.qy))
        conjugate_separation = float(np.hypot(first_spec.qx + second_spec.qx, first_spec.qy + second_spec.qy))
        resolution = max(abs(float(spatial_fft.qx[1] - spatial_fft.qx[0])), abs(float(spatial_fft.qy[1] - spatial_fft.qy[0])))
        if conjugate_separation < resolution:
            caution = "注意：两点接近 ±Q 共轭对；实数图像的两张振幅图会近乎重复，不能当作两种独立周期。"
        elif separation < (first_spec.radius + second_spec.radius) / 2:
            caution = "注意：两个 mask 明显重叠，合成图的共同亮区可能来自同一倒空间信号。"
        else:
            caution = "请同时检查两个输入图、峰位和 mask 宽度；共同亮区只表示振幅共现。"
        self.composite_caution_var.set(caution)
        first = np.abs(local_complex_component(spatial_fft.spectrum, spatial_fft.qx, spatial_fft.qy, first_spec))
        second = np.abs(local_complex_component(spatial_fft.spectrum, spatial_fft.qx, spatial_fft.qy, second_spec))
        method = COMPOSITE_METHODS[self.composite_method_var.get()]
        self.composite_data = combine_local_amplitudes(
            first, second, method=method, normalize=bool(self.composite_normalize_var.get())
        )
        result = self.composite_data
        _low, high = self._contrast()
        shared_high = float(np.percentile(np.maximum(result.first, result.second), high))
        shared_limits = (0.0, shared_high if shared_high > 0 else 1.0)
        source_note = "加窗源" if self.spatial_use_window_var.get() else "未加窗源"
        self.composite_first_panel.set_data(
            result.first, palette="viridis", limits=shared_limits, flip_y=self._spatial_flip_y(),
            title=f"{first_spec.label} 局域振幅（{source_note}）",
        )
        self.composite_second_panel.set_data(
            result.second, palette="viridis", limits=shared_limits, flip_y=self._spatial_flip_y(),
            title=f"{second_spec.label} 局域振幅（{source_note}）",
        )
        self.composite_result_panel.set_data(
            result.combined, palette="viridis", low=0, high=high, flip_y=self._spatial_flip_y(),
            title=f"{first_spec.label}、{second_spec.label}：{self.composite_method_var.get()}",
        )
        self.update_panel_decorations()

    def _mask_selection_changed(self, _event: tk.Event) -> None:
        index = self._selected_mask_index()
        current = self._keyboard_fft_target
        if index is not None and (current is None or current[0] != "mask" or current[1] != index):
            self._keyboard_fft_target = ("mask", index, 1)
        self.update_result()

    def _selected_mask_index(self) -> Optional[int]:
        selection = self.mask_list.curselection()
        return int(selection[0]) if selection else None

    def toggle_mask(self) -> None:
        index = self._selected_mask_index()
        if index is None:
            return
        self.state.masks[index].enabled = not self.state.masks[index].enabled
        self.refresh_mask_list()
        self.mask_list.selection_set(index)
        self.update_fft_overlays()
        self.update_result()

    def delete_mask(self) -> None:
        index = self._selected_mask_index()
        if index is None:
            return
        del self.state.masks[index]
        self.refresh_mask_list()
        self.update_fft_overlays()
        self.update_result()

    def clear_masks(self) -> None:
        self.state.masks.clear()
        if self._keyboard_fft_target and self._keyboard_fft_target[0] == "mask":
            self._keyboard_fft_target = None
        self.refresh_mask_list()
        self.update_fft_overlays()
        self.update_result()

    def update_result(self) -> None:
        if self.fft_result is None or self.reconstruction_fft_result is None or self.processed is None:
            return
        specs = [spec for spec in self.state.masks if spec.enabled]
        self.mask = make_frequency_mask(self.fft_result.qx, self.fft_result.qy, specs)
        spatial_fft = self.fft_result if self.spatial_use_window_var.get() else self.reconstruction_fft_result
        source_note = "加窗源" if self.spatial_use_window_var.get() else "未加窗源"
        self.reconstruction = reconstruct_from_mask(spatial_fft.spectrum, self.mask)
        selected_index = self._selected_mask_index()
        selected = self.state.masks[selected_index] if selected_index is not None and selected_index < len(self.state.masks) else None
        if selected is None or not selected.enabled:
            selected = specs[0] if specs else None
        self.complex_component = (
            local_complex_component(spatial_fft.spectrum, spatial_fft.qx, spatial_fft.qy, selected)
            if selected is not None
            else np.zeros_like(self.processed, dtype=np.complex128)
        )

        mode = RESULT_OPTIONS[self.result_var.get()]
        low, high = self._contrast()
        if mode == "reconstruction":
            array, palette, limits, title = self.reconstruction, "copper", None, f"成对 mask 的实数 iFFT（{source_note}）"
        elif mode == "amplitude":
            array, palette, limits, title = np.abs(self.complex_component), "viridis", None, f"单侧复包络的局域振幅（{source_note}）"
        elif mode == "phase":
            amplitude = np.abs(self.complex_component)
            threshold = np.percentile(amplitude, float(self.phase_floor_var.get())) if amplitude.size else 0.0
            array = np.angle(self.complex_component).copy()
            array[amplitude < threshold] = np.nan
            palette, limits, title = "phase", (-np.pi, np.pi), f"单侧复包络的局域相位（{source_note}）"
        elif mode == "mask":
            array, palette, limits, title = self.mask, "gray", (0.0, 1.0), "倒空间 mask"
        else:
            array = np.log1p(np.abs(spatial_fft.spectrum * self.mask))
            palette, limits, title = "inferno", None, f"mask 后 FFT 振幅（{source_note}）"
        reciprocal = mode in {"mask", "masked_fft"}
        self.result_panel.set_data(
            array,
            palette=palette,
            low=low,
            high=high,
            limits=limits,
            flip_x=False,
            flip_y=True if reciprocal else self._spatial_flip_y(),
            title=title,
        )
        self.update_composite_workspace()
        self.update_panel_decorations()

    def _result_motion(self, event: tk.Event) -> None:
        point = self.result_panel.canvas_to_data(event.x, event.y)
        if point is None or self.result_panel.array is None:
            return
        col, row = int(round(point[0])), int(round(point[1]))
        row = int(np.clip(row, 0, self.result_panel.array.shape[0] - 1))
        col = int(np.clip(col, 0, self.result_panel.array.shape[1] - 1))
        value = self.result_panel.array[row, col]
        self.status_var.set(f"结果像素 ({col},{row})，value={value:.5g}")

    def open_phase_histogram(self) -> None:
        if self.complex_component is None or not any(spec.enabled for spec in self.state.masks):
            messagebox.showinfo("相位分布", "请先在 FFT 中选择并启用一个 Q 点。")
            return
        dialog = tk.Toplevel(self)
        dialog.title("当前 Q 的相位分布")
        dialog.transient(self)
        bins_var = tk.IntVar(value=36)
        weighted_var = tk.BooleanVar(value=False)
        histogram_canvas = tk.Canvas(dialog, width=780, height=430, highlightthickness=0, cursor="sb_h_double_arrow")
        histogram_canvas.pack(fill="both", expand=True, padx=10, pady=(10, 4))
        stats_var = tk.StringVar()
        ttk.Label(dialog, textvariable=stats_var, justify="left").pack(fill="x", padx=12, pady=4)
        controls = ttk.Frame(dialog, padding=(10, 5, 10, 10))
        controls.pack(fill="x")
        ttk.Label(controls, text="bin 数").pack(side="left")
        ttk.Spinbox(controls, from_=8, to=180, textvariable=bins_var, width=6).pack(side="left", padx=(4, 10))
        ttk.Checkbutton(controls, text="按局域振幅加权", variable=weighted_var).pack(side="left")
        state: dict = {"phase_shift": 0.0, "drag_x": None, "drag_start_shift": 0.0}

        def wrap_phase(value: float) -> float:
            return float((value + np.pi) % (2 * np.pi) - np.pi)

        def render_display() -> None:
            if "result" not in state:
                return
            result = state["result"]
            shift = float(state["phase_shift"])
            image = self._phase_histogram_image(result, shift)
            photo = ImageTk.PhotoImage(image)
            histogram_canvas.delete("all")
            histogram_canvas.create_image(0, 0, image=photo, anchor="nw")
            histogram_canvas.image = photo
            state["image"] = image
            stats_var.set(
                f"纳入像素：{result['pixel_count']}；振幅阈值：{_format_number(result['threshold'])}（第 {float(self.phase_floor_var.get()):g} 百分位）\n"
                f"显示平移 Δφ：{shift:+.4g} rad\n"
                f"相位相干度 R：{result['resultant_length']:.4f}；圆标准差：{result['circular_std']:.4g} rad。水平拖动仅移动相位零点，原始复数数据不变。"
            )

        def refresh() -> None:
            try:
                result = phase_histogram(
                    self.complex_component,
                    amplitude_floor_percentile=float(self.phase_floor_var.get()),
                    bins=int(bins_var.get()),
                    amplitude_weighted=bool(weighted_var.get()),
                )
                state["result"] = result
                render_display()
            except Exception as exc:
                messagebox.showerror("相位统计失败", str(exc), parent=dialog)

        def drag_start(event: tk.Event) -> None:
            state["drag_x"] = float(event.x)
            state["drag_start_shift"] = float(state["phase_shift"])

        def drag_motion(event: tk.Event) -> None:
            if state["drag_x"] is None:
                return
            plot_width = 748 - 72
            delta = (float(event.x) - float(state["drag_x"])) / plot_width * 2 * np.pi
            state["phase_shift"] = wrap_phase(float(state["drag_start_shift"]) + delta)
            render_display()

        def drag_end(_event: tk.Event) -> None:
            state["drag_x"] = None

        def reset_shift() -> None:
            state["phase_shift"] = 0.0
            render_display()

        histogram_canvas.bind("<ButtonPress-1>", drag_start)
        histogram_canvas.bind("<B1-Motion>", drag_motion)
        histogram_canvas.bind("<ButtonRelease-1>", drag_end)

        def save_png() -> None:
            if "image" not in state:
                return
            path = filedialog.asksaveasfilename(parent=dialog, defaultextension=".png", filetypes=[("PNG", "*.png")])
            if path:
                state["image"].save(path)

        def save_csv() -> None:
            if "result" not in state:
                return
            path = filedialog.asksaveasfilename(parent=dialog, defaultextension=".csv", filetypes=[("CSV", "*.csv")])
            if not path:
                return
            result = state["result"]
            edges = result["edges"]
            values = result["histogram"]
            shift = float(state["phase_shift"])
            rows = [f"# display_phase_shift_rad={shift:.12g}", "original_bin_center_rad,displayed_bin_center_rad,value"]
            for index, value in enumerate(values):
                center = 0.5 * (edges[index] + edges[index + 1])
                displayed_center = wrap_phase(float(center) + shift)
                rows.append(f"{center:.12g},{displayed_center:.12g},{value:.12g}")
            Path(path).write_text("\n".join(rows) + "\n", encoding="utf-8")

        ttk.Button(controls, text="重新统计", command=refresh).pack(side="left", padx=8)
        ttk.Button(controls, text="复位相位平移", command=reset_shift).pack(side="left", padx=3)
        ttk.Button(controls, text="导出 PNG", command=save_png).pack(side="right", padx=3)
        ttk.Button(controls, text="导出 CSV", command=save_csv).pack(side="right", padx=3)
        refresh()

    def _phase_histogram_image(self, result: dict, phase_shift: float = 0.0) -> Image.Image:
        width, height = 780, 430
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        font = _cjk_font(15)
        title_font = _cjk_font(18)
        left, top, right, bottom = 72, 48, 748, 362
        values = np.asarray(result["histogram"], dtype=float)
        edges = np.asarray(result["edges"], dtype=float)
        maximum = float(np.max(values)) if values.size and np.max(values) > 0 else 1.0
        draw.text((left, 13), "局域复包络的相位分布", fill="black", font=title_font)
        draw.line((left, bottom, right, bottom), fill="black", width=2)
        draw.line((left, top, left, bottom), fill="black", width=2)
        lut = make_lut("phase")
        plot_width = right - left
        bin_width = plot_width / max(len(values), 1)
        shift_pixels = float(phase_shift) / (2 * np.pi) * plot_width
        for index, value in enumerate(values):
            if not np.isfinite(value) or value <= 0:
                continue
            x0 = left + index * bin_width + shift_pixels
            x0 = left + ((x0 - left) % plot_width)
            x1 = x0 + bin_width
            y0 = bottom - float(value) / maximum * (bottom - top)
            original_center = 0.5 * (edges[index] + edges[index + 1])
            # Keep each phase bin's color attached to that bin while dragging;
            # the colored distribution therefore moves and wraps as one object.
            color_index = int(np.clip(round((original_center + np.pi) / (2 * np.pi) * 255), 0, 255))
            color = tuple(int(channel) for channel in lut[color_index])
            if x1 <= right:
                draw.rectangle((x0 + 1, y0, max(x0 + 1, x1 - 1), bottom - 1), fill=color, outline=color)
            else:
                if x0 + 1 <= right:
                    draw.rectangle((x0 + 1, y0, right, bottom - 1), fill=color, outline=color)
                wrapped_right = left + (x1 - right)
                if wrapped_right - 1 >= left:
                    draw.rectangle((left, y0, wrapped_right - 1, bottom - 1), fill=color, outline=color)
        ticks = [(-np.pi, "−π"), (-np.pi / 2, "−π/2"), (0.0, "0"), (np.pi / 2, "+π/2"), (np.pi, "+π")]
        for phase, label in ticks:
            x = left + (phase + np.pi) / (2 * np.pi) * plot_width
            draw.line((x, bottom, x, bottom + 6), fill="black", width=1)
            draw.text((x, bottom + 10), label, fill="black", font=font, anchor="ma")
        draw.text(((left + right) / 2, height - 25), "相位 φ (rad)", fill="black", font=font, anchor="mm")
        y_label = "振幅加权计数" if result["amplitude_weighted"] else "像素数"
        draw.text((15, (top + bottom) / 2), y_label, fill="black", font=font, anchor="lm")
        draw.text((left - 8, top), _format_number(maximum), fill="black", font=font, anchor="ra")
        draw.text((left - 8, bottom), "0", fill="black", font=font, anchor="ra")
        return image

    def _crop_press(self, event: tk.Event) -> None:
        if self.click_action != "crop":
            return
        point = self.real_panel.canvas_to_data(event.x, event.y)
        if point is None:
            return
        self._crop_start = point
        self.real_panel.canvas.delete("crop_rect")

    def _crop_drag(self, event: tk.Event) -> None:
        if self.click_action != "crop" or self._crop_start is None:
            return
        current = self.real_panel.canvas_to_data(event.x, event.y)
        if current is None:
            return
        x0, y0 = self.real_panel.data_to_canvas(*self._crop_start)
        x1, y1 = self.real_panel.data_to_canvas(*current)
        self.real_panel.canvas.delete("crop_rect")
        self.real_panel.canvas.create_rectangle(x0, y0, x1, y1, outline="#53d8ff", width=2, dash=(5, 3), tags="crop_rect")

    def _crop_release(self, event: tk.Event) -> None:
        if self.click_action != "crop" or self._crop_start is None:
            return
        current = self.real_panel.canvas_to_data(event.x, event.y)
        if current is None:
            return
        x0, x1 = sorted((int(round(self._crop_start[0])), int(round(current[0]))))
        y0, y1 = sorted((int(round(self._crop_start[1])), int(round(current[1]))))
        if x1 - x0 < 4 or y1 - y0 < 4:
            self.status_var.set("裁剪区域太小，请重新框选。")
            return
        base = self.state.crop or (0, 0, self.sxm.scan_pixels[0], self.sxm.scan_pixels[1])
        self.pending_crop = (base[0] + x0, base[1] + y0, base[0] + x1, base[1] + y1)
        self.status_var.set(f"待应用裁剪：x={self.pending_crop[0]}:{self.pending_crop[2]}, y={self.pending_crop[1]}:{self.pending_crop[3]}")
        self._crop_start = None

    def apply_crop(self) -> None:
        if self.pending_crop is None:
            self.status_var.set("尚未框选裁剪区域。")
            return
        self.state.crop = self.pending_crop
        self.pending_crop = None
        self.click_action = "inspect"
        self.state.masks.clear()
        self.refresh_mask_list()
        self.update_analysis()

    def reset_crop(self) -> None:
        self.state.crop = None
        self.pending_crop = None
        self.state.masks.clear()
        self.refresh_mask_list()
        self.update_analysis()

    def reset_all_views(self) -> None:
        self.real_panel.reset_view()
        self.fft_panel.reset_view()
        self.result_panel.reset_view()
        self.status_var.set("三个图窗已恢复为完整视图。Ctrl+滚轮缩放，右键拖动平移。")

    def batch_export_all_q(self) -> None:
        if self.sxm is None or self.fft_result is None or self.reconstruction_fft_result is None:
            messagebox.showinfo("批量导出", "请先打开并分析一个 SXM 文件。")
            return
        specs = [spec for spec in self.state.masks if spec.enabled or not self.batch_enabled_only_var.get()]
        if not specs:
            messagebox.showinfo("批量导出", "没有符合条件的 Q 点。")
            return
        selected_modes = [
            mode
            for mode, variable in (
                ("amplitude", self.batch_amplitude_var),
                ("phase", self.batch_phase_var),
                ("reconstruction", self.batch_reconstruction_var),
                ("masked_fft", self.batch_masked_fft_var),
                ("mask", self.batch_mask_var),
            )
            if variable.get()
        ]
        if not selected_modes:
            messagebox.showinfo("批量导出", "请至少勾选一种结果。")
            return
        directory = filedialog.askdirectory(title="选择批量导出文件夹", initialdir=str(self.last_export_dir))
        if not directory:
            return
        output_dir = Path(directory)
        self.last_export_dir = output_dir
        source_name = self._safe_filename_part(Path(self.state.source_path).stem)
        exported: list[Path] = []
        failures: list[str] = []
        for index, original in enumerate(specs, start=1):
            spec = MaskSpec(
                qx=original.qx,
                qy=original.qy,
                radius=original.radius,
                shape=original.shape,
                include_conjugate=original.include_conjugate,
                enabled=True,
                label=original.label or f"Q{index}",
            )
            label = self._safe_filename_part(spec.label or f"Q{index}")
            for mode in selected_modes:
                try:
                    array, palette, limits, flip_x, flip_y, colorbar_label, ticks = self._batch_q_result(spec, mode)
                    image = colorize(_orient_array(array, flip_x, flip_y, self.rotation_quarters), palette, *self._contrast(), limits)
                    if mode in {"amplitude", "phase", "reconstruction"} and self.scale_visible_var.get():
                        image = self._draw_export_scale_bar(
                            image,
                            self.current_scan_nm[self.rotation_quarters%2],
                            "nm",
                            self.scale_style_var.get(),
                        )
                    panel = SimpleNamespace(
                        palette=palette,
                        colorbar_ticks=ticks,
                        colorbar_label=colorbar_label,
                    )
                    image = self._append_export_colorbar(image, panel)
                    mode_name = {
                        "amplitude": "amplitude",
                        "phase": "phase",
                        "reconstruction": "iFFT",
                        "masked_fft": "masked_FFT",
                        "mask": "mask",
                    }[mode]
                    path = output_dir / f"{label}_{mode_name}_result_({source_name}).png"
                    image.save(path)
                    exported.append(path)
                except Exception as exc:
                    failures.append(f"{label}/{mode}: {exc}")
        if failures:
            messagebox.showwarning(
                "批量导出完成但有错误",
                f"成功导出 {len(exported)} 个文件；失败 {len(failures)} 个。\n\n" + "\n".join(failures[:8]),
            )
        else:
            messagebox.showinfo("批量导出完成", f"已为 {len(specs)} 个 Q 点导出 {len(exported)} 个文件：\n{output_dir}")
        self.status_var.set(f"批量导出完成：{len(exported)} 个文件 → {output_dir}")

    def _batch_q_result(
        self,
        spec: MaskSpec,
        mode: str,
    ) -> tuple[np.ndarray, str, Optional[tuple[float, float]], bool, bool, str, tuple[str, str, str]]:
        spatial_fft = self.fft_result if self.spatial_use_window_var.get() else self.reconstruction_fft_result
        mask_spec = spec
        if mode == "reconstruction" and not spec.include_conjugate:
            mask_spec = MaskSpec(
                qx=spec.qx,
                qy=spec.qy,
                radius=spec.radius,
                shape=spec.shape,
                include_conjugate=True,
                enabled=True,
                label=spec.label,
            )
        mask = make_frequency_mask(spatial_fft.qx, spatial_fft.qy, [mask_spec])
        if mode == "reconstruction":
            array = reconstruct_from_mask(spatial_fft.spectrum, mask)
            palette, limits, flip_x, flip_y, label = "copper", None, False, self._spatial_flip_y(), "iFFT"
        elif mode in {"amplitude", "phase"}:
            component = local_complex_component(spatial_fft.spectrum, spatial_fft.qx, spatial_fft.qy, spec)
            amplitude = np.abs(component)
            if mode == "amplitude":
                array = amplitude
                palette, limits, flip_x, flip_y, label = "viridis", None, False, self._spatial_flip_y(), "振幅"
            else:
                threshold = np.percentile(amplitude, float(self.phase_floor_var.get())) if amplitude.size else 0.0
                array = np.angle(component).copy()
                array[amplitude < threshold] = np.nan
                palette, limits, flip_x, flip_y, label = "phase", (-np.pi, np.pi), False, self._spatial_flip_y(), "相位 (rad)"
        elif mode == "mask":
            array = mask
            palette, limits, flip_x, flip_y, label = "gray", (0.0, 1.0), False, True, "mask"
        elif mode == "masked_fft":
            array = np.log1p(np.abs(spatial_fft.spectrum * mask))
            palette, limits, flip_x, flip_y, label = "inferno", None, False, True, "log|FFT|"
        else:
            raise ValueError(f"未知批量导出类型：{mode}")
        if mode == "phase":
            ticks = ("+π", "0", "−π")
        else:
            lower, upper = limits or robust_limits(array, *self._contrast())
            ticks = (_format_number(upper), _format_number((upper + lower) / 2), _format_number(lower))
        return array, palette, limits, flip_x, flip_y, label, ticks

    @staticmethod
    def _safe_filename_part(value: str) -> str:
        forbidden = '<>:"/\\|?*'
        cleaned = "".join("_" if character in forbidden or ord(character) < 32 else character for character in value).strip(" .")
        return cleaned or "unnamed"

    def export_composite_data(self) -> None:
        if self.composite_data is None:
            messagebox.showinfo("导出合成数据", "请先选择两个不同且已启用的 Q 点。")
            return
        first_index = self._composite_choice_index(self.composite_first_var.get())
        second_index = self._composite_choice_index(self.composite_second_var.get())
        first_spec, second_spec = self.state.masks[first_index], self.state.masks[second_index]
        source_name = self._safe_filename_part(Path(self.state.source_path).stem)
        default_name = f"{first_spec.label}_{second_spec.label}_composite_amplitude_result_({source_name}).npz"
        path = filedialog.asksaveasfilename(
            defaultextension=".npz", initialfile=default_name, filetypes=[("NumPy arrays", "*.npz")]
        )
        if not path:
            return
        result = self.composite_data
        np.savez_compressed(
            path,
            first_amplitude=result.first,
            second_amplitude=result.second,
            composite_amplitude=result.combined,
            first_normalization_scale=result.first_scale,
            second_normalization_scale=result.second_scale,
            scan_size_nm=np.asarray(self.current_scan_nm),
            metadata=json.dumps(self._project_payload(), ensure_ascii=False),
        )
        self.status_var.set(f"合成振幅数值图已导出：{path}")

    def open_export_image_dialog(self, annotated: bool = False, composite: bool = False) -> None:
        if self.sxm is None or self.real_panel.array is None:
            messagebox.showinfo("导出图片", "请先打开并分析一个 SXM 文件。")
            return
        if composite and self.composite_data is None:
            messagebox.showinfo("导出合成图片", "请先选择两个不同且已启用的 Q 点。")
            return
        dialog = tk.Toplevel(self)
        dialog.title("导出带标注图片" if annotated else "导出图片")
        dialog.transient(self)
        dialog.grab_set()
        frame = ttk.Frame(dialog, padding=14)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="导出内容").grid(row=0, column=0, sticky="w")
        target_var = tk.StringVar(value="合成三联图" if composite else "三联图")
        targets = ["第一个 Q", "第二个 Q", "合成振幅", "合成三联图"] if composite else ["实空间", "FFT", "结果", "三联图"]
        ttk.Combobox(
            frame,
            textvariable=target_var,
            values=targets,
            state="readonly",
            width=22,
        ).grid(row=1, column=0, sticky="ew", pady=(3, 8))
        explanation = "圆圈、Q/G 标签会在完整 FFT 上重画；导出不受当前缩放和平移影响。" if annotated else "输出完整数据范围，不包含当前交互圆圈；导出不受当前缩放和平移影响。"
        ttk.Label(frame, text=explanation, wraplength=360).grid(row=2, column=0, sticky="w")

        def save() -> None:
            key = {
                "实空间": "real", "FFT": "fft", "结果": "result", "三联图": "triptych",
                "第一个 Q": "composite_first", "第二个 Q": "composite_second",
                "合成振幅": "composite_result", "合成三联图": "composite_triptych",
            }[target_var.get()]
            initialfile = None
            if composite:
                first_index = self._composite_choice_index(self.composite_first_var.get())
                second_index = self._composite_choice_index(self.composite_second_var.get())
                first = self._safe_filename_part(self.state.masks[first_index].label)
                second = self._safe_filename_part(self.state.masks[second_index].label)
                source = self._safe_filename_part(Path(self.state.source_path).stem)
                suffix = {
                    "composite_first": f"{first}_amplitude",
                    "composite_second": f"{second}_amplitude",
                    "composite_result": f"{first}_{second}_composite_amplitude",
                    "composite_triptych": f"{first}_{second}_composite_triptych",
                }[key]
                initialfile = f"{suffix}_result_({source}).png"
            path = filedialog.asksaveasfilename(
                parent=dialog,
                defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("TIFF", "*.tif *.tiff"), ("JPEG", "*.jpg *.jpeg")],
                initialfile=initialfile or "",
            )
            if not path:
                return
            try:
                image = self._render_export_image(key, annotated)
                suffix = Path(path).suffix.lower()
                if suffix in {".jpg", ".jpeg"}:
                    image = _rgb_on_background(image)
                image.save(path)
                dialog.destroy()
                self.status_var.set(f"已导出完整图片：{path}")
            except Exception as exc:
                messagebox.showerror("图片导出失败", str(exc), parent=dialog)

        buttons = ttk.Frame(frame)
        buttons.grid(row=3, column=0, sticky="e", pady=(12, 0))
        ttk.Button(buttons, text="取消", command=dialog.destroy).pack(side="left", padx=4)
        ttk.Button(buttons, text="选择位置并导出", command=save).pack(side="left")

    def _render_export_image(self, key: str, annotated: bool = False) -> Image.Image:
        if key in {"triptych", "composite_triptych"}:
            keys = ["real", "fft", "result"] if key == "triptych" else [
                "composite_first", "composite_second", "composite_result"
            ]
            images = [self._render_export_image(item, annotated) for item in keys]
            panels = {
                "real": self.real_panel, "fft": self.fft_panel, "result": self.result_panel,
                "composite_first": self.composite_first_panel,
                "composite_second": self.composite_second_panel,
                "composite_result": self.composite_result_panel,
            }
            titles = [panels[item].title_var.get() for item in keys]
            target_height = max(image.height for image in images)
            normalized: list[Image.Image] = []
            for image in images:
                if image.height != target_height:
                    width = max(1, round(image.width * target_height / image.height))
                    image = image.resize((width, target_height), Image.Resampling.LANCZOS)
                normalized.append(image)
            title_height, gap = 46, 8
            width = sum(image.width for image in normalized) + gap * (len(normalized) - 1)
            canvas = Image.new("RGB", (width, target_height + title_height), "white")
            draw = ImageDraw.Draw(canvas)
            font = _cjk_font(17)
            x = 0
            for image, title in zip(normalized, titles):
                canvas.paste(_rgb_on_background(image), (x, title_height))
                draw.text((x + 8, 11), title, fill="black", font=font)
                x += image.width + gap
            return canvas

        panel = {
            "real": self.real_panel, "fft": self.fft_panel, "result": self.result_panel,
            "composite_first": self.composite_first_panel,
            "composite_second": self.composite_second_panel,
            "composite_result": self.composite_result_panel,
        }[key]
        if panel.array is None:
            raise ValueError("当前图窗没有可导出的数据。")
        shown = _orient_array(panel.array, panel.flip_x, panel.flip_y, panel.rotation)
        image = colorize(shown, panel.palette, panel.low, panel.high, panel.limits)
        if annotated and key == "fft":
            image = self._draw_export_overlays(image, panel)
        if panel.scale_bar_visible:
            image = self._draw_export_scale_bar(image, panel.scale_bar_span, panel.scale_bar_unit, panel.scale_bar_style)
        if panel.colorbar_visible:
            image = self._append_export_colorbar(image, panel)
        return image

    def _draw_export_overlays(self, source: Image.Image, panel: ImagePanel) -> Image.Image:
        image = source.copy()
        draw = ImageDraw.Draw(image)
        ny, nx = panel.shown_shape()
        reference_size = min(image.size)
        # Annotation type is a user-selected fixed pixel size.  It does not
        # change with FFT zoom, source matrix size, or export resolution.
        font_size = self._annotation_font_size()
        font = _cjk_font(font_size)
        line_width = max(2, round(reference_size * 0.004))
        text_offset = max(3, round(font_size * 0.16))
        stroke_width = max(1, round(font_size * 0.055))
        for overlay in panel.overlays:
            col = float(overlay["x"])
            shown_col,shown_row=panel.to_shown(col,float(overlay["y"]))
            x = (shown_col + 0.5) / nx * image.width
            row = float(overlay["y"])
            shown_col,shown_row=panel.to_shown(col,row)
            y = (shown_row + 0.5) / ny * image.height
            radius = float(overlay.get("radius", 3.0)) * 0.5 * (image.width / nx + image.height / ny)
            color = overlay.get("color", "#55ff88")
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=line_width)
            text_value = overlay.get("text", "")
            if text_value:
                draw.text(
                    (x + radius + text_offset, y - radius - text_offset),
                    str(text_value),
                    fill=color,
                    font=font,
                    stroke_width=stroke_width,
                    stroke_fill="black",
                )
        return image

    def _draw_export_scale_bar(self, source: Image.Image, span: float, unit: str, style: str) -> Image.Image:
        if span <= 0:
            return source
        image = source.copy()
        draw = ImageDraw.Draw(image)
        value = _nice_scale(0.22 * span)
        length = value / span * image.width
        margin = max(12, round(image.width * 0.035))
        right = image.width - margin
        left = right - length
        y = image.height - margin
        font = _cjk_font(max(13, round(min(image.size) / 30)))
        label = f"{value:g} {unit}"
        if style == "blackwhite":
            draw.line((left, y, right, y), fill="white", width=max(6, round(image.height / 80)))
            draw.line((left, y, right, y), fill="black", width=max(3, round(image.height / 160)))
            draw.text(((left + right) / 2, y - 7), label, fill="black", font=font, anchor="ms", stroke_width=2, stroke_fill="white")
        else:
            color = "white" if style == "white" else "black"
            draw.line((left, y, right, y), fill=color, width=max(3, round(image.height / 130)))
            draw.text(((left + right) / 2, y - 7), label, fill=color, font=font, anchor="ms")
        return image

    def _append_export_colorbar(self, source: Image.Image, panel: ImagePanel) -> Image.Image:
        margin_width = max(105, round(source.width * 0.18))
        canvas = Image.new("RGBA", (source.width + margin_width, source.height), (0, 0, 0, 0))
        opaque_source = source.convert("RGBA")
        canvas.alpha_composite(opaque_source, (0, 0))
        draw = ImageDraw.Draw(canvas)
        bar_width = max(16, round(margin_width * 0.20))
        bar_height = max(60, round(source.height * 0.70))
        x = source.width + 12
        y = (source.height - bar_height) // 2
        gradient = make_lut(panel.palette)[::-1][:, None, :]
        gradient_image = Image.fromarray(gradient.astype(np.uint8), mode="RGB").resize((bar_width, bar_height), Image.Resampling.BILINEAR)
        canvas.paste(gradient_image, (x, y))
        font = _cjk_font(max(12, round(min(source.size) / 34)))
        top, middle, bottom = panel.colorbar_ticks
        text_x = x + bar_width + 7
        draw.text((text_x, y), top, fill="black", font=font, anchor="lm")
        draw.text((text_x, y + bar_height / 2), middle, fill="black", font=font, anchor="lm")
        draw.text((text_x, y + bar_height), bottom, fill="black", font=font, anchor="lm")
        if panel.colorbar_label:
            draw.text((x, max(2, y - font.size - 7)), panel.colorbar_label, fill="black", font=font)
        return canvas

    def open_ai_settings(self) -> None:
        dialog = tk.Toplevel(self)
        dialog.title("AI 设置 — OpenRouter")
        dialog.transient(self)
        dialog.grab_set()
        dialog.resizable(False, False)
        frame = ttk.Frame(dialog, padding=14)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)

        key_present = bool(resolve_openrouter_key(self.app_dir, self._session_api_key))
        key_status = "已检测到本地密钥" if key_present else "尚未设置密钥"
        ttk.Label(frame, text="提供方").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Label(frame, text="OpenRouter").grid(row=0, column=1, sticky="w", pady=5)
        ttk.Label(frame, text="密钥状态").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Label(frame, text=key_status).grid(row=1, column=1, sticky="w", pady=5)

        model_var = tk.StringVar(value=self.ai_config.model)
        endpoint_var = tk.StringVar(value=self.ai_config.endpoint)
        key_var = tk.StringVar()
        save_key_var = tk.BooleanVar(value=True)

        ttk.Label(frame, text="模型 ID").grid(row=2, column=0, sticky="w", pady=5)
        model_frame = ttk.Frame(frame)
        model_frame.grid(row=2, column=1, sticky="ew", pady=5)
        model_frame.columnconfigure(0, weight=1)
        model_combo = ttk.Combobox(model_frame, textvariable=model_var, width=35, state="normal")
        model_combo.grid(row=0, column=0, sticky="ew")
        ttk.Label(frame, text="API 地址").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=endpoint_var, width=46).grid(row=3, column=1, sticky="ew", pady=5)
        ttk.Label(frame, text="API key").grid(row=4, column=0, sticky="w", pady=5)
        key_entry = ttk.Entry(frame, textvariable=key_var, show="●", width=46)
        key_entry.grid(row=4, column=1, sticky="ew", pady=5)
        ttk.Checkbutton(frame, text="保存到程序目录的 .env.local（不会写入项目或导出）", variable=save_key_var).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(2, 7)
        )
        ttk.Label(
            frame,
            text="AI 分析会将当前三联图和参数摘要发送给所选模型。每次请求可能产生 OpenRouter 费用。",
            wraplength=500,
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(4, 10))

        buttons = ttk.Frame(frame)
        buttons.grid(row=7, column=0, columnspan=2, sticky="ew")
        ttk.Button(buttons, text="打开 OpenRouter 密钥页面", command=lambda: webbrowser.open("https://openrouter.ai/settings/keys")).pack(side="left")
        ttk.Button(buttons, text="模型目录", command=lambda: webbrowser.open("https://openrouter.ai/models")).pack(side="left", padx=4)

        fetch_button = ttk.Button(model_frame, text="获取图片模型")
        fetch_button.grid(row=0, column=1, padx=(5, 0))

        def finish_models(models: Optional[list[dict]], error: Optional[Exception]) -> None:
            if not dialog.winfo_exists():
                return
            fetch_button.configure(state="normal", text="获取图片模型")
            if error is not None:
                messagebox.showerror("模型列表", str(error), parent=dialog)
                return
            values = [item["id"] for item in (models or [])]
            model_combo.configure(values=values)
            if values and not model_var.get().strip():
                model_var.set(values[0])
            self.status_var.set(f"已从 OpenRouter 获取 {len(values)} 个支持图像输入和文本输出的模型。")

        def fetch_models() -> None:
            key = resolve_openrouter_key(self.app_dir, key_var.get().strip() or self._session_api_key)
            if not key:
                messagebox.showinfo("模型列表", "请先输入 OpenRouter API key，或打开模型目录手动查看 model ID。", parent=dialog)
                return
            fetch_button.configure(state="disabled", text="获取中…")

            def worker() -> None:
                try:
                    models = list_openrouter_vision_models(key)
                    self.after(0, lambda: finish_models(models, None))
                except Exception as exc:
                    self.after(0, lambda exc=exc: finish_models(None, exc))

            threading.Thread(target=worker, daemon=True).start()

        fetch_button.configure(command=fetch_models)

        def save_and_close() -> None:
            model = model_var.get().strip()
            endpoint = endpoint_var.get().strip()
            if not model or not endpoint:
                messagebox.showerror("AI 设置", "模型和 API 地址不能为空。", parent=dialog)
                return
            self.ai_config = AIConfig(provider="openrouter", endpoint=endpoint, model=model, timeout_seconds=120)
            save_ai_config(self.ai_config_path, self.ai_config)
            entered_key = key_var.get().strip()
            if entered_key:
                self._session_api_key = entered_key
                if save_key_var.get():
                    save_openrouter_key(self.ai_env_path, entered_key)
            dialog.destroy()
            self.status_var.set("AI 设置已保存。密钥不会进入项目文件或导出分析。")

        ttk.Button(buttons, text="取消", command=dialog.destroy).pack(side="right", padx=4)
        ttk.Button(buttons, text="保存", command=save_and_close).pack(side="right", padx=4)
        key_entry.focus_set()

    def _ensure_ai_window(self) -> None:
        if self.ai_window is not None and self.ai_window.winfo_exists():
            self.ai_window.deiconify()
            self.ai_window.lift()
            return
        window = tk.Toplevel(self)
        window.title("AI 分析当前 STM Fourier 结果")
        window.geometry("920x720")
        window.minsize(700, 520)
        self.ai_window = window
        window.protocol("WM_DELETE_WINDOW", self._close_ai_window)
        outer = ttk.Frame(window, padding=10)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(3, weight=1)
        ttk.Label(
            outer,
            text="仅在点击分析时发送当前三联图和数值摘要；不会发送原始 SXM、API key 或其他文件。",
            wraplength=860,
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(outer, text="希望 AI 回答的问题").grid(row=1, column=0, sticky="w", pady=(8, 2))
        question = tk.Text(outer, height=3, wrap="word")
        question.insert("1.0", "请分析当前 STM Fourier 结果：选中的峰对应什么空间分布，结果是否可能来自窗函数、mask、边界或缺陷，并建议下一步验证。")
        question.grid(row=2, column=0, sticky="ew")
        self.ai_question_text = question

        result_frame = ttk.Frame(outer)
        result_frame.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        result = tk.Text(result_frame, wrap="word", font=("Microsoft YaHei UI", 10))
        scrollbar = ttk.Scrollbar(result_frame, command=result.yview)
        result.configure(yscrollcommand=scrollbar.set)
        result.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.ai_result_text = result

        actions = ttk.Frame(outer)
        actions.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        self.ai_run_button = ttk.Button(actions, text="重新分析当前视图", command=self._start_ai_request)
        self.ai_run_button.pack(side="left")
        ttk.Button(actions, text="复制结果", command=self._copy_ai_result).pack(side="left", padx=5)
        ttk.Button(actions, text="保存 Markdown", command=self._save_ai_result).pack(side="left")

    def run_ai_analysis(self) -> None:
        if self.processed is None or self.fft_result is None:
            messagebox.showinfo("AI 分析", "请先打开并分析一个 SXM 文件。")
            return
        if not resolve_openrouter_key(self.app_dir, self._session_api_key):
            self.open_ai_settings()
            self.status_var.set("请先在 AI 设置中输入 OpenRouter API key。")
            return
        self._ensure_ai_window()
        self._start_ai_request()

    def _analysis_summary_for_ai(self) -> dict:
        channel = self.sxm.channels[self.state.channel_index]
        processed = np.asarray(self.processed)
        amplitude = np.abs(self.complex_component) if self.complex_component is not None else np.zeros_like(processed)

        def percentiles(array: np.ndarray) -> dict:
            finite = np.asarray(array)[np.isfinite(array)]
            if finite.size == 0:
                return {}
            p = np.percentile(finite, [1, 10, 50, 90, 99])
            return {"p01": float(p[0]), "p10": float(p[1]), "p50": float(p[2]), "p90": float(p[3]), "p99": float(p[4]), "std": float(np.std(finite))}

        masks = []
        for spec in self.state.masks:
            magnitude = math.hypot(spec.qx, spec.qy)
            item = spec.to_dict()
            item["q_magnitude_cycles_per_nm"] = magnitude
            item["period_nm"] = 1.0 / magnitude if magnitude > 0 else None
            if self.state.bragg_g1 and self.state.bragg_g2:
                try:
                    item["fractional_hk"] = fractional_coordinates((spec.qx, spec.qy), self.state.bragg_g1, self.state.bragg_g2)
                except ValueError:
                    pass
            masks.append(item)

        ny, nx = amplitude.shape
        center = amplitude[ny // 4 : 3 * ny // 4, nx // 4 : 3 * nx // 4]
        edge_width = max(1, min(nx, ny) // 10)
        edge_values = np.concatenate(
            [amplitude[:edge_width].ravel(), amplitude[-edge_width:].ravel(), amplitude[:, :edge_width].ravel(), amplitude[:, -edge_width:].ravel()]
        )
        edge_mean = float(np.mean(edge_values)) if edge_values.size else 0.0
        center_edge_ratio = float(np.mean(center) / edge_mean) if edge_mean > 0 and center.size else None

        fields = self.sxm.header_fields
        acquisition = {
            key: fields.get(key, [])[:3]
            for key in ("REC_DATE", "REC_TIME", "BIAS", "Z-CONTROLLER", "COMMENT")
            if fields.get(key)
        }
        summary = {
            "source_file": Path(self.state.source_path).name,
            "channel": {"name": channel.name, "unit": channel.unit, "direction": channel.direction},
            "scan": {
                "pixels": [int(processed.shape[1]), int(processed.shape[0])],
                "size_nm": [float(self.current_scan_nm[0]), float(self.current_scan_nm[1])],
                "crop_original_pixels": self.state.crop,
            },
            "acquisition_header_fields": acquisition,
            "processing": {
                "preprocess": self.state.preprocess,
                "fft_display_window": self.state.window,
                "spatial_reconstruction_uses_window": bool(self.spatial_use_window_var.get()),
                "fft_levels": self.fft_levels.copy(),
                "fft_view": self.fft_view_var.get(),
                "result_view": self.result_var.get(),
            },
            "bragg": {"g1_cycles_per_nm": self.state.bragg_g1, "g2_cycles_per_nm": self.state.bragg_g2},
            "masks": masks,
            "processed_statistics": percentiles(processed),
            "reconstruction_statistics": percentiles(self.reconstruction) if self.reconstruction is not None else {},
            "local_amplitude_statistics": percentiles(amplitude),
            "local_amplitude_center_to_edge_mean_ratio": center_edge_ratio,
            "phase_low_amplitude_mask_percentile": float(self.phase_floor_var.get()),
            "cautions": [
                "FFT 显示可加窗，但空间重建默认使用未加窗谱。",
                "局域振幅是所选 q 邻域的频率分量强度，不等同于结构含量。",
                "相位参考当前裁剪原点；低振幅区域相位不可靠。",
            ],
        }
        if self.workspace_notebook.index(self.workspace_notebook.select()) == 1 and self.composite_data is not None:
            first_index = self._composite_choice_index(self.composite_first_var.get())
            second_index = self._composite_choice_index(self.composite_second_var.get())
            summary["composite_workspace"] = {
                "selected_mask_indices": [first_index, second_index],
                "method": self.composite_data.method,
                "normalize_p90": self.composite_data.normalized,
                "normalization_scales": [self.composite_data.first_scale, self.composite_data.second_scale],
                "first_statistics": percentiles(self.composite_data.first),
                "second_statistics": percentiles(self.composite_data.second),
                "combined_statistics": percentiles(self.composite_data.combined),
                "caution": "这是不同 Q 局域振幅模长的合成，不是复数序参量或超导性的直接测量。",
            }
        return summary

    def _ai_snapshot_png(self) -> bytes:
        low, high = self._contrast()
        if self.workspace_notebook.index(self.workspace_notebook.select()) == 1 and self.composite_data is not None:
            source_panels = (self.composite_first_panel, self.composite_second_panel, self.composite_result_panel)
            panels = [colorize(_orient_array(panel.array, panel.flip_x, panel.flip_y, panel.rotation), panel.palette, panel.low, panel.high, panel.limits) for panel in source_panels]
            labels = [panel.title_var.get() for panel in source_panels]
        else:
            fft_array, fft_palette, fft_limits = self._fft_display_array()
            result_array = self.result_panel.array if self.result_panel.array is not None else np.zeros_like(self.processed)
            result_array = _orient_array(result_array, self.result_panel.flip_x, self.result_panel.flip_y, self.rotation_quarters)
            panels = [
                colorize(_orient_array(self._real_display_array(), False, self._spatial_flip_y(), self.rotation_quarters), self.real_palette_var.get(), low, high),
                colorize(_orient_array(fft_array, False, True, self.rotation_quarters), fft_palette, low, high, fft_limits),
                colorize(result_array, self.result_panel.palette, low, high, self.result_panel.limits),
            ]
            labels = [
                f"Real space: {self.sxm.channels[self.state.channel_index].name}",
                f"FFT: {self.fft_view_var.get()} (cycles/nm)",
                self.result_panel.title_var.get(),
            ]
        panel_size = 520
        header = 54
        canvas = Image.new("RGB", (panel_size * 3, panel_size + header), "white")
        draw = ImageDraw.Draw(canvas)
        font = _cjk_font(15)
        for index, (image, label) in enumerate(zip(panels, labels)):
            image.thumbnail((panel_size, panel_size), Image.Resampling.LANCZOS)
            x = index * panel_size + (panel_size - image.width) // 2
            y = header + (panel_size - image.height) // 2
            canvas.paste(image, (x, y))
            draw.text((index * panel_size + 8, 10), label, fill="black", font=font)
            if index == 0:
                draw.text((index * panel_size + 8, 30), f"{self.current_scan_nm[0]:.4g} × {self.current_scan_nm[1]:.4g} nm", fill="black", font=font)
        buffer = BytesIO()
        canvas.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()

    def _start_ai_request(self) -> None:
        key = resolve_openrouter_key(self.app_dir, self._session_api_key)
        if not key:
            self.open_ai_settings()
            return
        if self.ai_result_text is None or self.ai_question_text is None:
            return
        question = self.ai_question_text.get("1.0", "end").strip()
        if not question:
            question = "请分析当前 STM Fourier 结果。"
        try:
            summary = self._analysis_summary_for_ai()
            image_png = self._ai_snapshot_png()
            payload = build_openrouter_payload(self.ai_config, summary, image_png, question)
        except Exception as exc:
            messagebox.showerror("AI 分析准备失败", str(exc), parent=self.ai_window)
            return

        self.ai_result_text.delete("1.0", "end")
        self.ai_result_text.insert("1.0", f"正在调用 {self.ai_config.model} 分析当前视图……\n")
        if self.ai_run_button:
            self.ai_run_button.configure(state="disabled")
        self.status_var.set("AI 正在分析当前三联图和数值摘要……")

        def worker() -> None:
            try:
                text, usage = request_openrouter_analysis(self.ai_config, key, payload)
                self.after(0, lambda: self._finish_ai_request(text, usage, None))
            except Exception as exc:
                self.after(0, lambda exc=exc: self._finish_ai_request("", {}, exc))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_ai_request(self, text: str, usage: dict, error: Optional[Exception]) -> None:
        if self.ai_window is None or not self.ai_window.winfo_exists():
            self.ai_window = None
            self.ai_result_text = None
            self.ai_question_text = None
            self.ai_run_button = None
            self.status_var.set("AI 请求已结束；结果窗口已关闭。")
            return
        if self.ai_run_button:
            self.ai_run_button.configure(state="normal")
        if self.ai_result_text is None:
            return
        self.ai_result_text.delete("1.0", "end")
        if error is not None:
            self.ai_result_text.insert("1.0", f"分析失败：{error}\n\n请检查 API key、模型是否支持图像输入，以及 OpenRouter 账户余额。")
            self.status_var.set("AI 分析失败。")
            return
        usage_text = ""
        if usage:
            usage_text = f"\n\n---\n模型：{self.ai_config.model}\n用量：{json.dumps(usage, ensure_ascii=False)}"
        self.ai_result_text.insert("1.0", text + usage_text)
        self.status_var.set("AI 分析完成。")

    def _close_ai_window(self) -> None:
        if self.ai_window is not None and self.ai_window.winfo_exists():
            self.ai_window.destroy()
        self.ai_window = None
        self.ai_result_text = None
        self.ai_question_text = None
        self.ai_run_button = None

    def _copy_ai_result(self) -> None:
        if self.ai_result_text is None:
            return
        value = self.ai_result_text.get("1.0", "end").strip()
        self.clipboard_clear()
        self.clipboard_append(value)
        self.status_var.set("AI 结果已复制。")

    def _save_ai_result(self) -> None:
        if self.ai_result_text is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".md", filetypes=[("Markdown", "*.md"), ("文本", "*.txt")], parent=self.ai_window)
        if not path:
            return
        question = self.ai_question_text.get("1.0", "end").strip() if self.ai_question_text else ""
        result = self.ai_result_text.get("1.0", "end").strip()
        content = f"# STM Fourier AI 分析\n\n## 问题\n\n{question}\n\n## 回答\n\n{result}\n"
        Path(path).write_text(content, encoding="utf-8")
        self.status_var.set(f"AI 分析已保存：{path}")

    def _project_payload(self) -> dict:
        return {
            "format": "stm-fourier-analyzer-project-v1",
            "source_path": self.state.source_path,
            "channel_index": self.state.channel_index,
            "crop": self.state.crop,
            "preprocess": self.state.preprocess,
            "window": self.state.window,
            "masks": [spec.to_dict() for spec in self.state.masks],
            "bragg_g1": self.state.bragg_g1,
            "bragg_g2": self.state.bragg_g2,
            "display": {
                "rotation_quarters": self.rotation_quarters,
                "real_view": self.real_view_var.get(),
                "real_palette": self.real_palette_var.get(),
                "fft_levels": self.fft_levels.copy(),
                "fft_view": self.fft_view_var.get(),
                "result": self.result_var.get(),
                "low": self.low_var.get(),
                "high": self.high_var.get(),
                "spatial_use_window": self.spatial_use_window_var.get(),
                "scale_visible": self.scale_visible_var.get(),
                "scale_style": self.scale_style_var.get(),
                "annotation_font_size": self._annotation_font_size(),
                "composite": {
                    "first_index": self._composite_choice_index(self.composite_first_var.get()),
                    "second_index": self._composite_choice_index(self.composite_second_var.get()),
                    "method": COMPOSITE_METHODS.get(self.composite_method_var.get(), "harmonic"),
                    "normalize_p90": self.composite_normalize_var.get(),
                },
                "batch_export": {
                    "amplitude": self.batch_amplitude_var.get(),
                    "phase": self.batch_phase_var.get(),
                    "reconstruction": self.batch_reconstruction_var.get(),
                    "masked_fft": self.batch_masked_fft_var.get(),
                    "mask": self.batch_mask_var.get(),
                    "enabled_only": self.batch_enabled_only_var.get(),
                },
            },
        }

    def save_project(self) -> None:
        if not self.sxm:
            messagebox.showinfo("保存项目", "请先打开一个 SXM 文件。")
            return
        path = filedialog.asksaveasfilename(defaultextension=".stmfft.json", filetypes=[("STM FFT 项目", "*.stmfft.json"), ("JSON", "*.json")])
        if not path:
            return
        Path(path).write_text(json.dumps(self._project_payload(), ensure_ascii=False, indent=2), encoding="utf-8")
        self.status_var.set(f"项目已保存：{path}")

    def load_project(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("STM FFT 项目", "*.stmfft.json *.json"), ("所有文件", "*.*")])
        if not path:
            return
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            self.open_sxm(payload["source_path"])
            if not self.sxm:
                return
            self.state.channel_index = int(payload.get("channel_index", 0))
            self.channel_combo.current(self.state.channel_index)
            crop = payload.get("crop")
            self.state.crop = tuple(crop) if crop else None
            self.state.preprocess = payload.get("preprocess", "plane")
            self.state.window = payload.get("window", "hann")
            self.preprocess_var.set(next((key for key, value in PREPROCESS_OPTIONS.items() if value == self.state.preprocess), "稳健平面调平"))
            self.window_var.set(next((key for key, value in WINDOW_OPTIONS.items() if value == self.state.window), "Hann"))
            self.state.masks = [MaskSpec.from_dict(item) for item in payload.get("masks", [])]
            g1, g2 = payload.get("bragg_g1"), payload.get("bragg_g2")
            self.state.bragg_g1 = tuple(g1) if g1 else None
            self.state.bragg_g2 = tuple(g2) if g2 else None
            display = payload.get("display", {})
            self.real_view_var.set(display.get("real_view", "处理后"))
            self.real_palette_var.set(display.get("real_palette", "copper"))
            self.rotation_quarters=int(display.get("rotation_quarters",0))%4
            self.sync_rotation()
            self.fft_view_var.set(display.get("fft_view", "对数振幅"))
            self.fft_levels = display.get("fft_levels", {"low":1.0,"high":99.0,"limits":None})
            self._fft_levels_mode = self.fft_view_var.get()
            self.result_var.set(display.get("result", "成对 iFFT"))
            self.low_var.set(display.get("low", 1.0))
            self.high_var.set(display.get("high", 99.0))
            self.spatial_use_window_var.set(display.get("spatial_use_window", False))
            self.scale_visible_var.set(display.get("scale_visible", True))
            self.scale_style_var.set(display.get("scale_style", "blackwhite"))
            self.annotation_font_size_var.set(display.get("annotation_font_size", 13))
            batch_export = display.get("batch_export", {})
            self.batch_amplitude_var.set(batch_export.get("amplitude", True))
            self.batch_phase_var.set(batch_export.get("phase", True))
            self.batch_reconstruction_var.set(batch_export.get("reconstruction", False))
            self.batch_masked_fft_var.set(batch_export.get("masked_fft", False))
            self.batch_mask_var.set(batch_export.get("mask", False))
            self.batch_enabled_only_var.set(batch_export.get("enabled_only", True))
            self.refresh_mask_list()
            composite = display.get("composite", {})
            inverse_methods = {value: key for key, value in COMPOSITE_METHODS.items()}
            self.composite_method_var.set(inverse_methods.get(composite.get("method"), "共同振幅 · 调和平均"))
            self.composite_normalize_var.set(composite.get("normalize_p90", True))
            labels = dict(self._composite_choices)
            first_index = composite.get("first_index")
            second_index = composite.get("second_index")
            if first_index in labels:
                self.composite_first_var.set(labels[first_index])
            if second_index in labels and second_index != first_index:
                self.composite_second_var.set(labels[second_index])
            self._update_bragg_label()
            self.update_analysis()
            self.status_var.set(f"项目已载入：{path}")
        except Exception as exc:
            messagebox.showerror("项目载入失败", str(exc))

    def export_analysis(self) -> None:
        if self.processed is None or self.fft_result is None or self.reconstruction_fft_result is None or self.mask is None:
            messagebox.showinfo("导出分析", "当前没有可导出的分析。")
            return
        path = filedialog.asksaveasfilename(defaultextension=".npz", filetypes=[("NumPy analysis bundle", "*.npz")])
        if not path:
            return
        path_obj = Path(path)
        np.savez_compressed(
            path_obj,
            processed=self.processed,
            fft_windowed=self.fft_result.spectrum,
            fft_unwindowed=self.reconstruction_fft_result.spectrum,
            qx=self.fft_result.qx,
            qy=self.fft_result.qy,
            mask=self.mask,
            reconstruction=self.reconstruction,
            local_complex=self.complex_component,
            metadata=json.dumps(self._project_payload(), ensure_ascii=False),
        )
        overview = path_obj.with_name(path_obj.stem + "_overview.png")
        self._render_export_image("triptych", annotated=True).save(overview)
        project_json = path_obj.with_name(path_obj.stem + "_project.json")
        project_json.write_text(json.dumps(self._project_payload(), ensure_ascii=False, indent=2), encoding="utf-8")
        self.status_var.set(f"已导出 {path_obj.name}、{overview.name} 和项目 JSON。")


def main() -> None:
    app = STMFFTApp()
    app.mainloop()
