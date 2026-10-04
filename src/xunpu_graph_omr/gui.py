"""
Xunpu GraphOMR 桌面界面 (Tkinter)。

布局
----
  ┌──────────────────────────────────────────────────────┐
  │ 工具栏: 打开 · 引擎 · 识别 · 导出MusicXML/MIDI/JSON · 视图 │
  ├───────────────────────────┬──────────────────────────┤
  │                           │ 概要                      │
  │      乐谱图 (原图/叠加)     │──────────────────────────│
  │                           │ 音符 / 小节 / 处理步骤     │
  ├───────────────────────────┴──────────────────────────┤
  │ 状态栏                                                 │
  └──────────────────────────────────────────────────────┘

公开桌面版仅暴露仓库内的独立识别管线。外部 OMR 系统只用于可选的离线基线比较，
不作为本项目能力展示的一部分。识别在后台线程运行，避免界面卡死。

入口::

    xunpu-graph-omr-gui            # 安装后
    python -m xunpu_graph_omr.gui  # 源码运行
"""
from __future__ import annotations

import queue
import shutil
import sys
import tempfile
import threading
import traceback
from pathlib import Path
from tkinter import (
    BOTH,
    END,
    LEFT,
    RIGHT,
    TOP,
    X,
    Y,
    Canvas,
    StringVar,
    Tk,
    filedialog,
    messagebox,
    ttk,
)

from PIL import Image, ImageTk

from xunpu_graph_omr.vision.document import load_first_page_gray
from xunpu_graph_omr.engines import (
    EngineResult,
    available_engines,
    default_engine_key,
    get_engine,
)
from xunpu_graph_omr.vision.image_io import write_json
from xunpu_graph_omr.export.midi_simple import write_midi, write_midi_notes
from xunpu_graph_omr.export.musicxml_parse import read_musicxml_bytes
from xunpu_graph_omr.pipeline import write_debug_overlay

_SUPPORTED = [
    ("乐谱文件", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.pdf"),
    ("图片", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
    ("PDF", "*.pdf"),
    ("所有文件", "*.*"),
]


class OmrApp(Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Xunpu GraphOMR · Optical Music Recognition")
        self.geometry("1180x760")
        self.minsize(960, 640)

        self.input_path: Path | None = None
        self.engine_result: EngineResult | None = None
        self._tmp_dir = Path(tempfile.mkdtemp(prefix="xunpu_graph_omr_gui_"))
        self._photo = None  # keep a reference so the image is not GC'd
        self._pil_original: Image.Image | None = None
        self._pil_overlay: Image.Image | None = None
        self._queue: queue.Queue = queue.Queue()

        # engine selector: label -> engine
        self._engines = {self._engine_label(e): e for e in available_engines() if e.key == "xunpu"}
        self.engine_var = StringVar()
        self.view_mode = StringVar(value="original")
        self.status = StringVar(value="就绪 · 请打开一张乐谱图片或 PDF")

        self._build_toolbar()
        self._build_body()
        self._build_statusbar()
        self._select_default_engine()
        self._set_export_enabled(False)
        self._poll_queue()

    # ------------------------------------------------------------------ UI

    @staticmethod
    def _engine_label(engine) -> str:
        return f"{engine.label}  {'✓' if engine.available() else '（未安装）'}"

    def _select_default_engine(self) -> None:
        default_key = default_engine_key()
        for label, engine in self._engines.items():
            if engine.key == default_key:
                self.engine_var.set(label)
                return
        if self._engines:
            self.engine_var.set(next(iter(self._engines)))

    def _build_toolbar(self) -> None:
        outer = ttk.Frame(self, padding=(8, 6))
        outer.pack(side=TOP, fill=X)
        bar = ttk.Frame(outer)
        bar.pack(fill=X)

        ttk.Button(bar, text="📂 打开", command=self.on_open).pack(side=LEFT)
        ttk.Button(bar, text="示例", command=self.on_demo, width=6).pack(side=LEFT, padx=4)

        ttk.Label(bar, text="  引擎:").pack(side=LEFT)
        self.engine_box = ttk.Combobox(
            bar, textvariable=self.engine_var, state="readonly",
            values=list(self._engines.keys()), width=24,
        )
        self.engine_box.pack(side=LEFT, padx=(2, 0))

        self.btn_recognize = ttk.Button(bar, text="▶ 识别", command=self.on_recognize, state="disabled")
        self.btn_recognize.pack(side=LEFT, padx=(6, 0))

        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=120)
        self.progress.pack(side=RIGHT)
        bar = ttk.Frame(outer)
        bar.pack(fill=X, pady=(6, 0))

        self.btn_xml = ttk.Button(bar, text="导出 MusicXML", command=self.on_export_musicxml)
        self.btn_xml.pack(side=LEFT, padx=2)
        self.btn_midi = ttk.Button(bar, text="导出 MIDI", command=self.on_export_midi)
        self.btn_midi.pack(side=LEFT, padx=2)
        self.btn_json = ttk.Button(bar, text="导出 JSON", command=self.on_export_json)
        self.btn_json.pack(side=LEFT, padx=2)

        ttk.Separator(bar, orient="vertical").pack(side=LEFT, fill=Y, padx=10)
        ttk.Label(bar, text="视图:").pack(side=LEFT)
        ttk.Radiobutton(bar, text="叠加结果", value="overlay", variable=self.view_mode,
                        command=self._refresh_image).pack(side=LEFT)
        ttk.Radiobutton(bar, text="原图", value="original", variable=self.view_mode,
                        command=self._refresh_image).pack(side=LEFT)


    def _build_body(self) -> None:
        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill=BOTH, expand=True, padx=8, pady=(0, 6))

        left = ttk.Frame(paned)
        self.canvas = Canvas(left, background="#202327", highlightthickness=0)
        self.canvas.pack(fill=BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _e: self._refresh_image())
        paned.add(left, weight=3)

        right = ttk.Frame(paned)
        self._build_summary(right)
        nb = ttk.Notebook(right)
        nb.pack(fill=BOTH, expand=True, pady=(8, 0))

        self.tree_notes = self._make_tree(
            nb, columns=("idx", "pitch", "measure", "type", "extra"),
            headings=("#", "音高", "小节", "时值", "x/置信度"),
            widths=(40, 70, 60, 70, 90),
        )
        nb.add(self.tree_notes.master, text="音符")

        self.tree_measures = self._make_tree(
            nb, columns=("staff", "idx", "notes", "dur", "status"),
            headings=("谱表", "小节", "音符数", "时值", "状态"),
            widths=(50, 50, 70, 70, 90),
        )
        nb.add(self.tree_measures.master, text="小节")

        self.tree_steps = self._make_tree(
            nb, columns=("name", "status", "detail"),
            headings=("步骤", "状态", "详情"),
            widths=(90, 60, 260),
        )
        nb.add(self.tree_steps.master, text="处理步骤")

        paned.add(right, weight=2)

    def _build_summary(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="概要", padding=8)
        box.pack(fill=X)
        self.summary_vars = {
            "engine": StringVar(value="-"),
            "notes": StringVar(value="-"),
            "measures": StringVar(value="-"),
            "issues": StringVar(value="-"),
        }
        grid = ttk.Frame(box)
        grid.pack(fill=X)
        labels = [("引擎", "engine"), ("音符", "notes"), ("小节", "measures"), ("问题", "issues")]
        for col, (text, key) in enumerate(labels):
            cell = ttk.Frame(grid)
            cell.grid(row=0, column=col, padx=8, sticky="w")
            ttk.Label(cell, textvariable=self.summary_vars[key],
                      font=("Helvetica", 18, "bold")).pack(anchor="w")
            ttk.Label(cell, text=text, foreground="#777").pack(anchor="w")

    def _make_tree(self, parent, columns, headings, widths) -> ttk.Treeview:
        frame = ttk.Frame(parent)
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=12)
        for col, head, width in zip(columns, headings, widths):
            tree.heading(col, text=head)
            tree.column(col, width=width, anchor="center")
        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side=LEFT, fill=BOTH, expand=True)
        vsb.pack(side=RIGHT, fill=Y)
        return tree

    def _build_statusbar(self) -> None:
        bar = ttk.Frame(self, relief="sunken", padding=(8, 3))
        bar.pack(side=TOP, fill=X)
        ttk.Label(bar, textvariable=self.status).pack(side=LEFT)

    # -------------------------------------------------------------- actions

    def on_open(self) -> None:
        path = filedialog.askopenfilename(title="选择乐谱", filetypes=_SUPPORTED)
        if not path:
            return
        self.open_path(Path(path))

    def on_demo(self) -> None:
        from xunpu_graph_omr.evaluation.synthetic import draw_simple_score

        root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
        path = root / "examples/input/simple_score.png"
        if not path.exists():
            path = self._tmp_dir / "simple_score.png"
            draw_simple_score(path)
        self._select_default_engine()
        if self.open_path(path):
            self.on_recognize()

    def open_path(self, path: Path) -> bool:
        self.input_path = Path(path)
        self.engine_result = None
        self._pil_overlay = None
        self._set_export_enabled(False)
        try:
            self._pil_original = self._load_display_image(self.input_path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("打开失败", str(exc))
            return False
        self.view_mode.set("original")
        self._refresh_image()
        self._clear_results()
        self.btn_recognize.configure(state="normal")
        self.status.set(f"已加载: {self.input_path.name} · 选择引擎后点「识别」")
        return True

    def _current_engine(self):
        return self._engines.get(self.engine_var.get())

    def on_recognize(self) -> None:
        if not self.input_path:
            return
        engine = self._current_engine()
        if engine is None:
            return
        if not engine.available():
            messagebox.showwarning("引擎不可用", engine.unavailable_hint())
            return
        self.btn_recognize.configure(state="disabled")
        self.progress.start(12)
        self.status.set(f"识别中… (引擎: {engine.label})")
        thread = threading.Thread(
            target=self._recognize_worker, args=(self.input_path, engine), daemon=True
        )
        thread.start()

    def _recognize_worker(self, path: Path, engine) -> None:
        try:
            er = engine.recognize(path, self._tmp_dir)
            overlay = None
            if er.recognition is not None:
                overlay = self._tmp_dir / f"{path.stem}.overlay.png"
                write_debug_overlay(er.recognition, overlay)
            self._queue.put(("done", er, overlay))
        except Exception:  # noqa: BLE001
            self._queue.put(("error", traceback.format_exc(), None))

    def _poll_queue(self) -> None:
        try:
            kind, payload, extra = self._queue.get_nowait()
        except queue.Empty:
            pass
        else:
            self.progress.stop()
            if kind == "done":
                self._on_result(payload, extra)
            elif kind == "error":
                self.btn_recognize.configure(state="normal")
                self.status.set("识别失败")
                messagebox.showerror("识别失败", payload)
        self.after(80, self._poll_queue)

    def _on_result(self, er: EngineResult, overlay: Path | None) -> None:
        self.engine_result = er
        if overlay is not None and overlay.exists():
            self._pil_overlay = Image.open(overlay).copy()
            self.view_mode.set("overlay")
        else:
            self._pil_overlay = None
            self.view_mode.set("original")
        self._refresh_image()
        self._populate_results(er)
        self.btn_recognize.configure(state="normal")
        self._set_export_enabled(True)
        issues = len(er.recognition.rhythm_issues) if er.recognition is not None else 0
        self.status.set(
            f"完成 · 引擎={er.engine} · {len(er.notes)} 音符 · {er.measure_count} 小节 · {issues} 问题"
        )

    # -------------------------------------------------------------- exports

    def on_export_musicxml(self) -> None:
        def writer(p: str) -> None:
            src = self.engine_result.musicxml_path
            if Path(p).suffix.lower() == ".mxl":
                shutil.copyfile(src, p)                      # 用户明确要压缩格式
            else:
                # 解压成纯文本 XML，外部软件(MuseScore 等)才能打开
                Path(p).write_bytes(read_musicxml_bytes(src))
        self._export("MusicXML", ".musicxml", writer)

    def on_export_midi(self) -> None:
        def writer(p: str) -> None:
            er = self.engine_result
            if er.recognition is not None:
                write_midi(er.recognition, p)
            else:
                write_midi_notes(er.notes, p)
        self._export("MIDI", ".mid", writer)

    def on_export_json(self) -> None:
        def writer(p: str) -> None:
            er = self.engine_result
            if er.recognition is not None:
                write_json(p, er.recognition.to_dict())
            else:
                write_json(p, {
                    "engine": er.engine,
                    "image": er.image_path,
                    "measures": er.measure_count,
                    "notes": [
                        {"pitch": n.pitch_name, "measure": n.measure, "type": n.note_type,
                         "is_rest": n.is_rest}
                        for n in er.notes
                    ],
                })
        self._export("JSON", ".json", writer)

    def on_export_jianpu(self) -> None:
        def writer(p: str) -> None:
            from xunpu_graph_omr.export.jianpu import jianpu_from_result
            jp, key = jianpu_from_result(self.engine_result.notes, self.engine_result.musicxml_path)
            Path(p).write_text(key + "\n\n" + jp, encoding="utf-8")
        self._export("简谱", ".txt", writer)

    def _export(self, label: str, suffix: str, writer) -> None:
        if not self.engine_result:
            messagebox.showinfo("提示", "请先识别一张乐谱")
            return
        default = f"{self.input_path.stem}{suffix}"
        path = filedialog.asksaveasfilename(
            title=f"导出 {label}", defaultextension=suffix, initialfile=default,
            filetypes=[(label, f"*{suffix}"), ("所有文件", "*.*")],
        )
        if not path:
            return
        try:
            writer(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("导出失败", str(exc))
            return
        self.status.set(f"已导出 {label}: {path}")

    # ---------------------------------------------------------------- views

    def _set_export_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for btn in (self.btn_xml, self.btn_midi, self.btn_json):
            btn.configure(state=state)

    def _load_display_image(self, path: Path) -> Image.Image:
        if path.suffix.lower() == ".pdf":
            page = load_first_page_gray(path)
            return Image.fromarray(page.gray).convert("RGB")
        return Image.open(path).convert("RGB")

    def _refresh_image(self) -> None:
        image = self._pil_overlay if self.view_mode.get() == "overlay" else self._pil_original
        if image is None:
            image = self._pil_original
        if image is None:
            self.canvas.delete("all")
            return
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        scale = min(cw / image.width, ch / image.height, 4.0)
        new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        resized = image.resize(new_size, Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._photo)

    def _clear_results(self) -> None:
        for tree in (self.tree_notes, self.tree_measures, self.tree_steps):
            tree.delete(*tree.get_children())
        for var in self.summary_vars.values():
            var.set("-")

    def _populate_results(self, er: EngineResult) -> None:
        if er.recognition is not None:
            self._populate_own(er)
        else:
            self._populate_external(er)

    def _populate_own(self, er: EngineResult) -> None:
        result = er.recognition
        self._clear_results()
        self.summary_vars["engine"].set(er.engine)
        self.summary_vars["notes"].set(str(len(result.notes)))
        self.summary_vars["measures"].set(str(len(result.measures)))
        self.summary_vars["issues"].set(str(len(result.rhythm_issues)))

        for i, note in enumerate(
            sorted(result.notes, key=lambda n: (n.measure, n.staff_index, n.x)), start=1
        ):
            self.tree_notes.insert(
                "", END,
                values=(i, note.pitch_name, note.measure, note.note_type,
                        f"x={note.x:.0f} c={note.confidence:.2f}"),
            )
        for m in result.measures:
            self.tree_measures.insert(
                "", END,
                values=(m.staff_index, m.index, m.note_count,
                        f"{m.total_duration}/{m.expected_duration}", m.status),
            )
        for step in result.steps:
            self.tree_steps.insert("", END, values=(step.name, step.status, step.detail))

    def _populate_external(self, er: EngineResult) -> None:
        self._clear_results()
        self.summary_vars["engine"].set(er.engine)
        self.summary_vars["notes"].set(str(len(er.notes)))
        self.summary_vars["measures"].set(str(er.measure_count))
        self.summary_vars["issues"].set("-")

        per_measure: dict[int, int] = {}
        for i, n in enumerate(er.notes, start=1):
            self.tree_notes.insert(
                "", END,
                values=(i, n.pitch_name, n.measure, n.note_type, "-"),
            )
            if not n.is_rest:
                per_measure[n.measure] = per_measure.get(n.measure, 0) + 1

        for measure in sorted(per_measure):
            self.tree_measures.insert(
                "", END, values=("-", measure, per_measure[measure], "-", "-"),
            )
        self.tree_steps.insert("", END, values=(er.engine, "ok", f"MusicXML: {er.musicxml_path.name}"))
        self.tree_steps.insert("", END, values=("note", "info", "外部引擎仅提供 MusicXML，无叠加图/谱表几何"))


def main() -> int:
    app = OmrApp()
    app.after(150, app.on_demo)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
