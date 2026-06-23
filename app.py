"""Desktop UI for face-based image sorting."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from engine import DEFAULT_WORKERS, SortConfig, sort_images

PHASE_LABELS = {
    "models": "Loading models",
    "scan": "Scanning folder",
    "detect": "Detecting faces",
    "references": "Loading references",
    "match": "Comparing & grouping",
    "copy": "Copying files",
    "done": "Complete",
}


class FaceSorterApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Face Image Sorter")
        self.root.minsize(640, 520)
        self.root.geometry("760x620")

        self.events: queue.Queue[tuple[str, int, int, str, str] | None] = queue.Queue()
        self.cancel_event = threading.Event()
        self.worker: threading.Thread | None = None

        self.source_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.reference_var = tk.StringVar()
        self.use_reference_var = tk.BooleanVar(value=False)
        self.workers_var = tk.IntVar(value=DEFAULT_WORKERS)
        self.status_var = tk.StringVar(value="Ready")
        self.phase_var = tk.StringVar(value="Idle")
        self.progress_var = tk.DoubleVar(value=0.0)

        self._build_ui()
        self.root.after(100, self._poll_events)

    def _build_ui(self) -> None:
        padding = {"padx": 12, "pady": 6}
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(
            frame,
            text="Sort photos by similar faces",
            font=("Segoe UI", 14, "bold"),
        )
        title.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        ttk.Label(frame, text="From (source folder)").grid(row=1, column=0, sticky="w", **padding)
        ttk.Entry(frame, textvariable=self.source_var, width=62).grid(
            row=1, column=1, sticky="ew", **padding
        )
        ttk.Button(frame, text="Browse…", command=self._browse_source).grid(row=1, column=2, **padding)

        ttk.Label(frame, text="To (output folder)").grid(row=2, column=0, sticky="w", **padding)
        ttk.Entry(frame, textvariable=self.output_var, width=62).grid(
            row=2, column=1, sticky="ew", **padding
        )
        ttk.Button(frame, text="Browse…", command=self._browse_output).grid(row=2, column=2, **padding)

        ref_row = ttk.Frame(frame)
        ref_row.grid(row=3, column=0, columnspan=3, sticky="ew", padx=12, pady=4)
        ttk.Checkbutton(
            ref_row,
            text="Use reference folder (name known people)",
            variable=self.use_reference_var,
            command=self._toggle_reference,
        ).pack(side=tk.LEFT)
        self.reference_entry = ttk.Entry(ref_row, textvariable=self.reference_var, width=48)
        self.reference_entry.pack(side=tk.LEFT, padx=(12, 6), fill=tk.X, expand=True)
        self.reference_button = ttk.Button(ref_row, text="Browse…", command=self._browse_reference)
        self.reference_button.pack(side=tk.LEFT)
        self._toggle_reference()

        options = ttk.Frame(frame)
        options.grid(row=4, column=0, columnspan=3, sticky="w", padx=12, pady=4)
        ttk.Label(options, text="Speed (parallel workers):").pack(side=tk.LEFT)
        ttk.Spinbox(
            options,
            from_=1,
            to=16,
            width=4,
            textvariable=self.workers_var,
        ).pack(side=tk.LEFT, padx=(8, 0))

        action_row = ttk.Frame(frame)
        action_row.grid(row=5, column=0, columnspan=3, sticky="ew", padx=12, pady=(8, 4))
        self.start_button = ttk.Button(action_row, text="Start sorting", command=self._start)
        self.start_button.pack(side=tk.LEFT)
        self.cancel_button = ttk.Button(
            action_row, text="Cancel", command=self._cancel, state=tk.DISABLED
        )
        self.cancel_button.pack(side=tk.LEFT, padx=(8, 0))

        status_frame = ttk.LabelFrame(frame, text="Progress", padding=10)
        status_frame.grid(row=6, column=0, columnspan=3, sticky="ew", padx=12, pady=8)
        ttk.Label(status_frame, textvariable=self.phase_var, font=("Segoe UI", 10, "bold")).pack(
            anchor="w"
        )
        self.progress = ttk.Progressbar(
            status_frame,
            variable=self.progress_var,
            maximum=100,
            mode="determinate",
        )
        self.progress.pack(fill=tk.X, pady=(6, 4))
        ttk.Label(status_frame, textvariable=self.status_var, wraplength=680).pack(anchor="w")

        log_frame = ttk.LabelFrame(frame, text="Activity log", padding=8)
        log_frame.grid(row=7, column=0, columnspan=3, sticky="nsew", padx=12, pady=(0, 4))
        frame.rowconfigure(7, weight=1)
        frame.columnconfigure(1, weight=1)

        self.log = tk.Text(log_frame, height=16, wrap=tk.WORD, state=tk.DISABLED, font=("Consolas", 9))
        scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        hint = ttk.Label(
            frame,
            text=(
                "Auto mode groups similar faces into person_001, person_002, … "
                "Reference mode matches against named folders."
            ),
            wraplength=700,
            foreground="#555555",
        )
        hint.grid(row=8, column=0, columnspan=3, sticky="w", padx=12, pady=(4, 0))

    def _toggle_reference(self) -> None:
        enabled = self.use_reference_var.get()
        state = tk.NORMAL if enabled else tk.DISABLED
        self.reference_entry.configure(state=state)
        self.reference_button.configure(state=state)

    def _browse_source(self) -> None:
        path = filedialog.askdirectory(title="Select source folder")
        if path:
            self.source_var.set(path)

    def _browse_output(self) -> None:
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_var.set(path)

    def _browse_reference(self) -> None:
        path = filedialog.askdirectory(title="Select reference folder")
        if path:
            self.reference_var.set(path)

    def _append_log(self, text: str) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, text + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _set_running(self, running: bool) -> None:
        state = tk.DISABLED if running else tk.NORMAL
        self.start_button.configure(state=state)
        self.cancel_button.configure(state=tk.NORMAL if running else tk.DISABLED)
        self.reference_entry.configure(state=state if self.use_reference_var.get() else tk.DISABLED)
        self.reference_button.configure(
            state=state if self.use_reference_var.get() else tk.DISABLED
        )

    def _validate(self) -> SortConfig | None:
        source = self.source_var.get().strip()
        output = self.output_var.get().strip()
        if not source or not output:
            messagebox.showerror("Missing paths", "Choose both a source folder and an output folder.")
            return None

        source_path = Path(source).expanduser().resolve()
        output_path = Path(output).expanduser().resolve()

        if not source_path.exists():
            messagebox.showerror("Invalid source", f"Source folder does not exist:\n{source_path}")
            return None
        if not source_path.is_dir():
            messagebox.showerror("Invalid source", f"Source path is not a folder:\n{source_path}")
            return None
        if source_path == output_path:
            messagebox.showerror(
                "Same folder",
                "Source and output must be different folders.\n\n"
                "Put your photos in one folder (From) and choose a separate folder "
                "for sorted results (To).",
            )
            return None

        reference_path: Path | None = None
        auto_cluster = True
        if self.use_reference_var.get():
            reference = self.reference_var.get().strip()
            if not reference:
                messagebox.showerror("Missing reference", "Choose a reference folder or disable reference mode.")
                return None
            reference_path = Path(reference).expanduser().resolve()
            if not reference_path.exists():
                messagebox.showerror(
                    "Invalid reference", f"Reference folder does not exist:\n{reference_path}"
                )
                return None
            auto_cluster = False

        try:
            workers = int(self.workers_var.get())
        except tk.TclError:
            workers = DEFAULT_WORKERS

        return SortConfig(
            scan_folder=source_path,
            output_folder=output_path,
            reference_folder=reference_path,
            auto_cluster=auto_cluster,
            workers=max(1, min(16, workers)),
        )

    def _progress_callback(
        self, phase: str, current: int, total: int, message: str, detail: str = ""
    ) -> None:
        self.events.put((phase, current, total, message, detail))

    def _run_sort(self, config: SortConfig) -> None:
        try:
            result = sort_images(
                config,
                progress=self._progress_callback,
                cancel_event=self.cancel_event,
            )
            self.events.put(("result", result.image_count, result.group_count, str(result.report_path), ""))
        except InterruptedError:
            self.events.put(("cancelled", 0, 0, "Sorting cancelled.", ""))
        except Exception as exc:
            self.events.put(("error", 0, 0, str(exc), ""))
        finally:
            self.events.put(None)

    def _start(self) -> None:
        config = self._validate()
        if config is None:
            return

        self.cancel_event.clear()
        self.progress_var.set(0)
        self.phase_var.set("Starting…")
        self.status_var.set("Preparing…")
        self.log.configure(state=tk.NORMAL)
        self.log.delete("1.0", tk.END)
        self.log.configure(state=tk.DISABLED)
        self._set_running(True)

        self.worker = threading.Thread(target=self._run_sort, args=(config,), daemon=True)
        self.worker.start()

    def _cancel(self) -> None:
        self.cancel_event.set()
        self.status_var.set("Cancelling…")

    def _poll_events(self) -> None:
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break

            if event is None:
                self._set_running(False)
                continue

            phase, current, total, message, detail = event

            if phase == "error":
                self.phase_var.set("Error")
                self.status_var.set(message.split("\n", maxsplit=1)[0])
                self._append_log(f"ERROR:\n{message}")
                messagebox.showerror("Sorting failed", message)
                continue

            if phase == "cancelled":
                self.phase_var.set("Cancelled")
                self.status_var.set(message)
                self._append_log(message)
                continue

            if phase == "result":
                self.phase_var.set("Complete")
                self.progress_var.set(100)
                self.status_var.set(
                    f"Done — {current} images sorted into {total} groups. Report: {message}"
                )
                self._append_log(self.status_var.get())
                messagebox.showinfo(
                    "Sorting complete",
                    f"Processed {current} images into {total} face groups.\n\nReport:\n{message}",
                )
                continue

            label = PHASE_LABELS.get(phase, phase.title())
            self.phase_var.set(label)
            self.status_var.set(message)
            if total > 0:
                self.progress_var.set(min(100.0, (current / total) * 100.0))
            if detail:
                self._append_log(f"{message} — {detail}")
            elif phase in {"scan", "detect", "match", "copy"}:
                self._append_log(message)

        self.root.after(100, self._poll_events)


def main() -> None:
    root = tk.Tk()
    style = ttk.Style()
    if "vista" in style.theme_names():
        style.theme_use("vista")
    FaceSorterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
