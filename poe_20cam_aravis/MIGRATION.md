# Migration — `multi_cam_stream.py` (Hikvision MVS) → `poe_20cam_aravis` (Aravis)

## Nothing was removed

The original application is untouched and still runs:

```
multi_cam_stream.py        single_cam_stream.py     modules/
BasicDemo.py               CamOperation_class.py    single_cam_analysis.py
PyUICBasicDemo.py          PyUICBasicDemo.ui
```

The Aravis rewrite lives entirely in the new `poe_20cam_aravis/` folder. The
two can coexist in the same checkout; they share no imports, no config and no
log directory. Run whichever you need:

```bash
python multi_cam_stream.py            # original, needs the MVS SDK
./poe_20cam_aravis/run.sh             # new, needs Aravis 0.10
```

---

## Why Aravis

| | MVS SDK (V4.0.1) | Aravis 0.10 |
|---|---|---|
| Vendor lock | Hikrobot only | Any GenICam / GigE Vision camera |
| Install | Proprietary SDK, `MvImport` appended to `sys.path` | Distribution packages (`gir1.2-aravis-0.10`) |
| API style | `ctypes` structs, `MV_CC_*` int return codes | GObject Introspection, real Python objects |
| Buffers | Manual `c_ubyte` arrays + `MV_CC_ConvertPixelType` | Aravis buffer pool + a typed converter |
| Failure signal | Return code checked at each call site | Exception hierarchy + a camera state machine |
| Qt conflict | Ships its own `libQt5Core`, hijacks PyQt5 via `LD_LIBRARY_PATH` | No Qt of its own; the launcher strips `/opt/MVS` anyway |

---

## Structural changes

The old program was one 2 300-line module holding UI, threading, SDK calls,
GPU code and logging. The new one is layered, and each layer is testable on
its own:

```
ui/  ──────────►  services/  ──────────►  aravis_backend/  ──────────►  Aravis
      Qt only        orchestration            the ONLY code
   no SDK, no        + threading             that imports gi
   torch, no cv2
   pipelines             │
                         ▼
                     imaging/   (pure numpy/OpenCV, no Qt, no camera)
```

### Where each piece went

| Old (`multi_cam_stream.py`) | New |
|---|---|
| `MvCamera` + `MV_CC_EnumDevices` | `aravis_backend/discovery.py` |
| `MV_CC_CreateHandle/OpenDevice/StartGrabbing` | `aravis_backend/camera.py` (`CameraSession`) |
| `MV_CC_GetOneFrameTimeout` loop | `aravis_backend/stream.py` (buffer pool + worker) |
| `MV_CC_ConvertPixelType` | `aravis_backend/pixel_formats.py` |
| `MV_CC_Get/SetFloatValue`, `SetEnumValue` | `aravis_backend/feature_adapter.py` |
| `CameraThread` (grab **and** process) | `services/camera_worker.py` (process only; Aravis owns acquisition) |
| `GPUBatchProcessor` | `services/gpu_batch_processor.py` |
| `MonitorThread` | `services/hardware_monitor.py` |
| `append_log` / `read_log` / `pop_minute_avg` | `services/coverage_logger.py` |
| `CameraWidget` | `ui/camera_tile.py` |
| `CompactCameraCard` | `ui/compact_card.py` |
| `CoverageChart` | `ui/chart_widget.py` |
| `FullscreenDialog` | `ui/fullscreen_dialog.py` |
| `TR` / `tr()` | `ui/i18n.py` |
| `_apply_dark_theme` | `ui/theme.py` |
| `MainWindow` (everything else) | `ui/main_window.py` + `services/multi_camera_service.py` |
| module-level constants | `settings.py` + `config/default.yaml` |

The backend, domain and imaging layers are shared in design with the
single-camera Aravis application (`listenmudasir/Poe-single-camera`), so a fix
in one is a mechanical port to the other.

---

## Behaviour: kept, changed, added

### Kept identical

* **Frozen background snapshot**, never a running average — the V4.0.1 fix for
  long-run coverage drift. Disabling subtraction still clears the snapshot;
  the next frame after enabling re-freezes it.
* Coverage = Σ(external contour areas) / total pixels × 100, computed at
  640×480 after a 21×21 Gaussian blur and a 5×5 ellipse close+open.
* Sensitivity slider 1–100, default 5.
* Per-camera and global alert thresholds; 400 ms flashing red border.
* 2 cameras per page, 10 tabs; 4×5 overview matrix; double-click fullscreen;
  PiP difference view; per-camera coverage chart with a 1–5 day selector.
* Per-minute averaged coverage records.
* CPU / CUDA radio pair, locked while streaming.
* 中文 / English runtime toggle.

### Changed

| | Before | After |
|---|---|---|
| Camera identity | Enumeration index `0..19` | Aravis `device_id`; the index is a UI concern only |
| Log filenames | `coverage_logs/<serial>.txt`, tab-separated | `data/coverage/<serial>.csv` with a header and FPS/mode/alert/sample columns |
| Log averaging | Accumulated in the **UI widget** | Accumulated in `CoverageLogger`, flushed on one timer |
| Threads per camera | 1 (grab + process) | Aravis acquisition thread + 1 processing worker |
| Buffers per camera | SDK default | `camera.buffer_count`, default 8 |
| Errors | Printed to stdout, return codes ignored | Exception hierarchy → status bar + logging |
| Tuning constants | Literals in the source | `config/default.yaml` |
| Startup | 20 matplotlib canvases built eagerly | Charts built on first use |

### Added

* **Per-camera auto-reconnect** with exponential retry, reapplying exposure,
  gain, frame rate, ROI and pixel format, then restarting the stream.
* **Camera state machine** (`DISCONNECTED → CONNECTING → CONNECTED → STARTING
  → STREAMING → STOPPING`, plus `ERROR`) driving the per-tile status dot.
* **GigE reliability tuning at connect**: `GevGVSPExtendedIDMode` forced to
  `Off` (some cameras otherwise produce `PAYLOAD_NOT_SUPPORTED` buffers under
  Aravis 0.10), GVSP packet size auto-negotiated, and per-stream socket
  buffer / packet-resend / timeout settings.
* **`ui.display_max_width`** — a downscale-only cap on frames handed to the Qt
  thread. Twenty full-resolution frames per tick was the dominant UI cost.
* **Graceful degradation everywhere**: one camera failing to open or to accept
  a parameter no longer aborts "start all" or "apply to all"; a saturated GPU
  drops a frame instead of stalling acquisition.
* **Orderly shutdown** — workers joined, streams drained, buffers returned,
  final log interval flushed. No `terminate()`.
* **Tests** (`pytest`, 54 cases) and three operator scripts
  (`check_aravis.py`, `list_cameras.py`, `run_fake_cameras.py`).

### Not carried over

* `TriggerMode` is set through `FeatureAdapter`, not `MV_TRIGGER_MODE_OFF`;
  Aravis clears triggers by default so cameras free-run without extra setup.
* USB3 Vision devices appear only if Aravis was built with USB3 support
  (`libusb`). GigE needs nothing extra.
* The MVS-specific `BasicDemo.py` / `CamOperation_class.py` samples have no
  equivalent — they demonstrate the vendor SDK, which is what this migration
  removes.

---

## Mapping your existing config

| Old constant | New key |
|---|---|
| `MAX_CAMERAS = 20` | `camera.max_cameras` |
| `CAMS_PER_PAGE = 2` | `ui.cameras_per_page` |
| `OV_COLS = 4` | `ui.overview_columns` |
| `LOG_DIR` | `logging.directory` |
| `LOG_INTERVAL_SEC = 60` | `logging.interval_seconds` |
| `target_fps = 15` | `processing.target_fps` |
| `diff_threshold = 5` | `processing.difference_threshold` |
| `proc_w, proc_h = 640, 480` | `processing.analysis_width/height` |
| Gaussian `(21, 21)` | `processing.gaussian_kernel` |
| Morphology `(5, 5)` | `processing.morphology_kernel` |
| Alert spin default `50.0` | `processing.alert_threshold_percent` |

Old `coverage_logs/*.txt` files are **not** read by the new app — the format
gained columns. Keep them for reference; the new CSVs start fresh.
