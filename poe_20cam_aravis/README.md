# POE 20-Camera Monitor — Aravis Edition

Up to **20 PoE / GigE Vision cameras** in one dark-themed PyQt5 dashboard, with
frozen-background subtraction, per-camera coverage alerts, CPU **or** batched
CUDA processing, per-camera CSV logging and automatic reconnection.

This is a **new, self-contained application** living in `poe_20cam_aravis/`.
The original Hikvision-MVS programs at the repository root
(`multi_cam_stream.py`, `single_cam_stream.py`, `modules/`, …) are untouched
and still run exactly as before — see [`MIGRATION.md`](MIGRATION.md) for what
changed and why.

The vendor SDK is gone. Cameras are driven through **Aravis 0.10**, the open
GenICam/GigE-Vision library, so any GenICam-compliant camera works — Hikrobot,
Basler, FLIR, Allied Vision, IDS and others — on Windows or Linux, with no
`MvImport` on `sys.path`.

---

## Highlights

| | |
|---|---|
| **Vendor-neutral** | Aravis 0.10 via GObject Introspection; no MVS SDK |
| **20 cameras** | 10 tabs × 2 large tiles, plus a 4×5 live overview matrix |
| **Coverage engine** | Frozen background snapshot, contour-area coverage %, per-camera alert threshold with a flashing red border |
| **CPU or CUDA** | One worker per camera on CPU, or a single centralised GPU batch processor that stacks all 20 cameras into one CUDA upload |
| **Auto-reconnect** | Per-camera reconnect loop that reapplies exposure/gain/ROI and restarts the stream |
| **Coverage logging** | One CSV per camera keyed on serial number, one averaged row per minute, plotted in-app for up to 5 days |
| **Layered design** | `aravis_backend` → `imaging` → `services` → `ui`; the UI never touches Aravis, OpenCV pipelines or torch |
| **Bilingual** | 中文 / English, switched at runtime |

---

## Layout

```
poe_20cam_aravis/
├── config/default.yaml            every tunable, documented inline
├── run.sh                         launch from a source checkout
├── scripts/
│   ├── check_aravis.py            environment sanity check — run this first
│   ├── list_cameras.py            what Aravis sees + each camera's log key
│   └── run_fake_cameras.py        spawn N simulated cameras (no hardware)
├── src/poe_multi_aravis/
│   ├── app.py                     entry point + Aravis/Qt bootstrap
│   ├── settings.py                validated YAML config
│   ├── domain/                    models, error hierarchy, state machine
│   ├── aravis_backend/            the ONLY code that imports Aravis
│   │   ├── discovery.py           enumerate without opening devices
│   │   ├── camera.py              one CameraSession per camera
│   │   ├── stream.py              buffer pool + acquisition worker
│   │   ├── feature_adapter.py     safe GenICam feature access
│   │   └── pixel_formats.py       Mono/Bayer/RGB/YUV → BGR
│   ├── imaging/                   analyzer, white balance, resizer, coverage
│   ├── services/
│   │   ├── multi_camera_service.py  the orchestrator the UI talks to
│   │   ├── camera_worker.py         one processing QThread per camera
│   │   ├── gpu_batch_processor.py   single shared CUDA batch thread
│   │   ├── coverage_logger.py       per-camera CSV + registry
│   │   └── hardware_monitor.py      CPU / RAM / GPU sampling
│   └── ui/                        theme, i18n, tiles, overview, main window
└── tests/                         unit suite (no camera, no Qt required)
```

---

## Install

### 1. Aravis 0.10 + GObject Introspection

**Ubuntu / Debian**

```bash
sudo apt install -y aravis-tools gir1.2-aravis-0.10 libaravis-0.10-0 \
                    python3-gi python3-gi-cairo
```

If your distribution only ships Aravis 0.8, build 0.10 from source
(`meson setup build -Dintrospection=enabled`) and point the app at it:

```bash
export ARAVIS_TYPELIB_DIR=/usr/local/lib/x86_64-linux-gnu/girepository-1.0
```

**Windows** — install Aravis with its typelib plus PyGObject (MSYS2 is the
usual route), then make sure `GI_TYPELIB_PATH` includes the Aravis typelib
directory.

### 2. Python packages

```bash
pip install PyQt5 numpy opencv-python psutil matplotlib PyYAML
pip install torch          # optional — only for CUDA mode
```

PyGObject is a **system** package, not a pip dependency.

### 3. Verify

```bash
python3 scripts/check_aravis.py
```

It reports every component and lists the cameras Aravis can see.

---

## Network setup for 20 PoE cameras

Twenty cameras on one link is a bandwidth problem before it is a software
problem. On the host NIC:

```bash
# Jumbo frames — the single biggest reliability win for GigE Vision
sudo ip link set eth0 mtu 9000

# Large receive buffers (Aravis also raises these per stream)
sudo sysctl -w net.core.rmem_max=26214400
sudo sysctl -w net.core.rmem_default=26214400

# Link-local addressing so cameras self-assign 169.254.x.x
nmcli connection modify "YourConnection" ipv4.method link-local
```

Also worth doing:

* Spread cameras across **multiple NICs** — a single 1 GbE port cannot carry
  20 high-resolution streams at full frame rate.
* Lower each camera's frame rate (`Camera Parameters → Frame Rate`) or its
  ROI rather than dropping frames at the host.
* Increase the cameras' **GevSCPD** (inter-packet delay) if you see
  incomplete buffers; this staggers bursts from cameras sharing a switch.
* Keep `camera.buffer_count` modest (default **8**). Twenty cameras × 24
  buffers × a multi-megapixel payload reserves gigabytes before a single
  frame arrives.

---

## Run

```bash
./run.sh                       # normal
./run.sh --log-level DEBUG     # verbose
./run.sh --fake                # Aravis in-process fake camera

# 20 simulated cameras (needs aravis-tools), in two shells:
python3 scripts/run_fake_cameras.py 20
./run.sh
```

Or as a module: `PYTHONPATH=src python3 -m poe_multi_aravis.app`

### Typical session

1. **Refresh Devices** — cameras appear in the sidebar list and in every
   tile's dropdown.
2. Pick **CPU** or **CUDA (GPU)** (locked while streaming).
3. **Start All Cameras**.
4. Turn on **Subtract** per tile, or **Subtract All** in the sidebar. The
   first frame after enabling becomes the frozen background.
5. Set the alert threshold per tile or globally — a tile whose coverage
   crosses it flashes a red border, in the tab view and in the overview.
6. **Reset BG** re-freezes the background from the next frame.
7. **📊 Overview** switches to the 4×5 matrix; double-click any tile
   (either view) for fullscreen, Esc to exit.
8. **📈 Chart** flips a tile to its coverage history.

---

## Configuration

Everything lives in `config/default.yaml`; pass another file with
`--config PATH`. Missing or invalid keys fall back to validated defaults, so a
half-written config can never stop the app from starting.

| Key | Default | Notes |
|---|---|---|
| `camera.max_cameras` | 20 | Upper bound on discovered cameras |
| `camera.buffer_count` | 8 | Aravis buffers **per camera** |
| `camera.reconnect_enabled` | true | Per-camera reconnect loop |
| `camera.consecutive_failure_threshold` | 5 | Bad frames before a camera is declared down |
| `processing.mode` | `cpu` | `cpu` or `cuda` |
| `processing.target_fps` | 15 | Per-camera processing rate cap |
| `processing.analysis_width/height` | 640×480 | Coverage analysis resolution |
| `processing.difference_threshold` | 5 | Subtraction sensitivity (1–100) |
| `processing.alert_threshold_percent` | 50.0 | Default coverage alert level |
| `processing.gpu_max_batch` | 20 | Cameras coalesced into one CUDA batch |
| `processing.enable_image_analysis` | false | RGB/HSL/brightness stats per frame |
| `logging.interval_seconds` | 60 | One averaged CSV row per camera |
| `ui.cameras_per_page` | 2 | 20 / 2 = 10 tabs |
| `ui.display_max_width` | 960 | Downscale-only cap on frames sent to the UI |

---

## Coverage logs

One CSV per camera under `logging.directory` (default `./data/coverage`),
named from the camera's **serial number** (then IP, then device id), so a
camera keeps its file across restarts and re-enumerations. Run
`scripts/list_cameras.py` to see which file belongs to which camera.

```csv
timestamp,camera_id,coverage_percent,acquisition_fps,processing_fps,processing_mode,alert_active,samples
2026-08-07 14:31,SN0001,23.47,15.0,14.8,cpu,0,892
```

Each row is the **mean** over the interval, with `samples` recording how many
frames it averaged.

---

## CPU vs CUDA

**CPU** — each camera's worker runs the OpenCV pipeline on its own thread:
grayscale → resize → blur → abs-diff → threshold → close/open → contours.

**CUDA** — every worker hands its 640×480 grayscale to one shared
`GpuBatchProcessor`, which stacks up to `gpu_max_batch` cameras into a single
`(N, 1, H, W)` upload and runs the whole batch in one pass. Twenty independent
CUDA contexts would thrash both the GPU and the GIL; one batching thread does
not. The processor also sets `cudaDeviceScheduleBlockingSync` and
`torch.set_num_threads(1)`, without which waiting threads busy-spin and GPU
mode burns *more* CPU than CPU mode.

If PyTorch or CUDA is missing, the radio button is disabled and everything
stays on CPU — there is no silent half-GPU state.

---

## Tests

```bash
python3 -m pytest            # 54 tests, no camera or GPU required
```

The suite covers settings validation, the camera state machine, camera
identity/log-key rules, the coverage processor (frozen background, thresholds,
resolution changes, reset), coverage logging and averaging, and the imaging
helpers. Modules that need Aravis or Qt are exercised by the smoke path in
`scripts/check_aravis.py` and by running the app against fake cameras.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Failed to load the Aravis GObject-Introspection bindings` | Install `gir1.2-aravis-0.10`, or set `ARAVIS_TYPELIB_DIR` to the directory holding `Aravis-0.10.typelib` |
| No cameras found | Check the PoE link and MTU; confirm with `arv-tool-0.10`; make sure nothing else holds the camera open |
| Frames arrive then stop | Raise `camera.frame_timeout_ms`, enable jumbo frames, or increase the cameras' inter-packet delay |
| `PAYLOAD_NOT_SUPPORTED` / no usable frames | Handled automatically — `GevGVSPExtendedIDMode` is forced to `Off` at connect |
| Unsupported pixel format | Switch the camera to Mono8/16, RGB8, BGR8, a plain BayerXX, or YUV422 — *packed* variants are not supported |
| Qt fails to start / wrong `libQt5Core` | Already handled: the launcher strips `/opt/MVS` from `LD_LIBRARY_PATH` and re-execs once |
| CUDA radio disabled | `pip install torch` with CUDA support; verify with `scripts/check_aravis.py` |
| High CPU with 20 cameras | Lower `processing.target_fps`, lower `ui.display_max_width`, keep `enable_image_analysis` off, or switch to CUDA |

---

## License

MIT — same as the parent repository.
