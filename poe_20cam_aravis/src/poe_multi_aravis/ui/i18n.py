# -*- coding: utf-8 -*-
"""
ui/i18n.py
===========
Chinese / English localisation for the 20-camera Aravis monitor.
Switch language at runtime; the main window re-translates every label.
"""

from __future__ import annotations

_LANG = "zh"

TR: dict[str, tuple[str, str]] = {
    # ── window / top bar ──
    "win_title":          ("POE 多相機監控系統 (Aravis)  ─  最多 {n} 台 PoE 相機",
                           "POE Multi-Camera Monitor (Aravis)  ─  up to {n} PoE cameras"),
    "subtitle":           ("Aravis GigE Vision", "Aravis GigE Vision"),
    "lang_btn":           ("🌐 EN", "🌐 中"),
    "lang_tip":           ("切換中／英介面", "Switch Chinese / English"),

    # ── hardware monitor ──
    "hw_monitor":         ("硬體監控", "Hardware Monitor"),
    "cpu":                ("CPU", "CPU"),
    "ram":                ("RAM", "RAM"),
    "gpu":                ("GPU", "GPU"),
    "vram":               ("VRAM", "VRAM"),
    "streaming_n":        ("串流中：{n} 台", "Streaming: {n}"),

    # ── camera control ──
    "cam_control":        ("相機控制", "Camera Control"),
    "refresh_dev":        ("刷新設備列表", "Refresh Devices"),
    "compute_mode":       ("運算模式", "Compute Mode"),
    "cuda_na":            ("未偵測到 CUDA", "CUDA not detected"),
    "cuda_err":           ("CUDA 錯誤", "CUDA Error"),
    "cuda_fallback":      ("未偵測到 CUDA，已自動切換至 CPU 模式",
                           "CUDA not detected; switched to CPU mode"),
    "start_all":          ("▶  開啟全部相機", "▶  Start All Cameras"),
    "stop_all":           ("■  停止全部相機", "■  Stop All Cameras"),

    # ── global stream params ──
    "global_params":      ("全域串流參數", "Global Stream Params"),
    "fps_limit":          ("FPS 限制（每台）", "FPS Limit (per cam)"),
    "threshold":          ("相減靈敏度 (Threshold)", "Subtraction Sensitivity"),
    "sub_all_off":        ("🔁  全部影像相減 ON/OFF", "🔁  Subtract All ON/OFF"),
    "sub_all_on":         ("🔁  全部相減 ON", "🔁  Subtract All ON"),
    "reset_all_bg":       ("🔄  全部重設背景快照", "🔄  Reset All Backgrounds"),
    "reset_all_bg_tip":   ("清除所有相機的背景快照，下一幀重新拍攝",
                           "Clear all cameras' background snapshots; recapture next frame"),
    "alert_hdr":          ("全域覆蓋率 Alert 閾值", "Global Coverage Alert Threshold"),
    "global_alert_tip":   ("套用到所有相機的預設 Alert 閾值",
                           "Default alert threshold applied to all cameras"),
    "apply_all":          ("套用至全部", "Apply to All"),

    # ── camera parameters ──
    "cam_params":         ("相機參數", "Camera Parameters"),
    "exposure":           ("曝光 (µs)", "Exposure (µs)"),
    "gain":               ("增益 (dB)", "Gain (dB)"),
    "frame_rate":         ("幀率 (FPS)", "Frame Rate"),
    "get_param":          ("讀取參數", "Get Param"),
    "set_param":          ("套用參數", "Set Param"),
    "param_ready":        ("就緒", "Ready"),
    "param_read_ok":      ("已從 {name} 讀取", "Read from {name}"),
    "param_applied":      ("已套用至 {n} 台相機", "Applied to {n} camera(s)"),
    "param_no_cam":       ("尚未連線任何相機", "No camera connected"),
    "param_failed":       ("失敗：{items}", "Failed: {items}"),

    # ── detected devices ──
    "detected_dev":       ("偵測到的設備", "Detected Devices"),
    "no_device":          ("（未偵測到設備）", "(No devices found)"),
    "unassigned":         ("─ 未指派 ─", "─ Unassigned ─"),

    # ── camera tile ──
    "subtract":           ("影像相減", "Subtract"),
    "sub_on":             ("相減 ON", "Sub ON"),
    "reset_bg":           ("重設背景", "Reset BG"),
    "reset_bg_tip":       ("清除背景快照，下一幀重新拍攝（僅相減開啟時有效）",
                           "Clear background snapshot; recapture next frame (only when subtraction is on)"),
    "alert_spin_tip":     ("覆蓋率超過此值觸發 Alert",
                           "Trigger alert when coverage exceeds this"),
    "diff":               ("差異", "Diff"),
    "coverage_pct":       ("覆蓋率", "Coverage"),

    # ── status ──
    "status_disconnected": ("未連線", "Disconnected"),
    "status_connected":    ("已連線", "Connected"),
    "status_streaming":    ("串流中", "Streaming"),
    "status_stopped":      ("已停止", "Stopped"),
    "status_error":        ("錯誤", "Error"),
    "status_connecting":   ("連線中…", "Connecting…"),
    "status_reconnecting": ("重新連線中 {a}/{m}…", "Reconnecting {a}/{m}…"),
    "waiting_conn":        ("等待連線…", "Waiting for connection…"),
    "waiting_img":         ("等待影像…", "Waiting for image…"),

    # ── chart ──
    "show_chart":         ("📈 折線圖", "📈 Chart"),
    "show_live":          ("🎥 即時畫面", "🎥 Live"),
    "chart_days":         ("顯示天數", "Days"),
    "day_n":              ("{d} 天", "{d} d"),
    "chart_ylabel":       ("覆蓋率 (%)", "Coverage (%)"),
    "chart_title":        ("每分鐘平均覆蓋率", "Per-minute Avg Coverage"),
    "chart_no_data":      ("尚無紀錄資料\n（開啟『影像相減』後每分鐘自動記錄）",
                           "No data yet\n(records every minute once subtraction is ON)"),

    # ── overview ──
    "overview":           ("📊 總覽", "📊 Overview"),
    "overview_title":     ("📊  即時串流矩陣總覽", "📊  Live Stream Matrix Dashboard"),
    "back_to_tabs":       ("← 返回分頁", "← Back to Tabs"),
    "page_tab":           ("📷 P{p}  #{a:02d}-{b:02d}", "📷 P{p}  #{a:02d}-{b:02d}"),

    # ── fullscreen ──
    "fs_title":           ("全螢幕 — {title}", "Fullscreen — {title}"),
    "fs_hint":            ("按 Esc 或雙擊關閉", "Press Esc or double-click to close"),

    # ── messages ──
    "err":                ("錯誤", "Error"),
    "info":               ("訊息", "Info"),
}


def set_language(lang: str) -> None:
    global _LANG
    if lang in ("zh", "en"):
        _LANG = lang


def toggle_language() -> str:
    set_language("en" if _LANG == "zh" else "zh")
    return _LANG


def get_language() -> str:
    return _LANG


def tr(key: str, **kwargs) -> str:
    entry = TR.get(key)
    if entry is None:
        return key
    text = entry[1 if _LANG == "en" else 0]
    if kwargs:
        try:
            text = text.format(**kwargs)
        except Exception:
            pass
    return text
