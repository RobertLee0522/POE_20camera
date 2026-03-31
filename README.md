# Hikvision Multi-Camera Streaming System (V4.0)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyQt5](https://img.shields.io/badge/PyQt-5-green.svg)](https://www.riverbankcomputing.com/software/pyqt/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-orange.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A high-performance industrial camera monitoring solution supporting up to **20 PoE Area Scan Cameras** (GigE/USB3 Vision). Features real-time background subtraction, CUDA acceleration, and cross-platform support (Windows/Linux).

---

## 🌟 Key Features

- **Multi-Camera Management**: Supports up to 20 cameras with a scalable 2-per-page layout.
- **Background Subtraction Engine**: Real-time motion detection and coverage percentage calculation.
- **Dual Processing Modes**: Toggle between **CPU** and **CUDA (NVIDIA GPU)** for optimal performance.
- **Hardware Monitoring**: Built-in background thread for CPU, RAM, and GPU (NVIDIA) usage monitoring.
- **Cross-Platform**: Seamlessly works on Windows and Linux (Ubuntu/Debian) with automated SDK path detection.
- **Fullscreen Preview**: Double-click any camera slot for a high-resolution, full-screen live view.
- **Industrial UI**: Dark-themed, high-contrast interface designed for factory environments.

## 🛠 Tech Stack

- **Languge**: Python 3.10+
- **GUI**: PyQt5
- **Vision**: OpenCV (with CUDA support), Hikvision MVS SDK
- **Hardare Info**: `psutil`, `nvidia-smi`
- **Concurrency**: `QThread`, `threading.Lock`

## ⚙️ Prerequisites & Installation

### 1. Hikvision MVS SDK
You must install the **MVS (Machine Vision Software)** from Hikrobot.
- [Download MVS SDK](https://www.hikrobotics.com/en/machinevision/service/download?module=0)

### 2. Python Dependencies
```bash
pip install numpy opencv-python pyqt5 psutil
```
*Note: For CUDA acceleration, ensure you have an NVIDIA GPU and `opencv-python-headless` or a custom OpenCV build with CUDA enabled.*

### 3. Linux Network Configuration (PoE Cameras)
Industrial cameras often require **Jumbo Frames (MTU 9000)** and **Link-Local IP** addressing:
```bash
# Set MTU to 9000
sudo ip link set eth0 mtu 9000

# Set NetworkManager to link-local
nmcli connection modify "YourConnection" ipv4.method link-local
```

## 🚀 Quick Start

1. Connect your cameras to the PoE switch/network.
2. Run the main application:
   ```bash
   python multi_cam_stream.py
   ```
3. Click **"Refresh Devices"** to detect cameras.
4. Click **"Start All Cameras"** to begin streaming.

## 📸 Screenshots

*(Add your screenshots here)*

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 中文簡介 (Traditional Chinese)

這是一套為工業場景設計的高效能多路相機監控系統，基於海康威視（Hikvision）MVS SDK 開發。

**核心功能：**
1. **20 路並行處理**：支援最多 20 台面陣相機，提供流暢的預覽畫質。
2. **影像差異分析**：內建背景相減演算法，可即時偵測畫面變化並計算覆蓋率（Coverage %）。
3. **GPU 加速**：支援 CUDA 運算，顯著降低高解析度多路取像時的 CPU 負擔。
4. **硬體監控**：即時顯示 CPU、RAM 及 NVIDIA GPU 資源佔用。
5. **跨平台相容**：完美支援 Windows 與 Linux，並針對 Linux 環境下的 PoE 連線問題提供自動化處理方案。
