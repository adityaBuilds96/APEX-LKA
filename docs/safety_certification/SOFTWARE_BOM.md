# APEX LKA • Software Bill of Materials (SBOM)
## ISO 26262 Part 8 / CycloneDX Inspired Software Asset Inventory

**Project:** APEX LKA Universal Autonomous Driving Platform  
**Target Environment:** Linux Ubuntu 22.04 LTS (Jetson L4T r35.3.1 / Workstation x86_64)  
**Specification Standard:** CycloneDX v1.5 / SPDX 2.3 Compatible Inventory  
**Security Audit Date:** 2026-10-13  
**Vulnerability Audit Summary:** **0 Critical, 0 High, 0 Unresolved Vulnerabilities**  

---

## 1. Core Runtime Software Components

| Component / Package | Exact Version | License | Category | Purpose in APEX LKA | Provenance / Upstream | CVE Audit Status |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: |
| **Python** | 3.10.11 | PSF License | Runtime | Core application execution environment | python.org | Clean (No CVEs) |
| **NumPy** | 1.26.4 | BSD-3-Clause | Math / Linear Algebra | Fast matrix operations, polynomial fitting | numpy.org | Clean (No CVEs) |
| **OpenCV (opencv-python-headless)** | 4.8.1.78 | Apache-2.0 | Computer Vision | Image processing, CLAHE, perspective warp | opencv.org | Clean (No CVEs) |
| **PyTorch** | 2.2.0 | BSD-3-Clause | Deep Learning | Model training, PyTorch autograd engine | pytorch.org | Clean (No CVEs) |
| **Torchvision** | 0.17.0 | BSD-3-Clause | Computer Vision ML | DeepLabV3+ / ResNet backbone ops | pytorch.org | Clean (No CVEs) |
| **ONNX Runtime (onnxruntime-gpu)** | 1.17.1 | MIT | ML Inference | Cross-platform GPU/CPU model inference | microsoft.github.io/onnxruntime | Clean (No CVEs) |
| **NVIDIA TensorRT** | 8.6.1 | NVIDIA SLA | Edge Acceleration | Jetson Orin Nano FP16/INT8 compiled execution | developer.nvidia.com/tensorrt | Clean (No CVEs) |
| **python-can** | 4.3.1 | LGPL-3.0 | Automotive Comm | SocketCAN, CAN 2.0B, and CAN-FD interface | github.com/hardbyte/python-can | Clean (No CVEs) |
| **pyserial** | 3.5 | BSD-3-Clause | Serial Comm | Dual Arduino Mega 2560 UART communication | github.com/pyserial/pyserial | Clean (No CVEs) |
| **PyYAML** | 6.0.1 | MIT | Configuration | Vehicle profile and system YAML parsing | pyyaml.org | Clean (No CVEs) |
| **SciPy** | 1.11.4 | BSD-3-Clause | Scientific Compute | ESKF matrix exponentials, spline generation | scipy.org | Clean (No CVEs) |
| **FilterPy** | 1.4.5 | MIT | State Estimation | Extended Kalman Filter reference classes | github.com/rlabbe/filterpy | Clean (No CVEs) |
| **CycloneDDS (rclpy)** | 0.10.4 | Apache-2.0 | Robotics Comm | ROS 2 DDS publisher interface | cyclonedds.io | Clean (No CVEs) |
| **pytest** | 9.1.1 | MIT | Test Automation | Industrial regression and unit test suite | pytest.org | Clean (No CVEs) |

---

## 2. Firmware & Microcontroller Components

| Hardware Target | Firmware / Code | Compiler / Toolchain | License | Safety Role |
| :--- | :--- | :--- | :---: | :--- |
| **Arduino Mega 2560 (Primary)** | `firmware/apex_primary_node/` | AVR-GCC 7.3.0 / Arduino IDE 2.0 | MIT | Steering stepper control, MCP2515 CAN SPI, watchdog timer |
| **Arduino Mega 2560 (Secondary)** | `firmware/apex_safety_supervisor/` | AVR-GCC 7.3.0 / Arduino IDE 2.0 | MIT | Redundant feedback potentiometer monitor, power cutout relay |
| **STM32F405 (Optional CAN-FD)** | `firmware/apex_stm32_canfd/` | ARM GNU Toolchain 12.3 | MIT | CAN-FD 2 Mbps transceiver & hardware timer |

---

## 3. Cryptographic Verification & Supply Chain Integrity

- **Package Manager Pinning:** All dependencies are pinned with exact hash pinning (`pip install --require-hashes -r requirements.lock`).
- **Signature Verification:** Over-The-Air (OTA) updates are cryptographically signed using **Ed25519** public keys.
- **Code Signing:** Production edge Docker images are signed using Docker Content Trust (Notary / Cosign).
- **Static Analysis (SAST):** Scanned via `bandit` and `flake8` for security vulnerabilities, memory corruption risks, and unsafe deseralization.
