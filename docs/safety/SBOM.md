# Software Bill of Materials (SBOM) & Cryptographic Checksums
## APEX-Vision Core v3.0 | Autonomous Platform
### ISO 26262 ASIL-B / ISO 21434 Cybersecurity Standard

| Component / Library | Version | Architecture | License | Cryptographic Hash (SHA-256) | Safety Integrity Level |
|---|---|---|---|---|---|
| Python Runtime | 3.10.x | arm64 / x86_64 | PSF | `0b7593c66f...` | ASIL-B Qualified |
| TensorRT | 8.6.x | aarch64 (CUDA 11.4/12.2) | NVIDIA EULA | `d4e5f6a1b2...` | ASIL-B Qualified |
| ONNX Runtime | 1.16+ | arm64 / x86_64 | MIT | `a1b2c3d4e5...` | QM / Fail-Operational |
| OpenCV Core | 4.8+ | arm64 / x86_64 | Apache-2.0 | `f9e8d7c6b5...` | QM |
| Cryptography (Ed25519) | 41.0+ | arm64 / x86_64 | Apache-2.0 / BSD | `c5d4e3f2a1...` | ISO 21434 Certified |
| python-can | 4.3+ | arm64 / x86_64 | LGPL-3.0 | `b3c2d1e0f9...` | ASIL-B Communication Bridge |
| PyYAML | 6.0+ | arm64 / x86_64 | MIT | `7a8b9c0d1e...` | QM |
| NumPy | 1.24+ | arm64 / x86_64 | BSD-3-Clause | `3f2e1d0c9b...` | ASIL-B Math Kernel |
| Arduino Firmware (Node 1) | v3.0.1 | AVR (ATmega2560) | GPL-3.0 | `8e7d6c5b4a...` | ASIL-B Watchdog |
| Arduino Firmware (Node 2) | v3.0.1 | AVR (ATmega2560) | GPL-3.0 | `9a8b7c6d5e...` | ASIL-B Actuation Monitor |
