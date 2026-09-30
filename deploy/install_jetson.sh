#!/usr/bin/env bash
# ==============================================================================
# APEX-LKA One-Command Setup Script for NVIDIA Jetson Orin Nano
# Target: Ubuntu 22.04 LTS (JetPack 5.x / 6.x) | SAE BAJA 2027 Autonomous
# ==============================================================================

set -euo pipefail

echo "======================================================================"
echo "🏎️  APEX-LKA Jetson Orin Nano Automated Installation & Configuration"
echo "======================================================================"

if [ "$EUID" -ne 0 ]; then
  echo "[-] Please run this script with sudo: sudo ./install_jetson.sh"
  exit 1
fi

TARGET_USER="${SUDO_USER:-$USER}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[+] Target system user: ${TARGET_USER}"
echo "[+] Script directory: ${SCRIPT_DIR}"

# 1. Update Package Repositories
echo "[+] Updating apt repositories..."
apt-get update -y

# 2. Install System Dependencies & GStreamer
echo "[+] Installing system utilities, V4L2, GStreamer, and CAN utilities..."
apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    v4l-utils \
    can-utils \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    libgstreamer1.0-dev

# 3. Add User to Hardware Peripheral Groups
echo "[+] Granting permissions for Camera (video), Arduino Serial (dialout), and CAN bus..."
usermod -aG video "${TARGET_USER}"
usermod -aG dialout "${TARGET_USER}"

# 4. Install Edge Python Requirements
echo "[+] Installing minimal edge Python packages..."
python3 -m pip install --upgrade pip
python3 -m pip install -r "${SCRIPT_DIR}/requirements_edge.txt"

# 5. Configure SocketCAN can0 Interface
echo "[+] Configuring SocketCAN interface (can0 @ 500kbps)..."
modprobe can
modprobe can_raw
modprobe mttcan || modprobe can_dev

CAN_CONFIG_FILE="/etc/network/interfaces.d/can0"
mkdir -p /etc/network/interfaces.d
cat << 'EOF' > "${CAN_CONFIG_FILE}"
auto can0
iface can0 inet manual
    pre-up ip link set can0 type can bitrate 500000
    up ifconfig can0 up
    down ifconfig can0 down
EOF

# Bring up can0 if present
if ip link show can0 > /dev/null 2>&1; then
    ip link set can0 type can bitrate 500000 || true
    ip link set up can0 || true
    echo "[+] can0 interface activated at 500 kbps."
else
    echo "[!] can0 physical interface not currently detected. SocketCAN config saved for boot."
fi

# 6. Install and Enable Systemd Service
echo "[+] Installing systemd auto-boot service..."
SERVICE_SRC="${SCRIPT_DIR}/apex-lka.service"
SERVICE_DEST="/etc/systemd/system/apex-lka.service"

if [ -f "${SERVICE_SRC}" ]; then
    # Dynamically update paths for current user
    sed "s|/home/jetson|/home/${TARGET_USER}|g" "${SERVICE_SRC}" > "${SERVICE_DEST}"
    sed -i "s|User=jetson|User=${TARGET_USER}|g" "${SERVICE_DEST}"
    sed -i "s|Group=jetson|Group=${TARGET_USER}|g" "${SERVICE_DEST}"

    systemctl daemon-reload
    systemctl enable apex-lka.service
    echo "[+] apex-lka.service enabled on boot!"
else
    echo "[-] Warning: apex-lka.service file not found at ${SERVICE_SRC}"
fi

# 7. Hardware Self-Test
echo "======================================================================"
echo "[+] Hardware Self-Test:"
echo "----------------------------------------------------------------------"
echo "  [Camera Test]"
v4l2-ctl --list-devices || echo "    No USB cameras found on V4L2."

echo "  [CAN Bus Test]"
ip -details link show can0 || echo "    can0 not active."

echo "  [Arduino Serial Test]"
ls -l /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || echo "    No Arduino Serial ports currently plugged in."

echo "======================================================================"
echo "✅ APEX-LKA Jetson Orin Nano Setup Complete!"
echo "To test run manually: python3 ${SCRIPT_DIR}/jetson_run.py --debug"
echo "To inspect service: sudo systemctl status apex-lka.service"
echo "======================================================================"
