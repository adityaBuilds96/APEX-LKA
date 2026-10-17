#!/usr/bin/env bash
# ==============================================================================
# APEX LKA • Production Edge Container Entrypoint
# Handles hardware interface initialization, profile loading, and signal trapping
# ==============================================================================

set -e

echo "=========================================================="
echo " APEX LKA Autonomous Vehicle Edge Stack"
echo " Production Container Init: $(date -u)"
echo " Profile:      ${APEX_PROFILE:-baja_2027}"
echo " HAL Backend:  ${APEX_HAL_BACKEND:-auto}"
echo " Network:      ${APEX_NET_PROTOCOL:-can}"
echo "=========================================================="

# Trap termination signals for graceful vehicle stop
cleanup() {
    echo ""
    echo "[ENTRYPOINT] Caught termination signal (SIGINT/SIGTERM)."
    echo "[ENTRYPOINT] Transitioning vehicle actuators to NEUTRAL fail-safe state..."
    # Attempt neutral broadcast if python-can or socketcan is reachable
    python3 -c "
try:
    from src.hardware.net_adapter import NetworkAdapterManager
    adapter = NetworkAdapterManager(protocol='${APEX_NET_PROTOCOL:-can}')
    adapter.send_actuation_command(steering_deg=0.0, throttle=0.0, brake=1.0, confidence=0.0, mode=0)
    adapter.close()
    print('[ENTRYPOINT] Actuator fail-safe neutral command sent.')
except Exception as e:
    print(f'[ENTRYPOINT] Fail-safe broadcast skipped: {e}')
" 2>/dev/null || true
    echo "[ENTRYPOINT] Shutdown complete. Exiting."
    exit 0
}

trap cleanup SIGINT SIGTERM

# Verify active vehicle profile existence
PROFILE_PATH="/app/vehicle_profiles/${APEX_PROFILE:-baja_2027}.yaml"
if [ ! -f "$PROFILE_PATH" ]; then
    echo "[WARN] Specified profile '$PROFILE_PATH' not found. Falling back to 'baja_2027.yaml'."
    export APEX_PROFILE="baja_2027"
fi

# Initialize virtual CAN if physical CAN is absent and user has privileges
if command -v ip >/dev/null 2>&1; then
    if ! ip link show can0 >/dev/null 2>&1 && ! ip link show vcan0 >/dev/null 2>&1; then
        echo "[INFO] No CAN device detected. Attempting to bring up vcan0..."
        ip link add dev vcan0 type vcan 2>/dev/null || true
        ip link set up vcan0 2>/dev/null || true
    fi
fi

echo "[ENTRYPOINT] Starting APEX LKA runtime process: $@"
exec "$@"
