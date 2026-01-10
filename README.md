# ha_ghost_assistant
Home Assistant Voice Assistant and Animation 

## Home Assistant Wyoming integration
1. In Home Assistant, add the **Wyoming Protocol** integration.
2. Enter the host running this app and port `10700` (or the value from `--port` / `HA_GHOST_ASSISTANT_PORT`).
3. Start the app with `python -m ha_ghost_assistant` to begin listening for Wyoming connections.

### Integration status
This project currently implements a minimal Wyoming Protocol server that responds to `describe`, `run-satellite`, and `pause-satellite` events. Audio streaming, wake word detection, and full pipeline handling are not implemented yet, so Home Assistant will connect and see the satellite, but audio capture is not streamed to HA yet.
