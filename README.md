# ha_ghost_assistant
Home Assistant Voice Assistant and Animation 

## Home Assistant Wyoming integration
1. In Home Assistant, add the **Wyoming Protocol** integration.
2. Enter the host running this app and port `10700` (or the value from `--port` / `HA_GHOST_ASSISTANT_PORT`).
3. Start the app with `python -m ha_ghost_assistant` to begin listening for Wyoming connections.

### Push-to-talk
* Press **Space** or **Enter** on the fullscreen window to trigger a push-to-talk session.
* The app emits a `wake-word-detected` event and starts streaming microphone audio to Home Assistant.

### Integration status
The Wyoming Protocol server now responds to `describe`, `run-satellite`, and `pause-satellite` events, and streams microphone audio (`audio-start`, `audio-chunk`, `audio-stop`). Wake word detection is still stubbed, but hooks are in place to emit `wake-word-detected` for custom wake words.
