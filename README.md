# ha_ghost_assistant
Home Assistant Voice Assistant and Animation 

## Home Assistant Wyoming integration
1. In Home Assistant, add the **Wyoming Protocol** integration.
2. Enter the host running this app and port `10700` (or the value from `--port` / `HA_GHOST_ASSISTANT_PORT`).
3. Start the app with `python -m ha_ghost_assistant` to begin listening for Wyoming connections.

### Push-to-talk
* Press **Enter** on the fullscreen window to trigger a push-to-talk session.
* The app emits a `trigger` event and starts streaming microphone audio to Home Assistant.

### Wake word (openWakeWord)
* Install the `openwakeword` dependency (it is now listed in `pyproject.toml`).
* Provide a custom model file trained for the keyword **"Samantha"** and point the app at it:
  * `export HA_GHOST_ASSISTANT_WAKE_WORD_MODEL=/path/to/samantha.onnx`
  * `export HA_GHOST_ASSISTANT_WAKE_WORD_NAME=Samantha`
  * (Optional) `export HA_GHOST_ASSISTANT_WAKE_WORD_MODEL_NAME=alexa` to use a bundled openWakeWord model instead of a custom file.
* Optional tuning:
  * `HA_GHOST_ASSISTANT_WAKE_WORD_THRESHOLD` (default `0.6`)
  * `HA_GHOST_ASSISTANT_WAKE_WORD_COOLDOWN` (default `2.0` seconds)
  * `HA_GHOST_ASSISTANT_WAKE_WORD=0` to disable wake word detection.

### Training a custom "Samantha" model
This repository does not include training scripts. Use openWakeWord's training workflow to produce a `.onnx` model for the "Samantha" keyword, then point `HA_GHOST_ASSISTANT_WAKE_WORD_MODEL` at the resulting file. A typical workflow is:
1. Record short wake-word utterances ("Samantha") and background/negative examples at 16 kHz mono.
2. Follow the openWakeWord training guide to train/export a custom model (see https://github.com/dscripka/openWakeWord).
3. Copy the exported `samantha.onnx` to the device running this app and set `HA_GHOST_ASSISTANT_WAKE_WORD_MODEL` to its path.

### Integration status
The Wyoming Protocol server responds to `describe`, `run-satellite`, and `pause-satellite` events, and streams microphone audio (`audio-start`, `audio-chunk`, `audio-stop`). Wake word detection uses openWakeWord when configured, and the Enter key remains a fallback trigger.
