# ha_ghost_assistant
Home Assistant Voice Assistant and Animation

## Installation
These steps assume you can copy/paste commands. Use the section for your operating system.

### 1) Install Python 3.11+
* **Windows/macOS/Linux:** Download and install Python 3.11+ from https://www.python.org/downloads/
* **Linux package manager:** If you already have Python 3.11+ installed, you can skip this.

### 2) Create and activate a virtual environment
Pick **one** of the following (based on your OS). Run these commands in the repo folder.

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

**Windows (PowerShell)**
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**Windows (Command Prompt)**
```cmd
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
```

> **Tip:** If the `activate` command fails on Windows, reopen the terminal and try again. You can also run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` in PowerShell once to allow script activation.

### 3) Install dependencies
```bash
python -m pip install -r requirements.txt
```

### 4) Install the package (editable)
```bash
python -m pip install -e .
```

> **Note:** `sounddevice` requires system audio libraries (for example, PortAudio on Linux). Ensure your OS audio stack is installed before running the app.

## Usage
Run the assistant:
```bash
python -m ha_ghost_assistant
```

## Home Assistant Wyoming integration
1. In Home Assistant, add the **Wyoming Protocol** integration.
2. Enter the host running this app and port `10700` (or the value from `--port` / `HA_GHOST_ASSISTANT_PORT`).
3. Start the app with `python -m ha_ghost_assistant` to begin listening for Wyoming connections.

### Push-to-talk
* Press **Enter** on the fullscreen window to trigger a push-to-talk session.
* The app emits a `trigger` event and starts streaming microphone audio to Home Assistant.

### Wake word (openWakeWord)
* Install the `openwakeword` dependency (it is now listed in `pyproject.toml`).
* The bundled **"Samantha"** wake word model lives under `ha_ghost_assistant/wakewords/samantha` and is used automatically.
* To override with your own model file, point the app at it:
  * `export HA_GHOST_ASSISTANT_WAKE_WORD_MODEL=/path/to/samantha.onnx`
  * `export HA_GHOST_ASSISTANT_WAKE_WORD_NAME=Samantha`
  * (Optional) `export HA_GHOST_ASSISTANT_WAKE_WORD_MODEL_NAME=alexa` to use a bundled openWakeWord model instead of a custom file.
* Optional tuning:
  * `HA_GHOST_ASSISTANT_WAKE_WORD_THRESHOLD` (default `0.6`)
  * `HA_GHOST_ASSISTANT_WAKE_WORD_COOLDOWN` (default `2.0` seconds)
  * `HA_GHOST_ASSISTANT_WAKE_WORD=0` to disable wake word detection.

### Training a custom "Samantha" model
This repository does not include training scripts. Use the openWakeWord training workflow to produce a `.onnx` model for the "Samantha" keyword, then point `HA_GHOST_ASSISTANT_WAKE_WORD_MODEL` at the resulting file if you want to override the bundled model. A typical workflow is:
1. Record short wake-word utterances ("Samantha") and background/negative examples at 16 kHz mono.
2. Follow the openWakeWord training guide to train/export a custom model (see https://github.com/dscripka/openWakeWord and https://github.com/dscripka/openWakeWord/tree/main/docs).
3. Copy the exported `samantha.onnx` to the device running this app and set `HA_GHOST_ASSISTANT_WAKE_WORD_MODEL` to its path.

You can also use the provided Google Colab notebook to train a model:
https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb?usp=sharing#scrollTo=-Q9wEuRdwY_E

### Integration status
The Wyoming Protocol server responds to `describe`, `run-satellite`, and `pause-satellite` events, and streams microphone audio (`audio-start`, `audio-chunk`, `audio-stop`). Wake word detection uses openWakeWord when configured, and the Enter key remains a fallback trigger.
