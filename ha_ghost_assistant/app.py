"""Application entry point."""
from __future__ import annotations

import asyncio
import logging
import signal
from typing import Iterable

from ha_ghost_assistant.audio import AudioCapture
from ha_ghost_assistant.playback import AudioPlayback
from ha_ghost_assistant.renderer import FullscreenRenderer
from ha_ghost_assistant.wake_word import WakeWordDetector
from ha_ghost_assistant.wyoming import WyomingClient

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


async def log_audio_levels(
    stop_event: asyncio.Event, audio: AudioCapture, renderer: FullscreenRenderer
) -> None:
    while not stop_event.is_set():
        try:
            level = await asyncio.wait_for(audio.next_level(), timeout=1.0)
        except asyncio.TimeoutError:
            continue
        renderer.set_rms(level.rms)
        logging.getLogger(__name__).debug("Audio level RMS: %.6f", level.rms)


def _install_signal_handlers(loop: asyncio.AbstractEventLoop, stop_event: asyncio.Event) -> None:
    def _handle_stop() -> None:
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_stop)
        except NotImplementedError:
            signal.signal(sig, lambda *_: _handle_stop())


async def run() -> None:
    configure_logging()
    logger = logging.getLogger(__name__)
    stop_event = asyncio.Event()

    loop = asyncio.get_running_loop()
    _install_signal_handlers(loop, stop_event)

    audio = AudioCapture()
    playback = AudioPlayback()
    renderer = FullscreenRenderer()
    wake_word = WakeWordDetector()
    wyoming = WyomingClient()

    tasks: list[asyncio.Task[None]] = []
    try:
        audio.start(loop)
        renderer.set_state("listening")
        logger.info("Listening for audio")
        tasks.extend(
            [
                asyncio.create_task(log_audio_levels(stop_event, audio, renderer)),
                asyncio.create_task(playback.start()),
                asyncio.create_task(renderer.run(stop_event)),
                asyncio.create_task(wake_word.start()),
                asyncio.create_task(wyoming.connect()),
            ]
        )
        await stop_event.wait()
    finally:
        logger.info("Shutting down")
        await audio.stop()
        renderer.close()
        await playback.stop()
        await wake_word.stop()
        await wyoming.disconnect()
        for task in tasks:
            task.cancel()
        await _gather_safely(tasks)


async def _gather_safely(tasks: Iterable[asyncio.Task[None]]) -> None:
    if not tasks:
        return
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        pass
