"""Application entry point."""
from __future__ import annotations

import asyncio
import logging
import os
import signal
from typing import Iterable

from ha_ghost_assistant.audio import AudioCapture
from ha_ghost_assistant.playback import AudioPlayback
from ha_ghost_assistant.renderer import Renderer, build_renderer
from ha_ghost_assistant.wake_word import WakeWordDetector
from ha_ghost_assistant.wyoming_discovery import WyomingDiscovery
from ha_ghost_assistant.wyoming_server import WyomingInfo, WyomingServer

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


async def log_audio_levels(
    stop_event: asyncio.Event, audio: AudioCapture, renderer: Renderer
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


async def run(host: str, port: int) -> None:
    configure_logging()
    logger = logging.getLogger(__name__)
    stop_event = asyncio.Event()

    loop = asyncio.get_running_loop()
    _install_signal_handlers(loop, stop_event)

    audio = AudioCapture()
    playback = AudioPlayback()
    renderer = build_renderer()
    info = WyomingInfo()
    wyoming_server = WyomingServer(
        host=host,
        port=port,
        audio=audio,
        playback=playback,
        info=info,
        on_state=renderer.set_state,
    )
    discovery = WyomingDiscovery(host=host, port=port, info=info)
    wake_word = WakeWordDetector(
        on_detected=lambda name: loop.create_task(wyoming_server.trigger(name=name))
    )
    renderer.set_trigger(lambda: loop.create_task(wyoming_server.trigger()))
    renderer.set_stop(lambda: loop.create_task(wyoming_server.stop_streaming()))

    tasks: list[asyncio.Task[None]] = []
    try:
        audio.start(loop)
        renderer.set_state("idle")
        await wyoming_server.start()
        await discovery.start()
        wait_for_ha = os.getenv("HA_GHOST_ASSISTANT_WAIT_FOR_HA", "1").lower() in (
            "1",
            "true",
            "yes",
        )
        if wait_for_ha:
            logger.info("Waiting for Wyoming client connection...")
            wait_task = asyncio.create_task(wyoming_server.wait_for_client())
            stop_task = asyncio.create_task(stop_event.wait())
            done, pending = await asyncio.wait(
                {wait_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            await _gather_safely(pending)
            if stop_task in done:
                return
        tasks.extend(
            [
                asyncio.create_task(log_audio_levels(stop_event, audio, renderer)),
                asyncio.create_task(renderer.run(stop_event)),
                asyncio.create_task(wake_word.start()),
            ]
        )
        await stop_event.wait()
    finally:
        logger.info("Shutting down")
        await audio.stop()
        renderer.close()
        await playback.stop()
        await wake_word.stop()
        await wyoming_server.stop()
        await discovery.stop()
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
