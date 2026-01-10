"""Wyoming mDNS discovery broadcaster for Home Assistant."""
from __future__ import annotations

import logging
import socket

from zeroconf import IPVersion, ServiceInfo
from zeroconf.asyncio import AsyncZeroconf

from ha_ghost_assistant.wyoming_server import WyomingInfo

LOGGER = logging.getLogger(__name__)

SERVICE_TYPE = "_wyoming._tcp.local."


def _local_ip(host: str) -> str:
    if host not in ("0.0.0.0", "127.0.0.1", "::"):
        return host
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


class WyomingDiscovery:
    """Advertise the Wyoming satellite via mDNS/zeroconf."""

    def __init__(self, host: str, port: int, info: WyomingInfo) -> None:
        self._host = host
        self._port = port
        self._info = info
        self._zeroconf: AsyncZeroconf | None = None
        self._service_info: ServiceInfo | None = None

    async def start(self) -> None:
        if self._zeroconf is not None:
            return
        try:
            ip_address = _local_ip(self._host)
            properties: dict[str, str] = {
                "name": self._info.name,
                "description": self._info.description,
                "version": self._info.version,
                "attribution": self._info.attribution,
                "supports_trigger": str(self._info.supports_trigger).lower(),
                "has_vad": str(self._info.has_vad).lower(),
            }
            service_name = f"{self._info.name}.{SERVICE_TYPE}"
            self._service_info = ServiceInfo(
                SERVICE_TYPE,
                service_name,
                addresses=[socket.inet_aton(ip_address)],
                port=self._port,
                properties=properties,
                server=f"{socket.gethostname()}.local.",
            )
            self._zeroconf = AsyncZeroconf(ip_version=IPVersion.V4Only)
            await self._zeroconf.async_register_service(self._service_info)
            LOGGER.info("Wyoming discovery broadcast at %s:%s", ip_address, self._port)
        except Exception:
            LOGGER.exception("Failed to start Wyoming discovery")
            await self.stop()

    async def stop(self) -> None:
        if self._zeroconf is None:
            return
        if self._service_info is not None:
            await self._zeroconf.async_unregister_service(self._service_info)
        await self._zeroconf.async_close()
        self._zeroconf = None
        self._service_info = None
