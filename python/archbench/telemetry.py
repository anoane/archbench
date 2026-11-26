# SPDX-License-Identifier: AGPL-3.0-or-later
"""Optional NVML sampling. A sample covers the whole worker, not just kernel time."""
from __future__ import annotations
import ctypes as c
import json
import threading
import time
from pathlib import Path


class NvmlSampler:
    def __init__(self, output: Path, device_uuid: str | None, interval: float = 0.2):
        self.output = output
        self.interval = interval
        self.case_id = "setup"
        self.stop_event = threading.Event()
        self.error: str | None = None
        self.thread: threading.Thread | None = None
        self.handle = c.c_void_p()
        self.lib = None
        try:
            self.lib = c.CDLL("libnvidia-ml.so.1")
            if self.lib.nvmlInit_v2() != 0: raise RuntimeError("NVML initialization failed")
            if not device_uuid:
                raise RuntimeError("GPU UUID unavailable; refusing to guess NVML device index under CUDA_VISIBLE_DEVICES")
            # NVML UUID is GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.
            value = device_uuid
            if not value.startswith("GPU-"):
                value = "GPU-" + "-".join((value[:8], value[8:12], value[12:16], value[16:20], value[20:]))
            if self.lib.nvmlDeviceGetHandleByUUID(c.c_char_p(value.encode()), c.byref(self.handle)) != 0:
                raise RuntimeError("NVML could not resolve the CUDA GPU UUID")
        except (OSError, AttributeError, RuntimeError) as error:
            self.error = str(error)

    def _uint(self, name: str, *extra, wide=False):
        value = c.c_ulonglong() if wide else c.c_uint()
        try:
            code = getattr(self.lib, name)(self.handle, *extra, c.byref(value))
            return value.value if code == 0 else None
        except AttributeError:
            return None

    def sample(self):
        return {"unix_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns(), "case_id": self.case_id,
                "scope": "whole_worker_including_setup_and_validation",
                "power_mw": self._uint("nvmlDeviceGetPowerUsage"),
                "energy_mj": self._uint("nvmlDeviceGetTotalEnergyConsumption", wide=True),
                "sm_clock_mhz": self._uint("nvmlDeviceGetClockInfo", c.c_uint(1)),
                "memory_clock_mhz": self._uint("nvmlDeviceGetClockInfo", c.c_uint(2)),
                "temperature_c": self._uint("nvmlDeviceGetTemperature", c.c_uint(0)),
                "pstate": self._uint("nvmlDeviceGetPerformanceState"),
                "throttle_reasons": self._uint("nvmlDeviceGetCurrentClocksThrottleReasons", wide=True),
                "pcie_generation": self._uint("nvmlDeviceGetCurrPcieLinkGeneration"),
                "pcie_width": self._uint("nvmlDeviceGetCurrPcieLinkWidth")}

    def start(self):
        if self.error: return
        def loop():
            with self.output.open("a", encoding="utf-8") as stream:
                while not self.stop_event.is_set():
                    stream.write(json.dumps(self.sample(), allow_nan=False) + "\n")
                    stream.flush()
                    self.stop_event.wait(self.interval)
        self.thread = threading.Thread(target=loop, name="archbench_nvml", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread: self.thread.join(timeout=5)
        if self.lib:
            try: self.lib.nvmlShutdown()
            except (AttributeError, OSError): pass
