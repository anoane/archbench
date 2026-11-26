# SPDX-License-Identifier: AGPL-3.0-or-later
"""UUID resolution uses CUDA-visible ordinals and does not hide peer GPUs."""
import argparse
import os
from pathlib import Path
import unittest
from unittest.mock import patch
from archbench.runner import resolve_gpu


class GpuSelectionTests(unittest.TestCase):
    def test_uuid_resolves_visible_ordinal_without_modifying_environment(self):
        uuids = ["0" * 32, "1" * 32, "abcdef0123456789abcdef0123456789"]
        calls = []
        def worker(binary, argv, logs, stem, timeout):
            ordinal = int(argv[-1]); calls.append(ordinal)
            return {"status": "ok", "execution": "metadata_only", "metadata":
                    {"uuid_hex": uuids[ordinal], "visible_device_count": 3}}
        args = argparse.Namespace(device=0, gpu_uuid="GPU-abcdef01-2345-6789-abcd-ef0123456789", timeout=120)
        with patch("archbench.runner.invoke_worker", side_effect=worker), patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,0,1"}):
            before = dict(os.environ)
            ordinal, info = resolve_gpu(Path("worker"), args, Path("logs"))
            self.assertEqual(dict(os.environ), before)
        self.assertEqual(ordinal, 2)
        self.assertEqual(info["metadata"]["uuid_hex"], uuids[2])
        self.assertEqual(calls, [0, 1, 2])

    def test_ordinal_is_forwarded_directly(self):
        args = argparse.Namespace(device=7, gpu_uuid=None, timeout=120)
        with patch("archbench.runner.invoke_worker", return_value={"status": "ok", "metadata": {}}) as call:
            ordinal, _ = resolve_gpu(Path("worker"), args, Path("logs"))
        self.assertEqual(ordinal, 7)
        self.assertEqual(call.call_args.args[1], ["--info", "--device", "7"])

    def test_unknown_uuid_and_invalid_uuid_are_errors(self):
        args = argparse.Namespace(device=0, gpu_uuid="f" * 32, timeout=120)
        with patch("archbench.runner.invoke_worker", return_value={"status": "ok", "metadata": {"visible_device_count": 1, "uuid_hex": "0" * 32}}):
            with self.assertRaisesRegex(ValueError, "not visible"):
                resolve_gpu(Path("worker"), args, Path("logs"))
        args.gpu_uuid = "GPU-short"
        with patch("archbench.runner.invoke_worker") as call:
            with self.assertRaisesRegex(ValueError, "full 16-byte"):
                resolve_gpu(Path("worker"), args, Path("logs"))
            call.assert_not_called()
