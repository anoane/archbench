#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""TEST FIXTURE ONLY. Does not execute CUDA; all timing values are synthetic test data."""
import json
import sys
import time
args = sys.argv[1:]
opts = {}
i = 0
while i < len(args):
    key = args[i].lstrip('-')
    if i + 1 < len(args) and not args[i+1].startswith('--'):
        opts[key] = args[i+1]; i += 2
    else: opts[key] = 'true'; i += 1
if opts.get('op') == 'timeout': time.sleep(10)
if opts.get('op') == 'invalid': print('not json'); raise SystemExit(0)
result = {'schema_version': 1, 'status': 'ok', 'validation': 'passed', 'execution': 'mock_test_fixture',
          'samples_ms': [1.0, 1.2, 1.1], 'metadata': {}, 'metrics': {}, 'reason': '',
          'unit': 'Gop/s', 'work_per_iteration': 1, 'bytes_per_iteration': 0}
if opts.get('info'):
    result.update(validation='not_applicable', execution='metadata_only', samples_ms=[],
                  metadata={'uuid_hex':'0123456789abcdef0123456789abcdef', 'device_name':'TEST FIXTURE NO GPU',
                            'compute_capability':'0.0', 'binary_target':'mock'})
if opts.get('op') == 'skip': result.update(status='unsupported', validation='not_run', samples_ms=[])
if opts.get('op') == 'fail': result.update(status='validation_failed', validation='failed')
if opts.get('op') == 'xss': result['reason'] = '</script><img src=x onerror=alert(1)>'
print(json.dumps(result))
raise SystemExit(1 if opts.get('op') == 'fail' else 0)
