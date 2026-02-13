"""
OMEGA JSON Compatibility Wrapper - orjson for hot paths, stdlib fallback.
Drop-in replacement: import omega.json_compat as json
Rollback: OMEGA_USE_STDLIB_JSON=1
"""

import os
import json as _json

_USE_ORJSON = os.environ.get("OMEGA_USE_STDLIB_JSON") != "1"
try:
    if _USE_ORJSON:
        import orjson as _orjson
    else:
        _orjson = None
except ImportError:
    _orjson = None


def loads(s, **kwargs):
    if _orjson and not kwargs:
        return _orjson.loads(s if isinstance(s, (bytes, bytearray, memoryview)) else s.encode())
    return _json.loads(s, **kwargs)


def dumps(obj, *, indent=None, default=None, sort_keys=False, ensure_ascii=True, **kwargs):
    if indent is not None or not _orjson:
        return _json.dumps(
            obj, indent=indent, default=default, sort_keys=sort_keys, ensure_ascii=ensure_ascii, **kwargs
        )
    opts = 0
    if sort_keys:
        opts |= _orjson.OPT_SORT_KEYS
    orjson_default = default if callable(default) else (str if default == str else None)
    return _orjson.dumps(obj, default=orjson_default, option=opts or None).decode()


def load(fp, **kwargs):
    return loads(fp.read(), **kwargs)


def dump(obj, fp, *, indent=None, default=None, sort_keys=False, ensure_ascii=True, **kwargs):
    fp.write(dumps(obj, indent=indent, default=default, sort_keys=sort_keys, ensure_ascii=ensure_ascii, **kwargs))
