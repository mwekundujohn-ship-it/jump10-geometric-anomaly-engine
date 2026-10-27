"""
Parquet compatibility shim for this machine.

Situation (2026-10-27):
  * pyarrow's _compute.pyd is blocked by an Application Control policy.
  * fastparquet's cencoding.pyd is now blocked too (it worked for a while).
  * duckdb is NOT blocked and ships its own parquet reader/writer -> preferred.

Order of preference: duckdb -> pyarrow -> fastparquet (with the pure-python
cramjam stub so UNCOMPRESSED parquet still works where duckdb is missing).
"""
import sys
import os
import gzip
import types
import sqlite3


def _install_stub():
    if "cramjam" in sys.modules:
        return
    mod = types.ModuleType("cramjam")

    class Buffer(bytearray):
        pass

    def _unsupported(name):
        def _f(*a, **k):
            raise NotImplementedError(
                f"cramjam.{name} unavailable: the cramjam DLL is blocked by "
                f"Application Control on this host. Write parquet with compression=None."
            )
        return _f

    class _Gzip:
        @staticmethod
        def compress(data, level=6, **kw):
            return gzip.compress(bytes(data), compresslevel=int(level or 6))

        @staticmethod
        def decompress(data, output_len=None, **kw):
            return gzip.decompress(bytes(data))

        @staticmethod
        def decompress_into(data, out):
            b = gzip.decompress(bytes(data))
            mv = memoryview(out)
            mv[:len(b)] = b
            return len(b)

    class _Snappy:
        compress_raw = staticmethod(_unsupported("snappy.compress_raw"))
        decompress_raw = staticmethod(_unsupported("snappy.decompress_raw"))
        decompress_raw_into = staticmethod(_unsupported("snappy.decompress_raw_into"))
        compress = staticmethod(_unsupported("snappy.compress"))
        decompress = staticmethod(_unsupported("snappy.decompress"))

    class _Brotli:
        compress = staticmethod(_unsupported("brotli.compress"))
        decompress = staticmethod(_unsupported("brotli.decompress"))
        decompress_into = staticmethod(_unsupported("brotli.decompress_into"))

    class _Lz4:
        compress = staticmethod(_unsupported("lz4.compress"))
        decompress = staticmethod(_unsupported("lz4.decompress"))
        compress_block = staticmethod(_unsupported("lz4.compress_block"))
        decompress_block = staticmethod(_unsupported("lz4.decompress_block"))

    class _Zstd:
        compress = staticmethod(_unsupported("zstd.compress"))
        decompress = staticmethod(_unsupported("zstd.decompress"))
        decompress_into = staticmethod(_unsupported("zstd.decompress_into"))

    mod.Buffer = Buffer
    mod.gzip = _Gzip
    mod.snappy = _Snappy
    mod.brotli = _Brotli
    mod.lz4 = _Lz4
    mod.zstd = _Zstd
    sys.modules["cramjam"] = mod


def _duckdb():
    import duckdb
    return duckdb


def import_fastparquet():
    try:
        import fastparquet  # noqa: F401
        return fastparquet
    except ImportError as e:
        if "cramjam" not in str(e):
            raise
        _install_stub()
        import fastparquet  # noqa: F401
        return fastparquet


def engine():
    """'duckdb' | 'pyarrow' | 'fastparquet' | None"""
    try:
        _duckdb()
        return "duckdb"
    except Exception:
        pass
    try:
        import pyarrow  # noqa: F401
        return "pyarrow"
    except Exception:
        pass
    try:
        import_fastparquet()
        return "fastparquet"
    except Exception:
        return None


def _q(path):
    return "'" + str(path).replace("'", "''") + "'"


def write_parquet(df, path):
    path = os.path.abspath(str(path))
    if os.path.exists(path):
        os.remove(path)
    eng = engine()
    if eng == "duckdb":
        con = _duckdb().connect()
        try:
            con.register("_df", df)
            con.execute(f"COPY _df TO {_q(path)} (FORMAT PARQUET, COMPRESSION SNAPPY)")
        finally:
            con.close()
    elif eng == "pyarrow":
        df.to_parquet(path, index=False, engine="pyarrow")
    elif eng == "fastparquet":
        df.to_parquet(path, index=False, engine="fastparquet", compression=None)
    else:
        raise RuntimeError("no usable parquet engine (pyarrow/fastparquet blocked, duckdb missing)")
    return eng


def read_parquet(path):
    path = os.path.abspath(str(path))
    eng = engine()
    if eng == "duckdb":
        con = _duckdb().connect()
        try:
            return con.execute(f"SELECT * FROM read_parquet({_q(path)})").df()
        finally:
            con.close()
    if eng == "pyarrow":
        import pandas as pd
        return pd.read_parquet(path, engine="pyarrow")
    if eng == "fastparquet":
        import pandas as pd
        return pd.read_parquet(path, engine="fastparquet")
    raise RuntimeError("no usable parquet engine")


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    print("engine:", engine())
    df = pd.DataFrame({"a": np.arange(10), "b": np.linspace(0, 1, 10), "c": list("abcdefghij")})
    p = r"C:\cluster\phase2\_pq_test.parquet"
    print("write ->", write_parquet(df, p))
    back = read_parquet(p)
    print(back.head(3).to_string())
    print("roundtrip ok:", back.shape == df.shape)
