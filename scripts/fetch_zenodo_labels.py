"""Pull only the label files out of the Zenodo flight archives.

The BAMBI matched-dataset releases package each flight as one zip of frames
plus labels — 110 MB to 2.3 GB each, ~90 GB for the flights we care about. The
labels inside are a few hundred kB. ZIP stores a central directory at the end
of the file listing every member's offset, so with HTTP range requests we can
read that directory and then fetch only the label members: three small requests
per flight instead of a full download.

    py scripts\\fetch_zenodo_labels.py --flights 10,211,212,213 --out D:/zenodo_labels
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

RECORDS = {
    "21291168": "part1",
    "21298044": "part2",
}
BASE = "https://zenodo.org/api/records/{rec}/files/{name}/content"


# Zenodo rate-limits anonymous access; a full sweep of 86 archives is ~800
# requests and gets 429'd part-way through without pacing. The delay is applied
# before every request and the backoff is honoured per response, so a partial
# run can simply be repeated - completed flights are skipped.
_DELAY = 1.2
_MAX_RETRIES = 6


def _urlopen(req, timeout: int):
    for attempt in range(_MAX_RETRIES):
        time.sleep(_DELAY)
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == _MAX_RETRIES - 1:
                raise
            wait = int(e.headers.get("Retry-After") or 0) or (5 * 2 ** attempt)
            print(f"      429; waiting {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def http_range(url: str, start: int, end: int) -> bytes:
    """Bytes [start, end] inclusive, as a Range request."""
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with _urlopen(req, 120) as r:
        if r.status not in (200, 206):
            raise RuntimeError(f"HTTP {r.status} for {url}")
        return r.read()


def content_length(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with _urlopen(req, 60) as r:
        return int(r.headers["Content-Length"])


class RemoteZip:
    """Minimal random-access zip over HTTP range requests."""

    def __init__(self, url: str):
        self.url = url
        self.size = content_length(url)
        self._load_central_directory()

    def _load_central_directory(self) -> None:
        # EOCD is in the last 64 kB unless the comment is huge; ZIP64 needs the
        # locator that sits just before it.
        tail_len = min(65536 + 64, self.size)
        tail = http_range(self.url, self.size - tail_len, self.size - 1)
        idx = tail.rfind(b"PK\x05\x06")
        if idx < 0:
            raise RuntimeError("no end-of-central-directory found")
        cd_size, cd_off = struct.unpack("<II", tail[idx + 12:idx + 20])
        if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
            z64 = tail.rfind(b"PK\x06\x06")
            if z64 < 0:
                raise RuntimeError("ZIP64 record missing")
            cd_size, cd_off = struct.unpack("<QQ", tail[z64 + 40:z64 + 56])
        cd = http_range(self.url, cd_off, cd_off + cd_size - 1)
        # Reconstruct a zip whose central directory is intact; zipfile only
        # needs the directory to enumerate members.
        self._cd = cd
        self.entries = {}
        p = 0
        while p + 46 <= len(cd):
            if cd[p:p + 4] != b"PK\x01\x02":
                break
            f = struct.unpack("<HHHHHHIIIHHHHHII", cd[p + 4:p + 46])
            method, csize, usize = f[3], f[7], f[8]
            nlen, elen, clen, lho = f[9], f[10], f[11], f[15]
            name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
            extra = cd[p + 46 + nlen:p + 46 + nlen + elen]

            # Archives past 4 GB store 0xFFFFFFFF here and put the real 64-bit
            # values in the ZIP64 extra field, in a fixed order but only for
            # the fields that overflowed. Reading the placeholder as an offset
            # produces a range beyond the file and a confusing HTTP 416.
            if 0xFFFFFFFF in (csize, usize, lho):
                q = 0
                while q + 4 <= len(extra):
                    hid, hsz = struct.unpack("<HH", extra[q:q + 4])
                    body = extra[q + 4:q + 4 + hsz]
                    if hid == 0x0001:
                        vals = list(struct.unpack(
                            "<" + "Q" * (len(body) // 8), body[:len(body) // 8 * 8]))
                        it = iter(vals)
                        if usize == 0xFFFFFFFF:
                            usize = next(it, usize)
                        if csize == 0xFFFFFFFF:
                            csize = next(it, csize)
                        if lho == 0xFFFFFFFF:
                            lho = next(it, lho)
                        break
                    q += 4 + hsz

            self.entries[name] = {"offset": lho, "csize": csize,
                                  "usize": usize, "method": method}
            p += 46 + nlen + elen + clen

    def read(self, name: str) -> bytes:
        e = self.entries[name]
        # Local header is 30 bytes + name + extra; read a slab and parse.
        head = http_range(self.url, e["offset"], e["offset"] + 29)
        if head[:4] != b"PK\x03\x04":
            raise RuntimeError(f"bad local header for {name}")
        nlen, elen = struct.unpack("<HH", head[26:30])
        data_start = e["offset"] + 30 + nlen + elen
        blob = http_range(self.url, data_start, data_start + e["csize"] - 1)
        # Decompress the member's stream directly. Building a synthetic zip
        # around it needs a valid central directory too, which is more work than
        # calling zlib with a raw-deflate window.
        if e["method"] == 0:
            return blob
        if e["method"] == 8:
            return zlib.decompress(blob, -15)
        raise RuntimeError(f"unsupported compression method {e['method']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flights", default="10,211,212,213",
                    help="Comma-separated flight ids.")
    ap.add_argument("--out", type=Path, default=Path("D:/zenodo_labels"))
    ap.add_argument("--patterns", default="labels/,_labels.json,gt/,.json",
                    help="Substrings identifying members worth fetching.")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Which record holds which flight.
    where: dict[str, tuple[str, str]] = {}
    for rec in RECORDS:
        with urllib.request.urlopen(
                f"https://zenodo.org/api/records/{rec}", timeout=60) as r:
            meta = json.load(r)
        for f in meta.get("files", []):
            fid = f["key"].replace("flight_", "").replace(".zip", "")
            where[fid] = (rec, f["key"])

    pats = [p for p in args.patterns.split(",") if p]
    done = 0
    for fid in [f.strip() for f in args.flights.split(",") if f.strip()]:
        if fid not in where:
            print(f"flight {fid}: not present in either record")
            continue
        rec, name = where[fid]
        # Resume: a rate-limited sweep can just be rerun.
        if (args.out / f"{fid}_{fid}_labels.json").exists():
            done += 1
            continue
        url = BASE.format(rec=rec, name=name)
        print(f"\nflight {fid} ({RECORDS[rec]}, {name})", flush=True)
        try:
            rz = RemoteZip(url)
        except Exception as e:
            print(f"   could not read archive directory: {e}")
            continue
        members = [m for m in rz.entries
                   if any(p in m for p in pats)
                   and m.endswith((".json", ".txt"))]
        if not members:
            print(f"   no json members; {len(rz.entries)} entries, e.g. "
                  f"{list(rz.entries)[:3]}")
            continue
        for m in members:
            try:
                data = rz.read(m)
            except Exception as e:
                print(f"   {m}: FAILED ({e})")
                continue
            dst = args.out / f"{fid}_{Path(m).name}"
            dst.write_bytes(data)
            print(f"   {m}  ->  {dst.name}  ({len(data)/1024:.1f} kB)", flush=True)
    if done:
        print(f"\n{done} flight(s) already present, skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
