# Sign Language Recognition - working notes

Chronological log of decisions, data issues and results.

## Dataset

**WLASL100** (official subset of WLASL v0.3): the 100 glosses with the most
instances in the full 2000-gloss set, with the published train/val/test
splits, so results are comparable to the paper. 2038 video instances in total.

## Host reachability audit (this machine, 2026-10-22)

Every download attempt in WLASL points at one of 13 hosts. Probed each one
with curl using a browser User-Agent:

| Host | Instances | Result |
| --- | --- | --- |
| `www.youtube.com` | 589 | **works with yt-dlp `player_client=android`** |
| `youtu.be` | 96 | same |
| `www.signingsavvy.com` | 225 | 403 - bot wall (TLS fingerprint), retries do not help |
| `signstock.blob.core.windows.net` | 196 | 206 - direct HTTP |
| `www.handspeak.com` | 174 | connection failure from this network |
| `www.aslpro.com` | 162 | 404 - the SWF files are gone |
| `media.asldeafined.com` | 131 | 206 - direct HTTP |
| `www.aslsearch.com` | 114 | returns HTML, not video |
| `aslsignbank.haskins.yale.edu` | 108 | TLS/connection failure from this network |
| `s3-us-west-1.amazonaws.com` | 99 | 206 - direct HTTP |
| `media.spreadthesign.com` | 91 | 206 - direct HTTP |
| `aslbricks.org` | 45 | 206 - direct HTTP |
| `elementalaslconcepts.weebly.com` | 8 | 206 - direct HTTP |

So ~674 clips come over plain HTTP and ~685 more are reachable through yt-dlp;
the remaining ~680 are permanently unreachable from this network. Those hosts
are marked `host_blocked` in the manifest rather than retried, which turns a
multi-hour wall of timeouts into a few milliseconds each.

## Honesty rules

- The WLASL100 published splits are used as-is.
- Only classes with at least one downloaded instance can be evaluated; the
  class set actually used is recorded in the results table.
- Landmarks are cached to disk after extraction; training never re-decodes
  video.
- Mirroring augmentation swaps left/right hand landmarks.
