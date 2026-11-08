# R05 refresh error: trace and bounded acceptance

The matching historical Windows error is attributable to **Vite's WebSocket
proxy**. Its existing disconnect handling passes the focused checks. This
investigation establishes no new runtime defect and makes no executable change.

The reviewer's October report itself still has no bound process, application
revision or full stack. Independent matching logs establish a real historical
emitting path; they do not prove every occurrence of `ECONNABORTED`, or the
reviewer's particular session, has the same cause.

## Recovered history and call path

Three preserved Windows browser-test logs contain ten entries with the exact
message `Error: write ECONNABORTED`. Their prefixes are `[vite] ws proxy error:`
and `[vite] ws proxy socket error:`. Stacks include `afterWriteDispatched`,
`Socket._write` and `Socket.ondata`. Originals remain private with byte hashes.
Filesystem timestamps do not establish executed application source identity.

The current source trace explains the two diagnostic routes:

1. Browser `/ws` connections pass through the development proxy in the client's
   `vite.config.ts`.
2. Vite logs proxy errors and separately attaches an error listener to the
   browser-facing socket. A teardown can consequently produce both prefixes.
3. The client's `scripts/backend-proxy-lifecycle.ts`, introduced in commit
   `ebac31c`, tracks the actual socket and Error object. Only `ECONNRESET`,
   `ECONNABORTED` or `EPIPE` associated with that disconnected browser are quiet.
   Matching message text alone never authorizes filtering.
4. A connected upstream reset, connection refusal, unknown errors and unrelated
   Error objects remain reportable. The real-socket test checks a live upstream
   reset separately from browser exits.

This plugin applies only to Vite development serving. An integrated production
bundle has no Vite Node proxy. Production aborts need their own attribution.

Python paths remain separate. `WebServer.broadcast` bounds individual sends,
prunes failed session owners and releases only their pending callbacks. A stalled
browser cannot delay another browser's completion. Supervisor response writes
handle broken-pipe/reset/abort disconnects; unexpected I/O failures propagate.
Frontend connection generations cancel retired probes/timers and reject stale
callbacks. Validated ready health permits recovery; only a successful socket
handshake resets the retry budget. These mechanisms already existed.

## Evidence and results

Fresh tests used backend `6e93d140` and client `c568d342`, with product bytes
unchanged from R04. No model, running app, graph, installed package, cache setting
or saved user workflow was changed.

| Evidence                                          | Result and scope                                                                                                                                                                                                                       |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Deliberately unfiltered existing real-socket test | Expected failure: three browser resets produce six diagnostics across the two Vite loggers. The fresh code is `read ECONNRESET`, not a fabricated fresh `write ECONNABORTED` reproduction.                                             |
| Current client transport selection                | 91 passed, zero failed/skipped. Real proxy/Chromium navigation, connected upstream reset, known abort codes, retired handshakes, refused/unready/malformed health, timeout, stale reads, queue/output recovery and supervisor routing. |
| Backend disconnect selection                      | Six tests and six subtests passed; 43 deselected, one existing Torch deprecation warning. Includes an actual reset health client, unexpected I/O remaining an error and session-scoped callback cancellation.                          |
| Backend broadcast selection                       | Two passed; 19 deselected, one existing Torch deprecation warning. Independent delivery and bounded stalled-socket pruning.                                                                                                            |
| Reused Windows production idle record             | Five sessions with five navigations each; zero page/console errors, graph submissions or runtime mutations; no appended backend abort/fault entries.                                                                                   |
| Reused Windows Vite idle record                   | Five sessions with five navigations each; zero page/console errors, graph submissions or runtime mutations; no appended backend/Vite abort/fault entries.                                                                              |

The historical idle records remain at backend `661efe87` and their captured
client/bundle identities. Failed GETs remain present: 28 production and 39 Vite
requests carry the original bounded navigation/supersession classifications.
This is not a claim of zero network cancellations. Original log byte ranges and
hashes were checked separately from decoded UTF-8 copies; those copies are not
mislabelled original UTF-16LE bytes.

R03-T's editing/task-transition checks and R04's actual Linux task/reuse receipts
are reused at their stated scopes. A synthetic `retained-task` message in a
socket fixture is handshake/message-delivery coverage, not native inference or
proof of the reviewer's Windows hardware. No closed model was repeated.

### Exact focused commands

In MoDiff-client:

```text
node --test scripts/backend-proxy-lifecycle.test.mjs scripts/websocket-connection.test.mjs scripts/request-boundary.test.mjs scripts/shared-store-requests.test.mjs scripts/studio-output-requests.test.mjs scripts/server-actions.test.mjs
```

In MoDiff, using the existing reviewed test interpreter:

```text
python -m pytest -q tests/test_supervisor_control.py tests/test_server_security.py -k "response_write or aborted_health or signal_lookup or signal_disconnect or disconnected_browser"
python -m pytest -q tests/test_studio_history_responsiveness.py -k "slow_browser or stalled_browser"
```

The ignored R05 evidence directory retains exact interpreter paths, commands,
source/log hashes, original outcomes, historical stack excerpts and the isolated
no-filter control. Two private reader failures remain preserved: initial UTF-16
BOM decoding and comparing a decoded log copy with its original segment-byte
hash. Corrected readers changed no originals or application behavior.

## Remaining disposition

The historical Vite emitting path and bounded disconnect regression are verified.
The exact original reviewer session remains unbound. For recurrence, the client's
troubleshooting guide requests the emitting terminal/process, prefix, bounded
stack, versions, timestamp and direct/proxied health results. Do not stop/retry a
task merely to collect diagnostics, hide genuine failures or infer worker death
from an SSH observer failure.

Complete paired gates are reused at their actual unchanged product scope; this
documentation checkpoint needs no new full gate or deployment. Both existing
PR7s remain the delivery locations. All 43 missing-model cases remain user-held.
R06's creative, capability, hardware and publication decisions stay separate.
