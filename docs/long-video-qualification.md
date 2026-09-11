# Long-video qualification

MoDiff ships one qualification-only API graph template for the Phase 6
30-minute LTX continuation workload. It is not a Gallery workflow and has no
live-output claim. Run it only on an approved remote qualification host after
the exact model snapshot and opening image are present through normal app
workflows.

The reviewed graph uses:

- `Lightricks/LTX-Video-0.9.8-13B-distilled` at immutable revision
  `7c64400e1861cc0d7b98d570a1926d5408ec60cd`;
- `LTXConditionPipeline`, BF16, eight steps, guidance 1, and tiled/sliced VAE;
- a 1,800-second, 16 FPS plan split into 374 legal 81-frame jobs;
- 0.25-second continuation overlaps and last-frame handoff;
- retained, pinned segment assets and an opt-in durable loop checkpoint;
- file-native final concatenation and a six-hour execution ceiling.

## Materialize without running

Place the approved opening image under the app's managed `data/images`
directory. The local path passed to `--opening-image` and the portable app path
passed to `--opening-image-id` must resolve to the same file.

```bash
./scripts/with-runtime-env.sh ./.venv/bin/python \
  scripts/materialize_long_video_qualification.py \
  --opening-image data/images/qualification-opening.webp \
  --opening-image-id images/qualification-opening.webp \
  --sid long-video-qualification \
  --output ../ltx-30-minute.api.json
```

This command does not submit a graph or run inference. It binds the exact
opening-image byte hash into the graph, derives the durable `runInputHash`, and
writes a materialized API graph outside the repository.

## Submit through the app

Submission is an explicit long-running mutation and requires both flags below.
The helper accepts only a loopback app origin, verifies that the app reports
the exact immutable LTX snapshot complete and repair-free, and then queues the
graph through `POST /graph`.

```bash
./scripts/with-runtime-env.sh ./.venv/bin/python \
  scripts/materialize_long_video_qualification.py \
  --opening-image data/images/qualification-opening.webp \
  --opening-image-id images/qualification-opening.webp \
  --sid long-video-qualification \
  --app-url http://127.0.0.1:8088 \
  --submit \
  --consent-long-run
```

Monitor the run through the app or `GET /queue`. If the worker is replaced
after a segment checkpoint, resubmit with the same template, opening-image
bytes, app path, and workflow identity. The materializer derives the same
input hash and the durable loop resumes from retained segments. Changing the
opening image produces a different hash and starts an independent run.

Do not commit materialized graphs, opening images, generated segments, the
joined output, receipts containing local paths, model caches, or logs. Publish
reviewed output and bounded evidence through the remote asset workflow. A
successful remote run still needs duration/frame, non-black/finite output,
continuity, safety, cancellation/recovery, peak-memory, runtime, and asset
publication review before any Gallery or Auto activation.
