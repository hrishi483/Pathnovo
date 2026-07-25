# delta-chat

`delta-chat` is a FastAPI service for comparing revisions of engineering
documents, with an initial focus on P&ID drawings. This phase implements the
foundation: a loss-aware, format-agnostic canonical representation of one
document page.

The first goal is **not** to understand a P&ID semantically. It is to preserve:

1. deterministic textual information,
2. deterministic vector geometry, and
3. residual visual information that may require visual interpretation.

No LLM, VLM, network service, or proprietary API is involved in canonical
extraction.

## Canonical representation

Every ingested page has three content layers:

- **Text** — native PDF spans with exact extracted strings, font metadata,
  confidence, source, and bounding boxes. Spans are not merged.
- **Vector geometry** — raw lines, rectangles, Bézier curves, quads/polylines,
  and paths where PyMuPDF exposes them. Stroke, fill, layer, and coordinate
  data are retained when available. Geometry is not semantically classified.
- **Residual image content** — rendered pixels that are not covered by the
  extracted text and vector masks. Connected components become spatial image
  regions; the full residual page is also retained as a debug artifact.

This split prevents deterministic data from being delegated to a probabilistic
model while retaining raster symbols, embedded images, scan fragments, and
other content that PDF structure cannot explain.

## Coordinates

Canonical bounding boxes always use the original PDF page coordinate system:

- origin `(0, 0)` at the top-left,
- `x` increases to the right,
- `y` increases downward,
- `(x1, y1)` is the bottom-right.

Rendering uses a configurable zoom (default `2`). Pixel coordinates are used
only inside image processing and are converted back before a residual region
enters the canonical model. Pydantic validates that every element lies on the
PDF page.

## Residual extraction

The adapter renders the original PDF page and derives:

```text
residual = foreground(original render) - text mask - vector geometry mask
```

The residual is never reconstructed from parsed content. Configurable area,
padding, threshold, and morphology settings suppress anti-aliasing noise while
preserving useful connected components. Debug output is deterministic by
document SHA-256 prefix and includes:

- `original_render.png`
- `text_mask.png`
- `geometry_mask.png`
- `residual_mask.png`
- `residual_image.png`
- `overlay.png`
- one crop per retained residual region

## Run locally

Python 3.11 or newer is required.

```bash
cd delta-chat
make install
make run
```

Open `http://localhost:8000/docs`, or use:

```bash
curl http://localhost:8000/health
curl -F "file=@drawing.pdf" http://localhost:8000/ingest
```

By default artifacts are written to `data/artifacts/<document_id>/`. Copy
`.env.example` values into your environment to change the output directory,
render zoom, or minimum residual component area.

Docker is also supported:

```bash
docker compose up --build
```

## Tests

```bash
make test
make lint
```

Tests construct a one-page PDF containing native text, vector primitives, and
an embedded raster symbol. They cover model serialization, coordinate
round-trips, extraction, masks, residual subtraction, adapter artifacts, and
both API endpoints.

## System Dependencies

The project supports DWG ingestion through a DWG-to-DXF conversion step.

### macOS

Install LibreDWG using Homebrew:

```bash
brew install libredwg
which dwg2dxf
dwg2dxf --version
```

The DWG ingestion pipeline uses the converter to transform:

.dwg → .dxf → ezdxf → canonical document representation

## Current limitations and trade-offs

- Native PDFs must contain exactly one page. Multi-page aggregation is deferred.
- Text masking uses span bounding boxes. This is reliable and traceable but can
  mask non-text pixels that overlap a span box.
- Geometry masks approximate cubic Bézier curves with sampled line segments;
  canonical coordinates still retain all original control points.
- Foreground detection assumes a light page background. Threshold and cleanup
  settings are explicit because engineering exports vary.
- Connected components are pixel-based and may split one visual symbol or join
  nearby symbols. The full-page residual prevents information loss during
  debugging and future reprocessing.
- Embedded raster images are intentionally residual; they are not OCRed or
  interpreted during ingestion.
-Not able to show the raster for the .dwg files. Since I am unable to convert them to 
.png image files


The delta aligner, text/geometry/residual comparison engine, report renderer,
chat index, and provider-neutral `VisualInterpreter` are interfaces only.
Scanned-PDF OCR, DWG ingestion, semantic equipment/tag classification, VLM
interpretation, and full P&ID understanding are intentionally not implemented.
