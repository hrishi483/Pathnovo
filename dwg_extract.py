from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import ezdxf
from ezdxf.document import Drawing

# ---------------------------------------------------------------------------
# Entity classification
# ---------------------------------------------------------------------------

TEXT_TYPES = {"TEXT", "MTEXT", "ATTRIB", "ATTDEF"}

GEOMETRY_TYPES = {
    "LINE", "LWPOLYLINE", "POLYLINE", "CIRCLE", "ARC", "ELLIPSE", "SPLINE",
    "POINT", "SOLID", "TRACE", "3DFACE", "HATCH", "XLINE", "RAY", "MESH",
    "REGION", "3DSOLID", "WIPEOUT",
}

# Compound entities whose real content is only available via a rendered
# ("virtual") copy -- expanded and recursed into rather than read directly.
EXPANDABLE_TYPES = {"INSERT", "DIMENSION", "MLEADER", "LEADER", "ACAD_TABLE"}


# ---------------------------------------------------------------------------
# DWG -> DXF loading
# ---------------------------------------------------------------------------

class DWGConversionError(RuntimeError):
    pass


def _oda_is_installed() -> bool:
    from ezdxf.addons import odafc
    return odafc.is_installed()


def _libredwg_convert(path: Path) -> Path | None:
    exe = shutil.which("dwg2dxf")
    if exe is None:
        return None
    tmp_dir = Path(tempfile.mkdtemp(prefix="dwg2dxf_"))
    out_file = tmp_dir / (path.stem + ".dxf")
    try:
        subprocess.run([exe, "-o", str(out_file), str(path)], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise DWGConversionError(f"LibreDWG dwg2dxf failed: {e.stderr}") from e
    if not out_file.exists():
        raise DWGConversionError("LibreDWG dwg2dxf ran but produced no output file.")
    return out_file


def load_document(path: str | Path) -> Drawing:
    """Load a .dwg or .dxf file into an ezdxf Drawing. This is the only
    thing you need to point at your file -- it figures out the rest."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {path}")

    suffix = path.suffix.lower()
    if suffix == ".dxf":
        return ezdxf.readfile(str(path))

    if suffix != ".dwg":
        raise ValueError(f"Unsupported file extension {suffix!r}: expected .dwg or .dxf")

    if _oda_is_installed():
        from ezdxf.addons import odafc
        return odafc.readfile(str(path), audit=True)

    dxf_path = _libredwg_convert(path)
    if dxf_path is not None:
        return ezdxf.readfile(str(dxf_path))

    raise DWGConversionError(
        "Neither the ODA File Converter nor LibreDWG's `dwg2dxf` was found. "
        "Install one to read .dwg files -- see the docstring at the top of "
        "this file for links. (.dxf files don't need either.)"
    )


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _point3(v) -> tuple[float, float, float]:
    try:
        return (float(v.x), float(v.y), float(v.z))
    except AttributeError:
        seq = list(v)
        x = seq[0] if len(seq) > 0 else 0.0
        y = seq[1] if len(seq) > 1 else 0.0
        z = seq[2] if len(seq) > 2 else 0.0
        return (float(x), float(y), float(z))


def _extract_text(e, layout_name: str, source_block: str | None) -> dict | None:
    t = e.dxftype()
    color = getattr(e.dxf, "color", None)
    layer = getattr(e.dxf, "layer", "0")
    handle = getattr(e.dxf, "handle", "")

    if t == "TEXT":
        return {
            "handle": handle, "entity_type": "TEXT", "layout": layout_name, "layer": layer,
            "text": e.dxf.text, "insert": _point3(e.dxf.insert),
            "height": getattr(e.dxf, "height", None), "rotation": getattr(e.dxf, "rotation", 0.0),
            "style": getattr(e.dxf, "style", None), "color": color, "source_block": source_block,
        }
    if t == "MTEXT":
        text = e.plain_text() if hasattr(e, "plain_text") else e.text
        rotation = e.get_rotation() if hasattr(e, "get_rotation") else getattr(e.dxf, "rotation", 0.0)
        return {
            "handle": handle, "entity_type": "MTEXT", "layout": layout_name, "layer": layer,
            "text": text, "insert": _point3(e.dxf.insert),
            "height": getattr(e.dxf, "char_height", None), "rotation": rotation,
            "style": getattr(e.dxf, "style", None), "color": color, "source_block": source_block,
        }
    if t in ("ATTRIB", "ATTDEF"):
        text = getattr(e.dxf, "text", "") or getattr(e.dxf, "default", "")
        return {
            "handle": handle, "entity_type": t, "layout": layout_name, "layer": layer,
            "text": text, "insert": _point3(e.dxf.insert),
            "height": getattr(e.dxf, "height", None), "rotation": getattr(e.dxf, "rotation", 0.0),
            "style": getattr(e.dxf, "style", None), "color": color, "source_block": source_block,
        }
    return None


def _extract_geometry(e, layout_name: str, source_block: str | None) -> dict | None:
    t = e.dxftype()
    color = getattr(e.dxf, "color", None)
    layer = getattr(e.dxf, "layer", "0")
    handle = getattr(e.dxf, "handle", "")

    def make(geometry: dict) -> dict:
        return {
            "handle": handle, "entity_type": t, "layout": layout_name, "layer": layer,
            "geometry": geometry, "color": color, "source_block": source_block,
        }

    if t == "LINE":
        return make({"start": _point3(e.dxf.start), "end": _point3(e.dxf.end)})
    if t == "CIRCLE":
        return make({"center": _point3(e.dxf.center), "radius": e.dxf.radius})
    if t == "ARC":
        return make({"center": _point3(e.dxf.center), "radius": e.dxf.radius,
                      "start_angle": e.dxf.start_angle, "end_angle": e.dxf.end_angle})
    if t == "LWPOLYLINE":
        points = [(float(p[0]), float(p[1])) for p in e.get_points("xy")]
        return make({"points": points, "closed": bool(e.closed)})
    if t == "POLYLINE":
        points = [_point3(v.dxf.location) for v in e.vertices]
        return make({"points": points, "closed": bool(e.is_closed)})
    if t == "ELLIPSE":
        return make({"center": _point3(e.dxf.center), "major_axis": _point3(e.dxf.major_axis),
                      "ratio": e.dxf.ratio, "start_param": getattr(e.dxf, "start_param", 0.0),
                      "end_param": getattr(e.dxf, "end_param", 6.283185307179586)})
    if t == "SPLINE":
        return make({"control_points": [_point3(p) for p in e.control_points],
                      "fit_points": [_point3(p) for p in e.fit_points]})
    if t == "POINT":
        return make({"location": _point3(e.dxf.location)})
    if t in ("SOLID", "TRACE", "3DFACE"):
        pts = [_point3(getattr(e.dxf, n)) for n in ("vtx0", "vtx1", "vtx2", "vtx3") if e.dxf.hasattr(n)]
        return make({"points": pts})
    if t == "HATCH":
        paths = []
        for p in e.paths:
            if hasattr(p, "vertices"):
                paths.append([(float(v[0]), float(v[1])) for v in p.vertices])
            else:
                paths.append([])
        return make({"paths": paths, "solid_fill": bool(e.dxf.solid_fill) if e.dxf.hasattr("solid_fill") else None})
    if t in ("XLINE", "RAY"):
        return make({"point": _point3(e.dxf.start),
                      "direction": _point3(e.dxf.unit_vector) if e.dxf.hasattr("unit_vector") else None})
    if t == "MESH":
        verts = list(e.vertices) if hasattr(e, "vertices") else []
        return make({"vertices": [_point3(v) for v in verts]})
    if t in ("3DSOLID", "REGION", "WIPEOUT"):
        return make({})
    return None


def _block_name(e) -> str:
    return e.dxf.name if e.dxftype() == "INSERT" else e.dxftype()


def _walk(entities, layout_name, texts, geoms, expand_blocks, max_depth, depth, source_block):
    for e in entities:
        try:
            t = e.dxftype()
        except Exception:
            continue

        if t in TEXT_TYPES:
            item = _extract_text(e, layout_name, source_block)
            if item:
                texts.append(item)
        elif t in GEOMETRY_TYPES:
            item = _extract_geometry(e, layout_name, source_block)
            if item:
                geoms.append(item)
        elif t in EXPANDABLE_TYPES:
            if expand_blocks and depth < max_depth:
                try:
                    virtual = list(e.virtual_entities())
                except Exception:
                    virtual = []
                _walk(virtual, layout_name, texts, geoms, expand_blocks, max_depth, depth + 1, _block_name(e))
            elif t == "INSERT":
                geoms.append({
                    "handle": e.dxf.handle, "entity_type": "INSERT", "layout": layout_name,
                    "layer": e.dxf.layer, "geometry": {"insert": _point3(e.dxf.insert), "block": e.dxf.name},
                    "color": getattr(e.dxf, "color", None), "source_block": source_block,
                })
        # else: unsupported/annotation-only entity type, skipped


def extract(path: str | Path, layouts: str = "all", expand_blocks: bool = True, max_depth: int = 5) -> dict:
    """Extract text and geometry from a single .dwg/.dxf file.

    path:          the ONLY thing you need to supply.
    layouts:       "all" (default; modelspace + every paperspace layout),
                    "modelspace", or a comma-separated string of layout names.
    expand_blocks: expand block references (INSERT/DIMENSION/...) into their
                    rendered contents (with world coordinates applied). True by default.

    Returns: {"text": [ {...}, ... ], "geometry": [ {...}, ... ]}
    """
    doc = load_document(path)

    texts: list[dict] = []
    geoms: list[dict] = []

    if layouts == "all":
        layout_iter = [(layout.name, layout) for layout in doc.layouts]
    elif layouts == "modelspace":
        msp = doc.modelspace()
        layout_iter = [(msp.name, msp)]
    else:
        names = [n.strip() for n in layouts.split(",")]
        layout_iter = [(n, doc.layouts.get(n)) for n in names]

    for layout_name, layout in layout_iter:
        _walk(layout, layout_name, texts, geoms, expand_blocks, max_depth, 0, None)

    return {"text": texts, "geometry": geoms}


# PyMuPDF-style alias: `dwg_extract.open(path)` reads like `fitz.open(path)`.
open = extract


# ---------------------------------------------------------------------------
# CLI: python dwg_extract.py <path-to-file>  -- nothing else required
# ---------------------------------------------------------------------------

def _json_default(o):
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"Not JSON serializable: {o!r}")


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python dwg_extract.py <path-to-drawing.dwg-or-.dxf>")
        return 1

    path = Path(sys.argv[1])
    result = extract(path)

    text_out = path.with_name(path.stem + "_text.json")
    geom_out = path.with_name(path.stem + "_geometry.json")
    text_out.write_text(json.dumps(result["text"], indent=2, default=_json_default))
    geom_out.write_text(json.dumps(result["geometry"], indent=2, default=_json_default))

    print(f"Extracted {len(result['text'])} text entities and {len(result['geometry'])} geometry entities from {path}")
    print(f"  Text:     {text_out}")
    print(f"  Geometry: {geom_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())