bl_info = {
    "name": "GRP Enfusion Tools",
    "author": "GRP Tools",
    "version": (2, 2, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > GRP Enfusion Tools",
    "description": "Production-ready Enfusion prep: materials, textures, LODs, and colliders",
    "category": "3D View",
}

import math
import os
import re
import time
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import bmesh
import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
from bpy.types import Operator, Panel, PropertyGroup

EBT_AVAILABLE = True
try:
    from EnfusionBlenderTools.core.materials import accessors as ebt_accessors
    from EnfusionBlenderTools.core.materials.accessors import MaterialAccessor
    from EnfusionBlenderTools.core.materials.utils import SUPPORTED_SHADERS, TextureData, setup_enfusion_material
except Exception:
    EBT_AVAILABLE = False
    MaterialAccessor = None
    SUPPORTED_SHADERS = []
    TextureData = None
    setup_enfusion_material = None


DEFAULT_BASE_COLOR = (0.8, 0.8, 0.8, 1.0)
DEFAULT_NORMAL = (0.5, 0.5, 1.0, 1.0)
DEFAULT_ROUGHNESS = 1.0
DEFAULT_METALNESS = 0.0
DEFAULT_OCCLUSION = 1.0
DEFAULT_ALPHA_TEST = 0.5
SUPPORTED_IMAGE_EXTS = [".tif", ".tiff", ".tga", ".png", ".jpg", ".jpeg", ".dds"]


NAME_HINTS = {
    "bcr": ["_bcr"],
    "nmo": ["_nmo"],
    "base_color": ["albedo", "basecolor", "base_color", "diffuse", "_bc", "_d", "col", "color"],
    "normal": ["normal", "_n", "_nrm", "_nor"],
    "roughness": ["rough", "_r"],
    "metallic": ["metal", "metallic", "_m"],
    "ao": ["ao", "occlusion", "_o"],
    "emission": ["emiss", "emission", "_em"],
    "opacity": ["opacity", "alpha", "_a", "mask"],
    "orm": ["orm", "rma", "mrao", "occlusionroughnessmetallic"],
}

VEHICLE_PART_HINTS = {
    "door": ["door"],
    "window": ["window", "glass"],
    "hood": ["hood", "bonnet"],
    "trunk": ["trunk", "boot", "tailgate", "hatch"],
}

COLLIDER_PREFIXES = ("UBX_", "UCX_", "USP_", "UCS_", "UCL_", "UTM_", "OCC_", "COM_")


class ProgressReporter:
    def __init__(self, context, total_steps: int, label: str = ""):
        self.context = context
        self.total_steps = max(int(total_steps), 1)
        self.label = label
        self.step_index = 0
        self.last_update = 0.0

    def begin(self):
        wm = self.context.window_manager
        wm.progress_begin(0, self.total_steps)
        self._set_header(self.label)

    def step(self, message: str = "", advance: int = 1):
        self.step_index = min(self.step_index + advance, self.total_steps)
        self.context.window_manager.progress_update(self.step_index)
        if message:
            self._set_header(f"{self.label} - {message}" if self.label else message)
        self._maybe_redraw()

    def pulse(self, message: str = ""):
        if message:
            self._set_header(f"{self.label} - {message}" if self.label else message)
        self._maybe_redraw()

    def end(self):
        self.context.window_manager.progress_end()
        self._set_header(None)

    def _maybe_redraw(self):
        now = time.time()
        if now - self.last_update < 0.1:
            return
        try:
            bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
        except Exception:
            pass
        self.last_update = now

    def _set_header(self, text: Optional[str]):
        area = getattr(self.context, "area", None)
        if area and hasattr(area, "header_text_set"):
            area.header_text_set(text)
            return
        for area in self.context.screen.areas:
            if area.type == "PROPERTIES":
                area.header_text_set(text)
                break


def _normalize_name(image: bpy.types.Image) -> str:
    name = image.name
    if image.filepath:
        try:
            name = Path(image.filepath).stem
        except Exception:
            pass
    return name.lower()


def _name_has_hint(name: str, hints: Iterable[str]) -> bool:
    return any(h in name for h in hints)


def _strip_known_suffix(name: str) -> str:
    name = re.sub(r"(_bcr|_bc|_basecolor|_albedo|_diffuse|_nmo|_normal|_rough|_metal|_mrao|_orm)$", "", name, flags=re.IGNORECASE)
    return name


def _sanitize_name(name: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|]", "_", name)
    name = name.replace(".", "_")
    name = name.replace(" ", "_")
    return name


def _base_name_from_image(material: bpy.types.Material, image: Optional[bpy.types.Image]) -> str:
    if image:
        name = _normalize_name(image)
    else:
        name = material.name.lower()
    name = _strip_known_suffix(name)
    return _sanitize_name(name)


def _pick_reference_image(images: Dict[str, Optional[bpy.types.Image]]) -> Optional[bpy.types.Image]:
    for key in ("base_color", "normal", "roughness", "metallic", "ao", "orm", "emission", "opacity", "bcr", "nmo"):
        if images.get(key):
            return images[key]
    return None


def _is_data_colorspace(image: Optional[bpy.types.Image]) -> bool:
    if not image:
        return False
    try:
        return bool(image.colorspace_settings.is_data)
    except Exception:
        return "non-color" in image.colorspace_settings.name.lower()


def _linear_to_srgb_channel(value: float) -> float:
    if value <= 0.0031308:
        return 12.92 * value
    return 1.055 * pow(value, 1.0 / 2.4) - 0.055


def _linear_to_srgb(rgb: Tuple[float, float, float]) -> Tuple[float, float, float]:
    r, g, b = rgb
    r = min(max(_linear_to_srgb_channel(r), 0.0), 1.0)
    g = min(max(_linear_to_srgb_channel(g), 0.0), 1.0)
    b = min(max(_linear_to_srgb_channel(b), 0.0), 1.0)
    return (r, g, b)


def _find_existing_packed_image(base_image: Optional[bpy.types.Image], base_name: str, suffix: str) -> Optional[bpy.types.Image]:
    if not base_image or not base_image.filepath:
        return None

    try:
        base_dir = Path(base_image.filepath).resolve().parent
    except Exception:
        return None

    for ext in SUPPORTED_IMAGE_EXTS:
        candidate = base_dir / f"{base_name}{suffix}{ext}"
        if candidate.exists():
            try:
                return bpy.data.images.load(str(candidate), check_existing=True)
            except Exception:
                return None
    return None


def _image_has_alpha(image: Optional[bpy.types.Image]) -> bool:
    if not image or image.channels < 4:
        return False

    sampler = ImageSampler(image)
    if sampler.width == 0 or sampler.height == 0:
        return False

    grid = 8
    for y in range(grid):
        for x in range(grid):
            _, _, _, alpha = sampler.sample(x, y, grid, grid, (1.0, 1.0, 1.0, 1.0))
            if alpha < 0.999:
                return True
    return False


class ImageSampler:
    def __init__(self, image: Optional[bpy.types.Image]):
        self.image = image
        if not image:
            self.width = 0
            self.height = 0
            self.channels = 0
            self.pixels = []
            return

        image.pixels[0:1]
        self.width = int(image.size[0])
        self.height = int(image.size[1])
        self.channels = int(image.channels)
        self.pixels = list(image.pixels)

    def sample(self, x: int, y: int, dst_width: int, dst_height: int, default: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        if not self.image or self.width == 0 or self.height == 0:
            return default

        sx = min(max(int(x * self.width / dst_width), 0), self.width - 1)
        sy = min(max(int(y * self.height / dst_height), 0), self.height - 1)
        idx = (sy * self.width + sx) * self.channels

        if self.channels <= 0:
            return default
        if self.channels == 1:
            v = self.pixels[idx]
            return (v, v, v, 1.0)
        if self.channels == 2:
            r = self.pixels[idx]
            g = self.pixels[idx + 1]
            return (r, g, 0.0, 1.0)
        if self.channels == 3:
            r, g, b = self.pixels[idx : idx + 3]
            return (r, g, b, 1.0)
        r, g, b, a = self.pixels[idx : idx + 4]
        return (r, g, b, a)


def _average_luminance(image: Optional[bpy.types.Image], grid: int = 8) -> float:
    if not image:
        return 0.0
    sampler = ImageSampler(image)
    if sampler.width == 0 or sampler.height == 0:
        return 0.0

    total = 0.0
    count = 0
    for y in range(grid):
        for x in range(grid):
            r, g, b, _ = sampler.sample(x, y, grid, grid, DEFAULT_BASE_COLOR)
            total += 0.2126 * r + 0.7152 * g + 0.0722 * b
            count += 1
    return total / max(count, 1)


def _is_near_black(image: Optional[bpy.types.Image], threshold: float = 0.02) -> bool:
    return _average_luminance(image) < threshold


def _should_repack_bcr(existing: Optional[bpy.types.Image], base_color: Optional[bpy.types.Image], settings) -> bool:
    if settings.overwrite_packed:
        return True
    if not existing:
        return False
    if not _is_near_black(existing):
        return False
    if base_color and not _is_near_black(base_color, 0.05):
        return True
    return settings.create_missing_textures


def _should_repack_nmo(existing: Optional[bpy.types.Image], textures: Dict[str, Optional[bpy.types.Image]], settings) -> bool:
    if settings.overwrite_packed:
        return True
    if not existing:
        return False
    if not _is_near_black(existing):
        return False

    normal = textures.get("normal")
    if normal and not _is_near_black(normal, 0.05):
        return True
    if textures.get("orm") and not _is_near_black(textures.get("orm"), 0.05):
        return True
    if textures.get("metallic") and not _is_near_black(textures.get("metallic"), 0.05):
        return True
    if textures.get("ao") and not _is_near_black(textures.get("ao"), 0.05):
        return True
    return settings.create_missing_textures


def _find_principled_bsdf(material: bpy.types.Material) -> Optional[bpy.types.Node]:
    if not material or not material.use_nodes:
        return None

    nt = material.node_tree
    for node in nt.nodes:
        if node.type == "OUTPUT_MATERIAL" and node.is_active_output:
            if node.inputs.get("Surface") and node.inputs["Surface"].is_linked:
                src = node.inputs["Surface"].links[0].from_node
                if src.type == "BSDF_PRINCIPLED":
                    return src

    for node in nt.nodes:
        if node.type == "BSDF_PRINCIPLED":
            return node
    return None


def _find_upstream_image(socket: bpy.types.NodeSocket, visited=None) -> Optional[bpy.types.Image]:
    if visited is None:
        visited = set()
    if socket is None or not socket.is_linked:
        return None
    for link in socket.links:
        from_node = link.from_node
        if from_node in visited:
            continue
        visited.add(from_node)

        if from_node.type == "TEX_IMAGE" and getattr(from_node, "image", None):
            return from_node.image

        if from_node.type == "NORMAL_MAP" and from_node.inputs.get("Color"):
            img = _find_upstream_image(from_node.inputs["Color"], visited)
            if img:
                return img

        for input_socket in from_node.inputs:
            img = _find_upstream_image(input_socket, visited)
            if img:
                return img
    return None


def _scan_images_by_name(material: bpy.types.Material) -> Dict[str, Optional[bpy.types.Image]]:
    found = {key: None for key in NAME_HINTS.keys()}
    if not material or not material.use_nodes:
        return found

    for node in material.node_tree.nodes:
        if node.type != "TEX_IMAGE" or not getattr(node, "image", None):
            continue
        image = node.image
        name = _normalize_name(image)

        if _name_has_hint(name, NAME_HINTS["bcr"]):
            found["bcr"] = image
        if _name_has_hint(name, NAME_HINTS["nmo"]):
            found["nmo"] = image
        if _name_has_hint(name, NAME_HINTS["orm"]):
            found["orm"] = image
        if _name_has_hint(name, NAME_HINTS["base_color"]):
            found["base_color"] = image
        if _name_has_hint(name, NAME_HINTS["normal"]):
            found["normal"] = image
        if _name_has_hint(name, NAME_HINTS["roughness"]):
            found["roughness"] = image
        if _name_has_hint(name, NAME_HINTS["metallic"]):
            found["metallic"] = image
        if _name_has_hint(name, NAME_HINTS["ao"]):
            found["ao"] = image
        if _name_has_hint(name, NAME_HINTS["emission"]):
            found["emission"] = image
        if _name_has_hint(name, NAME_HINTS["opacity"]):
            found["opacity"] = image

    return found


def _gather_material_textures(material: bpy.types.Material) -> Dict[str, Optional[bpy.types.Image]]:
    textures = {
        "base_color": None,
        "normal": None,
        "roughness": None,
        "metallic": None,
        "ao": None,
        "emission": None,
        "opacity": None,
        "orm": None,
        "bcr": None,
        "nmo": None,
        "roughness_value": None,
        "metallic_value": None,
    }

    if not material or not material.use_nodes:
        return textures

    bsdf = _find_principled_bsdf(material)
    if bsdf:
        textures["base_color"] = _find_upstream_image(bsdf.inputs.get("Base Color"))
        textures["roughness"] = _find_upstream_image(bsdf.inputs.get("Roughness"))
        textures["metallic"] = _find_upstream_image(bsdf.inputs.get("Metallic"))
        textures["normal"] = _find_upstream_image(bsdf.inputs.get("Normal"))
        textures["emission"] = _find_upstream_image(bsdf.inputs.get("Emission"))
        textures["opacity"] = _find_upstream_image(bsdf.inputs.get("Alpha"))
        textures["roughness_value"] = bsdf.inputs.get("Roughness").default_value if bsdf.inputs.get("Roughness") else None
        textures["metallic_value"] = bsdf.inputs.get("Metallic").default_value if bsdf.inputs.get("Metallic") else None

    name_hits = _scan_images_by_name(material)
    for key in textures.keys():
        if textures[key] is None and key in name_hits:
            textures[key] = name_hits[key]

    return textures


def _find_texture_on_disk(
    base_dir: Optional[Path],
    base_name: str,
    include_hints: Iterable[str],
    exclude_hints: Iterable[str] = (),
    colorspace: str = "sRGB",
) -> Optional[bpy.types.Image]:
    if not base_dir or not base_dir.exists():
        return None

    base_name = (base_name or "").lower()
    include_hints = [h.lower() for h in include_hints]
    exclude_hints = [h.lower() for h in exclude_hints]

    best_path = None
    best_score = -1
    for path in base_dir.iterdir():
        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTS:
            continue
        name = path.stem.lower()
        if exclude_hints and any(h in name for h in exclude_hints):
            continue
        if not any(h in name for h in include_hints):
            continue

        score = 0
        if base_name and base_name in name:
            score += 10
        if base_name and name.startswith(base_name):
            score += 5
        if best_path is None or score > best_score:
            best_path = path
            best_score = score

    if not best_path:
        return None

    try:
        image = bpy.data.images.load(str(best_path), check_existing=True)
    except Exception:
        return None

    _ensure_colorspace(image, colorspace)
    image.alpha_mode = "CHANNEL_PACKED"
    return image


def _resolve_source_textures_from_disk(textures: Dict[str, Optional[bpy.types.Image]], material: bpy.types.Material) -> Dict[str, Optional[bpy.types.Image]]:
    reference = textures.get("base_color") or textures.get("normal") or textures.get("bcr") or textures.get("nmo") or _pick_reference_image(textures)

    base_dir = None
    if reference and reference.filepath:
        try:
            base_dir = Path(reference.filepath).resolve().parent
        except Exception:
            base_dir = None

    base_name = _base_name_from_image(material, reference)

    exclude_packed = NAME_HINTS["bcr"] + NAME_HINTS["nmo"] + NAME_HINTS["orm"]

    if textures.get("base_color") is None or textures.get("base_color") == textures.get("bcr"):
        found = _find_texture_on_disk(base_dir, base_name, NAME_HINTS["base_color"], exclude_hints=exclude_packed, colorspace="sRGB")
        if found:
            textures["base_color"] = found

    if textures.get("normal") is None or textures.get("normal") == textures.get("nmo"):
        found = _find_texture_on_disk(base_dir, base_name, NAME_HINTS["normal"], exclude_hints=exclude_packed, colorspace="Non-Color")
        if found:
            textures["normal"] = found

    if textures.get("orm") is None:
        found = _find_texture_on_disk(base_dir, base_name, NAME_HINTS["orm"], exclude_hints=exclude_packed, colorspace="Non-Color")
        if found:
            textures["orm"] = found

    if textures.get("roughness") is None and textures.get("orm") is None:
        found = _find_texture_on_disk(base_dir, base_name, NAME_HINTS["roughness"], exclude_hints=exclude_packed, colorspace="Non-Color")
        if found:
            textures["roughness"] = found

    if textures.get("metallic") is None and textures.get("orm") is None:
        found = _find_texture_on_disk(base_dir, base_name, NAME_HINTS["metallic"], exclude_hints=exclude_packed, colorspace="Non-Color")
        if found:
            textures["metallic"] = found

    if textures.get("ao") is None and textures.get("orm") is None:
        found = _find_texture_on_disk(base_dir, base_name, NAME_HINTS["ao"], exclude_hints=exclude_packed, colorspace="Non-Color")
        if found:
            textures["ao"] = found

    if textures.get("opacity") is None:
        found = _find_texture_on_disk(base_dir, base_name, NAME_HINTS["opacity"], exclude_hints=exclude_packed, colorspace="Non-Color")
        if found:
            textures["opacity"] = found

    if textures.get("emission") is None:
        found = _find_texture_on_disk(base_dir, base_name, NAME_HINTS["emission"], exclude_hints=exclude_packed, colorspace="sRGB")
        if found:
            textures["emission"] = found

    return textures


def _detect_shader_class(material: bpy.types.Material, has_opacity: bool) -> str:
    name = material.name.lower() if material else ""
    if "decal" in name:
        return "MatPBRDecal"
    if "glass" in name or "window" in name or "transparent" in name:
        return "MatPBRBasicGlass"
    if "trunk" in name:
        return "MatPBRTreeTrunk"
    if "crown" in name or "leaf" in name or "leaves" in name:
        return "MatPBRTreeCrown"
    if "camo" in name:
        return "MatPBRCamo"
    if "skin" in name:
        return "MatPBRSkinProfile"
    if has_opacity and "MatPBRBasicGlass" in SUPPORTED_SHADERS:
        return "MatPBRBasicGlass"
    return "MatPBRBasic"


@dataclass
class PackedResult:
    image: Optional[bpy.types.Image]
    created: bool


def _ensure_colorspace(image: bpy.types.Image, colorspace: str):
    try:
        image.colorspace_settings.name = colorspace
    except Exception:
        pass


def _is_writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        test_file = path / ".write_test"
        with open(test_file, "w", encoding="utf-8") as handle:
            handle.write("test")
        try:
            test_file.unlink()
        except FileNotFoundError:
            pass
        return True
    except Exception:
        return False


def _choose_writable_dir(*candidates: Path) -> Optional[Path]:
    for candidate in candidates:
        if candidate and _is_writable_dir(candidate):
            return candidate
    return None


def _apply_alpha_handling(material: bpy.types.Material, shader_class: str, has_opacity: bool):
    if not has_opacity:
        return

    if shader_class in {"MatPBRBasicGlass", "MatPBRDecal"}:
        if hasattr(material, "BlendMode"):
            material.BlendMode = "AlphaBlend"
        if hasattr(material, "AlphaTest"):
            material.AlphaTest = 0.0
        return

    if hasattr(material, "AlphaTest"):
        material.AlphaTest = DEFAULT_ALPHA_TEST
    if hasattr(material, "BlendMode"):
        material.BlendMode = "None"


def _prepare_texture_data(image: bpy.types.Image, postfix: str, colorspace: str) -> TextureData:
    if image is None:
        return None

    _ensure_colorspace(image, colorspace)
    image.alpha_mode = "CHANNEL_PACKED"

    return TextureData(
        resource_name=image.name,
        texture_source_path=image.filepath or image.name,
        post_fix=postfix,
        color_space=colorspace,
        conversion="None",
        contain_mips=False,
    )


def _get_or_create_packed_image(name: str, width: int, height: int, overwrite: bool) -> PackedResult:
    existing = bpy.data.images.get(name)
    if existing and not overwrite:
        return PackedResult(existing, False)
    if existing:
        bpy.data.images.remove(existing)

    image = bpy.data.images.new(name=name, width=width, height=height, alpha=True, float_buffer=False)
    return PackedResult(image, True)


def _build_output_paths(base_dir: Path, base_name: str, suffix: str) -> Path:
    safe_name = _sanitize_name(base_name)
    return (base_dir / f"{safe_name}{suffix}.tif").resolve()


def _resolve_output_dir(reference_image: Optional[bpy.types.Image]) -> Path:
    candidates: list[Path] = []
    if reference_image and reference_image.filepath:
        try:
            candidates.append(Path(reference_image.filepath).resolve().parent)
        except Exception:
            pass

    blend_dir = Path(bpy.path.abspath("//"))
    if blend_dir.exists():
        candidates.append(blend_dir)

    candidates.append(Path(bpy.app.tempdir))
    candidates.append(Path.home())

    chosen = _choose_writable_dir(*candidates)
    return chosen or Path(bpy.app.tempdir)


def _resolve_unpacked_dir() -> Tuple[Path, bool]:
    blend_dir = Path(bpy.path.abspath("//"))
    if blend_dir.exists():
        candidate = blend_dir / "textures"
        if _is_writable_dir(candidate):
            return candidate, False

    temp_candidate = Path(bpy.app.tempdir) / "enfusion_textures"
    if _is_writable_dir(temp_candidate):
        return temp_candidate, True

    home_candidate = Path.home() / "enfusion_textures"
    _is_writable_dir(home_candidate)
    return home_candidate, True


def _pack_bcr(material: bpy.types.Material, textures: Dict[str, Optional[bpy.types.Image]], settings, progress: Optional[ProgressReporter] = None) -> PackedResult:
    base_reference = textures.get("base_color") or _pick_reference_image(textures)
    base_name = _base_name_from_image(material, textures.get("base_color") or base_reference)
    existing_bcr = textures.get("bcr")
    if existing_bcr and not _should_repack_bcr(existing_bcr, textures.get("base_color") or base_reference, settings):
        return PackedResult(existing_bcr, False)

    existing_bcr = _find_existing_packed_image(base_reference, base_name, "_BCR")
    if existing_bcr and not _should_repack_bcr(existing_bcr, textures.get("base_color") or base_reference, settings):
        return PackedResult(existing_bcr, False)

    if not settings.pack_textures:
        return PackedResult(textures.get("base_color"), False)

    if textures.get("base_color") is None and not settings.create_missing_textures:
        return PackedResult(None, False)

    reference = base_reference
    if not reference and not settings.create_missing_textures:
        return PackedResult(None, False)

    width = int(reference.size[0]) if reference else 1024
    height = int(reference.size[1]) if reference else 1024

    if textures.get("base_color"):
        _ensure_colorspace(textures["base_color"], "sRGB")
    if textures.get("roughness"):
        _ensure_colorspace(textures["roughness"], "Non-Color")
    if textures.get("orm"):
        _ensure_colorspace(textures["orm"], "Non-Color")

    base_color_sampler = ImageSampler(textures.get("base_color"))
    roughness_sampler = ImageSampler(textures.get("roughness"))
    orm_sampler = ImageSampler(textures.get("orm"))

    packed_name = f"{base_name}_BCR"
    packed = _get_or_create_packed_image(packed_name, width, height, settings.overwrite_packed)

    pixels = array("f", [0.0]) * (width * height * 4)
    update_stride = max(1, height // 120)
    for y in range(height):
        for x in range(width):
            idx = (y * width + x) * 4
            base_rgba = base_color_sampler.sample(x, y, width, height, DEFAULT_BASE_COLOR)

            if textures.get("roughness"):
                roughness = roughness_sampler.sample(x, y, width, height, (DEFAULT_ROUGHNESS, 0, 0, 0))[0]
            elif textures.get("orm"):
                roughness = orm_sampler.sample(x, y, width, height, (0, DEFAULT_ROUGHNESS, 0, 0))[1]
            else:
                roughness = textures.get("roughness_value") if textures.get("roughness_value") is not None else DEFAULT_ROUGHNESS

            rgb = (base_rgba[0], base_rgba[1], base_rgba[2])

            pixels[idx] = rgb[0]
            pixels[idx + 1] = rgb[1]
            pixels[idx + 2] = rgb[2]
            pixels[idx + 3] = roughness

        if progress and y % update_stride == 0:
            percent = int((y / max(height - 1, 1)) * 100)
            progress.pulse(f"Packing BCR {material.name} ({percent}%)")

    try:
        packed.image.pixels.foreach_set(pixels)
    except Exception:
        packed.image.pixels = pixels
    packed.image.update()
    _ensure_colorspace(packed.image, "sRGB")
    packed.image.alpha_mode = "CHANNEL_PACKED"

    if settings.save_packed_textures:
        output_dir = _resolve_output_dir(reference)
        filepath = _build_output_paths(output_dir, base_name, "_BCR")
        packed.image.filepath_raw = str(filepath)
        packed.image.file_format = "TIFF"
        if settings.overwrite_packed or not filepath.exists():
            packed.image.save()

    return packed


def _pack_nmo(material: bpy.types.Material, textures: Dict[str, Optional[bpy.types.Image]], settings, progress: Optional[ProgressReporter] = None) -> PackedResult:
    normal_reference = textures.get("normal") or _pick_reference_image(textures)
    base_name = _base_name_from_image(material, textures.get("normal") or normal_reference)
    existing_nmo = textures.get("nmo")
    if existing_nmo and not _should_repack_nmo(existing_nmo, textures, settings):
        return PackedResult(existing_nmo, False)

    existing_nmo = _find_existing_packed_image(normal_reference, base_name, "_NMO")
    if existing_nmo and not _should_repack_nmo(existing_nmo, textures, settings):
        return PackedResult(existing_nmo, False)

    if not settings.pack_textures:
        return PackedResult(textures.get("normal"), False)

    if textures.get("normal") is None and not settings.create_missing_textures:
        if textures.get("metallic") or textures.get("ao") or textures.get("orm"):
            pass
        else:
            return PackedResult(None, False)

    reference = normal_reference
    if not reference and not settings.create_missing_textures:
        return PackedResult(None, False)

    width = int(reference.size[0]) if reference else 1024
    height = int(reference.size[1]) if reference else 1024

    if textures.get("normal"):
        _ensure_colorspace(textures["normal"], "Non-Color")
    if textures.get("metallic"):
        _ensure_colorspace(textures["metallic"], "Non-Color")
    if textures.get("ao"):
        _ensure_colorspace(textures["ao"], "Non-Color")
    if textures.get("orm"):
        _ensure_colorspace(textures["orm"], "Non-Color")

    normal_sampler = ImageSampler(textures.get("normal"))
    metallic_sampler = ImageSampler(textures.get("metallic"))
    ao_sampler = ImageSampler(textures.get("ao"))
    orm_sampler = ImageSampler(textures.get("orm"))

    packed_name = f"{base_name}_NMO"
    packed = _get_or_create_packed_image(packed_name, width, height, settings.overwrite_packed)

    pixels = array("f", [0.0]) * (width * height * 4)
    update_stride = max(1, height // 120)
    for y in range(height):
        for x in range(width):
            idx = (y * width + x) * 4
            normal_rgba = normal_sampler.sample(x, y, width, height, DEFAULT_NORMAL)

            if textures.get("metallic"):
                metalness = metallic_sampler.sample(x, y, width, height, (DEFAULT_METALNESS, 0, 0, 0))[0]
            elif textures.get("orm"):
                metalness = orm_sampler.sample(x, y, width, height, (0, 0, DEFAULT_METALNESS, 0))[2]
            else:
                metalness = textures.get("metallic_value") if textures.get("metallic_value") is not None else DEFAULT_METALNESS

            if textures.get("ao"):
                occlusion = ao_sampler.sample(x, y, width, height, (DEFAULT_OCCLUSION, 0, 0, 0))[0]
            elif textures.get("orm"):
                occlusion = orm_sampler.sample(x, y, width, height, (DEFAULT_OCCLUSION, 0, 0, 0))[0]
            else:
                occlusion = DEFAULT_OCCLUSION

            pixels[idx] = normal_rgba[0]
            pixels[idx + 1] = normal_rgba[1]
            pixels[idx + 2] = metalness
            pixels[idx + 3] = occlusion

        if progress and y % update_stride == 0:
            percent = int((y / max(height - 1, 1)) * 100)
            progress.pulse(f"Packing NMO {material.name} ({percent}%)")

    try:
        packed.image.pixels.foreach_set(pixels)
    except Exception:
        packed.image.pixels = pixels
    packed.image.update()
    _ensure_colorspace(packed.image, "Non-Color")
    packed.image.alpha_mode = "CHANNEL_PACKED"

    if settings.save_packed_textures:
        output_dir = _resolve_output_dir(reference)
        filepath = _build_output_paths(output_dir, base_name, "_NMO")
        packed.image.filepath_raw = str(filepath)
        packed.image.file_format = "TIFF"
        if settings.overwrite_packed or not filepath.exists():
            packed.image.save()

    return packed


def _pack_opacity_from_base_color(material: bpy.types.Material, base_color: Optional[bpy.types.Image], settings, progress: Optional[ProgressReporter] = None) -> PackedResult:
    if base_color is None:
        return PackedResult(None, False)

    base_name = _base_name_from_image(material, base_color)
    existing_alpha = _find_existing_packed_image(base_color, base_name, "_A")
    if existing_alpha:
        return PackedResult(existing_alpha, False)

    if not settings.pack_textures:
        return PackedResult(None, False)

    width = int(base_color.size[0]) if base_color else 1024
    height = int(base_color.size[1]) if base_color else 1024
    sampler = ImageSampler(base_color)

    packed_name = f"{base_name}_A"
    packed = _get_or_create_packed_image(packed_name, width, height, settings.overwrite_packed)

    pixels = array("f", [0.0]) * (width * height * 4)
    update_stride = max(1, height // 120)
    for y in range(height):
        for x in range(width):
            idx = (y * width + x) * 4
            _, _, _, alpha = sampler.sample(x, y, width, height, (1.0, 1.0, 1.0, 1.0))
            pixels[idx] = alpha
            pixels[idx + 1] = alpha
            pixels[idx + 2] = alpha
            pixels[idx + 3] = 1.0

        if progress and y % update_stride == 0:
            percent = int((y / max(height - 1, 1)) * 100)
            progress.pulse(f"Packing Opacity {material.name} ({percent}%)")

    packed.image.pixels = pixels
    _ensure_colorspace(packed.image, "Non-Color")
    packed.image.alpha_mode = "CHANNEL_PACKED"

    if settings.save_packed_textures:
        output_dir = _resolve_output_dir(base_color)
        filepath = _build_output_paths(output_dir, base_name, "_A")
        packed.image.filepath_raw = str(filepath)
        packed.image.file_format = "TIFF"
        if settings.overwrite_packed or not filepath.exists():
            packed.image.save()

    return packed


def _assign_texture_slot(slot, image: Optional[bpy.types.Image], postfix: str, colorspace: str):
    if slot is None:
        return
    if image is None:
        slot.set(None, None)
        return

    tex_data = _prepare_texture_data(image, postfix, colorspace)
    slot.set(image, tex_data)


def _ensure_preview_shader(
    material: bpy.types.Material,
    bcr_image: Optional[bpy.types.Image],
    opacity_image: Optional[bpy.types.Image],
    uv_map: str = "UVMap0",
):
    if not material:
        return

    material.use_nodes = True
    nt = material.node_tree

    def get_or_create(node_type: str, name: str):
        node = nt.nodes.get(name)
        if node and node.type != node_type:
            nt.nodes.remove(node)
            node = None
        if node is None:
            node = nt.nodes.new(node_type)
            node.name = name
            node.label = name
        return node

    preview_img_node = get_or_create("ShaderNodeTexImage", "Preview_BCR")
    preview_bsdf = get_or_create("ShaderNodeBsdfPrincipled", "Preview_BSDF")
    preview_out = get_or_create("ShaderNodeOutputMaterial", "Preview_Output")
    preview_uv = get_or_create("ShaderNodeUVMap", "Preview_UV")

    preview_opacity = None
    if opacity_image:
        preview_opacity = get_or_create("ShaderNodeTexImage", "Preview_Opacity")

    if bcr_image:
        preview_img_node.image = bcr_image
        try:
            preview_img_node.image.colorspace_settings.name = "sRGB"
        except Exception:
            pass

    preview_uv.uv_map = uv_map

    if opacity_image and preview_opacity:
        preview_opacity.image = opacity_image
        try:
            preview_opacity.image.colorspace_settings.name = "Non-Color"
        except Exception:
            pass

    def remove_links_to(node):
        for link in list(nt.links):
            if link.from_node == node or link.to_node == node:
                nt.links.remove(link)

    for node in [preview_img_node, preview_bsdf, preview_out, preview_uv]:
        remove_links_to(node)
    if preview_opacity:
        remove_links_to(preview_opacity)

    nt.links.new(preview_uv.outputs.get("UV"), preview_img_node.inputs.get("Vector"))
    nt.links.new(preview_img_node.outputs.get("Color"), preview_bsdf.inputs.get("Base Color"))

    if preview_opacity:
        nt.links.new(preview_uv.outputs.get("UV"), preview_opacity.inputs.get("Vector"))
        alpha_out = preview_opacity.outputs.get("Alpha") or preview_opacity.outputs.get("Color")
        if alpha_out and preview_bsdf.inputs.get("Alpha"):
            nt.links.new(alpha_out, preview_bsdf.inputs.get("Alpha"))
            material.blend_method = "BLEND"
            material.shadow_method = "CLIP"

    nt.links.new(preview_bsdf.outputs.get("BSDF"), preview_out.inputs.get("Surface"))
    preview_out.is_active_output = True


def _collect_materials(context, scope: str) -> Iterable[bpy.types.Material]:
    mats = []

    if scope == "ACTIVE" and getattr(context, "material", None):
        mats.append(context.material)
    elif scope == "OBJECT" and context.object:
        for slot in context.object.material_slots:
            if slot.material:
                mats.append(slot.material)
    elif scope == "SCENE":
        mats.extend([mat for mat in bpy.data.materials if mat])
    else:
        if getattr(context, "material", None):
            mats.append(context.material)

    return list(dict.fromkeys(mats))


def _collect_objects(context, scope: str) -> Iterable[bpy.types.Object]:
    if scope == "ACTIVE" and context.object:
        return [context.object]
    if scope == "SELECTED":
        return list(context.selected_objects)
    return list(context.scene.objects)


def _unique_name(name: str, existing: set[str]) -> str:
    if name not in existing:
        existing.add(name)
        return name

    index = 1
    while True:
        candidate = f"{name}_{index:02d}"
        if candidate not in existing:
            existing.add(candidate)
            return candidate
        index += 1


def _rename_objects(objects: Iterable[bpy.types.Object], rename_data: bool = True):
    existing_obj_names = set(obj.name for obj in bpy.data.objects)
    existing_data_names = set(block.name for block in bpy.data.meshes)

    for obj in objects:
        new_name = _sanitize_name(obj.name)
        new_name = _unique_name(new_name, existing_obj_names)
        if new_name != obj.name:
            obj.name = new_name

        if rename_data and obj.type == "MESH" and obj.data:
            data_name = _sanitize_name(obj.data.name)
            data_name = _unique_name(data_name, existing_data_names)
            if data_name != obj.data.name:
                obj.data.name = data_name


def _rename_materials(materials: Iterable[bpy.types.Material]):
    existing_mat_names = set(mat.name for mat in bpy.data.materials)
    for mat in materials:
        if not mat:
            continue
        new_name = _sanitize_name(mat.name)
        new_name = _unique_name(new_name, existing_mat_names)
        if new_name != mat.name:
            mat.name = new_name


def _unify_uv_maps(objects: Iterable[bpy.types.Object]):
    for obj in objects:
        if obj.type != "MESH" or not obj.data:
            continue
        for index, uv_layer in enumerate(obj.data.uv_layers):
            uv_layer.name = f"UVMap{index}"


def _collect_images_from_materials(materials: Iterable[bpy.types.Material]) -> list[bpy.types.Image]:
    images = set()
    for mat in materials:
        if not mat or not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type == "TEX_IMAGE" and getattr(node, "image", None):
                images.add(node.image)
    return list(images)


def _rename_images(images: Iterable[bpy.types.Image]):
    existing = set(img.name for img in bpy.data.images)
    for image in images:
        if not image:
            continue
        base_name = Path(image.filepath).stem if image.filepath else image.name
        new_name = _sanitize_name(base_name)
        new_name = _unique_name(new_name, existing)
        if new_name != image.name:
            image.name = new_name


def _make_paths_relative(images: Iterable[bpy.types.Image]):
    for image in images:
        if not image or not image.filepath:
            continue
        if os.path.isabs(image.filepath):
            image.filepath = bpy.path.relpath(image.filepath)


def _guess_image_extension(image: bpy.types.Image) -> str:
    if image.filepath:
        ext = Path(image.filepath).suffix
        if ext:
            return ext

    format_map = {
        "PNG": ".png",
        "TARGA": ".tga",
        "TIFF": ".tif",
        "JPEG": ".jpg",
        "OPEN_EXR": ".exr",
        "OPEN_EXR_MULTILAYER": ".exr",
        "DDS": ".dds",
        "BMP": ".bmp",
        "WEBP": ".webp",
    }
    return format_map.get(image.file_format, ".png")


def _unpack_packed_images(images: Iterable[bpy.types.Image], output_dir: Path, overwrite: bool = False) -> Optional[Path]:
    if not _is_writable_dir(output_dir):
        fallback = _choose_writable_dir(Path(bpy.app.tempdir) / "enfusion_textures", Path.home() / "enfusion_textures")
        if not fallback:
            return None
        output_dir = fallback

    for image in images:
        if not image or not image.packed_file:
            continue

        base_name = Path(image.filepath).stem if image.filepath else image.name
        file_name = f"{_sanitize_name(base_name)}{_guess_image_extension(image)}"
        target_path = (output_dir / file_name).resolve()

        image.filepath_raw = str(target_path)
        method = "OVERWRITE" if overwrite else "WRITE_LOCAL"
        try:
            image.unpack(method=method)
        except Exception:
            try:
                image.unpack(method="WRITE_LOCAL")
            except Exception:
                continue
    return output_dir


def _find_lod_suffix(name: str) -> Optional[int]:
    match = re.search(r"_LOD(\d+)$", name, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _strip_lod_suffix(name: str) -> str:
    return re.sub(r"_LOD\d+$", "", name, flags=re.IGNORECASE)


def _generate_lods(objects: Iterable[bpy.types.Object], lod_count: int, ratio_step: float, apply_modifiers: bool = True) -> list[bpy.types.Object]:
    created: list[bpy.types.Object] = []

    if lod_count < 2:
        return created

    for obj in objects:
        if not obj or obj.type != "MESH":
            continue

        base_name = _sanitize_name(_strip_lod_suffix(obj.name))
        if not base_name:
            base_name = _sanitize_name(obj.name)

        # Ensure base object is LOD0
        if _find_lod_suffix(obj.name) != 0:
            obj.name = f"{base_name}_LOD0"
            if obj.data:
                obj.data.name = f"{base_name}_LOD0"

        for level in range(1, lod_count):
            lod_name = f"{base_name}_LOD{level}"
            if bpy.data.objects.get(lod_name):
                continue

            lod_obj = obj.copy()
            lod_obj.data = obj.data.copy()
            lod_obj.name = lod_name
            lod_obj.data.name = lod_name
            lod_obj.parent = obj.parent
            lod_obj.matrix_world = obj.matrix_world.copy()

            # Add decimate modifier
            mod = lod_obj.modifiers.new(name=f"LOD_{level}_Decimate", type="DECIMATE")
            mod.ratio = max(0.02, min(1.0, ratio_step ** level))
            mod.use_collapse_triangulate = True

            # Link to the same collections
            if obj.users_collection:
                for col in obj.users_collection:
                    col.objects.link(lod_obj)
            else:
                bpy.context.scene.collection.objects.link(lod_obj)

            if apply_modifiers:
                try:
                    depsgraph = bpy.context.evaluated_depsgraph_get()
                    eval_obj = lod_obj.evaluated_get(depsgraph)
                    mesh = bpy.data.meshes.new_from_object(eval_obj, preserve_all_data_layers=True, depsgraph=depsgraph)
                    lod_obj.modifiers.clear()
                    lod_obj.data = mesh
                except Exception:
                    pass

            created.append(lod_obj)

    return created


def _is_collider_name(name: str) -> bool:
    return name.upper().startswith(COLLIDER_PREFIXES)


def _is_visual_mesh(obj: bpy.types.Object) -> bool:
    return obj and obj.type == "MESH" and not _is_collider_name(obj.name)


def _set_custom_prop(obj: bpy.types.Object, prop_name: str, value: str):
    if not obj or not prop_name:
        return
    obj[prop_name] = value


def _split_mesh_by_material_hints(obj: bpy.types.Object, hints: list[str], suffix: str) -> Optional[bpy.types.Object]:
    if not obj or obj.type != "MESH" or not obj.data or not obj.material_slots:
        return None

    hint_lc = [h.strip().lower() for h in hints if h.strip()]
    if not hint_lc:
        return None

    material_indices = []
    for i, slot in enumerate(obj.material_slots):
        if not slot.material:
            continue
        name = slot.material.name.lower()
        if any(h in name for h in hint_lc):
            material_indices.append(i)

    if not material_indices:
        return None

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    bm_new = bm.copy()
    bm_new.faces.ensure_lookup_table()
    delete_faces = [f for f in bm_new.faces if f.material_index not in material_indices]
    if delete_faces:
        bmesh.ops.delete(bm_new, geom=delete_faces, context="FACES")

    if len(bm_new.faces) == 0:
        bm_new.free()
        bm.free()
        return None

    delete_faces_main = [f for f in bm.faces if f.material_index in material_indices]
    if delete_faces_main:
        bmesh.ops.delete(bm, geom=delete_faces_main, context="FACES")

    bm.to_mesh(obj.data)
    bm.free()

    new_mesh = bpy.data.meshes.new(_sanitize_name(obj.name + suffix))
    bm_new.to_mesh(new_mesh)
    bm_new.free()

    new_mesh.materials.clear()
    for mat in obj.data.materials:
        new_mesh.materials.append(mat)

    new_obj = bpy.data.objects.new(new_mesh.name, new_mesh)
    for col in obj.users_collection:
        col.objects.link(new_obj)
    if not obj.users_collection:
        bpy.context.scene.collection.objects.link(new_obj)

    new_obj.matrix_world = obj.matrix_world.copy()
    new_obj.parent = obj.parent
    return new_obj


def _bbox_world(obj: bpy.types.Object):
    from mathutils import Vector

    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_v = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
    max_v = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    return min_v, max_v


def _create_box_collider(name: str, target_obj: bpy.types.Object) -> bpy.types.Object:
    from mathutils import Vector

    min_v, max_v = _bbox_world(target_obj)
    center = (min_v + max_v) * 0.5
    size = max_v - min_v

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=center)
    col = bpy.context.active_object
    col.name = name
    col.data.name = name
    col.scale = Vector((size.x * 0.5, size.y * 0.5, size.z * 0.5))
    col.rotation_euler = target_obj.rotation_euler
    col.display_type = "WIRE"
    col.hide_render = True
    col.show_in_front = True
    return col


def _create_trimesh_collider(name: str, target_obj: bpy.types.Object, decimate_ratio: float) -> bpy.types.Object:
    col = target_obj.copy()
    col.data = target_obj.data.copy()
    col.name = name
    col.data.name = name

    if target_obj.users_collection:
        for colc in target_obj.users_collection:
            colc.objects.link(col)
    else:
        bpy.context.scene.collection.objects.link(col)

    col.display_type = "WIRE"
    col.hide_render = True
    col.show_in_front = True

    mod = col.modifiers.new(name="FireDecimate", type="DECIMATE")
    mod.ratio = max(0.02, min(1.0, decimate_ratio))
    mod.use_collapse_triangulate = True

    try:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = col.evaluated_get(depsgraph)
        mesh = bpy.data.meshes.new_from_object(eval_obj, preserve_all_data_layers=True, depsgraph=depsgraph)
        col.modifiers.clear()
        col.data = mesh
    except Exception:
        pass

    return col


def _create_empty(name: str, location) -> bpy.types.Object:
    empty = bpy.data.objects.new(name, None)
    empty.empty_display_type = "ARROWS"
    empty.location = location
    bpy.context.scene.collection.objects.link(empty)
    return empty


def _classify_vehicle_part(name: str) -> Optional[str]:
    lowered = name.lower()
    for part, hints in VEHICLE_PART_HINTS.items():
        if any(h in lowered for h in hints):
            return part
    return None


def _collect_vehicle_parts(objects: Iterable[bpy.types.Object]) -> list[tuple[bpy.types.Object, str]]:
    parts: list[tuple[bpy.types.Object, str]] = []
    for obj in objects:
        if not obj or obj.type != "MESH":
            continue
        part = _classify_vehicle_part(obj.name)
        if part:
            parts.append((obj, part))
    return parts


def _axis_index(axis: str) -> int:
    axis = (axis or "Z").upper()
    if axis == "X":
        return 0
    if axis == "Y":
        return 1
    return 2


def _ensure_vehicle_armature(name: str, location) -> bpy.types.Object:
    armature_name = _sanitize_name(name or "Vehicle_Rig")
    existing = bpy.data.objects.get(armature_name)
    if existing and existing.type == "ARMATURE":
        return existing

    bpy.ops.object.armature_add(enter_editmode=False, location=location)
    arm = bpy.context.active_object
    arm.name = armature_name
    arm.data.name = armature_name
    return arm


def _ensure_root_bone(armature: bpy.types.Object, root_name: str) -> str:
    root_name = root_name or "Root"
    bpy.ops.object.mode_set(mode="EDIT")
    root = armature.data.edit_bones.get(root_name)
    if root is None:
        root = armature.data.edit_bones.new(root_name)
        root.head = (0.0, 0.0, 0.0)
        root.tail = (0.0, 0.0, 0.1)
    if (root.tail - root.head).length < 0.01:
        root.tail.z += 0.1
    return root.name


def _create_part_bones(armature: bpy.types.Object, parts: list[tuple[bpy.types.Object, str]], root_bone_name: str) -> dict[str, str]:
    from mathutils import Vector

    bone_map: dict[str, str] = {}
    arm_inv = armature.matrix_world.inverted()

    bpy.ops.object.mode_set(mode="EDIT")
    root = armature.data.edit_bones.get(root_bone_name)

    existing_names = {bone.name for bone in armature.data.edit_bones}

    for obj, _part in parts:
        bone_name = _unique_name(_sanitize_name(obj.name), existing_names)
        bone = armature.data.edit_bones.get(bone_name)
        if bone is None:
            bone = armature.data.edit_bones.new(bone_name)

        head = arm_inv @ obj.matrix_world.translation
        bone.head = head
        bone.tail = head + Vector((0.0, 0.0, 0.1))
        if root:
            bone.parent = root
        bone_map[obj.name] = bone.name

    return bone_map


def _parent_object_to_bone(obj: bpy.types.Object, armature: bpy.types.Object, bone_name: str):
    obj.parent = armature
    obj.parent_type = "BONE"
    obj.parent_bone = bone_name
    obj.matrix_parent_inverse = armature.matrix_world.inverted() @ obj.matrix_world


def _ensure_action(name: str, overwrite: bool) -> bpy.types.Action:
    action_name = _sanitize_name(name)
    action = bpy.data.actions.get(action_name)
    if action and overwrite:
        bpy.data.actions.remove(action)
        action = None
    if action is None:
        action = bpy.data.actions.new(action_name)
    return action


def _apply_part_keyframes(
    armature: bpy.types.Object,
    bone_name: str,
    action: bpy.types.Action,
    start_frame: int,
    end_frame: int,
    mode: str,
    axis_index: int,
    amount: float,
    reverse: bool,
):
    armature.animation_data_create()
    armature.animation_data.action = action
    pose_bone = armature.pose.bones.get(bone_name)
    if not pose_bone:
        return False

    start_value = amount if reverse else 0.0
    end_value = 0.0 if reverse else amount

    if mode == "rotation":
        pose_bone.rotation_mode = "XYZ"
        rot = list(pose_bone.rotation_euler)
        rot[axis_index] = start_value
        pose_bone.rotation_euler = rot
        pose_bone.keyframe_insert(data_path="rotation_euler", frame=start_frame)

        rot = list(pose_bone.rotation_euler)
        rot[axis_index] = end_value
        pose_bone.rotation_euler = rot
        pose_bone.keyframe_insert(data_path="rotation_euler", frame=end_frame)
    else:
        loc = list(pose_bone.location)
        loc[axis_index] = start_value
        pose_bone.location = loc
        pose_bone.keyframe_insert(data_path="location", frame=start_frame)

        loc = list(pose_bone.location)
        loc[axis_index] = end_value
        pose_bone.location = loc
        pose_bone.keyframe_insert(data_path="location", frame=end_frame)

    return True


def _apply_material_fixes(materials: Iterable[bpy.types.Material], shader_mode: str, settings, progress: Optional[ProgressReporter] = None) -> tuple[int, int, int, list[str]]:
    processed = 0
    packed_count = 0
    skipped = 0
    warnings: list[str] = []

    for material in materials:
        if material is None:
            continue

        if progress:
            progress.step(f"Processing {material.name}")

        textures = _gather_material_textures(material)
        textures = _resolve_source_textures_from_disk(textures, material)
        has_opacity = bool(textures.get("opacity"))

        shader_class = shader_mode
        if shader_class == "AUTO":
            shader_class = _detect_shader_class(material, has_opacity)

        if shader_class not in SUPPORTED_SHADERS:
            warnings.append(f"Material {material.name}: Unsupported shader {shader_class}")
            skipped += 1
            continue

        assigned_material_data = {"ShaderClass": shader_class}
        messages = setup_enfusion_material(material, assigned_material_data, "")
        if messages:
            warnings.extend(messages)

        material.ebt_resource_name = ""

        try:
            accessor = MaterialAccessor.get(material)
        except Exception as exc:
            warnings.append(f"Material {material.name}: {exc}")
            skipped += 1
            continue

        bcr_result = _pack_bcr(material, textures, settings, progress=progress)
        nmo_result = _pack_nmo(material, textures, settings, progress=progress)

        if bcr_result.image:
            _assign_texture_slot(getattr(accessor, "BCRMap", None), bcr_result.image, "_BCR", "sRGB")
            if bcr_result.created:
                packed_count += 1

        if nmo_result.image:
            _assign_texture_slot(getattr(accessor, "NMOMap", None), nmo_result.image, "_NMO", "Non-Color")
            if nmo_result.created:
                packed_count += 1

        opacity_result = None
        if textures.get("opacity"):
            _assign_texture_slot(getattr(accessor, "OpacityMap", None), textures["opacity"], "_A", "Non-Color")
        elif settings.use_base_alpha_opacity and _image_has_alpha(textures.get("base_color")):
            opacity_result = _pack_opacity_from_base_color(material, textures.get("base_color"), settings, progress=progress)
            if opacity_result.image:
                _assign_texture_slot(getattr(accessor, "OpacityMap", None), opacity_result.image, "_A", "Non-Color")

        if textures.get("emission"):
            _assign_texture_slot(getattr(accessor, "EmissiveMap", None), textures["emission"], "_EM", "sRGB")

        _apply_alpha_handling(material, shader_class, has_opacity)

        if getattr(settings, "enable_preview_shader", False):
            preview_bcr = bcr_result.image
            if preview_bcr and _is_near_black(preview_bcr) and textures.get("base_color") and not _is_near_black(textures.get("base_color"), 0.05):
                preview_bcr = textures.get("base_color")
            if preview_bcr is None:
                preview_bcr = textures.get("base_color")
            preview_opacity = textures.get("opacity") or (opacity_result.image if opacity_result else None)
            _ensure_preview_shader(material, preview_bcr, preview_opacity)

        processed += 1

    return processed, packed_count, skipped, warnings


class EBTMaterialFixerSettings(PropertyGroup):
    shader_mode: EnumProperty(
        name="Shader",
        items=[
            ("AUTO", "Auto", "Detect shader type from material"),
            ("MatPBRBasic", "PBR Basic", ""),
            ("MatPBRMulti", "PBR Multi", ""),
            ("MatPBRDecal", "PBR Decal", ""),
            ("MatPBRBasicGlass", "PBR Basic Glass", ""),
            ("MatPBRTreeTrunk", "PBR Tree Trunk", ""),
            ("MatPBRTreeCrown", "PBR Tree Crown", ""),
            ("MatPBR2Layers", "PBR 2 Layers", ""),
            ("MatPBRCamo", "PBR Camo", ""),
            ("MatPBRSkinProfile", "PBR Skin Profile", ""),
        ],
        default="AUTO",
    )

    scope: EnumProperty(
        name="Scope",
        items=[
            ("ACTIVE", "Active Material", "Fix only active material"),
            ("OBJECT", "Active Object", "Fix all materials on active object"),
            ("SCENE", "All Materials", "Fix all materials in the file"),
        ],
        default="OBJECT",
    )

    pack_textures: BoolProperty(
        name="Pack BCR/NMO",
        default=True,
        description="Generate Enfusion packed textures when missing",
    )

    save_packed_textures: BoolProperty(
        name="Save Packed Textures",
        default=True,
        description="Save generated textures to disk as .tif",
    )

    overwrite_packed: BoolProperty(
        name="Overwrite Existing",
        default=False,
        description="Overwrite existing packed textures with the same name",
    )

    create_missing_textures: BoolProperty(
        name="Create Missing Defaults",
        default=True,
        description="Create default BCR/NMO textures even when inputs are missing",
    )

    use_base_alpha_opacity: BoolProperty(
        name="Use Base Alpha as Opacity",
        default=True,
        description="Create OpacityMap from base color alpha when no opacity map is present",
    )

    enable_preview_shader: BoolProperty(
        name="Enable Preview Shader",
        default=True,
        description="Create a Blender preview shader so the material isn't black",
    )


ASSET_TYPE_SHADER = {
    "GENERIC": "AUTO",
    "DECAL": "MatPBRDecal",
    "GLASS": "MatPBRBasicGlass",
    "VEGETATION_TRUNK": "MatPBRTreeTrunk",
    "VEGETATION_CROWN": "MatPBRTreeCrown",
    "SKIN": "MatPBRSkinProfile",
    "CAMO": "MatPBRCamo",
    "MULTI": "MatPBRMulti",
    "TWO_LAYERS": "MatPBR2Layers",
}


class EBTOneTapSettings(PropertyGroup):
    asset_type: EnumProperty(
        name="Asset Type",
        items=[
            ("GENERIC", "Generic", "Use auto-detected shaders"),
            ("DECAL", "Decal", "Force decal shader"),
            ("GLASS", "Glass", "Force glass shader"),
            ("VEGETATION_TRUNK", "Vegetation Trunk", "Force tree trunk shader"),
            ("VEGETATION_CROWN", "Vegetation Crown", "Force tree crown shader"),
            ("SKIN", "Skin", "Force skin profile shader"),
            ("CAMO", "Camo", "Force camo shader"),
            ("MULTI", "Multi", "Force multi shader"),
            ("TWO_LAYERS", "Two Layers", "Force 2-layer shader"),
        ],
        default="GENERIC",
    )

    scope: EnumProperty(
        name="Scope",
        items=[
            ("ACTIVE", "Active Object", "Only active object"),
            ("SELECTED", "Selected Objects", "Only selected objects"),
            ("SCENE", "All Objects", "All objects in the scene"),
        ],
        default="SCENE",
    )

    rename_objects: BoolProperty(
        name="Rename Objects",
        default=True,
        description="Sanitize object and mesh names",
    )

    rename_materials: BoolProperty(
        name="Rename Materials",
        default=True,
        description="Sanitize material names",
    )

    rename_images: BoolProperty(
        name="Rename Images",
        default=True,
        description="Rename images to match file stems",
    )

    unpack_embedded_textures: BoolProperty(
        name="Unpack Embedded Textures",
        default=True,
        description="Extract packed textures embedded in FBX/Blend",
    )

    overwrite_unpacked_textures: BoolProperty(
        name="Overwrite Unpacked Textures",
        default=False,
        description="Overwrite existing files when unpacking embedded textures",
    )

    make_paths_relative: BoolProperty(
        name="Make Paths Relative",
        default=True,
        description="Convert image paths to relative paths",
    )

    unify_uvs: BoolProperty(
        name="Unify UV Names",
        default=True,
        description="Rename UV maps to UVMap0/UVMap1",
    )

    cleanup_orphans: BoolProperty(
        name="Cleanup Orphans",
        default=True,
        description="Purge orphaned data blocks",
    )

    generate_lods: BoolProperty(
        name="Generate LODs",
        default=False,
        description="Create LOD meshes using decimation",
    )

    lod_count: IntProperty(
        name="LOD Count",
        default=3,
        min=2,
        max=6,
        description="Number of LOD levels to generate (including LOD0)",
    )

    lod_ratio_step: FloatProperty(
        name="LOD Ratio Step",
        default=0.5,
        min=0.05,
        max=0.95,
        description="Decimate ratio multiplier per LOD level",
    )

    pack_textures: BoolProperty(
        name="Pack BCR/NMO",
        default=True,
        description="Generate Enfusion packed textures when missing",
    )

    save_packed_textures: BoolProperty(
        name="Save Packed Textures",
        default=True,
        description="Save generated textures to disk as .tif",
    )

    overwrite_packed: BoolProperty(
        name="Overwrite Existing",
        default=False,
        description="Overwrite existing packed textures with the same name",
    )

    create_missing_textures: BoolProperty(
        name="Create Missing Defaults",
        default=True,
        description="Create default BCR/NMO textures even when inputs are missing",
    )

    use_base_alpha_opacity: BoolProperty(
        name="Use Base Alpha as Opacity",
        default=True,
        description="Create OpacityMap from base color alpha when no opacity map is present",
    )

    enable_preview_shader: BoolProperty(
        name="Enable Preview Shader",
        default=True,
        description="Create a Blender preview shader so the material isn't black",
    )


class EBTVehicleAnimSettings(PropertyGroup):
    scope: EnumProperty(
        name="Scope",
        items=[
            ("ACTIVE", "Active Object", "Only active object"),
            ("SELECTED", "Selected Objects", "Only selected objects"),
            ("SCENE", "All Objects", "All objects in the scene"),
        ],
        default="SELECTED",
    )

    armature_name: StringProperty(
        name="Armature Name",
        default="Vehicle_Rig",
    )

    root_bone_name: StringProperty(
        name="Root Bone",
        default="Root",
    )

    create_actions: BoolProperty(
        name="Create Open/Close Actions",
        default=True,
    )

    overwrite_actions: BoolProperty(
        name="Overwrite Existing Actions",
        default=False,
    )

    anim_start_frame: IntProperty(
        name="Start Frame",
        default=0,
        min=0,
    )

    anim_end_frame: IntProperty(
        name="End Frame",
        default=20,
        min=1,
    )

    rotate_axis: EnumProperty(
        name="Rotate Axis",
        items=[("X", "X", ""), ("Y", "Y", ""), ("Z", "Z", "")],
        default="Y",
    )

    door_open_angle: FloatProperty(
        name="Door Open Angle",
        default=70.0,
        min=0.0,
        max=180.0,
        subtype="ANGLE",
    )

    hood_open_angle: FloatProperty(
        name="Hood Open Angle",
        default=50.0,
        min=0.0,
        max=180.0,
        subtype="ANGLE",
    )

    trunk_open_angle: FloatProperty(
        name="Trunk Open Angle",
        default=50.0,
        min=0.0,
        max=180.0,
        subtype="ANGLE",
    )

    window_axis: EnumProperty(
        name="Window Axis",
        items=[("X", "X", ""), ("Y", "Y", ""), ("Z", "Z", "")],
        default="Z",
    )

    window_slide_distance: FloatProperty(
        name="Window Slide Distance",
        default=0.2,
        min=0.0,
    )


class EBTVehiclePrepSettings(PropertyGroup):
    scope: EnumProperty(
        name="Scope",
        items=[
            ("ACTIVE", "Active Object", "Only active object"),
            ("SELECTED", "Selected Objects", "Only selected objects"),
            ("SCENE", "All Objects", "All objects in the scene"),
        ],
        default="SELECTED",
    )

    apply_scale: BoolProperty(
        name="Apply Scale",
        default=True,
        description="Apply scale to selected objects",
    )

    auto_orient_y: BoolProperty(
        name="Auto‑orient to Y+",
        default=False,
        description="Rotate objects 90° if X dimension is longer than Y",
    )

    enforce_lod_suffix: BoolProperty(
        name="Ensure _LOD0",
        default=True,
        description="Add _LOD0 suffix to visual meshes missing LOD suffix",
    )

    split_glass_materials: BoolProperty(
        name="Split Glass by Material",
        default=True,
        description="Separate glass/window materials into their own object",
    )

    glass_material_hints: StringProperty(
        name="Glass Material Hints",
        default="glass,window",
        description="Comma‑separated hints used to detect glass materials",
    )

    create_collision_collider: BoolProperty(
        name="Create Collision Collider",
        default=True,
    )

    collision_prefix: StringProperty(
        name="Collision Prefix",
        default="UBX",
    )

    collision_usage: StringProperty(
        name="Collision Usage",
        default="Collision",
    )

    collision_layer_preset: StringProperty(
        name="Collision Layer Preset",
        default="VehicleSimple",
    )

    create_fire_collider: BoolProperty(
        name="Create Fire Geometry",
        default=True,
    )

    fire_prefix: StringProperty(
        name="Fire Prefix",
        default="UTM",
    )

    fire_decimate_ratio: FloatProperty(
        name="Fire Decimate Ratio",
        default=0.6,
        min=0.02,
        max=1.0,
    )

    fire_usage: StringProperty(
        name="Fire Usage",
        default="Fire",
    )

    fire_layer_preset: StringProperty(
        name="Fire Layer Preset",
        default="FireGeo",
    )

    create_occluder: BoolProperty(
        name="Create Occluder",
        default=False,
    )

    occluder_prefix: StringProperty(
        name="Occluder Prefix",
        default="OCC",
    )

    occluder_usage: StringProperty(
        name="Occluder Usage",
        default="Occluder",
    )

    occluder_layer_preset: StringProperty(
        name="Occluder Layer Preset",
        default="ViewGeo",
    )

    create_center_of_mass: BoolProperty(
        name="Create COM",
        default=True,
    )

    com_prefix: StringProperty(
        name="COM Prefix",
        default="COM",
    )

    create_snap_points_empty: BoolProperty(
        name="Create SnapPoints Empty",
        default=True,
    )

    snap_points_name: StringProperty(
        name="SnapPoints Name",
        default="SNAPPOINTS",
    )

    usage_property: StringProperty(
        name="Collider Usage Property",
        default="usage",
    )

    layer_preset_property: StringProperty(
        name="Layer Preset Property",
        default="LayerPreset",
    )


class EBT_OT_fix_enfusion_materials(Operator):
    bl_idname = "ebt.fix_enfusion_materials"
    bl_label = "Fix Enfusion Materials"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        if not EBT_AVAILABLE:
            self.report({"ERROR"}, "EnfusionBlenderTools is not available. Enable the add-on first.")
            return {"CANCELLED"}

        ebt_accessors.register()

        settings = context.scene.ebt_material_fixer_settings
        materials = _collect_materials(context, settings.scope)
        progress = ProgressReporter(context, total_steps=len(materials) + 1, label="Enfusion Material Fixer")
        progress.begin()
        processed, packed_count, skipped, warnings = _apply_material_fixes(materials, settings.shader_mode, settings, progress=progress)
        progress.end()

        if warnings:
            for warning in warnings[:10]:
                self.report({"WARNING"}, warning)

        self.report({"INFO"}, f"Fixed {processed} materials. Packed {packed_count} textures. Skipped {skipped}.")
        return {"FINISHED"}


class EBT_OT_one_tap_enfusion_setup(Operator):
    bl_idname = "ebt.one_tap_enfusion_setup"
    bl_label = "One Tap Enfusion Setup"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        if not EBT_AVAILABLE:
            self.report({"ERROR"}, "EnfusionBlenderTools is not available. Enable the add-on first.")
            return {"CANCELLED"}

        ebt_accessors.register()

        settings = context.scene.ebt_one_tap_settings
        objects = _collect_objects(context, settings.scope)

        materials = set()
        for obj in objects:
            for slot in obj.material_slots:
                if slot.material:
                    materials.add(slot.material)

        material_list = list(materials)
        total_steps = len(material_list) + 8
        progress = ProgressReporter(context, total_steps=total_steps, label="One Tap Enfusion")
        progress.begin()

        if settings.unify_uvs:
            progress.step("Unifying UVs")
            _unify_uv_maps(objects)

        if settings.rename_objects:
            progress.step("Renaming objects")
            _rename_objects(objects, rename_data=True)

        if settings.rename_materials:
            progress.step("Renaming materials")
            _rename_materials(material_list)

        if settings.generate_lods:
            progress.step("Generating LODs")
            _generate_lods(objects, settings.lod_count, settings.lod_ratio_step, apply_modifiers=True)

        shader_mode = ASSET_TYPE_SHADER.get(settings.asset_type, "AUTO")

        progress.step("Fixing materials")
        processed, packed_count, skipped, warnings = _apply_material_fixes(material_list, shader_mode, settings, progress=progress)

        images = _collect_images_from_materials(material_list)
        if settings.unpack_embedded_textures:
            progress.step("Unpacking embedded textures")
            output_dir, used_fallback = _resolve_unpacked_dir()
            actual_dir = _unpack_packed_images(images, output_dir, overwrite=settings.overwrite_unpacked_textures)
            if actual_dir is None:
                warnings.append("Unable to unpack embedded textures (no writable directory found).")
            elif used_fallback:
                warnings.append(f"Unpacked embedded textures to '{actual_dir}'.")

            images = _collect_images_from_materials(material_list)
        if settings.rename_images:
            progress.step("Renaming images")
            _rename_images(images)
        if settings.make_paths_relative:
            progress.step("Making paths relative")
            _make_paths_relative(images)

        if settings.cleanup_orphans:
            progress.step("Cleaning orphans")
            bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)

        progress.end()

        if warnings:
            for warning in warnings[:10]:
                self.report({"WARNING"}, warning)

        self.report({"INFO"}, f"One tap complete. Fixed {processed} materials, packed {packed_count}. Skipped {skipped}.")
        return {"FINISHED"}


class EBT_OT_setup_vehicle_anims(Operator):
    bl_idname = "ebt.setup_vehicle_anims"
    bl_label = "Setup Vehicle Animations"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.ebt_vehicle_anim_settings
        objects = _collect_objects(context, settings.scope)
        parts = _collect_vehicle_parts(objects)

        if not parts:
            self.report({"ERROR"}, "No door/window/hood/trunk objects found (name them with door/window/hood/trunk).")
            return {"CANCELLED"}

        base_obj = context.object or parts[0][0]
        armature = _ensure_vehicle_armature(settings.armature_name, base_obj.location)

        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:
            pass
        bpy.ops.object.select_all(action="DESELECT")
        armature.select_set(True)
        context.view_layer.objects.active = armature

        root_bone_name = _ensure_root_bone(armature, settings.root_bone_name)
        bone_map = _create_part_bones(armature, parts, root_bone_name)
        bpy.ops.object.mode_set(mode="OBJECT")

        for obj, _part in parts:
            bone_name = bone_map.get(obj.name)
            if bone_name:
                _parent_object_to_bone(obj, armature, bone_name)

        created_actions = 0
        if settings.create_actions:
            rot_axis = _axis_index(settings.rotate_axis)
            win_axis = _axis_index(settings.window_axis)
            start_frame = settings.anim_start_frame
            end_frame = settings.anim_end_frame

            for obj, part in parts:
                bone_name = bone_map.get(obj.name)
                if not bone_name:
                    continue

                if part == "window":
                    mode = "location"
                    amount = settings.window_slide_distance
                    axis = win_axis
                elif part == "hood":
                    mode = "rotation"
                    amount = math.radians(settings.hood_open_angle)
                    axis = rot_axis
                elif part == "trunk":
                    mode = "rotation"
                    amount = math.radians(settings.trunk_open_angle)
                    axis = rot_axis
                else:
                    mode = "rotation"
                    amount = math.radians(settings.door_open_angle)
                    axis = rot_axis

                open_action = _ensure_action(f"{bone_name}_open", settings.overwrite_actions)
                close_action = _ensure_action(f"{bone_name}_close", settings.overwrite_actions)

                if _apply_part_keyframes(armature, bone_name, open_action, start_frame, end_frame, mode, axis, amount, reverse=False):
                    created_actions += 1
                if _apply_part_keyframes(armature, bone_name, close_action, start_frame, end_frame, mode, axis, amount, reverse=True):
                    created_actions += 1

        self.report({"INFO"}, f"Vehicle animation setup complete. Parts: {len(parts)}. Actions: {created_actions}.")
        return {"FINISHED"}


class EBT_OT_vehicle_prep(Operator):
    bl_idname = "ebt.vehicle_prep"
    bl_label = "Vehicle Asset Prep"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.ebt_vehicle_prep_settings
        objects = _collect_objects(context, settings.scope)
        if not objects:
            self.report({"ERROR"}, "No objects found for vehicle prep.")
            return {"CANCELLED"}

        visual_meshes = [obj for obj in objects if _is_visual_mesh(obj)]

        _rename_objects(objects, rename_data=True)

        if settings.split_glass_materials:
            hints = settings.glass_material_hints.split(",")
            for obj in list(visual_meshes):
                new_obj = _split_mesh_by_material_hints(obj, hints, "_window")
                if new_obj:
                    visual_meshes.append(new_obj)

        if settings.enforce_lod_suffix:
            for obj in visual_meshes:
                if _find_lod_suffix(obj.name) is None:
                    base_name = _sanitize_name(obj.name)
                    obj.name = f"{base_name}_LOD0"
                    if obj.data:
                        obj.data.name = obj.name

        if settings.apply_scale:
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
                bpy.ops.object.select_all(action="DESELECT")
                for obj in objects:
                    obj.select_set(True)
                context.view_layer.objects.active = objects[0]
                bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            except Exception:
                pass

        if settings.auto_orient_y and visual_meshes:
            # Heuristic: if X extent is larger than Y, rotate 90° around Z
            from mathutils import Vector

            min_v = Vector((1e9, 1e9, 1e9))
            max_v = Vector((-1e9, -1e9, -1e9))
            for obj in visual_meshes:
                bmin, bmax = _bbox_world(obj)
                min_v.x = min(min_v.x, bmin.x)
                min_v.y = min(min_v.y, bmin.y)
                min_v.z = min(min_v.z, bmin.z)
                max_v.x = max(max_v.x, bmax.x)
                max_v.y = max(max_v.y, bmax.y)
                max_v.z = max(max_v.z, bmax.z)

            extent = max_v - min_v
            if extent.x > extent.y:
                for obj in visual_meshes:
                    if obj.parent is None:
                        obj.rotation_euler.z += math.radians(90)

        created_colliders = 0
        created_helpers = 0

        for obj in visual_meshes:
            if _find_lod_suffix(obj.name) not in (None, 0):
                continue

            base_name = _strip_lod_suffix(obj.name)

            if settings.create_collision_collider:
                col_name = f"{settings.collision_prefix}_{base_name}_00"
                if not bpy.data.objects.get(col_name):
                    col = _create_box_collider(col_name, obj)
                    created_colliders += 1
                else:
                    col = bpy.data.objects.get(col_name)

                if col:
                    _set_custom_prop(col, settings.usage_property, settings.collision_usage)
                    _set_custom_prop(col, settings.layer_preset_property, settings.collision_layer_preset)

            if settings.create_fire_collider:
                fire_name = f"{settings.fire_prefix}_{base_name}_00"
                if not bpy.data.objects.get(fire_name):
                    fire = _create_trimesh_collider(fire_name, obj, settings.fire_decimate_ratio)
                    created_colliders += 1
                else:
                    fire = bpy.data.objects.get(fire_name)

                if fire:
                    _set_custom_prop(fire, settings.usage_property, settings.fire_usage)
                    _set_custom_prop(fire, settings.layer_preset_property, settings.fire_layer_preset)

            if settings.create_occluder:
                occ_name = f"{settings.occluder_prefix}_{base_name}_00"
                if not bpy.data.objects.get(occ_name):
                    occ = _create_box_collider(occ_name, obj)
                    created_helpers += 1
                else:
                    occ = bpy.data.objects.get(occ_name)

                if occ:
                    _set_custom_prop(occ, settings.usage_property, settings.occluder_usage)
                    _set_custom_prop(occ, settings.layer_preset_property, settings.occluder_layer_preset)

            if settings.create_center_of_mass:
                com_name = f"{settings.com_prefix}_{base_name}"
                if not bpy.data.objects.get(com_name):
                    min_v, max_v = _bbox_world(obj)
                    center = (min_v + max_v) * 0.5
                    _create_empty(com_name, center)
                    created_helpers += 1

        if settings.create_snap_points_empty and settings.snap_points_name:
            snap_name = _sanitize_name(settings.snap_points_name)
            if not bpy.data.objects.get(snap_name):
                _create_empty(snap_name, (0.0, 0.0, 0.0))
                created_helpers += 1

        self.report({"INFO"}, f"Vehicle prep complete. Colliders: {created_colliders}, Helpers: {created_helpers}.")
        return {"FINISHED"}


class EBT_PT_enfusion_material_fixer(Panel):
    bl_label = "GRP Enfusion Tools"
    bl_idname = "EBT_PT_enfusion_material_fixer"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GRP Enfusion Tools"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.ebt_material_fixer_settings
        one_tap = context.scene.ebt_one_tap_settings
        vehicle = context.scene.ebt_vehicle_anim_settings
        vehicle_prep = context.scene.ebt_vehicle_prep_settings
        status_row = layout.row()
        status_row.alert = not EBT_AVAILABLE
        status_row.label(text="EnfusionBlenderTools not enabled" if not EBT_AVAILABLE else "Ready", icon="ERROR" if not EBT_AVAILABLE else "CHECKMARK")

        box = layout.box()
        box.label(text="One‑Tap Enfusion Setup")
        box.prop(one_tap, "asset_type")
        box.prop(one_tap, "scope")
        col = box.column(align=True)
        col.prop(one_tap, "rename_objects")
        col.prop(one_tap, "rename_materials")
        col.prop(one_tap, "rename_images")
        col.prop(one_tap, "unpack_embedded_textures")
        col.prop(one_tap, "overwrite_unpacked_textures")
        col.prop(one_tap, "make_paths_relative")
        col.prop(one_tap, "unify_uvs")
        col.prop(one_tap, "pack_textures")
        col.prop(one_tap, "save_packed_textures")
        col.prop(one_tap, "overwrite_packed")
        col.prop(one_tap, "create_missing_textures")
        col.prop(one_tap, "use_base_alpha_opacity")
        col.prop(one_tap, "enable_preview_shader")
        col.prop(one_tap, "cleanup_orphans")
        col.prop(one_tap, "generate_lods")
        if one_tap.generate_lods:
            col.prop(one_tap, "lod_count")
            col.prop(one_tap, "lod_ratio_step")

        box.operator("ebt.one_tap_enfusion_setup", icon="CHECKMARK")

        layout.separator()
        box = layout.box()
        box.label(text="Material Fixer (Manual)")
        box.prop(settings, "scope")
        box.prop(settings, "shader_mode")
        box.prop(settings, "pack_textures")
        box.prop(settings, "save_packed_textures")
        box.prop(settings, "overwrite_packed")
        box.prop(settings, "create_missing_textures")
        box.prop(settings, "use_base_alpha_opacity")
        box.prop(settings, "enable_preview_shader")

        box.operator("ebt.fix_enfusion_materials", icon="CHECKMARK")

        layout.separator()
        box = layout.box()
        box.label(text="Vehicle Asset Prep")
        box.prop(vehicle_prep, "scope")
        box.prop(vehicle_prep, "apply_scale")
        box.prop(vehicle_prep, "auto_orient_y")
        box.prop(vehicle_prep, "enforce_lod_suffix")
        box.prop(vehicle_prep, "split_glass_materials")
        if vehicle_prep.split_glass_materials:
            box.prop(vehicle_prep, "glass_material_hints")

        box.separator()
        box.prop(vehicle_prep, "create_collision_collider")
        if vehicle_prep.create_collision_collider:
            box.prop(vehicle_prep, "collision_prefix")
            box.prop(vehicle_prep, "collision_usage")
            box.prop(vehicle_prep, "collision_layer_preset")

        box.prop(vehicle_prep, "create_fire_collider")
        if vehicle_prep.create_fire_collider:
            box.prop(vehicle_prep, "fire_prefix")
            box.prop(vehicle_prep, "fire_usage")
            box.prop(vehicle_prep, "fire_layer_preset")
            box.prop(vehicle_prep, "fire_decimate_ratio")

        box.prop(vehicle_prep, "create_occluder")
        if vehicle_prep.create_occluder:
            box.prop(vehicle_prep, "occluder_prefix")
            box.prop(vehicle_prep, "occluder_usage")
            box.prop(vehicle_prep, "occluder_layer_preset")

        box.prop(vehicle_prep, "create_center_of_mass")
        if vehicle_prep.create_center_of_mass:
            box.prop(vehicle_prep, "com_prefix")

        box.prop(vehicle_prep, "create_snap_points_empty")
        if vehicle_prep.create_snap_points_empty:
            box.prop(vehicle_prep, "snap_points_name")

        box.prop(vehicle_prep, "usage_property")
        box.prop(vehicle_prep, "layer_preset_property")

        box.operator("ebt.vehicle_prep", icon="MODIFIER")

        layout.separator()
        box = layout.box()
        box.label(text="Vehicle Animation Setup")
        box.prop(vehicle, "scope")
        box.prop(vehicle, "armature_name")
        box.prop(vehicle, "root_bone_name")
        box.prop(vehicle, "create_actions")
        if vehicle.create_actions:
            box.prop(vehicle, "overwrite_actions")
            box.prop(vehicle, "anim_start_frame")
            box.prop(vehicle, "anim_end_frame")
            box.prop(vehicle, "rotate_axis")
            box.prop(vehicle, "door_open_angle")
            box.prop(vehicle, "hood_open_angle")
            box.prop(vehicle, "trunk_open_angle")
            box.prop(vehicle, "window_axis")
            box.prop(vehicle, "window_slide_distance")

        box.operator("ebt.setup_vehicle_anims", icon="ARMATURE_DATA")


classes = [
    EBTMaterialFixerSettings,
    EBTOneTapSettings,
    EBTVehicleAnimSettings,
    EBTVehiclePrepSettings,
    EBT_OT_fix_enfusion_materials,
    EBT_OT_one_tap_enfusion_setup,
    EBT_OT_setup_vehicle_anims,
    EBT_OT_vehicle_prep,
    EBT_PT_enfusion_material_fixer,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.ebt_material_fixer_settings = PointerProperty(type=EBTMaterialFixerSettings)
    bpy.types.Scene.ebt_one_tap_settings = PointerProperty(type=EBTOneTapSettings)
    bpy.types.Scene.ebt_vehicle_anim_settings = PointerProperty(type=EBTVehicleAnimSettings)
    bpy.types.Scene.ebt_vehicle_prep_settings = PointerProperty(type=EBTVehiclePrepSettings)


def unregister():
    if hasattr(bpy.types.Scene, "ebt_material_fixer_settings"):
        del bpy.types.Scene.ebt_material_fixer_settings
    if hasattr(bpy.types.Scene, "ebt_one_tap_settings"):
        del bpy.types.Scene.ebt_one_tap_settings
    if hasattr(bpy.types.Scene, "ebt_vehicle_anim_settings"):
        del bpy.types.Scene.ebt_vehicle_anim_settings
    if hasattr(bpy.types.Scene, "ebt_vehicle_prep_settings"):
        del bpy.types.Scene.ebt_vehicle_prep_settings

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
