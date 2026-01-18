# GRP Enfusion Tools

Production-ready Blender add-on for preparing Enfusion (Arma Reforger) assets. It streamlines materials, texture packing, naming, LODs, vehicle colliders, and basic vehicle animations from a single UI panel.

> Panel location: **View3D → Sidebar → GRP Enfusion Tools**

---

## Features

### One‑Tap Enfusion Setup
- Sanitizes names (no periods, no invalid characters)
- Unifies UV sets (`UVMap0`, `UVMap1`…)
- Packs **BCR/NMO** textures (with safe repacking if existing packs are invalid)
- Optional preview shader (prevents black materials in Blender)
- Optional LOD generation
- Optional unpacking of embedded textures
- Optional relative paths and orphan cleanup

### Material Fixer (Manual)
- Converts materials to Enfusion shaders
- Builds and assigns **BCR** and **NMO** packs
- Resolves original textures from disk when re-running
- Optional opacity map from base color alpha
- Optional preview shader for Blender viewport

### Vehicle Asset Prep
Automates common vehicle asset prep steps:
- Orientation check with optional auto‑rotate to **Y+** (heuristic)
- Enforces `_LOD0` suffix for visual meshes
- Splits glass/window materials into separate objects
- Creates colliders with correct prefixes and custom properties:
  - **Collision** (`UBX_`, `UCX_`, `USP_`, `UCS_`, `UCL_`, `UTM_`)
  - **Fire geometry** (trimesh)
  - **Occluders** (`OCC_`)
  - **Center of Mass** (`COM_`)
- Adds **SnapPoints** empty
- Writes collider custom properties like `usage` and `LayerPreset`

### Vehicle Animation Setup
- Creates a vehicle armature and bones for named parts
- Parents doors/windows/hood/trunk to bones
- Generates open/close actions per part
- Configurable axis, angles, frame ranges, and window slide distance

---

## Requirements

- **Blender 4.2+**
- **Enfusion Blender Tools** add‑on enabled (for shader accessors and material setup)

---

## Installation

1. Place `GRPTools.py` into your Blender add‑ons folder.
2. In Blender: **Edit → Preferences → Add-ons → Install** (or enable from file list).
3. Enable **GRP Enfusion Tools**.
4. Ensure **Enfusion Blender Tools** is also enabled.

---

## Quick Start

1. Open the **GRP Enfusion Tools** panel in the 3D View sidebar.
2. Run **One‑Tap Enfusion Setup** for general assets.
3. Use **Vehicle Asset Prep** for vehicles (colliders, COM, snap points, glass split).
4. Use **Vehicle Animation Setup** to generate door/window/hood/trunk actions.

---

## Vehicle Prep Guidance (from Bohemia docs)

Key rules implemented by the tool:
- **Model forward = Y+** (Blender / 3ds Max)
- **Use LOD suffixes**: `_LODx` (up to 8 LODs)
- **Collider prefixes**:
  - `UBX_` box, `UCX_` convex, `USP_` sphere, `UCS_` capsule, `UCL_` cylinder, `UTM_` trimesh
- **Occluder prefix**: `OCC_`
- **Center of Mass prefix**: `COM_`
- **Colliders must have a `usage` property**
- **LayerPreset** should be defined on colliders

> The tool exposes **usage** and **LayerPreset** property names and values as configurable options in the UI.

---

## Naming Conventions

The tool enforces safe naming to avoid engine/import issues:
- No periods (`.`) in names
- Invalid characters replaced with `_`
- LOD suffix enforced where applicable

---

## Vehicle Animation Naming

Vehicle part detection uses object name hints:
- `door`
- `window` / `glass`
- `hood` / `bonnet`
- `trunk` / `boot` / `tailgate` / `hatch`

Rename your meshes accordingly for automatic rigging and action generation.

---

## Export Notes (FBX)

When exporting animated vehicles:
- Include **Empty**, **Armature**, and **Mesh** object types
- **Custom Properties** must be exported (for LayerPreset and usage)
- Disable **Leaf Bones**

These recommendations align with the official Enfusion guidance.

---

## Troubleshooting

**Material turns black after One‑Tap**
- Ensure original source textures are still on disk (same folder as BCR/NMO)
- Re‑run One‑Tap with **Enable Preview Shader** turned on

**Colliders not recognized**
- Confirm correct prefix (`UBX_`, `UCX_`, etc.)
- Ensure `usage` property is present and correct

**Vehicle animations missing**
- Make sure parts are named with the expected hints (door/window/hood/trunk)
- Ensure the armature is present and the part objects are parented to bones

---

## License

See `LICENSE`.

---

## Support

Discord: https://discord.gg/DkVheGS2na

---

## Credits

Built for GRP workflows using Blender + Enfusion Blender Tools.
References: Bohemia Interactive Community documentation on vehicle asset preparation and FBX import.
