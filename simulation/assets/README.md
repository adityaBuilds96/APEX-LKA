# APEX-LKA Digital Twin Asset Creation & Photogrammetry Guide
## High-Fidelity 3D Reconstruction of Real-World Testing Environments for CARLA 0.9.15+

This guide details the complete industrial pipeline for capturing real-world environments (such as your university campus, proving ground, or BAJA SAE track) and converting them into high-fidelity Unreal Engine 4 / CARLA Digital Twins.

---

### Phase 1: Aerial Photogrammetry Acquisition

1. **Flight Hardware & Sensor Rig:**
   - Drone: DJI Matrice 300 RTK / Phantom 4 RTK (or equivalent with RTK GNSS).
   - Camera: Minimum 20 MP mechanical shutter (e.g. Zenmuse P1 / 1-inch CMOS).
   - Ground Sample Distance (GSD): $\le 1.5\text{ cm/pixel}$ target resolution.
2. **Flight Pattern:**
   - **Double-Grid Nadir + Oblique Flight:**
     - Altitude: 45–60 meters above ground level (AGL).
     - Front overlap: 80%.
     - Side overlap: 75%.
     - Camera Pitch: 60° (oblique) to capture building facades, curbs, and lane markings, combined with 90° (nadir) top-down passes.
3. **Ground Control Points (GCPs):**
   - Place 8–12 surveyed high-contrast GCP targets across the campus.
   - Record millimeter-accurate RTK coordinates (WGS84 / UTM).

---

### Phase 2: 3D Point Cloud & Mesh Reconstruction

1. **Photogrammetry Processing (RealityCapture / Meshroom / Agisoft Metashape):**
   - Import all RAW/JPEG aerial frames + RTK EXIF coordinates.
   - Align photos using High Quality / SIFT tie-point extraction.
   - Optimize camera bundle adjustment using surveyed GCPs.
   - Generate High-Density 3D Mesh ($\ge 10\text{ million}$ triangles for 500m area).
   - Texture baking: Generate $8192 \times 8192$ (8K) PBR diffuse, normal, and roughness maps.
2. **Decimation & LOD Pipeline:**
   - Export decimated high-detail OBJ/FBX mesh ($\approx 250,000$ triangles per 100m grid tile).
   - Generate collision meshes (`UCX_` prefix for Unreal Engine convex collision).

---

### Phase 3: OpenDRIVE (.xodr) Road Network Generation

1. **RoadRunner (MathWorks) / VectorZero:**
   - Import georeferenced orthophoto and digital elevation model (DEM).
   - Trace road centerlines, lane boundaries, curbs, crosswalks, and speed bumps.
   - Assign OpenDRIVE lane types (`driving`, `sidewalk`, `shoulder`, `parking`).
   - Export ASAM OpenDRIVE v1.6 XML:
     ```bash
     # Output to simulation maps directory:
     simulation/maps/college_campus.xodr
     ```
2. **Validation:**
   - Validate OpenDRIVE file using `odrViewer` or CARLA `PythonAPI/util/opendrive_features.py`.

---

### Phase 4: Unreal Engine 4 (UE4) / CARLA Map Cooking

1. **Importing into CARLA UE4 Environment:**
   ```bash
   # In CARLA UE4 source tree (UnrealEngine 4.26):
   cp simulation/maps/college_campus.xodr CarlaUE4/Content/Carla/Maps/OpenDrive/
   make import-map
   ```
2. **Road & Vegetation Dressing:**
   - Apply CARLA PBR Asphalt and Wetness materials (`M_Road_Asphalt`).
   - Paint procedural grass, trees, and street furniture (signs, lights, barrels).
   - Ensure `SkyAtmosphere`, `DirectionalLight` (Sun), and `ExponentialHeightFog` are dynamic to support all 8 weather presets.
3. **Baking & Cooking:**
   - Bake static lighting or enable dynamic lighting (Ray Tracing / Cascade Shadow Maps).
   - Package/Cook Level:
     ```bash
     make package ARGS="--packages=CollegeCampus"
     ```
   - Export level file to `simulation/maps/college_campus.umap`.

---

### Phase 5: Loading into APEX-LKA

Run live simulation with your newly cooked digital twin:
```bash
# Run single scenario on digital twin:
python deploy/sim_run.py --map college_campus --weather dusk_golden

# Run regression validation:
python run.py sim --map college_campus --regression
```
