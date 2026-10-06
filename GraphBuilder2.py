"""
GraphBuilder2.py -- adapted from the ORIGINAL GraphBuilder.py, keeping its
class-based design and organ-by-organ processing (memory-safe: only one
organ's voxel coordinates are ever materialized at once, via np.argwhere,
exactly like the original build_supervoxels), but fixing the issues found
during the simulation campaign:

  1. SUPERVOXEL SIZE IN MILLIMETRES, not a fixed voxel window. The phantoms
     have different (anisotropic) spacings, so a fixed voxel window covers
     different physical volumes in different phantoms -- fatal for
     cross-anatomy comparison. `window_size_mm` + `spacing` -> voxel window
     is computed per phantom.

  2. DOSE AGGREGATION SUMS ENERGY-PROPORTIONAL MASS-WEIGHTED DOSE, i.e.
         D_sv = sum(D_i * m_i) / sum(m_i)
     not a plain mean of per-voxel dose. Equivalent for organ-pure blocks of
     equal-density voxels, but robust in general and cheap to do right.

  3. UNCERTAINTY IS CARRIED THROUGH per supervoxel (combined in quadrature,
     mass-weighted), enabling inverse-variance loss weighting.

  4. TARGET is log10(dose) -- the dose range spans ~8 orders of magnitude.

  5. Small nodes (near-zero mass, near-zero dose) are dropped: they add cost
     with no training signal.

Usage:
    from GraphBuilder2 import GraphBuilder2
    gb = GraphBuilder2(window_mm=10.0, ignore_tag=0)
    gb.build_supervoxels(phantom, spacing_xyz_mm)
    feats = gb.build_supervoxel_features(phantom, densities, spacing_xyz_mm)
    target, sigma = gb.build_target(dose_map, unc_map)
    edge_index = gb.build_supervoxel_edges()
"""

import numpy as np

# Diffuse "background" tissues: they fill most of the body, are almost never
# the clinical target of interest, and at fine resolution they dominate the
# graph (adipose+muscle+skin were 74% of all nodes in an earlier build). They
# still matter physically -- gammas traverse them -- so they're kept as
# COARSE-resolution context nodes rather than dropped.
BACKGROUND_LABELS = {1, 43, 57, 62, 63, 64}                # adipose, skin, air,
                                                            # vessels, muscle
BACKGROUND_LABELS |= set(range(128, 135))                  # cartilage / discs
BACKGROUND_LABELS |= set(range(151, 185))                  # bone sites
BACKGROUND_LABELS |= set(range(201, 235))                  # marrow (sp-/mc-)


def tissue_tier(label):
    return "background" if int(label) in BACKGROUND_LABELS else "organ"


class GraphBuilder2:
    def __init__(self, window_mm=10.0, bg_window_mm=None, ignore_tag=0,
                min_volume_cm3=0.3, keep_zero_dose=False, dose_floor=None,
                floor_weight_frac=0.1):
        """
        keep_zero_dose : keep supervoxels that received no dose at all.
        dose_floor     : Gy/decay assigned to those nodes (Option B). Should be
                         the Monte Carlo detection limit for your primary count,
                         i.e. roughly the smallest dose a single scoring event
                         produces. Nodes below it are "not resolvable", which is
                         a real and predictable physical state.
        floor_weight_frac : loss weight given to floored nodes, as a fraction of
                         the median weight of real nodes. They carry information
                         ("negligible dose here") but are censored observations,
                         not measurements, so they should not dominate.
        """
        self.window_mm = window_mm
        # background tissue defaults to 2.5x coarser than organ tissue
        self.bg_window_mm = bg_window_mm if bg_window_mm else window_mm * 2.5
        self.ignore_tag = ignore_tag
        self.min_volume_cm3 = min_volume_cm3
        self.keep_zero_dose = keep_zero_dose
        self.dose_floor = dose_floor
        self.floor_weight_frac = floor_weight_frac
        self.supervoxel_map = None
        self.supervoxel_to_voxels = None   # kept for compatibility / debugging
        self.supervoxel_organ = {}
        self.window_size = None            # last-used (wx, wy, wz) in VOXELS

    # ------------------------------------------------------------------ core
    def _window_voxels(self, spacing_xyz, mm):
        """Voxel window size (x,y,z) so each supervoxel is ~mm per side.
        Clamped to >=1 voxel per axis."""
        return tuple(max(1, int(round(mm / s))) for s in spacing_xyz)

    def build_supervoxels(self, phantom, spacing_xyz):
        """Same organ-by-organ, sliding-window design as the ORIGINAL
        GraphBuilder.build_supervoxels -- this is what keeps memory bounded:
        only one organ's coordinate list is ever in memory, never a
        whole-volume index array. TWO-TIER: organ tissue uses `window_mm`,
        background tissue (fat/muscle/skin/bone/marrow) uses `bg_window_mm`,
        coarser so it doesn't dominate the graph."""
        print("Building supervoxels (mm-based, two-tier window)...")
        wx_o, wy_o, wz_o = self._window_voxels(spacing_xyz, self.window_mm)
        wx_b, wy_b, wz_b = self._window_voxels(spacing_xyz, self.bg_window_mm)
        print(f"  spacing={spacing_xyz} mm | organ window=({wx_o},{wy_o},{wz_o}) "
             f"~{self.window_mm}mm | background window=({wx_b},{wy_b},{wz_b}) "
             f"~{self.bg_window_mm}mm")
        self.window_size = (wx_o, wy_o, wz_o)

        X, Y, Z = phantom.shape  # NOTE: caller must pass phantom as (X,Y,Z) to
                                 # match the ORIGINAL convention (see loader);
                                 # if you have (Z,Y,X) from ITK, transpose first.
        supervoxel_map = -np.ones_like(phantom, dtype=np.int32)
        supervoxel_to_voxels = {}
        sv_id = 0
        unique_tags = np.unique(phantom)
        n_organ_tags = n_bg_tags = 0

        for organ_id in unique_tags:
            organ_id = int(organ_id)
            if organ_id == self.ignore_tag:
                continue
            tier = tissue_tier(organ_id)
            wx, wy, wz = (wx_o, wy_o, wz_o) if tier == "organ" else (wx_b, wy_b, wz_b)
            if tier == "organ":
                n_organ_tags += 1
            else:
                n_bg_tags += 1

            organ_mask = (phantom == organ_id)
            coords = np.argwhere(organ_mask)
            if coords.size == 0:
                continue
            x_min, y_min, z_min = coords.min(axis=0)
            x_max, y_max, z_max = coords.max(axis=0)
            del coords

            for x in range(x_min, x_max + 1, wx):
                for y in range(y_min, y_max + 1, wy):
                    for z in range(z_min, z_max + 1, wz):
                        x_end = min(x + wx, X)
                        y_end = min(y + wy, Y)
                        z_end = min(z + wz, Z)
                        window_mask = organ_mask[x:x_end, y:y_end, z:z_end]
                        local = np.argwhere(window_mask)
                        if local.size == 0:
                            continue
                        gx = local[:, 0] + x
                        gy = local[:, 1] + y
                        gz = local[:, 2] + z
                        supervoxel_map[gx, gy, gz] = sv_id
                        supervoxel_to_voxels[sv_id] = (gx, gy, gz)
                        self.supervoxel_organ[sv_id] = organ_id
                        sv_id += 1

        self.supervoxel_map = supervoxel_map
        self.supervoxel_to_voxels = supervoxel_to_voxels
        print(f"Created {sv_id} supervoxels "
             f"({n_organ_tags} organ tags, {n_bg_tags} background tags).")
        return supervoxel_map, supervoxel_to_voxels

    # -------------------------------------------------------------- features
    def build_supervoxel_features(self, phantom, densities, spacing_xyz,
                                  source_labels=()):
        """densities: {label: g/cm3}. source_labels: labels of the source
        organ (marks is_source). Returns dict sv_id -> feature array."""
        print("Building supervoxel features...")
        vox_vol_cm3 = (spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2]) / 1000.0
        src_set = set(source_labels)

        feats = {}
        for sv_id, (gx, gy, gz) in self.supervoxel_to_voxels.items():
            organ_id = self.supervoxel_organ[sv_id]
            n = len(gx)
            vol_cm3 = n * vox_vol_cm3
            if vol_cm3 < self.min_volume_cm3:
                continue
            rho = densities.get(organ_id, 1.0)
            mass_g = vol_cm3 * rho
            cx = (gx.mean() * spacing_xyz[0]) / 10.0   # cm
            cy = (gy.mean() * spacing_xyz[1]) / 10.0
            cz = (gz.mean() * spacing_xyz[2]) / 10.0
            feats[sv_id] = {
                "organ": organ_id, "nvox": n, "volume_cm3": vol_cm3,
                "mass_g": mass_g, "density": rho,
                "cx": cx, "cy": cy, "cz": cz,
                "is_source": organ_id in src_set,
                "voxels": (gx, gy, gz),
            }
        # Free the full coordinate table now: everything downstream (build_target,
        # edges) uses either `feats[...]["voxels"]` or `self.supervoxel_map`, not
        # this dict. On real-size phantoms it holds index arrays for 100k+
        # supervoxels (most filtered out) and was still alive during edge
        # building, which contributed to a segfault at full scale.
        self.supervoxel_to_voxels = None
        print(f"  {len(feats)} supervoxels pass the volume filter "
             f"(of {len(self.supervoxel_organ)})")
        return feats

    # ---------------------------------------------------------------- target
    def build_target(self, feats, dose_map, unc_map, spacing_xyz):
        """dose_map/unc_map: numpy arrays SAME (X,Y,Z) orientation as phantom.
        Returns (feats with dose/sigma added, filtered to nonzero-mass/dose)."""
        vox_vol_cm3 = (spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2]) / 1000.0
        out = {}
        n_bad = 0
        for sv_id, f in feats.items():
            try:
                gx, gy, gz = f["voxels"]
                # ravel() guarantees 1-D: for a supervoxel holding a single
                # voxel, fancy indexing can yield a 0-d scalar, which then
                # breaks np.where's axis remapping.
                gx = np.asarray(gx).ravel()
                gy = np.asarray(gy).ravel()
                gz = np.asarray(gz).ravel()
                if gx.size == 0:
                    continue
                dd = np.asarray(dose_map[gx, gy, gz], dtype=np.float64).ravel()
                uu = np.asarray(unc_map[gx, gy, gz], dtype=np.float64).ravel()
                dd = np.where(np.isfinite(dd), dd, 0.0)
                uu = np.where(np.isfinite(uu), uu, 0.0)
                sigma_vox = uu * dd
                m = f["mass_g"] / f["nvox"]     # per-voxel mass (organ-pure block)
                total_m = m * f["nvox"]
                dose_sv = float((dd * m).sum() / total_m) if total_m > 0 else 0.0
                sig_sv = float(np.sqrt(((sigma_vox * m) ** 2).sum()) / total_m) \
                    if total_m > 0 else 0.0
            except Exception:
                n_bad += 1
                continue
            if dose_sv <= 0:
                if not self.keep_zero_dose:
                    continue
                # OPTION B: this supervoxel received no scoring event at all.
                # That is a real, predictable physical state ("below the MC
                # detection limit"), so keep it with the floor as its target and
                # flag it as censored rather than measured.
                dose_sv = float(self.dose_floor) if self.dose_floor else 0.0
                sig_sv = dose_sv          # 100% relative: it's an upper limit
                censored = True
            else:
                censored = False
            f = dict(f)
            f.pop("voxels")
            f["dose"] = dose_sv
            f["sigma"] = sig_sv
            f["censored"] = censored
            out[sv_id] = f
        if n_bad:
            print(f"  [warn] {n_bad} supervoxel(s) skipped on target error")
        print(f"  {len(out)} supervoxels have target dose "
             f"(of {len(feats)} volume-filtered)")
        return out

    # ----------------------------------------------------------------- edges
    def build_supervoxel_edges(self):
        """6-connected face adjacency. Vectorized end-to-end -- pairs are
        encoded as single int64 ids and deduplicated with np.unique, never
        materialized as a Python set of a million tuples (which is what
        segfaulted on real-size phantoms)."""
        print("Building supervoxel edges...")
        X, Y, Z = self.supervoxel_map.shape
        neighbors6 = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0),
                     (0, 0, -1), (0, 0, 1)]
        m = self.supervoxel_map
        # id encoding needs a value bigger than any supervoxel id
        max_id = int(m.max()) + 2

        all_pairs = []
        for dx, dy, dz in neighbors6:
            a = m[max(0, -dx):X - max(0, dx),
                 max(0, -dy):Y - max(0, dy),
                 max(0, -dz):Z - max(0, dz)]
            b = m[max(0, dx):X - max(0, -dx),
                 max(0, dy):Y - max(0, -dy),
                 max(0, dz):Z - max(0, -dz)]
            valid = (a != -1) & (b != -1) & (a != b)
            pa = a[valid].astype(np.int64)
            pb = b[valid].astype(np.int64)
            lo = np.minimum(pa, pb)
            hi = np.maximum(pa, pb)
            del a, b, valid, pa, pb
            code = lo * max_id + hi          # unique per (lo,hi) pair
            all_pairs.append(np.unique(code))
            del lo, hi, code

        codes = np.unique(np.concatenate(all_pairs)) if all_pairs else np.array([])
        del all_pairs
        lo = codes // max_id
        hi = codes % max_id
        print(f"Created {len(codes)} unique edges.")

        if len(codes) == 0:
            return np.zeros((2, 0), dtype=np.int64)
        # undirected: both directions
        edges = np.concatenate([
            np.stack([lo, hi]), np.stack([hi, lo])
        ], axis=1).astype(np.int64)
        return edges
