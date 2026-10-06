"""
source_organs.py -- MIRD-style source-organ definitions for the production batch.
=================================================================================
Labels VERIFIED against _mastertable_ref_size.xlsx (arm phantoms). Many organs are
split in the phantom (left/right, wall/contents, cortex/medulla/pelvis); emitting
from only one sub-label would be physically wrong, so each source organ below is
a GROUP of labels that are activated together.

Used by run_pipeline.py:  SOURCE_ORGANS = {name: [labels...]}
"""

SOURCE_ORGANS = {
    # --- primary therapy-relevant organs -----------------------------------
    "liver":            [25],
    "kidneys":          [17, 18, 19, 20, 21, 22],   # cortex+medulla+pelvis, L+R
    "spleen":           [45],
    "lungs":            [26, 27],                    # L + R
    # Label 15 ("Heart") is EMPTY in these phantoms -- this release uses the
    # cardiac substructure model instead: chambers + myocardium.
    "heart":            [16, 87, 88, 89, 90],        # LA, RA, LV, RV, LV myocardium
    "pancreas":         [32],
    "stomach":          [46, 47],                    # wall + contents
    "small_intestine":  [41, 42],                    # wall + contents (major GI source)
    "rectosigmoid":     [37, 38],                    # wall + contents
    "urinary_bladder":  [54, 55],                    # wall + contents
    "thyroid":          [50],
    "brain":            [4],
    "spinal_cord":      [44],

    # --- PSMA / PRRT relevant ----------------------------------------------
    "salivary_glands":  [39, 60, 61],                # parotid+submaxillary+sublingual
    "adrenals":         [2, 3],                      # L + R
    "gall_bladder":     [13, 14],                    # wall + contents
    "prostate":         [36],                        # male phantoms only
    "testes":           [48],                        # male only
    "ovaries":          [31],                        # female only
    "uterus":           [56],                        # female only

    # --- GI tract / other ---------------------------------------------------
    "right_colon":      [7, 8],                      # wall + contents
    "left_colon":       [58, 59],                    # wall + contents
    "esophagus":        [10],
    "thymus":           [49],
    "breast_glandular": [65],                        # female phantoms
    # Dropped as sources: pituitary (~120 voxels), lens, trachea, eye_balls.
    # Too small to hold meaningful activity in any real biodistribution; their
    # S-values are pure self-dose and the cross-organ field is noise. They remain
    # TARGETS in every dose map -- only their use as a SOURCE is dropped.
}

# Organs absent in some phantoms (sex-specific or age-specific) are skipped
# automatically by the pipeline: if a label has zero voxels, that run is skipped.

def flat_labels():
    """All labels used, for a quick sanity check."""
    out = []
    for v in SOURCE_ORGANS.values():
        out.extend(v)
    return sorted(set(out))


if __name__ == "__main__":
    print(f"{len(SOURCE_ORGANS)} source organs, "
          f"{len(flat_labels())} distinct labels")
    for name, labels in SOURCE_ORGANS.items():
        print(f"  {name:20s} {labels}")
