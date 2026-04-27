"""
envelope.py — Compute maximum buildable GFA per parcel.

Enforces three constraints in order:
  1. FAR limit (lot_area × FAR), or lot-occupancy × stories if no explicit FAR
  2. Zoning height limit
  3. Height Act of 1910 limit (function of street ROW width and use type)

Returns the binding (minimum) GFA, plus metadata on which constraint binds.
"""

import numpy as np
import pandas as pd


# ── Height Act computation ────────────────────────────────────────────────────

def height_act_limit(street_width_ft: float, use_type: str, address: str = "") -> float:
    """Return Height Act allowable height in feet.

    Commercial: ROW width + 20 ft, max 130 ft.
    Residential: ROW width − 10 ft, max 90 ft.
    Pennsylvania Ave NW: 160 ft exception applies for certain frontages.

    Args:
        street_width_ft: Total cross-section width of the fronting street (ft).
                         If NaN, returns the statutory maximum for the use type.
        use_type:        "commercial" or "residential"
        address:         Parcel address string (for Pennsylvania Ave exception check).
    """
    penn_ave = address and "PENNSYLVANIA AVE" in str(address).upper()

    if pd.isna(street_width_ft) or street_width_ft <= 0:
        # No street width data — fall back to statutory maximums
        if penn_ave:
            return 160.0
        return 130.0 if use_type == "commercial" else 90.0

    if use_type == "commercial":
        limit = min(street_width_ft + 20, 130.0)
    else:
        limit = min(max(street_width_ft - 10, 20.0), 90.0)

    if penn_ave:
        limit = max(limit, 160.0)

    return limit


# ── Use-type classification ───────────────────────────────────────────────────

# DC ITSPE use codes: first digit classifies broad use.
# 0xx = miscellaneous/special; 1xx = residential; 2xx = apartments;
# 3xx = commercial office; 4xx = retail; 5xx = industrial; 6xx = institutional;
# 7xx = utility; 8xx = vacant/other
_COMMERCIAL_USE_PREFIXES = ("3", "4", "5", "7")
_RESIDENTIAL_USE_PREFIXES = ("1", "2")

# Zones that use the commercial Height Act formula (street_width + 20 ft)
# regardless of USECODE. Most 0xx parcels in these zones are mixed-use or
# industrial-converted and would be built as commercial-class structures.
_COMMERCIAL_ZONE_PREFIXES = ("MU", "RA", "CG", "C-", "CR", "D-", "PDR")


def parcel_use_type(usecode: str, zone: str = "") -> str:
    """Classify parcel as "residential" or "commercial" for Height Act purposes.

    Zone takes precedence: MU, RA, CG, C-*, CR, and D- zones are commercial.
    Falls back to USECODE prefix classification when zone is ambiguous.
    """
    zone_str = str(zone or "").strip()
    if any(zone_str.startswith(p) for p in _COMMERCIAL_ZONE_PREFIXES):
        return "commercial"
    if not usecode or pd.isna(usecode):
        return "residential"
    prefix = str(usecode).strip()[:1]
    return "commercial" if prefix in _COMMERCIAL_USE_PREFIXES else "residential"


# ── Construction type ─────────────────────────────────────────────────────────

def construction_type(stories: float) -> str:
    if pd.isna(stories) or stories <= 0:
        return "type_v"
    if stories <= 4:
        return "type_v"
    if stories <= 8:
        return "type_i_mid"
    return "type_i"


# ── Setback-adjusted footprint ────────────────────────────────────────────────

def buildable_footprint(lot_area: float, lot_width_ft: float,
                        front_setback: float, rear_setback: float,
                        side_setback: float, lot_occupancy: float | None) -> float:
    """Estimate the buildable footprint after setbacks and lot occupancy.

    Uses a simplified rectangular lot model. For non-zero setbacks, subtracts
    setback rectangles from the lot area assuming lot_width_ft is known.

    Args:
        lot_area:      Total lot area (SF).
        lot_width_ft:  Minimum lot width from ZDS (used to estimate lot depth).
        front_setback, rear_setback, side_setback: Required setbacks in feet.
        lot_occupancy: Maximum fraction of lot that can be covered (0–1); None = 1.0.
    """
    if lot_area <= 0:
        return 0.0

    if lot_width_ft and lot_width_ft > 0:
        lot_depth = lot_area / lot_width_ft
        usable_width = max(lot_width_ft - 2 * side_setback, 0)
        usable_depth = max(lot_depth - front_setback - rear_setback, 0)
        footprint_setback = usable_width * usable_depth
    else:
        # No lot width data — apply setback as a rough fractional reduction
        setback_reduction = min((front_setback + rear_setback + 2 * side_setback) / 100, 0.4)
        footprint_setback = lot_area * (1 - setback_reduction)

    if lot_occupancy and 0 < lot_occupancy <= 1:
        footprint_occ = lot_area * lot_occupancy
        return min(footprint_setback, footprint_occ)

    return footprint_setback


# ── Main envelope function ────────────────────────────────────────────────────

def compute_envelope(row: pd.Series, cfg: dict) -> dict:
    """Compute maximum buildable GFA and related envelope metrics for one parcel.

    Returns a dict with keys:
      max_buildable_gfa, max_buildable_stories, construction_type,
      height_act_ft, effective_height_ft, binding_constraint,
      net_rentable_sf, max_units
    """
    lot_area = float(row.get("lot_area_sf") or 0)
    if lot_area <= 0:
        return _zero_envelope()

    # Zoning parameters
    zoning_height = row.get("zoning_height_ft")
    zoning_far    = row.get("zoning_far")
    zoning_stories = row.get("zoning_stories")
    lot_occupancy = row.get("lot_occupancy")
    front_setback  = float(row.get("front_setback") or 0)
    rear_setback   = float(row.get("rear_setback") or 0)
    side_setback   = float(row.get("side_setback") or 0)
    min_lot_width  = row.get("min_lot_width")

    floor_to_floor = cfg.get("floor_to_floor_ft", 12)

    # Determine use type for Height Act
    use_type = parcel_use_type(row.get("USECODE", ""), row.get("zone_normalized", ""))

    # Height Act limit
    ha_ft = height_act_limit(
        row.get("street_width_ft", np.nan),
        use_type,
        str(row.get("PREMISEADD", "")),
    )

    # Policy overrides
    if cfg.get("height_act_amendment") == "affordable_only":
        # Lewis George: raise cap for affordable-only projects by 20%
        ha_ft = ha_ft * 1.20
    elif cfg.get("height_act_amendment") == "metro_half_mile":
        # McDuffie: near-Metro parcels can use commercial cap regardless of street
        # (metro_adjacent flag must be pre-computed on the row)
        if row.get("metro_adjacent", False):
            ha_ft = cfg.get("height_act_commercial_max_ft", 130)

    # Effective height: minimum of zoning height and Height Act
    if zoning_height and not pd.isna(zoning_height):
        effective_height = min(float(zoning_height), ha_ft)
        binding = "zoning" if float(zoning_height) <= ha_ft else "height_act"
    else:
        effective_height = ha_ft
        binding = "height_act"

    max_stories_from_height = effective_height / floor_to_floor

    # Buildable footprint
    footprint = buildable_footprint(
        lot_area,
        float(min_lot_width) if min_lot_width and not pd.isna(min_lot_width) else 0,
        front_setback, rear_setback, side_setback,
        float(lot_occupancy) if lot_occupancy and not pd.isna(lot_occupancy) else None,
    )

    # GFA: compare FAR-constrained vs. height-constrained
    if zoning_far and not pd.isna(zoning_far) and float(zoning_far) > 0:
        gfa_by_far = lot_area * float(zoning_far)
        binding_far = True
    else:
        # Residential fallback: lot_occupancy × stories
        if zoning_stories and not pd.isna(zoning_stories) and float(zoning_stories) > 0:
            effective_stories = min(float(zoning_stories), max_stories_from_height)
        else:
            effective_stories = max_stories_from_height
        gfa_by_far = footprint * effective_stories
        binding_far = False

    gfa_by_height = footprint * max_stories_from_height

    if binding_far:
        max_gfa = min(gfa_by_far, gfa_by_height)
        if gfa_by_far <= gfa_by_height:
            binding = "far"
    else:
        max_gfa = gfa_by_height

    if max_gfa <= 0:
        return _zero_envelope()

    actual_stories = max_gfa / footprint if footprint > 0 else max_stories_from_height
    c_type = construction_type(actual_stories)

    net_rentable = max_gfa * cfg.get("efficiency_ratio", 0.82)
    max_units = net_rentable / cfg.get("avg_unit_sf", 1000)

    return {
        "max_buildable_gfa": max_gfa,
        "max_buildable_stories": actual_stories,
        "construction_type": c_type,
        "height_act_ft": ha_ft,
        "effective_height_ft": effective_height,
        "envelope_binding": binding,
        "buildable_footprint_sf": footprint,
        "net_rentable_sf": net_rentable,
        "max_units": max_units,
    }


def _zero_envelope() -> dict:
    return {
        "max_buildable_gfa": 0.0,
        "max_buildable_stories": 0.0,
        "construction_type": "type_v",
        "height_act_ft": 0.0,
        "effective_height_ft": 0.0,
        "envelope_binding": "zero_lot",
        "buildable_footprint_sf": 0.0,
        "net_rentable_sf": 0.0,
        "max_units": 0.0,
    }


def apply_envelopes(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Vectorized wrapper: apply compute_envelope to every row of df."""
    results = df.apply(compute_envelope, axis=1, cfg=cfg, result_type="expand")
    return pd.concat([df, results], axis=1)
