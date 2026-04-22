"""
proforma.py — Pro-forma analysis for one parcel.

Computes revenue, costs, and three feasibility thresholds:
  1. Yield-on-cost ≥ exit cap rate + spread (default 125 bps)
  2. Untrended return on cost ≥ min_roc (default 6.5%)
  3. Residual land value ≥ assessed land value

Also runs a social housing variant (parallel track, Lewis George scenario).
"""

import math
import numpy as np
import pandas as pd


# ── IZ rent calculation ───────────────────────────────────────────────────────

def iz_rent_per_sf(ami_pct: float, ami_4person: float,
                   income_share: float = 0.30, unit_sf: float = 1000) -> float:
    """Monthly rent per SF for an IZ unit at given AMI percentage.

    Assumes household size of 2.5 (HUD adjustment ≈ 0.875 × 4-person AMI).
    Renters pay income_share of income toward rent.
    """
    ami_household = ami_4person * 0.875   # approximate 2-person AMI
    monthly_income = ami_household * ami_pct / 12
    monthly_rent = monthly_income * income_share
    return monthly_rent / unit_sf


# ── Construction timeline ─────────────────────────────────────────────────────

def construction_months(units: float, stories: float,
                         entitlement_months: float = 3,
                         is_historic: bool = False,
                         hprb_months: float = 2) -> float:
    """Estimate total development timeline in months (entitlement + construction).

    Construction duration scales with units, loosely following Terner assumptions.
    """
    if units <= 0:
        return 0
    # Construction phase
    if stories <= 2:
        build_months = 9
    elif stories <= 6:
        build_months = 18
    elif stories <= 12:
        build_months = 24
    else:
        base = 24
        extra = math.ceil((stories - 12) / 2)
        build_months = base + extra

    hprb_add = hprb_months if is_historic else 0
    return entitlement_months + hprb_add + build_months


# ── Core pro-forma ────────────────────────────────────────────────────────────

def run_proforma(row: pd.Series, cfg: dict) -> dict:
    """Run a market-rate multifamily rental pro-forma for one parcel.

    Args:
        row: A row from the parcel master with envelope columns already computed.
        cfg: Merged assumptions dict (assumptions.yaml + scenario overrides).

    Returns:
        Dict with feasibility flag, binding constraint, and key financial metrics.
    """
    max_gfa = float(row.get("max_buildable_gfa") or 0)
    max_units = float(row.get("max_units") or 0)
    c_type = row.get("construction_type", "type_v")
    stories = float(row.get("max_buildable_stories") or 0)
    is_historic = bool(row.get("is_historic", False))
    lot_area = float(row.get("lot_area_sf") or 0)

    if max_gfa <= 0 or max_units < 1:
        return _infeasible("no_envelope")

    # ── Revenue ───────────────────────────────────────────────────────────────
    # Use parcel-level submarket rent if available; fall back to cfg citywide default.
    _sp = row.get("submarket_rent_psf")
    rent_psf = float(_sp) if (_sp is not None and _sp == _sp and float(_sp) > 0) else cfg["rent_per_sf_per_month"]
    unit_sf  = cfg.get("avg_unit_sf", 1000)
    vacancy  = cfg.get("vacancy_rate", 0.06)
    opex_pu  = cfg.get("opex_per_unit_annual", 8500)

    # IZ set-aside
    iz_threshold = cfg.get("iz_threshold_units", 10)
    if max_units >= iz_threshold:
        iz_pct = (cfg.get("iz_setaside_wood_pct", 0.11)
                  if c_type == "type_v"
                  else cfg.get("iz_setaside_concrete_pct", 0.09))
        # Scenario override
        if "iz_setaside_pct" in cfg:
            iz_pct = cfg["iz_setaside_pct"]
    else:
        iz_pct = 0.0

    market_units = max_units * (1 - iz_pct)
    affordable_units = max_units * iz_pct

    iz_rent = iz_rent_per_sf(
        cfg.get("iz_ami_tier", 0.60),
        cfg.get("ami_4person", 142300),
        unit_sf=unit_sf,
    )

    gross_market = market_units * unit_sf * rent_psf * 12
    gross_iz     = affordable_units * unit_sf * iz_rent * 12
    egi = (gross_market + gross_iz) * (1 - vacancy)
    noi = egi - max_units * opex_pu

    # ── Stabilized value ──────────────────────────────────────────────────────
    going_in_cap = cfg["going_in_cap_rate"]
    if noi <= 0:
        return _infeasible("negative_noi")
    stabilized_value = noi / going_in_cap

    # ── Costs ─────────────────────────────────────────────────────────────────
    cost_key = {
        "type_v":     "hard_cost_type_v_per_sf",
        "type_i_mid": "hard_cost_type_i_mid_per_sf",
        "type_i":     "hard_cost_type_i_per_sf",
    }.get(c_type, "hard_cost_type_v_per_sf")
    hard_cost_psf = cfg[cost_key]
    hard_cost = max_gfa * hard_cost_psf * (1 + cfg.get("contingency_pct", 0.10))
    soft_cost = hard_cost * cfg.get("soft_cost_pct", 0.25)

    # DC Water SAF
    water_saf = (
        max_units * cfg.get("dc_water_saf_per_unit_market", 22580)
        - affordable_units * cfg.get("dc_water_saf_credit_affordable", 3944)
    )
    # DOB permit fees
    dob_fees = hard_cost * cfg.get("dob_permit_pct_of_construction", 0.02) * (
        1 + cfg.get("dob_permit_surcharge", 0.10)
    )
    fees = water_saf + dob_fees

    # Land cost
    land_cost = float(row.get("land_cost") or 0)

    # Entitlement timeline
    entitlement_mo = cfg.get("entitlement_months_byright", 3)
    if row.get("has_pud", False):
        entitlement_mo = cfg.get("entitlement_months_zc", 15)
    # Scenario reduction (McDuffie 50% cut)
    reduction_factor = cfg.get("entitlement_months_reduction", 0)
    entitlement_mo = entitlement_mo * (1 - reduction_factor)
    # TOPA friction add (McDuffie scenario)
    entitlement_mo += cfg.get("topa_friction_months", 0)

    total_months = construction_months(
        max_units, stories, entitlement_mo, is_historic,
        cfg.get("hprb_review_months", 2),
    )

    # Financing carry (interest on drawn debt during construction)
    debt = (hard_cost + soft_cost) * cfg.get("loan_to_cost", 0.65)
    carry = debt * cfg.get("construction_loan_rate", 0.075) * (total_months / 12) * 0.60
    # 0.60 = average draw factor (debt drawn progressively, not all day-1)

    dev_fee = (hard_cost + soft_cost + fees) * cfg.get("developer_fee_pct", 0.035)

    tdc = land_cost + hard_cost + soft_cost + fees + carry + dev_fee

    # ── Return tests ──────────────────────────────────────────────────────────
    exit_cap = cfg["exit_cap_rate"]
    # TOPA exit cap premium (McDuffie scenario)
    exit_cap += cfg.get("topa_exit_cap_premium", 0.0)

    yoc_spread   = cfg.get("yoc_spread_over_exit_cap", 0.0125)
    min_roc      = cfg.get("min_return_on_cost", 0.065)
    assessed_lv  = float(row.get("assessed_land_value") or 0)

    yoc           = noi / tdc
    return_on_cost = stabilized_value / tdc - 1
    rlv           = stabilized_value - (tdc - land_cost)

    yoc_ok  = yoc >= exit_cap + yoc_spread
    roc_ok  = return_on_cost >= min_roc
    rlv_ok  = rlv >= assessed_lv

    feasible = yoc_ok and roc_ok and rlv_ok

    if not yoc_ok:
        binding_constraint = "yoc"
    elif not roc_ok:
        binding_constraint = "roc"
    elif not rlv_ok:
        binding_constraint = "rlv"
    else:
        binding_constraint = None

    # ── Rent-to-pencil (reservation price) ────────────────────────────────────
    # Market rent $/SF/mo at which YoC threshold is just cleared, holding TDC fixed.
    # Solve: NOI(rent*) / TDC = exit_cap + spread
    # NOI = [(market_units × unit_sf × rent* + affordable_units × unit_sf × iz_rent)
    #         × 12 × (1-vacancy)] - opex_total
    target_noi = tdc * (exit_cap + yoc_spread)
    opex_total = max_units * opex_pu
    iz_gross = affordable_units * unit_sf * iz_rent * 12 * (1 - vacancy)
    if market_units * unit_sf > 0:
        rent_to_pencil = (
            (target_noi + opex_total) / (1 - vacancy)
            - iz_gross / (1 - vacancy) * (1 - vacancy)
        ) / (market_units * unit_sf * 12)
        # Simpler: directly from NOI equation
        rent_to_pencil = (
            (target_noi + opex_total) / ((1 - vacancy) * 12)
            - affordable_units * unit_sf * iz_rent
        ) / (market_units * unit_sf)
    else:
        rent_to_pencil = np.nan

    return {
        "feasible": feasible,
        "binding_constraint": binding_constraint,
        "noi": noi,
        "stabilized_value": stabilized_value,
        "tdc": tdc,
        "hard_cost": hard_cost,
        "soft_cost": soft_cost,
        "carry_cost": carry,
        "yoc": yoc,
        "return_on_cost": return_on_cost,
        "rlv": rlv,
        "rent_to_pencil_psf": rent_to_pencil,
        "iz_pct_applied": iz_pct,
        "market_units": market_units,
        "affordable_units": affordable_units,
        "construction_months_total": total_months,
    }


# ── Social housing pro-forma ──────────────────────────────────────────────────

def run_social_housing_proforma(row: pd.Series, cfg: dict) -> dict:
    """Pro-forma for a social housing project on a public-land parcel.

    Differences from market-rate:
    - Land cost = $0 (public land) or assessed value (acquired land)
    - Required yield = cost of tax-exempt capital (~4–5%)
    - Blended AMI revenue from cfg['social_housing_ami_weights']
    - No IZ set-aside (all units are affordable by definition)
    - Developer fee = social_housing_developer_fee_pct
    """
    max_gfa = float(row.get("max_buildable_gfa") or 0)
    max_units = float(row.get("max_units") or 0)
    c_type = row.get("construction_type", "type_v")
    stories = float(row.get("max_buildable_stories") or 0)
    is_historic = bool(row.get("is_historic", False))

    if max_gfa <= 0 or max_units < 1:
        return _infeasible_social("no_envelope")

    owner_type = row.get("owner_type", "private")
    land_cost_override = cfg.get("social_housing_land_cost_override", 0)
    if owner_type == "public_or_nonprofit":
        land_cost = land_cost_override
    else:
        land_cost = float(row.get("land_cost") or 0)

    unit_sf   = cfg.get("avg_unit_sf", 1000)
    vacancy   = cfg.get("vacancy_rate", 0.06)
    opex_pu   = cfg.get("opex_per_unit_annual", 8500)
    ami_4p    = cfg.get("ami_4person", 142300)
    ami_weights = cfg.get("social_housing_ami_weights", [
        {"ami_pct": 0.60, "share": 1.0}
    ])

    blended_rent_psf = sum(
        iz_rent_per_sf(w["ami_pct"], ami_4p, unit_sf=unit_sf) * w["share"]
        for w in ami_weights
    )

    gross_revenue = max_units * unit_sf * blended_rent_psf * 12
    egi = gross_revenue * (1 - vacancy)
    noi = egi - max_units * opex_pu

    if noi <= 0:
        return _infeasible_social("negative_noi")

    cost_key = {
        "type_v":     "hard_cost_type_v_per_sf",
        "type_i_mid": "hard_cost_type_i_mid_per_sf",
        "type_i":     "hard_cost_type_i_per_sf",
    }.get(c_type, "hard_cost_type_v_per_sf")
    hard_cost = max_gfa * cfg[cost_key] * (1 + cfg.get("contingency_pct", 0.10))
    soft_cost = hard_cost * cfg.get("soft_cost_pct", 0.25)

    water_saf = max_units * cfg.get("dc_water_saf_credit_affordable", 3944)  # all affordable
    fees = water_saf
    dev_fee = (hard_cost + soft_cost) * cfg.get("social_housing_developer_fee_pct", 0.025)

    entitlement_mo = cfg.get("entitlement_months_byright", 3)
    total_months = construction_months(max_units, stories, entitlement_mo, is_historic, 2)
    debt = (hard_cost + soft_cost) * cfg.get("loan_to_cost", 0.65)
    carry = debt * cfg.get("construction_loan_rate", 0.075) * (total_months / 12) * 0.60

    tdc = land_cost + hard_cost + soft_cost + fees + carry + dev_fee

    required_yield = cfg.get("social_housing_required_yield", 0.045)
    yoc = noi / tdc
    feasible_social = yoc >= required_yield

    return {
        "social_feasible": feasible_social,
        "social_yoc": yoc,
        "social_required_yield": required_yield,
        "social_noi": noi,
        "social_tdc": tdc,
        "social_land_cost": land_cost,
        "social_blended_rent_psf": blended_rent_psf,
        "social_units": max_units,
    }


def _infeasible(reason: str) -> dict:
    return {
        "feasible": False,
        "binding_constraint": reason,
        "noi": np.nan, "stabilized_value": np.nan, "tdc": np.nan,
        "hard_cost": np.nan, "soft_cost": np.nan,
        "carry_cost": np.nan, "yoc": np.nan, "return_on_cost": np.nan,
        "rlv": np.nan, "rent_to_pencil_psf": np.nan,
        "iz_pct_applied": np.nan, "market_units": np.nan,
        "affordable_units": np.nan, "construction_months_total": np.nan,
    }


def _infeasible_social(reason: str) -> dict:
    return {
        "social_feasible": False,
        "social_yoc": np.nan, "social_required_yield": np.nan,
        "social_noi": np.nan, "social_tdc": np.nan, "social_land_cost": np.nan,
        "social_blended_rent_psf": np.nan, "social_units": np.nan,
        "binding_constraint": reason,
    }
