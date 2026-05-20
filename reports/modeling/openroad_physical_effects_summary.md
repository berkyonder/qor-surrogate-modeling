# OpenROAD Physical Effects Summary

- Parsed OpenROAD runs: 18
- Successfully parsed runs: 18

## Physical/timing closure metrics inspected
- `openroad_tns`
- `openroad_wns`
- `openroad_worst_slack`
- `openroad_clock_period_min`
- `openroad_fmax`
- `openroad_critical_path_slack`
- `openroad_setup_violation_count`
- `openroad_hold_violation_count`
- `openroad_max_slew_violation_count`
- `openroad_max_cap_violation_count`

## Violation counts
- `openroad_setup_violation_count` total: 0, max per design: 0
- `openroad_hold_violation_count` total: 0, max per design: 0
- `openroad_max_slew_violation_count` total: 0, max per design: 0
- `openroad_max_cap_violation_count` total: 3, max per design: 2

## Note
The current parser captures OpenROAD timing/electrical closure indicators including WNS, TNS, worst slack, setup/hold violations, and slew/capacitance violations. Explicit routed wirelength/congestion extraction can be added as future work from DEF/routing reports.