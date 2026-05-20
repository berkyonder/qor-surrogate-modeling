# OpenROAD Physical-Effect Correlation Summary

- Rows analyzed: 18
- Physical metrics analyzed: 6

## Physical metrics
- `openroad_def_routed_wirelength_microns`
- `openroad_def_coordinate_pair_count`
- `openroad_def_routed_net_count`
- `openroad_route_guide_nonempty_lines`
- `openroad_route_guide_segment_like_lines`
- `openroad_route_drc_violation_keyword_count`

## Strongest correlations with final QoR metrics

| Physical metric | QoR metric | Pearson correlation |
|---|---|---:|
| `openroad_def_routed_net_count` | `openroad_synth_chip_area` | 0.9978 |
| `openroad_route_guide_nonempty_lines` | `openroad_synth_chip_area` | 0.9974 |
| `openroad_route_guide_segment_like_lines` | `openroad_synth_chip_area` | 0.9968 |
| `openroad_def_coordinate_pair_count` | `openroad_synth_chip_area` | 0.9894 |
| `openroad_def_routed_wirelength_microns` | `openroad_synth_chip_area` | 0.9315 |
| `openroad_def_coordinate_pair_count` | `openroad_critical_path_delay` | 0.7149 |
| `openroad_route_guide_segment_like_lines` | `openroad_critical_path_delay` | 0.7083 |
| `openroad_route_guide_nonempty_lines` | `openroad_critical_path_delay` | 0.7049 |
| `openroad_def_routed_net_count` | `openroad_critical_path_delay` | 0.6934 |
| `openroad_def_routed_wirelength_microns` | `openroad_critical_path_delay` | 0.5540 |

## Strongest correlations with early-stage features
- No non-constant early-feature correlations available.

## Interpretation note
This is an auxiliary physical-effect analysis. The main surrogate targets remain final OpenROAD area and critical-path delay. Wirelength, route-guide indicators, and DRC report indicators are inspected to evaluate whether physical implementation effects are related to early-stage features and final QoR.