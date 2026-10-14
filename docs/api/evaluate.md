# Evaluation

Object-level accuracy assessment of field polygons. See
[Evaluation](../user-guide/evaluation.md) for the definitions in one place.
The `ignore` argument of `evaluate`, `evaluate_frame`, `chance_baselines` and
`paired_bootstrap` leaves out the predictions on ground excluded from the
score (see [Ignoring predictions on excluded
ground](../user-guide/evaluation.md#ignoring-predictions-on-excluded-ground)).
`paired_bootstrap` compares two prediction layers against one reference,
with intervals from resamples of reference fields or of square spatial
blocks that both layers share (see [Comparing two layers: paired
bootstrap](../user-guide/evaluation.md#comparing-two-layers-paired-bootstrap)).

::: agribound.evaluate
    options:
      members:
        - evaluate
        - evaluate_frame
        - paired_bootstrap
        - pixels_per_field
        - chance_baselines
        - proj_network_enabled
        - MATCHING_METHODS
        - BOUNDARY_SAMPLE_SPACING_M
        - BOUNDARY_MAX_SAMPLES
        - BOOTSTRAP_CONFIDENCE_LEVEL
        - BLIND_GRID_OFFSET_STEPS
        - BLIND_GRID_MAX_CELLS
