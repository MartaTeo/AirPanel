# KL Sampling Progression

A tiny AirPanel helper for answering:

> How many responses do we need before the brand winner becomes stable?

The script is:

```bash
kl_sampling_progression.py
```

It repeatedly samples smaller groups of panelists, compares each sampled preference distribution to the full 800-response distribution with KL divergence, and checks whether the sampled brand winner matches the full-sample brand winner.

## Quick Run

From the AirPanel folder:

```bash
python kl_sampling_progression.py
```

This creates:

```text
kl_sampling_progression_outputs/
```

## Main Outputs

```text
kl_sampling_progression_plot.png
```

The main visual. Top panel shows KL convergence. Bottom panel shows winner stability.

```text
kl_sampling_progression_decision_table.csv
```

The cute little answer table: estimated number of responses needed per dimension.

```text
kl_sampling_progression_summary.csv
```

Detailed KL and stability statistics for every sample size.

```text
kl_sampling_progression_replicates.csv
```

All bootstrap runs, useful if you want to inspect uncertainty more deeply.

## Useful Runs

Fast test:

```bash
python kl_sampling_progression.py --reps 100 --output_dir kl_sampling_progression_outputs_fast
```

More precise:

```bash
python kl_sampling_progression.py --reps 2000 --output_dir kl_sampling_progression_outputs_precise
```

Only humour:

```bash
python kl_sampling_progression.py --dims humour --output_dir kl_sampling_progression_outputs_humour
```

Stricter decision rule:

```bash
python kl_sampling_progression.py --kl_threshold 0.005 --winner_threshold 0.99 --output_dir kl_sampling_progression_outputs_strict
```

Older 3-dimension preference file:

```bash
python kl_sampling_progression.py --input preference_annotations_multidim.csv --dims creative_preference,commercial_preference,overall_preference --output_dir kl_sampling_progression_outputs_multidim
```

## How To Read It

Low KL means the sampled distribution looks close to the full distribution.

High winner stability means the sampled winner is usually the same as the full-sample winner.

If a dimension needs many responses, it probably means the race is close. In the current run, humour is the delicate one.
