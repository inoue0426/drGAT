#!/bin/bash
for script in jobs/run_*.sh; do
    sbatch "$script"
done
