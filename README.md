# Reproduction code

This directory reproduces Figure 1B and the softmax panel of Figure 2 from scratch.
Each script runs the numerical experiments, aggregates the results, and draws the figure.

## Usage

```sh
cd reproduction
python -m pip install numpy matplotlib torch
python figure1b.py
python figure2_softmax.py
```

Figures, CSV results, and run metadata are written to `output/`.
