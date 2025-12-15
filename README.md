# Numerical PDE

This is my homeworks and explorations of solving PDEs with ngsolve. 

The repo is organized in a way that is sane to me:
- keep the code in a python package
- use notebooks purely as frontend and dont develop in notebooks
- keep documentation within the repo

As this is work in progress expect the repo not as well organized as I wish.

## Package

The package is managed with `uv` and is including utilities I want to reuse within my explorations and then has "scripts" for different exercises which I can run as is or import into a notebook and render there.
For most things I will avoid notebooks - `matplotlib` plots will usually be rendered in `kitty` terminal for my own environment.
If I need/want to Draw with `ngsolve` then I have to use Notebooks as it utilizes JavaScript.
The scripts will return objects or plots but never Draw - this can then be done in the Notebooks.

## Notebooks

These are for presentation only - If i want to show in class or Draw with ngsolve - but I really dont like working with Notebooks and thus dont want to build on them.

## Docs

Here I have Markdown writeups of my exercises and my own observations with ngsolve and dump useful parts of ngsolve docs (so Notebooks can be here too).

## Usage
### Running Jupyter with uv
if using `uv` then this command is necessary to run jupyter with the correct kernel of the venv

```bash
uv run --with jupyter jupyter lab
```
