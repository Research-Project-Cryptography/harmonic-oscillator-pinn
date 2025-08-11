# Harmonic Oscilator demonstration

Implementation of under damped harmonic oscilator:

- https://beltoforion.de/en/harmonic_oscillator/
- https://en.wikipedia.org/wiki/Harmonic_oscillator

In short, we simulate this with PINN:

<img src="Damped_spring.gif" width="50">

Experiment samples will be presented as png files, combine them into a clip with:

```bash
# within the plot subfolders
cat $(find . -name '*.png' | sort -V) | ffmpeg -hwaccel cuda -framerate 60 -i - -c:v libx264 -pix_fmt yuv420p -s 4000x800 out.mp4

# if use nvidia cards
cat $(find . -name '*.png' | sort -V) | ffmpeg -hwaccel cuda  -hwaccel_output_format cuda -framerate 60 -i - -c:v h264_nvenc -pix_fmt yuv420p -s 4000x800 out.mp4
```

# Install this repo

### Install pyenv and Pipenv (one-time setup)

- Why: `pyenv` lets you install and switch Python versions (we use Python 3.10). `pipenv` manages the virtual environment and dependencies.

  - Official docs: [pyenv installation guide](https://github.com/pyenv/pyenv#installation) · [Pipenv installation](https://pipenv.pypa.io/en/latest/installation/)

- macOS (Homebrew):

```bash
brew update
brew install pyenv pipenv
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
exec $SHELL -l
pyenv install 3.10.9
pyenv global 3.10.9   # or run in project dir: pyenv local 3.10.13
```

- Linux (bash):

```bash
curl https://pyenv.run | bash
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
exec $SHELL -l
pyenv install 3.10.9
# Install pipenv (pick one)
pipx install pipenv  # recommended if pipx available
# or: python3 -m pip install --user pipenv
```

### Install with Pipenv (recommended)

- Prereqs: Python 3.10 and Pipenv (macOS: `brew install pipenv`)

1. Clone and enter the project

2. Create the environment and install dependencies (matches Pipfile's Python 3.10)

```bash
pipenv shell
pipenv install
```

3. Launch your IDE to run:

Jupyter Lab

```bash
pipenv run jupyter lab
```

Cursor, vscode install the extension [ms-python](https://marketplace.cursorapi.com/items/?itemName=ms-python.python) then open or run any notebook.
