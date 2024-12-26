# Full-body control

<img src="./assets/box.webp" width="256">

# Disclaimer

Only tested and verified on Arch Linux. For any other distro your mileage may vary.

# Python environment

**Version: 3.8**

```bash
python3.8 -m venv .venv
source .venv/bin/activate
```

# Install FBX

[Source](https://github.com/nv-tlabs/ASE/issues/61)

## Download necessary files

- [FBX Sdk](https://www.autodesk.com/content/dam/autodesk/www/adn/fbx/2020-3-2/fbx202032_fbxsdk_linux.tar.gz)
- [FBX Python Bindings](https://www.autodesk.com/content/dam/autodesk/www/adn/fbx/2020-3-2/fbx202032_fbxpythonbindings_linux.tar.gz)


## Install SIP v4.19.3

Get the source from [GitHub](https://github.com/Python-SIP/sip/releases/tag/4.19.3).

```bash
python build.py
python configure.py
make
make install
export SIP_ROOT=<THE_FOLDER_YOU_EXTRACTED_TO>
```

## Install FBX Sdk

```bash
tar -zxvf fbx202032_fbxsdk_linux.tar.gz --one-top-level
cd fbx202032_fbxsdk_linux
./fbx202032_fbxsdk_linux
export FBXSDK_ROOT=<YOUR_FBXSDK_PATH>
```

## Install FBX Python Bindings

```bash
tar -zxvf fbx202032_fbxpythonbindings_linux.tar.gz --one-top-level
cd fbx202032_fbxpythonbindings_linux
./fbx202032_fbxpythonbindings_linux
python PythonBindings.py Python3_x64 buildsip
```

Copy `<YOUR_FBX_PYTHON_BINDING_PATH>/build/Distrib/site-packages/fbx/*` to your `site-packages` in the virtual environment. Make sure it's the files directly on the site-packages. Eg: `.venv/lib/python3.8/site-packages/fbx.so` not `.venv/lib/python3.8/site-packages/fbx/fbx.so` and so on.

# Run retargeting

We retarget data from the given CMU human shape to the H1 humanoid. We use H1 since there's no working T-Pose model for XBot-L as far as I'm aware.

## Download CMU FBX dataset

Download the dataset from [here](https://data.4tu.nl/datasets/0448aab2-3332-449f-a8e2-d208cb58c7df), then unzip the `CMU_fbx.zip` archive into `poselib/data/cmu`.

## Test

To make sure it works, before you spend a lot of time waiting for retargeting make sure it works by running the poselib retarget test:

```bash
cd poselib
python test_retarget.py <MOTION_TO_TEST>
```

Where `MOTION_TO_TEST` would be `01_01` or any other motion. For information on how to control the visualization refer to [PoseLib Docs](./poselib/README.md#poselibvisualization).

## Parse CMU FBX to PoseLib format

Run the following script to parse the CMU FBX dataset to PoseLib. This might take a while

```bash
python fbx_to_poselib.py
```

## Retarget to H1

Then retarget to the H1 humanoid by running

```bash
python retarget_motions.py
```

## Pre-retargeted

The pre-retargeted dataset can also be found [here](TODO).