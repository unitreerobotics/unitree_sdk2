# Usage

The current `crc_module.so` is compiled on arm64 cpu and python3.8 (compatible with the default Go2 environment). Please compile the library yourself if you are running on a different platform/python environment.

To use it in your Python script (both sdk and ros), please put this file to a directory included in `$PYTHONPATH` and refer to `test_crc.py` for an example. You can also put `crc_module.pyi` to the same directory as `crc_module.so` to enable python type hint.

# Compiling

Activate the correct python environment and use `pip install pybind11` to install pybind.
Find the pybind cmake path and replace `/home/unitree/.local/lib/python3.8/site-packages/pybind11/share/cmake/pybind11` in CMakeLists.txt

```bash
cd python
mkdir build && cd build
cmake ..
make
```
