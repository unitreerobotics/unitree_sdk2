# unitree_sdk2
Unitree robot sdk version 2 modified by @yihuai-gao

## Additional features
- python/crc_module.cpython-310-aarch64-linux-gnu.so to calculate crc for each message pack when running python ros2
- build/disable_sports_mode_go2 to disable sports mode before running self-built controller

usage: [README](python/README.md)


### Prebuild environment
* OS  (Ubuntu 20.04 LTS)  
* CPU  (aarch64 and x86_64)   
* Compiler  (gcc version 9.4.0) 

### Installation
```bash
sudo ./install.sh

```

### Build examples
```bash
mkdir build
cd build
cmake ..
make
```

### Notice
For more reference information, please go to [Unitree Document Center](https://support.unitree.com/home/zh/developer).
