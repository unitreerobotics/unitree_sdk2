# H2X low-level examples

These examples drive the H2X through `unitree_hg` DDS messages at 500 Hz: they
write `rt/lowcmd` and read `rt/lowstate`. They are built as part of the SDK
(`example/CMakeLists.txt` adds this directory).

`h2x_dual_arm_example` needs yaml-cpp; `h2x_termination` needs Eigen3.

## Programs

### `h2x_ankle_swing_example <network_interface>`

Drives all 31 joints: 3 s to the zero posture, then an ankle swing in PR mode,
then an ankle swing in AB mode, repeating.

### `h2x_dual_arm_example <network_interface>`

Replays the 14 arm joints from `low_level/behavior_lib/motion.seq` (500 fps).

### `h2x_termination [-n <network_interface>]`

Read-only: subscribes to `rt/lowstate` and prints a warning on excessive
orientation or when no state message arrives.

The network interface is the DDS interface on the robot side (for example
`eth0`). The examples send motor commands directly, so keep the robot suspended
or otherwise protected and make sure no other controller is driving the motors.
