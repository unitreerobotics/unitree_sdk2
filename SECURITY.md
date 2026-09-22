# Security Policy

## Supported Versions

Security reports should target the current `main` branch and any actively maintained release branches or tags. If your deployment uses an older SDK revision, include the exact commit, robot model, firmware version, DDS configuration, and operating environment in the report.

## Reporting a Vulnerability

Please do not publish exploit details, packet captures, or unsafe robot-control steps in public issues, pull requests, or discussions. Use GitHub private vulnerability reporting if it is enabled for this repository, or contact the maintainers through the official Unitree support/security channel with:

- affected SDK commit or release;
- affected robot model and firmware version;
- DDS domain, interface, and security configuration;
- a concise impact description;
- minimal reproduction details that avoid unsafe public disclosure;
- suggested mitigations, if known.

Maintainers should acknowledge reports, triage safety impact, coordinate fixes with affected robot/software releases, and publish advisories when users need to take action.

## DDS/RTPS Trust Boundary

This SDK communicates over DDS/RTPS and exposes generic publish/subscribe channels plus RPC-style request/response APIs. Deployments should treat the DDS domain and robot-control network as a security boundary.

Unless DDS Security, network segmentation, and access controls are configured externally, peers on the same reachable DDS domain may be able to discover topics, publish data, subscribe to telemetry, or interfere with command/control traffic. Do not expose robot-control DDS traffic to untrusted networks.

Recommended deployment controls:

- run robot-control clients only on trusted, isolated control networks;
- configure DDS Security authentication, encryption, governance, and permissions where supported;
- restrict host firewalls, VLANs, VPNs, and routing so only intended controllers can reach the robot DDS domain;
- use unique DDS domain IDs and explicit network interfaces for each deployment;
- apply DDS QoS/resource limits for strings, byte arrays, history, queues, and samples;
- monitor dependency advisories for the bundled DDS/runtime libraries.

## Command-Control Safety Guidance

Applications built on this SDK should validate all operator, UI, network, cloud, and automation inputs before forwarding them to robot-control APIs.

Recommended application guardrails:

- require an explicit control lease or authorization gate before publishing motion, arm, hand, or low-level motor commands;
- verify a fresh, synchronized robot state sample before using sensor state to seed trajectories;
- reject non-finite numbers, out-of-range velocities, excessive durations, invalid FSM/action IDs, and malformed trajectories;
- bound all RPC strings, JSON parameters, binary payloads, audio/video/map frames, and generated IDL vectors;
- fail closed on DDS/RPC errors, stale telemetry, missing interfaces, and unknown robot/control state;
- treat CRC values as accidental-corruption checks, not as authentication or tamper protection;
- avoid copying example programs into production without adding synchronization, input validation, arming controls, bounded run times, and cleanup on error or interrupt.

## Examples

The examples are useful for learning SDK APIs, but they may publish real robot commands. Operators should review each example before running it on hardware and should prefer dry-run, finite-duration, and explicitly armed execution modes for any example that writes command topics.

## Dependency and Binary Provenance

This repository includes prebuilt SDK and DDS libraries. Release artifacts should document the source version, build configuration, supported platforms, and known security advisories for bundled runtime dependencies so users can make deployment and update decisions.
