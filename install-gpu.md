This only works for Nvidia GPUs above like ????

In Docker Desktop: Settings, Docker Engine
Add this line into the daemon config:
"runtimes": { "nvidia": { "path": "nvidia-container-runtime", "runtimeArgs": [] } }

This assumes you already have Nvidia graphic drivers installed for your machine.
Run install-gpu.sh in the native OS (inside WSL2 on Windows)
