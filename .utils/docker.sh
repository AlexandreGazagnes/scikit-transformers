#! /bin/sh

# docker base
docker build -f .utils/Dockerfile.base -t skres:base .

# docker build
docker build --no-cache -f .utils/Dockerfile -t skres:latest .

# docker run
# docker run -ti skres:latest /bin/bash
docker run -ti skres:latest python3 -m IPython