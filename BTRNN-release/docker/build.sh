#!/bin/bash
docker build -t btrnn-dev --build-arg authorized_keys="$(cat ~/.ssh/authorized_keys)" .
