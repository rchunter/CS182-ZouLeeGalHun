#!/bin/bash

# Final training invocation

./train.py --model=denoise --print-every=100 --log=logs/denoise-final.pt --params=params/denoise-final.pt --max-epochs=2
