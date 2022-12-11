#!/bin/bash
for i in {1..64}
do
   python main.py --config_file ./configs/minatar/AdaMMQL_fixed_minatar.json --config_idx $i
done
for i in {1..16}
do
   python main.py --config_file ./configs/minatar/AdaMMQL_adaptive_minatar.json --config_idx $i
done
for i in {1..16}
do
   python main.py --config_file ./configs/pixelcopter/AdaMMQL_fixed_pixelcopter.json --config_idx $i
done
for i in {1..4}
do
   python main.py --config_file ./configs/pixelcopter/AdaMMQL_adaptive_pixelcopter.json --config_idx $i
done