#!/bin/bash
python generate_mask.py --image "./input_images/bagpack.jpg" --object "bagpack" --output "./output_images/masked_image_bagpack.png"
python generate_mask.py --image "./input_images/stool.jpeg" --object "stool" --output "./output_images/masked_image_stool.png"
python generate_mask.py --image "./input_images/wall hanging.jpg" --object "wall hanging" --output "./output_images/masked_image_wallhanging.png"

