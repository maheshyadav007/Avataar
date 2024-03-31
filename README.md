# Avataar Assignment

## Description of Files
1. generate_mask.py - Generates mask for a given image
2. shift_object.py - inpaints the objected to shifted and then moves the given object by x and y
3. utils.py - util funtions for all the functionality
4. task1_script.sh - a script to do task 1 of the assignment on all the given images
5. task2_script.sh - a script to do task 2 of the assignment on all the given images
6. requirement.txt - dependency for the assignment
7. Avataar.ipynb - Notebook with all the functionity[Ignore]

## Folder Structure
1. input_images - all the input images are present in this folder
2. output_images - all the generated images are present in this folder.
   - masked_image_{image_name}.png: generated image with generate_mask.py. These images have mask on the object as required in Task 1
   - shifted_image_{image_name}.png: Generated using shift_object.py. These object specified are shift by x and y coordinate as required for Task 2

## Usage
Install the dependency with **requirements.txt**

To generate mask on images:
<code>$python generate_mask.py --image "./input_images/bagpack.jpg" --object "bagpack" --output "./output_images/masked_image_bagpack.png" </code>

To shift object:

<code>$python shift_object.py --image "./input_images/bagpack.jpg" --object "bagpack" --x 162 --y 0 --output "./output_images/shifted_image_bagpack.png" </code>

Note: If you want to run for all the 3 images directly we can use **task1_script.sh** and **task2_script.sh**

## Observations
Successful Cases

