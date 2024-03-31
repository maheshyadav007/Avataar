import argparse
from utils import *

def main(args):
 
    image_path = args.image
    object_name = args.object
    output_path = args.output
    x_shift = int(args.x)
    y_shift = int(args.y)

    if 'hanging' in image_path:
        mask_pad = 20
    elif 'bag' in image_path:
        mask_pad = 20
    elif 'stool' in image_path:
        mask_pad = 30
    else:
        mask_pad = 5

    
    # image_name = image_path.split(".")[0]
    # image_extension = image_path.split(".")[1]
    mask_prompt = [object_name]

    image = read_image(image_path = image_path)

    processor, model = load_segment_pipeline()
    pipeline = load_inpaint_pipeline()


    mask = get_mask(image = image, prompts = mask_prompt, processor = processor, model = model)
    discrete_mask = (mask >= .6).to(torch.float)
    image = resize_image(image, shape = (512,512), n_channel = 3)
    discrete_mask = resize_image(discrete_mask, shape = (512,512), n_channel = 1)

    discrete_mask_padded = add_padding_mask(discrete_mask, k = mask_pad)

    shifted_image, shifted_mask = move_content(image = copy.deepcopy(image), mask = discrete_mask, x_shift = x_shift, y_shift=y_shift)
    pil_image = tensor_to_pil(image)
    pil_mask_padded = tensor_to_pil(discrete_mask_padded)



    # discrete_mask_padded = add_padding_mask(discrete_mask, k = mask_pad)

    masked_image = apply_mask_to_image(image, discrete_mask)
    pil_masked_image = tensor_to_pil(masked_image)



    inpaint_prompt = ""
    temperature = .4
    strength = 1
    guidance_scale = 2
    negative_prompt = ""
    final_image = inpaint_image(pipeline, prompt = inpaint_prompt, image = pil_image, mask = pil_mask_padded, temperature = temperature, strength = strength, guidance_scale = guidance_scale, negative_prompt = negative_prompt)



    transform = transforms.ToTensor()
    tensor_image = transform(final_image)
    tensor_image = tensor_image.permute(1,2,0)
    merged_image = merge_images(tensor_image, shifted_image, shifted_mask)
    merged_image = tensor_to_pil(merged_image)
    

    merged_image.save(output_path)
    print(f"Shifted Image generated and save in {output_path}")
   


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your script")

    parser.add_argument("--image", type=str, help="Image path")
    parser.add_argument("--object", type=str, help="Object name")
    parser.add_argument("--x", type=float, help="Object name")
    parser.add_argument("--y", type=float, help="Object name")

    parser.add_argument("--output", type=str, help="generated image path")

    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args)
