import argparse
from utils import *

def main(args):
    # Your main function code here
    print("Name:", args.image)
    print("Image:", args.image)
    # print(args.image.rsplit('.'))
    image_path = args.image
    object_name = args.object
    output_path = args.output

    
    # image_name = image_path.split(".")[0]
    # image_extension = image_path.split(".")[1]
    mask_prompt = [object_name]

    image = read_image(image_path = image_path)

    processor, model = load_segment_pipeline()

    mask = get_mask(image = image, prompts = mask_prompt, processor = processor, model = model)
    discrete_mask = (mask >= .6).to(torch.float)
    image = resize_image(image, shape = (512,512), n_channel = 3)
    discrete_mask = resize_image(discrete_mask, shape = (512,512), n_channel = 1)

    # shifted_image, shifted_mask = move_content(image = copy.deepcopy(image), mask = discrete_mask, x_shift = x_shift, y_shift=y_shift)



    # discrete_mask_padded = add_padding_mask(discrete_mask, k = mask_pad)

    masked_image = apply_mask_to_image(image, discrete_mask)
    pil_masked_image = tensor_to_pil(masked_image)
    pil_masked_image.save(output_path)
    print(f"Masked Image generated and save in {output_path}")
   


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your script")

    parser.add_argument("--image", type=str, help="Image path")
    parser.add_argument("--object", type=str, help="Object name")
    parser.add_argument("--output", type=str, help="generated image path")

    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args)
