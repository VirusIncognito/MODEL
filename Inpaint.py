import json
import os
from PIL import Image
import requests
import time
from fsam import Inference

#@title Connect to the Stability API

import getpass
# @markdown To get your API key visit https://platform.stability.ai/account/keys
STABILITY_KEY = getpass.getpass('Enter your API Key: ')


# Function to send the generation request
def send_generation_request(
    host,
    params,
):
    headers = {
        "Accept": "image/*",
        "Authorization": f"Bearer {STABILITY_KEY}"
    }

    # Encode parameters
    files = {}
    image = params.pop("image", None)
    mask = params.pop("mask", None)
    if image is not None and image != '':
        files["image"] = open(image, 'rb')
    if mask is not None and mask != '':
        files["mask"] = open(mask, 'rb')
    if len(files) == 0:
        files["none"] = ''

    # Send request
    print(f"Sending REST request to {host}...")
    response = requests.post(
        host,
        headers=headers,
        files=files,
        data=params
    )
    if not response.ok:
        raise Exception(f"HTTP {response.status_code}: {response.text}")

    return response

def send_async_generation_request(
    host,
    params,
):
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {STABILITY_KEY}"
    }

    # Encode parameters
    files = {}
    if "image" in params:
        image = params.pop("image")
        files = {"image": open(image, 'rb')}

    # Send request
    print(f"Sending REST request to {host}...")
    response = requests.post(
        host,
        headers=headers,
        files=files,
        data=params
    )
    if not response.ok:
        raise Exception(f"HTTP {response.status_code}: {response.text}")

    # Process async response
    response_dict = json.loads(response.text)
    generation_id = response_dict.get("id", None)
    assert generation_id is not None, "Expected id in response"

    # Loop until result or timeout
    timeout = int(os.getenv("WORKER_TIMEOUT", 500))
    start = time.time()
    status_code = 202
    while status_code == 202:
        response = requests.get(
            f"{host}/result/{generation_id}",
            headers={
                **headers,
                "Accept": "image/*"
            },
        )

        if not response.ok:
            raise Exception(f"HTTP {response.status_code}: {response.text}")
        status_code = response.status_code
        time.sleep(10)
        if time.time() - start > timeout:
            raise Exception(f"Timeout after {timeout} seconds")

    return response

# Define paths and parameters
image = "C:\\Users\\KIIT\\Desktop\\BITS Pilani Research Docs\\MODEL\\images\\cat.jpg"  # Update this path to your image location
mask_prompt = input("Enter Mask Prompt: ")
inpaint_prompt = input("Enter Inpaint Prompt: ")  # Your inpaint prompt
negative_prompt = "bad quality, does not follow original prompt, blurry, changes the whole image, does not follow mask, changes original color, not coherent" # Your negative prompt
seed = 0  # Seed value
output_format = "jpeg"  # Output format: webp, jpeg, or png

# Assuming you have a function Inference.create_mask() and Inference.get_predefined_input()
mask = Inference.create_mask(Inference.get_predefined_input(image, mask_prompt))

host = "https://api.stability.ai/v2beta/stable-image/edit/inpaint"

params = {
    "image": image,
    "mask": mask,
    "negative_prompt": negative_prompt,
    "seed": seed,
    "mode": "mask",
    "output_format": output_format,
    "prompt": inpaint_prompt
}

# Send the request
response = send_generation_request(
    host,
    params
)

# Decode response
output_image = response.content
finish_reason = response.headers.get("finish-reason")
seed = response.headers.get("seed")

# Check for NSFW classification
if finish_reason == 'CONTENT_FILTERED':
    raise Warning("Generation failed NSFW classifier")

# Create output directory if it doesn't exist
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save and display result
filename, _ = os.path.splitext(os.path.basename(image))
edited = os.path.join(output_dir, f"edited_{filename}_{seed}.{output_format}")
with open(edited, "wb") as f:
    f.write(output_image)
print(f"Saved image {edited}")

# Display images
original_image = Image.open(image)
result_image = Image.open(edited)

print("Original image:")
original_image.show()
print("Result image:")
result_image.show()
