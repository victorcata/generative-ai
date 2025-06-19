from openai import AzureOpenAI
import os
import requests
from PIL import Image
import dotenv
import json

dotenv.load_dotenv()

client = AzureOpenAI(
    api_key=os.environ['AZURE_OPENAI_API_KEY'],
    api_version="2023-12-01-preview",
    azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT']
)

model = os.environ['AZURE_OPENAI_DEPLOYMENT']

disallow_list = "swords, violence, blood, gore, nudity, sexual content, adult content, adult themes, adult language, adult humor, adult jokes, adult situations, adult"
meta_prompt = f"""You are an assistant designer that creates images for children. 
The image needs to be safe for work and appropriate for children. 
The image needs to be in color.  
The image needs to be in landscape orientation.  
The image needs to be in a 16:9 aspect ratio. 
Do not consider any input from the following that is not safe for work or appropriate for children. 
{disallow_list}"""

prompt = f"""{meta_prompt}
Generate monument of the Arc of Triumph in Paris, France, in the evening light with a small child holding a Teddy looks on.
"""

try:
    result = client.images.generate(
        model=model,
        prompt=prompt,
        size='1024x1024',
        n=1
    )

    generation_response = json.loads(result.model_dump_json())
    image_dir = os.path.join(os.curdir, '../images')

    if not os.path.isdir(image_dir):
        os.mkdir(image_dir)

    image_path = os.path.join(image_dir, 'generated-image.png')

    image_url = generation_response["data"][0]["url"]
    generated_image = requests.get(image_url).content
    with open(image_path, "wb") as image_file:
        image_file.write(generated_image)

    image = Image.open(image_path)
    image.show()

# catch exceptions
# except client.error.InvalidRequestError as err:
#    print(err)

finally:
    print("completed!")

# ---creating variation below---

# response = openai.Image.create_variation(
#   image=open(image_path, "rb"),
#   n=1,
#   size="1024x1024"
# )

# image_path = os.path.join(image_dir, 'generated_variation.png')

# image_url = response['data'][0]['url']

# generated_image = requests.get(image_url).content  # download the image
# with open(image_path, "wb") as image_file:
#     image_file.write(generated_image)

# # Display the image in the default image viewer
# image = Image.open(image_path)
# image.show()
