import base64
import urllib.parse
from typing import List
import os
import requests
from io import BytesIO
from PIL import Image

from cog import BasePredictor, Input, File, emit_metric, Path
from api_client import APIClient


class Predictor(BasePredictor):
    client: APIClient

    def setup(self, weights: str) -> None:
        if not weights:
            raise ValueError(
                "API token must be provided. "
                "Set COG_WEIGHTS environment variable to a "
                "base64-encoded data URI containing the API key."
            )

        parsed_uri = urllib.parse.urlparse(weights)
        if not parsed_uri.scheme == "data":
            raise ValueError(
                "Invalid data URI. Expected a data URI with a base64-encoded API key."
            )

        _, data = parsed_uri.path.split(",", 1)
        try:
            api_key = base64.b64decode(data).decode("utf-8")
        except Exception as e:
            raise ValueError(f"Failed to decode API key: {str(e)}") from e

        self.client = APIClient(api_key)

    def aspect_ratio_to_width_height(self, aspect_ratio: str):
        aspect_ratios = {
            "1:1": (1024, 1024),
            "16:9": (1344, 768),
            "3:2": (1216, 832),
            "2:3": (832, 1216),
            "4:5": (896, 1088),
            "5:4": (1088, 896),
            "9:16": (768, 1344),
        }
        return aspect_ratios.get(aspect_ratio)

    async def predict(
        self,
        prompt: str = Input(description="Text prompt for image generation"),
        aspect_ratio: str = Input(
            description="Aspect ratio for the generated image",
            choices=["1:1", "16:9", "2:3", "3:2", "4:5", "5:4", "9:16"],
            default="1:1",
        ),
        steps: int = Input(
            description="Number of diffusion steps", ge=1, le=50, default=25
        ),
        guidance: float = Input(description="Controls the balance between adherence to the text prompt and image quality/diversity. Higher values make the output more closely match the prompt but may reduce overall image quality. Lower values allow for more creative freedom but might produce results less relevant to the prompt.", ge=2, le=5, default=3),
        seed: int = Input(description="Random seed. Set for reproducible generation", default=None)
    ) -> Path:
        
        if not seed:
            seed = int.from_bytes(os.urandom(2), "big")
        self.log(f"Using seed: {seed}\n")
        
        try:
            self.log("Running prediction... \n")
            width, height = self.aspect_ratio_to_width_height(aspect_ratio)

            image_url = await self.client.predict(
                prompt=prompt,
                width=width,
                height=height,
                steps=steps,
                guidance=guidance,
                seed=seed,
                log=self.log
            )

            req = requests.get(image_url)

            img = Image.open(BytesIO(req.content))
            img_path = "./output.jpg"
            img.save(img_path)

            emit_metric("width", width)
            emit_metric("height", height)
            emit_metric("steps", steps)
            emit_metric("num_images", 1)

            return Path(img_path)

        except Exception as e:
            raise ValueError(f"Error generating image: {str(e)}") from e
