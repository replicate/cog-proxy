import asyncio
import aiohttp


class APIClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://my-api.com/v1"

    async def predict(
        self, prompt, width, height, steps, guidance, seed, log
    ):
        async with aiohttp.ClientSession() as session:
            # Create image request (or whatever)
            create_response = await self._create_image_request(
                session=session,
                prompt=prompt,
                width=width,
                height=height,
                steps=steps,
                guidance=guidance,
                seed=seed,
                log=log
            )
            if "id" not in create_response:
                raise ValueError(str(create_response))
            request_id = create_response["id"]
            log("Generating image...\n")

            # buffer so we don't immediately hammer the API;
            await asyncio.sleep(10)

            # Poll for result
            while True:
                result = await self._get_result(session, request_id)
                if "status" in result and result["status"] == "OK":
                    return result["result"]
                if "status" in result and result["status"] in ("Error", "Task not found"):
                    log("Error generating image\n")
                    raise Exception("Error generating image", result)
                await asyncio.sleep(0.5)

    async def _create_image_request(
        self,
        session,
        prompt,
        width,
        height,
        steps,
        guidance,
        seed,
        log
    ):
        url = f"{self.base_url}/foo"
        headers = {
            "accept": "application/json",
            "Authorization": self.api_key,
            "Content-Type": "application/json",
        }
        data = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance": guidance,
            "seed": seed
        }
        max_retries = 4
        retries = 0
        delay = 1

        while retries <= max_retries:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status == 429:
                    if retries == max_retries:
                        log("Queue is full; wait and retry request\n")
                        raise Exception("Queue is full; wait and retry request")
                    retries += 1
                    log(f"Queue is full, retrying in {delay} seconds. Attempt {retries}/{max_retries}\n")
                    await asyncio.sleep(delay)
                    delay *= 2  # Exponential backoff
                    continue
                response.raise_for_status()  # This will raise an exception for other HTTP errors
                return await response.json()
        
        raise Exception("Queue is full; wait and retry request")

    async def _get_result(self, session, request_id):
        url = f"{self.base_url}/get_foo"
        headers = {"accept": "application/json", "Authorization": self.api_key}
        params = {"id": request_id}
        async with session.get(url, headers=headers, params=params) as response:
            return await response.json()
