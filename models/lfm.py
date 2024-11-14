import json
import logging
import urllib.request
from abc import ABC, abstractmethod

from azure.identity import AzureCliCredential, get_bearer_token_provider
from .secret_key_utils import GetKey

import anthropic

class LFM(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def generate(self, text_prompt, query_images=None):
        pass

    def base64encode(self, query_images):
        import base64
        from io import BytesIO

        encoded_images = []

        for query_image in query_images:

            buffered = BytesIO()
            query_image.save(buffered, format="JPEG")
            base64_bytes = base64.b64encode(buffered.getvalue())
            base64_string = base64_bytes.decode("utf-8")

            encoded_images.append(base64_string)

        return encoded_images


class KeyBasedAuthentication:
    def get_api_key(self, config):
        """
        either api_key (str) or secret_key_params (dict) must be provided in the config.
        if api_key is not provided, secret_key_params must be provided to get the api_key using GetKey method.
        """
        try:
            api_key = config["api_key"]
        except KeyError:
            if "secret_key_params" not in config:
                raise ValueError("Either api_key (str) or secret_key_params (dict) must be provided.")
            api_key = GetKey(**config["secret_key_params"])
        return api_key


class EndpointModels(LFM, ABC):
    def __init__(self, config):
        self.url = config.get("url")
        self.max_tokens = config.get("max_tokens", 1000)
        self.temperature = config.get("temperature", 0)
        self.top_p = config.get("top_p", 0.95)
        self.num_retries = config.get("num_retries", 3)
        self.model_name = config.get("model_name")
        self.frequency_penalty = config.get("frequency_penalty", 0)
        self.presence_penalty = config.get("presence_penalty", 0)

    @abstractmethod
    def create_request(self, text_prompt, query_images=None, system_message=None):
        raise NotImplementedError

    @abstractmethod
    def get_response(self, request):
        raise NotImplementedError

    def generate(self, query_text, query_images=None, system_message=None):
        """
        Calls the endpoint to generate the model response.
        args:
            query_text (str): the text prompt to generate the response.
            query_images (list): list of images in base64 bytes format to be included in the request.
            system_message (str): the system message to be included in the request.
        returns:
            response (str): the generated response.
            is_valid (bool): whether the response is valid.
        """
        request = self.create_request(query_text, query_images, system_message)

        attempts = 0
        while attempts < self.num_retries:
            try:
                response = self.get_response(request)
                break
            except Exception as e:
                logging.warning(f"Attempt {attempts+1}/{self.num_retries} failed: {e}")
                response, is_valid, do_return = self.handle_request_error(e)
                if do_return:
                    return response, is_valid
                attempts += 1
        else:
            logging.warning("All attempts failed.")
            return None, False
        return response, True

    @abstractmethod
    def handle_request_error(self, e):
        raise NotImplementedError


class GCREndpointModels(EndpointModels, KeyBasedAuthentication):
    """Tested for Phi3 instruct and Llama3 instruct GCR endpoints."""

    def __init__(self, config):
        super().__init__(config)
        self.do_sample = config.get("do_sample", True)
        # This class uses key-based authentication for now until we receive guidance
        # from GCR on how to authenticate with Azure to the GCR endpoints.
        self.api_key = self.get_api_key(config)

    def create_request(self, text_prompt, query_images=None, system_message=None):
        data = {
            "input_data": {
                "input_string": [
                    {
                        "role": "user",
                        "content": text_prompt,
                    }
                ],
                "parameters": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "do_sample": self.do_sample,
                    "max_new_tokens": self.max_tokens,
                },
            }
        }
        if system_message:
            data["input_data"]["input_string"] = [{"role": "system", "content": system_message}] + data["input_data"][
                "input_string"
            ]
        if query_images:
            raise NotImplementedError("Images are not supported for GCR endpoints yet.")

        body = str.encode(json.dumps(data))
        # The azureml-model-deployment header will force the request to go to a specific deployment.
        # Remove this header to have the request observe the endpoint traffic rules
        headers = {
            "Content-Type": "application/json",
            "Authorization": ("Bearer " + self.api_key),
            "azureml-model-deployment": self.model_name,
        }

        return urllib.request.Request(self.url, body, headers)

    def get_response(self, request):
        response = urllib.request.urlopen(request)
        res = json.loads(response.read())
        return res["output"]

    def handle_request_error(self, e):
        if isinstance(e, urllib.error.HTTPError):
            logging.info("The request failed with status code: " + str(e.code))
            # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
            logging.info(e.info())
            logging.info(e.read().decode("utf8", "ignore"))
        return None, False, False


class OpenAIModels(EndpointModels, ABC):
    """
    This class defines the request and response handling for OpenAI models.
    This is an abstract class and should not be used directly. Child classes should implement the get_client
    method and handle_request_error method.
    """

    def __init__(self, config):
        super().__init__(config)
        self.client = self.get_client(config)
        self.model_name = config.get("model_name")
        self.seed = config.get("seed", 0)

    @abstractmethod
    def get_client(self, config):
        pass

    def create_request(self, prompt, query_images, system_message):
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        user_content = {"role": "user", "content": prompt}
        if query_images:
            encoded_images = self.base64encode(query_images)
            user_content["content"] = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_images[0]}",
                    },
                },
            ]
        messages.append(user_content)
        return {"messages": messages}

    def get_response(self, request):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            top_p=self.top_p,
            seed=self.seed,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **request,
        )
        openai_response = completion.model_dump()
        return openai_response["choices"][0]["message"]["content"]

    @abstractmethod
    def handle_request_error(self, e):
        pass


class OpenAIModelsAzure(OpenAIModels):
    """This class is used to interact with Azure OpenAI models."""

    def get_client(self, config):
        from openai import AzureOpenAI

        token_provider = get_bearer_token_provider(AzureCliCredential(), "https://cognitiveservices.azure.com/.default")
        return AzureOpenAI(
            azure_endpoint=config.get("url"),
            api_version=config.get("api_version", "2023-06-01-preview"),
            azure_ad_token_provider=token_provider,
        )

    def handle_request_error(self, e):
        # if the error is due to a content filter, there is no need to retry
        if e.code == "content_filter":
            logging.warning("Content filtered.")
            response = None
            return response, False, True
        return None, False, False

class TnRModels(OpenAIModelsAzure, KeyBasedAuthentication):
    """This class is used to interact with TnR proxy to Azure endpoint models."""

    def __init__(self, config):
        self.api_key = self.get_api_key(config)
        super().__init__(config)

    def get_client(self, config):
        from openai import AzureOpenAI

        return AzureOpenAI(
            azure_endpoint=config.get("url"),
            api_version=config.get("api_version", "2023-06-01-preview"),
            api_key=self.api_key,
        )


class OpenAIModelsOAI(OpenAIModels, KeyBasedAuthentication):
    """This class is used to interact with OpenAI models dirctly (not through Azure)"""

    def __init__(self, config):
        self.api_key = self.get_api_key(config)
        super().__init__(config)

    def get_client(self, config):
        from openai import OpenAI

        return OpenAI(
            api_key=self.api_key,
        )

    def handle_request_error(self, e):
        logging.warning(e)
        return None, False, False

class GeminiModels(EndpointModels, KeyBasedAuthentication):
    """This class is used to interact with Gemini models throught the python api."""

    def __init__(self, config):
        import google.generativeai as genai
        from google.generativeai.types import HarmCategory, HarmBlockThreshold

        super().__init__(config)
        self.api_key = self.get_api_key(config)
        self.timeout = config.get("timeout", 60)
        genai.configure(api_key=self.api_key)       
        # Safety config, turning off all filters for direct experimentation with the model only
        self.safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }
        self.gen_config = genai.GenerationConfig(
            max_output_tokens = self.max_tokens,
            temperature = self.temperature,
            top_p=self.top_p
        )  

    def create_request(self, text_prompt, query_images=None, system_message=None):
        import google.generativeai as genai

        self.model = genai.GenerativeModel(self.model_name, system_instruction=system_message)
        if query_images:
            return [text_prompt] + query_images
        else:
            return text_prompt
    
    def get_response(self, request):
        gemini_response = self.model.generate_content(request, 
                                generation_config=self.gen_config, request_options={"timeout": self.timeout}, 
                                safety_settings=self.safety_settings,
                                )
        return gemini_response.parts[0].text
    
    def handle_request_error(self, e):
        return None, False, False
    
class HuggingFaceLM(LFM):
    def __init__(self, config, apply_model_template=True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = config.get("model_name")
        self.temperature = config.get("temperature", 0)
        self.top_p = config.get("top_p", 0.95)
        self.max_tokens = config.get("max_tokens", 1000)
        self.do_sample = config.get("do_sample", False)

        self.apply_model_template = apply_model_template
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)

    def generate(self, text_prompt, query_images=None):
        if self.apply_model_template:
            text_prompt = self.model_template_fn(text_prompt)
        if text_prompt is None:
            return None, False
        try:
            inputs = self.tokenizer(text_prompt, return_tensors="pt").to(self.device)
            output_ids = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.do_sample,
            )
            sequence_length = inputs["input_ids"].shape[1]
            new_output_ids = output_ids[:, sequence_length:]
            answer_text = self.tokenizer.batch_decode(
                new_output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            return answer_text, True
        except Exception as e:
            logging.warning(e)
            return None, False

    def model_template_fn(self, text_prompt):
        return text_prompt


class Phi3HF(HuggingFaceLM):
    def __init__(self, config, apply_model_template=True):
        if "microsoft/Phi-3" not in config.get("model_name"):
            logging.warning(
                "This model class applies a template to the prompt that is specific to Phi-3 models"
                "but your model is not a Phi-3 model."
            )
        super().__init__(config, apply_model_template=apply_model_template)

    def model_template_fn(self, text_prompt):
        return f"<|user|>\n{text_prompt}<|end|>\n<|assistant|>"


class LLaVAHuggingFaceMM(LFM):
    def __init__(self, config):
        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = config.get("model_name")
        self.temperature = config.get("temperature", 0)
        self.top_p = config.get("top_p", 0.95)
        self.max_tokens = config.get("max_tokens", 1000)
        self.do_sample = config.get("do_sample", False)
        self.model = LlavaForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def generate(self, text_prompt, query_images=None):
        text_prompt = "USER: <image>\n%s ASSISTANT:" % text_prompt

        inputs = self.processor(text=text_prompt, images=query_images[0], return_tensors="pt").to(self.device)
        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=self.do_sample,
        )
        answer_text = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        assistant_index = answer_text.find("ASSISTANT:") + len("ASSISTANT:")
        result = answer_text[assistant_index:].strip()

        answer_text = result

        return answer_text, True


class ClaudeModels(EndpointModels, KeyBasedAuthentication):
    """This class is used to interact with Claude models through the python api."""

    def __init__(self, config):
        
        super().__init__(config)
        self.api_key = self.get_api_key(config)
        self.timeout = config.get("timeout", 60.0)
        self.client = anthropic.Anthropic(            
            api_key=self.api_key,
            timeout=self.timeout,
        )

    def create_request(self, prompt, query_images, system_message):
        messages = []
        # This seems to be buggy ...
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        user_content = {"role": "user", "content": prompt}

        if query_images:
            encoded_images = self.base64encode(query_images)
            user_content["content"] = [
                {"type": "text", "text": prompt},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": encoded_images[0],
                    },
                },
            ]
        messages.append(user_content)
        return {"messages": messages}

    def get_response(self, request):
        completion = self.client.messages.create(
            model=self.model_name,
            **request,
            temperature=self.temperature,
            top_p=self.top_p,            
            max_tokens=self.max_tokens,
        )
        claude_response = completion.content[0].text
        return claude_response

    def handle_request_error(self, e):
        return None, False, False


"""
Need LFM, need get_key from data/secret_key_utils, correct model config (which is in config/model_config.py), and keys in key/aifeval....json
"""