from .lfm import OpenAIModelsOAI, GeminiModels, ClaudeModels
from PIL import Image
from azure.identity import DefaultAzureCredential, get_bearer_token_provider, AzureCliCredential
from openai import AzureOpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, MllamaForConditionalGeneration, AutoProcessor
import torch

"""
When adding a new model, the following things are required:
    * this method must be implemented: answer_question(self, question: str, system_message: Optional[str], image: Optional[PIL.Image]) -> str
    * this field must be defined: self.modelname (str)
    * the constructor must not require any arguments. This will live in _MODELS_DICT

Then, add a line to _MODELS_DICT at the end of this file. The key should be the modelname, and the value should be the constructor.
This way, the model can be initialized by calling _MODELS_DICT[modelname]()
Outputs of this model will be saved in a directory called modelname.

(I know I should probably implement an abstract class for this... todo)
"""

##### Lots of code for endpoint models is modified from LFM-Eval-Understand: https://dev.azure.com/msresearch/LFM-Eval-Understand

#### Google Gemini models
GEMINI_SECRET_KEY_PARAMS = {
    "key_name": "aif-eval-gemini",
    "local_keys_path": "/home/t-mmoayeri/keys/aifeval-vault-azure-net.json",
    "key_vault_url": "https://aifeval.vault.azure.net",
}

class Gemini:
    def __init__(self):
        self.config = {
            "secret_key_params": GEMINI_SECRET_KEY_PARAMS,
        }

    def answer_question(self, question, system_message, image):
        if self.modelname == 'gemini-1.0-pro' and len(system_message) > 0: # system message not supported for v1
            question = system_message + '\n\n' + question
            system_message = ''
        return self.model.generate(query_text=question, query_images = [image] if image else None, system_message=system_message if len(system_message) > 0 else None)[0]

class GeminiV15Pro(Gemini):
    def __init__(self):
        super().__init__()
        self.config['model_name'] = "gemini-1.5-pro"
        self.model = GeminiModels(self.config)
        self.modelname = 'gemini-1.5-pro'

class GeminiV1Pro(Gemini):
    def __init__(self):
        super().__init__()
        self.config['model_name'] = "gemini-1.0-pro"
        self.model = GeminiModels(self.config)
        self.modelname = 'gemini-1.0-pro'


#### Anthropic Claude models
CLAUDE_SECRET_KEY_PARAMS = {
    "key_name": "aif-eval-claude",
    "local_keys_path": "/home/t-mmoayeri/keys/aifeval-vault-azure-net.json",
    "key_vault_url": "https://aifeval.vault.azure.net",
}

class Claude:
    def __init__(self):
        self.config = {
            "secret_key_params": CLAUDE_SECRET_KEY_PARAMS,
            # "model_name": config_model_name
        }
        # self.model = ClaudeModels(self.config)

    def answer_question(self, question, system_message, image):
        ### system_message handling seems to be buggy on the lfm side...
        if image and max(image.size) > 8000:
            new_size = [int(x*(8000/max(image.size))) for x in image.size]
            image = image.resize(new_size)

        if len(system_message) > 0:
            question = system_message + '\n' + question
        return self.model.generate(query_text=question, query_images = [image] if image else None, system_message=None)[0]

class ClaudeSonnet(Claude):
    def __init__(self):
        super().__init__()
        self.config['model_name'] = "claude-3-5-sonnet-20240620"
        self.model = ClaudeModels(self.config)
        self.modelname = "claude-sonnet"

class ClaudeOpus(Claude):
    def __init__(self):
        super().__init__()
        self.config['model_name'] = "claude-3-opus-20240229"
        self.model = ClaudeModels(self.config)
        self.modelname = "claude-opus"

#### OpenAI GPT models
OPENAI_SECRET_KEY_PARAMS = {
    "key_name": "openai",
    "local_keys_path": "/home/t-mmoayeri/keys/aifeval-vault-azure-net.json",
    "key_vault_url": "https://aifeval.vault.azure.net",
}

class GPTEndPoint:
    def __init__(self, modelname: str = "gpt-4o", azure_endpoint_url: str = "https://openai-models-west-us3.openai.azure.com/"):

        acceptable_model_names = ['gpt-4o', 'gpt-4-july']
        assert modelname in acceptable_model_names, "model_name must be in "+', '.join(acceptable_model_names)
        self.modelname = modelname

        token_provider = get_bearer_token_provider(
            AzureCliCredential(), "https://cognitiveservices.azure.com/.default"
            )
            
        self.client = AzureOpenAI(
            api_version="2023-06-01-preview",
            azure_endpoint=azure_endpoint_url,
            azure_ad_token_provider=token_provider
            )

    def create_request(self, image, prompt, system_message):
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        user_content = {"role": "user", "content": prompt}
        if image:
            user_content["content"] = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpg;base64,{image}",
                    },
                },
            ]
        messages.append(user_content)
        return {"messages": messages}

    def get_response(self, request):
        completion = self.client.chat.completions.create(
            model=self.modelname,
            ### For now let's just use default values
            # top_p=self.top_p,
            # seed=self.seed,
            # frequency_penalty=self.frequency_penalty,
            # presence_penalty=self.presence_penalty,
            # temperature=self.temperature,
            # max_tokens=self.max_tokens,
            **request,
        )
        openai_response = completion.model_dump()
        return openai_response["choices"][0]["message"]["content"]
    
    def answer_question(self, question, system_message, image):
        msgs = self.create_request(image, prompt=question, system_message=system_message)
        answer = self.get_response(msgs)
        return answer

    def answer_question_w_ic_egs(self, texts, system_message, images):
        messages = [{"role": "system", "content": system_message}]
        content = [
            [
                {"type": "text", "text": text}, 
                {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{img}"}}
            ]
            for text, img in zip(texts, images)
        ]
        return self.get_response({"messages": messages})

class GPT4v:
    def __init__(self):
        self.config = {
            "model_name": "gpt-4-turbo-2024-04-09",
            "secret_key_params": OPENAI_SECRET_KEY_PARAMS,
        }

        self.model = OpenAIModelsOAI(self.config)
        self.modelname = 'gpt-4v'

    def answer_question(self, question, system_message, image):
        return self.model.generate(query_text=question, query_images = [image] if image else None, system_message=system_message)[0]

#### Open source models

class Phi3:
    def __init__(self):
        from transformers import AutoModelForCausalLM 
        from transformers import AutoProcessor 

        self.modelname = 'phi3'
        self.hf_model_name = "microsoft/Phi-3-vision-128k-instruct" 
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_model_name, device_map="cuda", trust_remote_code=True, 
            torch_dtype="auto", _attn_implementation='flash_attention_2'
        )

        self.processor = AutoProcessor.from_pretrained(self.hf_model_name, trust_remote_code=True) 

    def answer_question(self, question, system_message, image):
        if '<image_1>' in question:
            question = question.replace('<image_1>', '<|image_1|>')
        elif '<image>' in question:
            question = question.replace('<image_1>', '<|image_1|>')
        elif image:
            question = '<|image_1|>\n'+question

        ### Phi3 doesn't seem to follow system messages very well, so I will just add it to the question.
        if len(system_message) > 0:
            question = system_message + '\n' + question

        messages = [
            # {"role": "system", "content": system_message},
            {"role": "user", "content": question}
        ]

        prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if image:
            inputs = self.processor(prompt, [image], return_tensors="pt").to("cuda:0")
        else:
            inputs = self.processor(prompt, return_tensors="pt").to("cuda:0")
        
        generation_args = { 
            "max_new_tokens": 1000, 
            "temperature": 0.0, 
            "do_sample": False, 
        } 

        generate_ids = self.model.generate(**inputs, eos_token_id=self.processor.tokenizer.eos_token_id, **generation_args) 

        # remove input tokens 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 
        return response

class Llama32v_Chat_HF:
    def __init__(self):
        model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.modelname = 'llama3.2-11b-instruct-hf'
    
    def answer_question(self, question, system_message, image):
        ### Note: 
        if image:
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text":"SYSTEM MESSAGE:\n"+system_message},
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]}
            ]
        else:
            messages = [
                {"role": "system", "content": [
                    {"type": "text", "text": system_message}
                ]},
                {"role": "user", "content": [
                    {"type": "text", "text": question}
                ]}
            ]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)

        raw_output = self.model.generate(**inputs, max_new_tokens=1000)
        output = self.processor.decode(raw_output[0])
        return output


### Language only models
class Orca2:
    def __init__(self, modelsize='13b'):
        self.model = AutoModelForCausalLM.from_pretrained(f"microsoft/Orca-2-{modelsize}", device_map='auto')
        self.tokenizer = AutoTokenizer.from_pretrained(
            f"microsoft/Orca-2-{modelsize}",
            use_fast=False
        )

    def answer_question(self, question, system_message, image=None):
        if image:
            raise ValueError("Orca2 models are language only; you cannot pass an image.")
        prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"
        inputs = self.tokenizer(prompt, return_tensors='pt')
        output_ids = self.model.generate(inputs["input_ids"].cuda(), max_new_tokens=256)
        answer = self.tokenizer.batch_decode(output_ids[:, inputs['input_ids'].shape[1]:])[0]
        return answer

class Orca2_13b(Orca2):
    def __init__(self):
        super().__init__()
        self.modelname = 'orca2-13b'

class Orca2_7b(Orca2):
    def __init__(self):
        super().__init__(modelsize='7b')
        self.modelname = 'orca2-7b'

class Llama2Chat:
    def __init__(self, modelsize='13b'):
        self.model = AutoModelForCausalLM.from_pretrained(f"meta-llama/Llama-2-{modelsize}-chat-hf", device_map='auto')
        self.tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/Llama-2-{modelsize}-chat-hf")

    def answer_question(self, question, system_message, image=None):
        if image:
            raise ValueError("Llama2 models are language only; you cannot pass an image.")
        prompt = f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{question} [/INST]"
        inputs = self.tokenizer(prompt, return_tensors='pt')
        with torch.no_grad():
            output_ids = self.model.generate(inputs["input_ids"].cuda(), max_new_tokens=256)
        answer = self.tokenizer.batch_decode(output_ids[:, inputs['input_ids'].shape[1]:])[0]
        return answer

class Llama2Chat_13b(Llama2Chat):
    def __init__(self):
        super().__init__()
        self.modelname = 'llama2-13b-chat'

class Llama2Chat_7b(Llama2Chat):
    def __init__(self):
        super().__init__()
        self.modelname = 'llama2-7b-chat'


_MODELS_DICT = {
    'gpt-4o': GPTEndPoint,
    'gpt-4v': GPT4v,
    'phi3': Phi3,
    'gemini-1.5-pro': GeminiV15Pro,
    'gemini-1.0-pro': GeminiV1Pro,
    'claude-sonnet': ClaudeSonnet,
    'claude-opus': ClaudeOpus,
    'orca2-13b': Orca2_13b,
    'orca2-7b': Orca2_7b,
    'llama2chat-13b': Llama2Chat_13b,
    'llama2chat-7b': Llama2Chat_7b,
    'llama32v-chat-11b': Llama32v_Chat_HF
}