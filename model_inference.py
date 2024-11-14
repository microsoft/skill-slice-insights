from dsets import _DSET_DICT
# from models import _MODELS_DICT
from constants import _SYS_MSGS, _CACHE_ROOT
import pandas as pd
import os
from tqdm import tqdm
import openai
import time
import numpy as np
import torch

"""
This file is for getting model outputs on datasets where we've annotated skills.
This will enable uncovering skill deficiencies that exist across models, as well as 
identifying individual strengths and weaknesses of each model. 

We will also utilize this file for our prompting experiments, used to validate 
relevance of annotated skills. 
"""

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

class QwenVL:
    def __init__(self):
        # default: Load the model on the available device(s)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
        )

        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        self.modelname = 'Qwen2-VL-7B-Instruct'
    
    def answer_question(self, question, system_message, image):
        ########### THIS IS THE ONLY CODE CHANGE WE NEEDED TO MODIFY THE EXAMPLE INFERENCE CODE ############
        if image:
            messages = [{"role":"system", "content": [{"type":"text", "text": system_message}]},
                    {"role": "user", "content": [{"type":"image", "image": image}, {"type":"text", "text":question}]}]
            image_inputs, video_inputs = process_vision_info(messages)
        else:
            messages = [{"role":"system", "content": [{"type":"text", "text": system_message}]},
                    {"role": "user", "content": [{"type":"text", "text":question}]}] 
            image_inputs, video_inputs = None, None

        ### Standard code from huggingface example inference docs: https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

_MODELS_DICT = {'Qwen2-VL-7B-Instruct': QwenVL}

def eval_model(model, dsetname, prompt_key='standard_prompt', finish=True, sub_sample=False):
    dset = _DSET_DICT[dsetname]()
    print(f"{dsetname} dataset loaded.")
    s = _SYS_MSGS[prompt_key]

    if sub_sample:
        path = os.path.join(_CACHE_ROOT, 'model_outputs', prompt_key, model.modelname, 'SUBSAMPLE__' + dset.dsetname + '.csv')
        full_todo = list(range(0, len(dset), max(int(np.round(len(dset) / 300)), 1)))
        print("SUBSAMPLING")
    else:
        path = os.path.join(_CACHE_ROOT, 'model_outputs', prompt_key, model.modelname, dset.dsetname + '.csv')
        full_todo = list(range(len(dset)))
    os.makedirs('/'.join(path.split('/')[:-1]), exist_ok = True)

    if os.path.exists(path) and finish:
        df = pd.read_csv(path)
        df = df.dropna()
        to_do = [i for i in full_todo if i not in list(df['index'])]
        results = list(df[['index', 'system_message', 'response', 'question', 'answer']].values)
    else:
        results = []
        to_do = full_todo

    for i in tqdm(to_do):
        q = dset[i]
        retry_cnt, retry = 5, True # in case of rate limit error
        while retry:
            retry = False
            try:
                with torch.no_grad():
                    ans = model.answer_question(q['prompt'], s, reduce_size(q['image'], 512) if q['image'] else None)
                if not ans: # Gemini will just return None instead of throwing an error; we don't want to save these
                    continue

                results.append([i, s, ans, q['prompt'], q['answer']])
                df = pd.DataFrame(results, columns=['index', 'system_message', 'response', 'question', 'answer'])
                df.to_csv(path, index=False)
            except openai.RateLimitError as e:
                print(e)
                time.sleep(5)
                retry_cnt -= 1
                retry = retry_cnt > 0
            except Exception as e:
                print(e)


def eval_model_many_prompts_and(modelname, dsetnames, prompt_keys, sub_sample=False):
    print(f"Running inference for {modelname} on datasets {', '.join(dsetnames)} for the following prompts: " + ', '.join(prompt_keys))
    model = _MODELS_DICT[modelname]()
    
    for dsetname in tqdm(dsetnames):
        for prompt_key in prompt_keys:
            prompt_args = get_prompt_args(dsetname, prompt_key)
            dset = _DSET_DICT[dsetname]()
            eval_model(model, dset, prompt_key=prompt_key, prompt_args=prompt_args, sub_sample=sub_sample)
            print('done with '+prompt_key)

if __name__ == '__main__':
    # for modelname in ['gpt-4o']:
    #     for prompt_key in ['list_attributes_content_only']:
    #         eval_w_prompt(modelname, ['mmc', 'mmbench', 'mmtbench'], prompt_key)


    import submitit
    log_folder = '../logs/%j'
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(timeout_min=1200)

    dsetnames = _DSET_DICT.keys()#['mmlu_pro'] # or _DSET_DICT.keys(), to do all datasets
    modelnames = ['qwen_vl2'] # or ['gpt-4o', 'claude-sonnet', 'gemini-1.5-pro'] to match analysis from paper
    prompt_keys = ['standard_prompt']
    jobs = []
    with executor.batch():
        for modelname in modelnames:
            for dsetname in dsetnames:
                for prompt_key in prompt_keys:
                    jobs.append(executor.submit(eval_model, modelname, dsetnames, prompt_keys))

