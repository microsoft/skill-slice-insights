from torch.utils.data import Dataset
import pandas as pd
from datasets import load_dataset
import os
from io import BytesIO
import base64
from PIL import Image
from tqdm import tqdm
import json
import numpy as np
from constants import _DATA_ROOT
import glob
import requests
import shutil
import tarfile

###### A quick util for saving images to disc
def reduce_size(image, target_size=-1):
    if target_size > 0 and max(image.size) > target_size:
        new_size = [int(x*(target_size/max(image.size))) for x in image.size]
        image.thumbnail(tuple(new_size))

def decode_base64_to_image(base64_string, target_size=-1, use_b64decode=True):
    if use_b64decode:
        base64_string = base64.b64decode(base64_string)
    image = Image.open(BytesIO(base64_string))
    if image.mode in ('RGBA', 'P', 'LA'):
        image = image.convert('RGB')
    reduce_size(image, target_size)
    return image

###### Datasets
class SEEDBench(Dataset):
    def __init__(self):
        ds = load_dataset("lmms-lab/SEED-Bench")
        df = ds['test'].to_pandas()
        self.df = df[df.data_type == 'image']
        self.images_root = os.path.join(_DATA_ROOT, 'seed_bench', 'images')
        self.dsetname = 'seedbench'

    def __getitem__(self, ind):
        row = self.df.iloc[ind]
        answer, question = [row[x] for x in ['answer', 'question']]
        qtype_ind, og_q_ind = [row[x] for x in ['question_type_id', 'question_id']]
        img_path = f"{self.images_root}/qtype_ind_{qtype_ind}__og_q_ind_{og_q_ind}__ind_{ind}.jpg"
        if not os.path.exists(img_path):
            os.makedirs(self.images_root, exist_ok=True)
            img = decode_base64_to_image(row['image'][0]['bytes'], use_b64decode=False, target_size=512)
            img.save(img_path)
        image = Image.open(img_path)

        options = []
        for opt in ['choice_a', 'choice_b', 'choice_c', 'choice_d']:
            if row[opt]:
                options.append(opt[-1].upper() + '. ' + row[opt])
        options = '\n'.join(options)
        return dict({
            'image': image, 'prompt': question+'\n'+options,
            'question': question,  'index': ind,
            'answer': answer, 'options': options
        })  

    def __len__(self):
        return len(self.df)


class MME(Dataset):
    def __init__(self):
        ds = load_dataset("lmms-lab/MME")
        self.df = ds['test'].to_pandas()
        self.images_root = os.path.join(_DATA_ROOT, 'mme', 'images')
        self.dsetname = 'mme'

    def __getitem__(self, ind):
        row = self.df.iloc[ind]
        answer, question = [row[x] for x in ['answer', 'question']]
        question = question.replace('Please answer yes or no.', '\nA. Yes\nB. No')
        og_q_ind = row['question_id'].replace('/','_')
        img_path = f"{self.images_root}/{og_q_ind}__ind_{ind}.jpg"
        if not os.path.exists(img_path):
            os.makedirs(self.images_root, exist_ok=True)
            img = decode_base64_to_image(row['image']['bytes'], use_b64decode=False, target_size=512)
            img.save(img_path)
        image = Image.open(img_path)

        return dict({
            'image': image, 'prompt': question,
            'index': ind, 'answer': answer, 'options': '\nA. Yes\nB. No'
        })  

    def __len__(self):
        return len(self.df)

class MathVista(Dataset):
    def __init__(self):
        ds = load_dataset("AI4Math/MathVista")
        self.df = ds['testmini'].to_pandas()
        self.images_root = os.path.join(_DATA_ROOT, 'mathvista', 'images')
        self.dsetname = 'mathvista'

    def __getitem__(self, ind):
        row = self.df.iloc[ind]
        answer, question = [row[x] for x in ['answer', 'query']]
        img_path = f"{self.images_root}/{row['pid']}.jpg"
        if not os.path.exists(img_path):
            os.makedirs(self.images_root, exist_ok=True)
            img = decode_base64_to_image(row['decoded_image']['bytes'], use_b64decode=False, target_size=512)
            img.save(img_path)
        image = Image.open(img_path)

        return dict({
            'image': image, 'prompt': question,
            'index': ind, 'answer': answer 
        })  

    def __len__(self):
        return len(self.df)


class MMVet(Dataset):
    def __init__(self):
        ds = load_dataset("lmms-lab/MMVet")
        self.df = ds['test'].to_pandas()
        self.images_root = os.path.join(_DATA_ROOT, 'mmvet', 'images')
        self.dsetname = 'mmvet'

    def __getitem__(self, ind):
        row = self.df.iloc[ind]
        answer, question, capability = [row[x] for x in ['answer', 'question', 'capability']]
        img_path = f"{self.images_root}/{row['question_id']}.jpg"
        if not os.path.exists(img_path):
            os.makedirs(self.images_root, exist_ok=True)
            img = decode_base64_to_image(row['image']['bytes'], use_b64decode=False, target_size=1024)
            img.save(img_path)
        image = Image.open(img_path)

        return dict({
            'image': image, 'prompt': question,
            'index': ind, 'answer': answer, 'capability': capability
        })  

    def __len__(self):
        return len(self.df)


class MMTBench(Dataset):
    def __init__(self):
        self.root = os.path.join(_DATA_ROOT, 'mmtbench')

        metadata_df_path = os.path.join(self.root, 'MMT-Bench_VAL.tsv')
        if not os.path.exists(metadata_df_path): 
            os.makedirs(self.root, exist_ok=True)
            os.makedirs(os.path.join(self.root, 'images'))
            response = requests.get("https://huggingface.co/datasets/OpenGVLab/MMT-Bench/resolve/main/MMT-Bench_VAL.tsv?download=true")
            with open(metadata_df_path, "wb") as file:
                file.write(response.content)
            
        self.df = pd.read_csv(f'{self.root}/MMT-Bench_VAL.tsv', sep='\t')
        self.dsetname = 'mmtbench'

    def __getitem__(self, ind):
        row = self.df.iloc[ind]
        img_path = f"{self.root}/images/{row['l2-category']}__ind_{row['index']}.jpg"
        if not os.path.exists(img_path):
            img_bytes = row['image']
            image = decode_base64_to_image(img_bytes)
            image.save(img_path)

        image = Image.open(img_path)

        options = []
        for opt in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
            if row[opt] and not pd.isnull(row[opt]):
                options.append(opt + '. ' + str(row[opt]))
        options = '\n'.join(options)

        question, answer = [row[x] for x in ['question', 'answer']]
        prompt = question + '\n' + options
        categories = [row[x].replace('_', ' ') for x in ['l2-category', 'category']]
        return dict({
            'prompt': prompt, 'answer':answer, 'index': ind, 'categories': categories,
            'image': image, 'category': row['l2-category'], 'options': options
            })
    
    def __len__(self):
        return len(self.df)
    

class MMBench(Dataset):
    def __init__(self, lite: bool=False):
        ds = load_dataset("HuggingFaceM4/MMBench")
        df = ds['validation'].to_pandas()
        self.lite = lite
        self.images_root = os.path.join(_DATA_ROOT, 'mmbench', 'images')
        self.dsetname = 'mmbench'

        if self.lite:
            lite_df = pd.DataFrame(columns = df.columns)
            for category, sub_df in df.groupby('category'):
                lite_df = pd.concat([lite_df, sub_df.iloc[:10]])
            self.df = lite_df
        else:
            self.df = df

    def __getitem__(self, ind):
        row = self.df.iloc[ind]
        l2_cat, cat = [row[x] for x in ['l2-category', 'category']]
        # answer, image, question = [row[x] for x in ['answer', 'image', 'question']]
        answer, question = [row[x] for x in ['answer', 'question']]
        img_path = f"{self.images_root}/{cat}__{l2_cat}__ind_{ind}.jpg"
        if not os.path.exists(img_path):
            os.makedirs(self.images_root, exist_ok=True)
            img = decode_base64_to_image(row['image'])
            img.save(img_path)
        image = Image.open(img_path)

        if row['hint']:
            question += '\n'+row['hint']
        options = []
        for opt in ['A', 'B', 'C', 'D']:
            if row[opt]:
                options.append(opt + '. ' + row[opt])
        options = '\n'.join(options)
        return dict({
            'image': image, 'prompt': question+'\n'+options,
            'question': question, 'skills': [l2_cat, cat], 
            'answer': answer, 'options': options,
            'l2_category': l2_cat, 'category': cat, 'index': ind
        })  

    def __len__(self):
        return len(self.df)


class MMMU(Dataset):
    def __init__(self):
        self.dsetname = 'mmmu'
        self.images_root = os.path.join(_DATA_ROOT, 'mmmu')

        self.subjects = [
            'Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 'Art_Theory', 
            'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine', 'Computer_Science',
            'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics', 'Energy_and_Power', 
            'Finance', 'Geography', 'History', 'Literature', 'Manage', 'Marketing', 'Materials', 'Math', 
            'Mechanical_Engineering', 'Music', 'Pharmacy', 'Physics', 'Psychology', 'Public_Health', 'Sociology'
        ]

        df = pd.DataFrame(columns = ['id', 'question', 'options', 'explanation', 'image_1', 'image_2',
                                    'image_3', 'image_4', 'image_5', 'image_6', 'image_7', 'img_type',
                                    'answer', 'topic_difficulty', 'question_type', 'subfield', 'field'])

        for subject in tqdm(self.subjects):
            ds = load_dataset("MMMU/MMMU", subject)
            subject_df = ds['validation'].to_pandas()
            subject_df['field'] = [subject.replace('_', ' ')] * len(subject_df)
            df = pd.concat([df, subject_df])
        
        self.df = df

    def __getitem__(self, ind):
        row = self.df.iloc[ind]
        field, subfield, image = [row[x] for x in ['field', 'subfield', 'image_1']]
        question, options, answer = [row[x] for x in ['question', 'options', 'answer']]
        options = eval(options)
        options = '\n'.join([f'{chr(65+i)} {opt}' for i,opt in enumerate(options)])

        img_path = os.path.join(self.images_root, image['path']).replace('.png', '.jpg')
        if not os.path.exists(img_path):
            buffered = BytesIO(image['bytes'])
            img = Image.open(buffered).convert("RGB")
            os.makedirs('/'.join(img_path.split('/')[:-1]), exist_ok=True)
            img.save(img_path)

        return dict({
            'image': Image.open(img_path), 'img_path': img_path, 
            'prompt': question+'\n'+options,
            'question': question, 'skills': [field, subfield], 
            'answer': answer, 'options': options,
            'field': field, 'subfield': subfield
        })

    def __len__(self):
        return len(self.df)


class MMC(Dataset):
    def __init__(self):
        self.dsetname = 'mmc'
        self.data_root = os.path.join(_DATA_ROOT, 'mmc')

        metadata_df_path = os.path.join(self.data_root, 'mmc_benchmark_text.jsonl')
        if not os.path.exists(metadata_df_path): # dataset is not yet downloaded
            ### create directory for dataset
            os.makedirs(self.data_root, exist_ok=True)
            ### donwload metadata jsonl
            response = requests.get("https://huggingface.co/datasets/xywang1/MMC/resolve/main/MMC-Benchmark/mmc_benchmark_text.jsonl?download=true")
            with open(metadata_df_path, "wb") as file:
                file.write(response.content)
            ### create subdirectory for images and download all images
            os.makedirs(os.path.join(self.data_root, 'images'), exist_ok=True)
            response = requests.get("https://huggingface.co/datasets/xywang1/MMC/resolve/main/MMC-Benchmark/mmc_benchmark_images.tar.gz?download=true")
            tar_gz_file_path = os.path.join(self.data_root, 'mmc_benchmark_images.tar.gz')
            with open(tar_gz_file_path, "wb") as file:
                file.write(response.content)
            # extract images
            with tarfile.open(tar_gz_file_path, "r:gz") as tar:
                tar.extractall(path=self.data_root)
            # remove tar_gz_file
            os.remove(tar_gz_file_path)

        self.df = pd.read_json(metadata_df_path, lines=True)

    def __getitem__(self, ind):
        row = self.df.iloc[ind]
        question, answer = [row[x] for x in ['instruction', 'label']]
        img_path = os.path.join(self.data_root,'mmc_benchmark_images/',row['image_id'])
        image = Image.open(img_path).convert('RGB')
        prompt = '\n'.join([question, 'A. True', 'B. False'])
        return dict({
            'prompt': prompt, 'answer': answer, 
            'image': image, 'img_path': img_path, 'options': 'A. True\nB. False'
        })

    def __len__(self):
        return len(self.df)


class VibeEval(Dataset):
    def __init__(self):
        self.df = load_dataset('RekaAI/VibeEval')['test'].to_pandas()
        self.dsetname = 'reka_vibe'
        self.images_root = os.path.join(_DATA_ROOT, 'reka_vibe', 'images')

    def __getitem__(self, ind):
        row = self.df.iloc[ind]
        prompt, reference, image = [row[x] for x in ['prompt', 'reference', 'image']]

        img_path = os.path.join(self.images_root, image['path']).replace('.png', '.jpg')
        if not os.path.exists(img_path):
            os.makedirs(self.images_root, exist_ok=True)
            buffered = BytesIO(image['bytes'])
            img = Image.open(buffered).convert("RGB")
            os.makedirs('/'.join(img_path.split('/')[:-1]), exist_ok=True)
            img.save(img_path)
        
        return dict({
            'prompt': prompt, 'answer': reference, 
            'image': Image.open(img_path),'img_path': img_path
        })

    def __len__(self):
        return len(self.df)


class RealWorldQA(Dataset):
    def __init__(self):
        self.df = load_dataset('xai-org/RealworldQA')['test'].to_pandas()
        self.dsetname = 'realworld_qa'
        self.images_root = os.path.join(_DATA_ROOT, 'realworld_qa', 'images')

    def __getitem__(self, ind):
        row = self.df.iloc[ind]
        question, answer, image = [row[x] for x in ['question', 'answer', 'image']]

        question_str = question.split('?')[0].replace(' ', '_')
        img_path = os.path.join(self.images_root, f'image_{ind}__{question_str}.jpg')
        if not os.path.exists(img_path):
            buffered = BytesIO(image['bytes'])
            img = Image.open(buffered).convert("RGB")
            os.makedirs('/'.join(img_path.split('/')[:-1]), exist_ok=True)
            img.save(img_path)

        # Let's take out the last line, which says to only provide the answer (and not)
        question = '\n'.join(question.split('\n')[:-1])
        
        return dict({
            'prompt': question, 'answer': answer, 
            'image': Image.open(img_path),'img_path': img_path
        })

    def __len__(self):
        return len(self.df)


class MMVP(Dataset):
    def __init__(self):
        self.dsetname = 'mmvp'
        self.categories = ['camera_perspective', 'color', 'orientation', 'presence', 'quantity', 'spatial', 'state', 'structural_character', 'text']
        self.root = os.path.join(_DATA_ROOT, 'MMVP')

        questions_path = os.path.join(self.root, 'mmvp_questions.csv')
        if not os.path.exists(questions_path):
            os.makedirs(self.root, exist_ok=True)
            response = requests.get("https://huggingface.co/datasets/MMVP/MMVP/resolve/main/Questions.csv?download=true")
            with open(questions_path, "wb") as file:
                file.write(response.content)

        self.questions_df = pd.read_csv(questions_path).set_index('Index')#.sort_values(['Type', 'Question ID'])
        self.images_root = os.path.join(self.root, 'images/')

        # huggingface saves the images in an odd order
        self.images_df = load_dataset('MMVP/MMVP')['train'].to_pandas()
        num_strs = [str(x) for x in range(1,301)]
        num_strs.sort()
        self.images_df['image_id'] = [int(x) for x in num_strs]
        self.images_df = self.images_df.set_index('image_id')

    def __getitem__(self, ind):
        if ind >= len(self):
            raise IndexError(f"Index {ind} out of range for dataset of length {len(self)}")

        row = self.questions_df.loc[ind+1]
        image = self.images_df.loc[ind+1]['image']
        question, answer, options = [row[x] for x in ['Question', 'Correct Answer', 'Options']]

        prompt = f'{question} {options}' # this is how they do it in the official repo

        question_str = question.split('?')[0].replace(' ', '_').replace('/','').replace('\'','')
        img_path = os.path.join(self.images_root, f'image_{ind+1}__{question_str}.jpg')
        if not os.path.exists(img_path):
            # buffered = BytesIO(image['bytes'])
            # img = Image.open(buffered).convert("RGB")
            os.makedirs('/'.join(img_path.split('/')[:-1]), exist_ok=True)
            shutil.copy(image['path'], img_path)
            # img.save(img_path)
        
        return dict({
            'prompt': prompt, 'answer': answer, 
            'image': Image.open(img_path),'img_path': img_path, 'options': options
        })

    def __len__(self):
        return len(self.images_df)


#### Language only datasets
class MmluPro(Dataset):
    def __init__(self, lite=False):
        self.df = load_dataset("TIGER-Lab/MMLU-Pro")['test'].to_pandas()
        self.dsetname = 'mmlu_pro'
        if lite:
            self.dsetname = 'mmlu_pro__lite'
            self.df = self.df.iloc[::10]
    
    def __getitem__(self, ind):
        row = self.df.iloc[ind]
        question, answer, options, category= [row[x] for x in ['question', 'answer', 'options', 'category']]

        options_str = ""
        for i,o in enumerate(options):
            options_str += f"{chr(65+i)}. {o}\n"

        return dict({
            'prompt': question+'\n'+options_str, 'ind': ind, 
            'answer': answer, 'options': options, 
            'question': question, 'image': None
        })

    def __len__(self):
        return len(self.df)


class MATH(Dataset):
    def __init__(self):
        ### Dowmload data and eval code from here: github.com/hendrycks/math
        self.dsetname = 'math'
        self.ind_to_path_fname = os.path.join(_DATA_ROOT, 'math', 'metadata.json')
        if not os.path.exists(self.ind_to_path_fname):
            ### First you must download the math dataset and place it in _DATA_ROOT. Rename 'test' dir to 'math'
            fs = glob.glob(os.path.join(_DATA_ROOT, 'math', '*', '*'))
            fixed_path_list = [f.split('../data/math/')[-1] for f in fs]
            with open(self.ind_to_path_fname, 'w') as f:
                json.dump(fixed_path_list, f)
        
        with open(self.ind_to_path_fname, 'r') as f:
            self.ind_to_path = json.load(f)

        train_prompt = "Given a mathematics problem, determine the answer. Simplify your answer as much as possible." + "\n" + "Problem: What is $\left(\\frac{7}{8}\\right)^3 \cdot \left(\\frac{7}{8}\\right)^{-3}$?" + "\n" + "Answer: $1$"
        train_prompt += "\n" + "###" + "\n" + "Problem: In how many ways can 4 books be selected from a shelf of 6 books if the order in which the books are selected does not matter?" + "\n" +"Answer: $15$"
        train_prompt += "\n" +"###" + "\n" + "Problem: Find the distance between the points $(2,1,-4)$ and $(5,8,-3).$" + "\n" + "Answer: $\sqrt{59}$"
        train_prompt += "\n" + "###" + "\n" + "Problem: The faces of an octahedral die are labeled with digits $1$ through $8$. What is the probability, expressed as a common fraction, of rolling a sum of $15$ with a pair of such octahedral dice?" + "\n" + "Answer: $\\frac{1}{32}$"
        train_prompt += "\n" + "###" + "\n" + "Problem: The first three terms of an arithmetic sequence are 1, 10 and 19, respectively. What is the value of the 21st term?" + "\n" + "Answer: $181$"
        train_prompt += "\n" + "###" + "\n" + "Problem: Calculate $6 \\cdot 8\\frac{1}{3}" + "\n" + "Answer: $50$"
        train_prompt += "\n" + "###" + "\n" + "Problem: When the binary number $100101110010_2$ is divided by 4, what is the remainder (give your answer in base 10)?" + "\n" + "Answer: $2$"
        train_prompt += "\n" + "###" + "\n" + "Problem: How many zeros are at the end of the product 25 $\\times$ 240?" + "\n" + "Answer: $3$" + "\n" + "###"
        self.train_prompt = train_prompt

    def __getitem__(self, ind):
        problem_path = self.ind_to_path[ind]
        with open(problem_path, 'r') as f:
            problem_dict = json.load(f)

        problem, answer, problem_type, problem_level = [problem_dict[x] for x in ['problem', 'solution', 'type', 'level']]
        prompt = '\n'.join([self.train_prompt, problem, 'Answer: $'])

        return dict({
            'prompt': prompt, 'ind': ind, 'answer':answer, 'image': None,
            'problem_type': problem_type, 'problem_level': problem_level
        })
        

    def __len__(self):
        return len(self.ind_to_path)



_DSET_DICT = {
    'mathvista': MathVista,
    'mmbench': MMBench,
    'mmc': MMC,
    'mmtbench': MMTBench,
    'mme': MME,
    'mmmu': MMMU,
    'mmvet': MMVet,
    'mmvp': MMVP,
    'realworld_qa': RealWorldQA,
    'reka_vibe': VibeEval,
    'seedbench': SEEDBench,
    #### langauge datasets
    'mmlu_pro': MmluPro,
    # 'math': MATH
}