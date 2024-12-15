import numpy as np
from dsets import _DSET_DICT
from models import _MODELS_DICT
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import util
from collections import defaultdict
from tqdm import tqdm
import os
from constants import _SYS_MSGS, _CACHE_ROOT
import pickle
import openai
import time


class SkillAnnotator:

    def __init__(self, prompt_name='long_prompt', annotator_model_name='gpt-4o', probe_model_name=None):
        """
        We will use a strong model (GPT-4o) to understand the skills required to 
        solve each instance in a benchmark. These skills should be model agnostic
        (as they are a property of the data), but we will utilize these skills to
        diagnose skill deficiencies *of a specific model*. 
        """
        # self.dset = dset
        self.annotator_model_name = annotator_model_name
        self.annotator_model = _MODELS_DICT[self.annotator_model_name]()
        ### Probe model is only used for generating probe questions to isolate a skill (see line 350)
        self.probe_model_name = probe_model_name if probe_model_name else self.annotator_model_name
        self.probe_model = _MODELS_DICT[self.probe_model_name]() 
        
        self.sys_msg_name = prompt_name
        self.sys_msg = _SYS_MSGS[self.sys_msg_name]
        self.cache_root = _CACHE_ROOT
        self.loaded_dsets = dict()

        self.text_embedder_and_tokenizer = None
        self.subsample_ext = '' # runs on small subsamples of each dataset, for lightweight analyses

    def turn_on_subsampling(self):
        if self.subsample_ext == 'SUBSAMPLE__':
            print('Subsampling was already on.')
        else:
            print('Turning subsampling on.')
            self.subsample_ext = 'SUBSAMPLE__'

    def turn_off_subsampling(self):
        if self.subsample_ext != 'SUBSAMPLE__':
            print('Turning subsampling on.')
            self.subsample_ext = 'SUBSAMPLE__'
        else:
            self.subsample_ext = ''
            print('Subsampling was already off.')

    #####################################################################
    ### Annotating skills (via getting + parsing rationales)
    #####################################################################
    def compute_rationales(self, path, dsetname, responses_df=None):
        dset = _DSET_DICT[dsetname]()
        full_todo = list(range(len(dset)))
        if self.subsample_ext != '':
            full_todo = list(range(0, len(dset), max(int(np.round(len(dset) / 300)), 1)))

        if responses_df is None:
            os.makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)
            results = []
            to_do = full_todo
        else: # in case anything was done incompletely
            responses_df = responses_df.dropna()
            to_do = [i for i in full_todo if i not in list(responses_df['question_ind'])]
            results = list(responses_df[['question_ind', 'response', 'system_message', 'question', 'answer']].values)
            
        for i in tqdm(to_do):
            q = dset[i]
            retry_cnt, retry = 5, True # in case of openai rate limit error
            while retry:
                retry = False
                try:
                    response = self.annotator_model.answer_question(q['prompt'], self.sys_msg, q['image'])
                    results.append([i, response, self.sys_msg, q['prompt'], q['answer']])
                    
                    df = pd.DataFrame(results, columns=['question_ind', 'response', 'system_message', 'question', 'answer'])
                    df.to_csv(path, index=False)
                except openai.RateLimitError as e:
                    print(e)
                    time.sleep(5)
                    retry_cnt -= 1
                    retry = retry_cnt > 0
                except Exception as e:
                    print(e)


    def get_rationales(self, dsetname, finish=False):
        path = os.path.join(self.cache_root, 'rationales', self.annotator_model_name, self.subsample_ext + dsetname+'.csv')
        if not os.path.exists(path):
            self.compute_rationales(path, dsetname)
        elif finish:
            responses_df = pd.read_csv(path)
            self.compute_rationales(path, dsetname, responses_df)

        responses_df = pd.read_csv(path)
        responses_df['id'] = responses_df.question_ind.apply(lambda q_ind: f"{dsetname}__{q_ind}")
        return responses_df

    def parse_skills_and_spans(self, dsetname):
        path = os.path.join(self.cache_root, 'parsed_skills', self.annotator_model_name, self.subsample_ext + dsetname+".csv")
        if not os.path.exists(path):
            os.makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)
            responses_df = self.get_rationales(dsetname)
            all_skills = []
            for i, row in responses_df.iterrows():
                question_ind, response = [row[x] for x in ['question_ind', 'response']]
                response = response.lower()
                spans = response.split('\n\n')
                span_start = 0
                for span in spans:
                    if 'skill' in span:
                        skill_names_str = span.split('skill')[-1].split('\n')[0].replace(':**','').replace('**', '').strip().replace(':',',').replace(' -',',')
                        if len(skill_names_str) < 5: # sometimes the word skill is mentioned before listing any steps
                            continue
                        skill_names_list = [s.strip() for s in skill_names_str.split(',') if len(s.strip()) > 0]

                        span_end = span_start + len(span)
                        for skill in skill_names_list:
                            all_skills.append([skill, question_ind, span_start, span_end, span])
                    span_start += len(span) + len('\n\n')

            skills_df = pd.DataFrame(all_skills, columns=['skill', 'question_ind', 'span_start', 'span_end', 'span'])
            skills_df.to_csv(path, index=False)
        
        skills_df = pd.read_csv(path)
        skills_df['id'] = skills_df.question_ind.apply(lambda q_ind: f"{dsetname}__{q_ind}")
        return skills_df

    #####################################################################
    ### Aggregation (dedup via clustering of embeddings)
    #####################################################################
    def embed_strs(self, strs, max_length=256, batch_size=64):
        ### Typically, strs is the list of skill names, but it might also be a list of skill cluster names or other general strs
        if not self.text_embedder_and_tokenizer:
            tokenizer = AutoTokenizer.from_pretrained('Salesforce/SFR-Embedding-2_R')
            model = AutoModel.from_pretrained('Salesforce/SFR-Embedding-2_R').cuda()
            self.text_embedder_and_tokenizer = (model, tokenizer)
        model, tokenizer = self.text_embedder_and_tokenizer

        def last_token_pool(last_hidden_states: Tensor,
                        attention_mask: Tensor) -> Tensor:
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                return last_hidden_states[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_states.shape[0]
                return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

        vecs = []
        for i in tqdm(range(0, len(strs), batch_size)):
            batch = strs[i:i+batch_size]
            batch_dict = {k:v.cuda() for k,v in tokenizer(batch, max_length=max_length, padding=True, truncation=True, return_tensors="pt").items()}
            with torch.no_grad():
                outputs = model(**batch_dict)
            batch_vecs = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            batch_vecs = (batch_vecs / batch_vecs.norm(dim=1, keepdim=True)).detach().cpu()
            del outputs; torch.cuda.empty_cache()
            vecs.append(batch_vecs)

        return torch.vstack(vecs)

    def get_skill_embeddings(self, dsetname):
        skills_df = self.parse_skills_and_spans(dsetname)
        
        skill_vecs_path = os.path.join(self.cache_root, 'skill_vecs', self.annotator_model_name, self.subsample_ext + dsetname+'.pkl')
        if not os.path.exists(skill_vecs_path):
            os.makedirs('/'.join(skill_vecs_path.split('/')[:-1]), exist_ok=True)
            skills_list = list(set(skills_df.skill))
            skill_vecs = self.embed_strs(skills_list)

            skills_vecs_dict = {'skills_list': skills_list, 'skill_vecs': skill_vecs}
            with open(skill_vecs_path, 'wb') as f:
                pickle.dump(skills_vecs_dict, f)
        
        with open(skill_vecs_path, 'rb') as f:
            skill_vecs_dict = pickle.load(f)
            skills_list, skill_vecs = [skill_vecs_dict[x] for x in ['skills_list', 'skill_vecs']]

        return skills_df, skills_list, skill_vecs

    def name_cluster(self, cluster, vecs, skills, counts, mode='weighted_mean'):
        ### Options for mode are 'mean', 'weighted_mean', and 'count' (mode)
        cluster_vecs = vecs[cluster]
        if 'mean' in mode:
            if mode == 'mean':
                mean_vec = cluster_vecs.mean()
            elif mode == 'weighted_mean':
                tot_weight = sum(counts)
                mean_vec = torch.stack([w*vecs[s] for w,s in zip(counts, cluster)]).sum(0) / tot_weight
            dists = torch.norm(cluster_vecs - mean_vec)
            ind = torch.argmin(dists) # most central
        elif mode == 'count':
            ind = np.argmax(counts) # most common

        return skills[cluster[ind]]

    def compute_cluster_closeness(self, qs_by_cluster, skill_vecs, skills_list, closeness_thresh=0.975):
        ### Here we compute the fraction of cluster names that have a pair that is too close
        skill_to_ind = {s:i for i,s in enumerate(skills_list)}
        cluster_names = list(qs_by_cluster.keys()) # note that cluster names are always selected from og skill_list
        c_idx = [skill_to_ind[s] for s in cluster_names]
        cluster_vecs = skill_vecs[c_idx]
        cluster_sims = cluster_vecs @ cluster_vecs.T
        cluster_sims = cluster_sims - torch.eye(cluster_sims.shape[0])

        return (cluster_sims.max(1).values > closeness_thresh).float().mean(), cluster_names, cluster_vecs

    def cluster_once(self, qs_by_skill, skills_list, skill_vecs, sim_thresh=0.95):
        clusters = util.community_detection(skill_vecs, threshold=sim_thresh, min_community_size=1)
        print(f"{len(clusters)} clusters currently")
        
        qs_by_cluster = defaultdict(list)
        skills_by_cluster = defaultdict(list)
        for cluster in clusters:
            weights = [len(qs_by_skill[skills_list[i]]) for i in cluster]
            representative_skill = self.name_cluster(cluster, skill_vecs, skills_list, weights)
            for skill_ind in cluster:
                skill = skills_list[skill_ind]
                qs_by_cluster[representative_skill].extend(qs_by_skill[skill])
                skills_by_cluster[representative_skill].append(skill)

        # remove any duplicate questions within a cluster
        qs_by_cluster = {cluster: list(set(qs)) for cluster, qs in qs_by_cluster.items()}
        return qs_by_cluster, skills_by_cluster

    def cluster_skills(self, skills_df, skills_list, skill_vecs, sim_thresh=0.95, max_overlap_frac=0.01):
        qs_by_skill = defaultdict(list)
        for _, row in skills_df.iterrows():
            qs_by_skill[row['skill']].append(row['id'])

        ### We cluster a few times to ensure deduplication
        curr_qs_by_skill, curr_sl, curr_sv = qs_by_skill, skills_list, skill_vecs
        # originally, every skill has its own cluster containing only itself
        prev_skills_by_cluster = {skill:[skill] for skill in skills_list}
        frac_too_close_clusters = 1 # definitely cluster at least once
        while (frac_too_close_clusters > max_overlap_frac):
            # cluster current clusters
            qs_by_cluster, skills_by_cluster = self.cluster_once(curr_qs_by_skill, curr_sl, curr_sv, sim_thresh=sim_thresh)
            # curr_sl is now cluster_names, curr_sv is now vecs corresponding to cluster_names
            frac_too_close_clusters, curr_sl, curr_sv = self.compute_cluster_closeness(qs_by_cluster, skill_vecs, skills_list)
            print(frac_too_close_clusters, len(qs_by_cluster))
            # last book keeping: update skills_by_cluster from (cluster name, last round of cluster names) to (cluster name, list of og skills)            
            merged_skills_by_cluster = {}
            for cluster_name, last_round_cluster_names in skills_by_cluster.items():
                skills_in_new_cluster = []
                for last_round_cluster_name in last_round_cluster_names:
                    skills_in_new_cluster.extend(prev_skills_by_cluster[last_round_cluster_name])
                merged_skills_by_cluster[cluster_name] = skills_in_new_cluster
            skills_by_cluster = merged_skills_by_cluster

            curr_qs_by_skill = qs_by_cluster
            prev_skills_by_cluster = skills_by_cluster

        return qs_by_cluster, skills_by_cluster

    #####################################################################
    ### Putting everything together
    #####################################################################
    def build_skillset(self, dsetnames=list(_DSET_DICT.keys())):
        skills_df, skills_list, skill_vecs = [], [], []
        for dsetname in dsetnames:
            sd, sl, sv = self.get_skill_embeddings(dsetname)
            sd['dsetname'] = [dsetname] * len(sd)
            skills_df.append(sd)
            skills_list.extend(sl)
            skill_vecs.append(sv)

        skill_to_ind = {}
        keep_idx = []
        for i,s in enumerate(skills_list):
            if s not in skill_to_ind:
                skill_to_ind[s] = i
                keep_idx.append(i)
        
        corpus_vecs = torch.vstack(skill_vecs)[keep_idx]
        corpus_skills = [skills_list[i] for i in keep_idx]
        corpus_skills_df = pd.concat(skills_df)
        return corpus_skills_df, corpus_skills, corpus_vecs


    def annotate_skills(self, dsetname='all', sim_thresh=0.95):
        """
        This function implicitly can call the whole (first half of the) pipeline, including:
        (i) getting detailed rationales (ii) parsing skills & spans (iii) embedding skills (iv) dedup/clustering them

        Everything is cached too. 
        The params control the clustering, they can be dataset specific. 
        """

        if dsetname == 'all':
            skills_df, skills_list, skill_vecs = self.build_skillset()
        else:    
            skills_df, skills_list, skill_vecs = self.get_skill_embeddings(dsetname)
        
        qs_by_cluster, skills_by_cluster = self.cluster_skills(skills_df, skills_list, skill_vecs, sim_thresh)
        return qs_by_cluster, skills_by_cluster, skills_df

    #####################################################################
    ### Utils for analysis
    #####################################################################
    def summarize_skill_list(self, skill_list, mode='sentence', instr=None):
        skill_list_str = '\n- '.join(['']+skill_list)
        if instr:
            pass
        elif mode == 'sentence':
            instr = "Answer in one sentence."
        elif 'phrase' in mode:
            instr = f"Answer in a single {mode}." #could be short phrase
        elif mode == 'py_dict':
            instr = "Answer in the form of a python dictionary."
        else:
            raise ValueError(f"Mode {mode} not recognized.")
        
        prompt = f"Summarize the following list of skills. {instr}\nSkills:{skill_list_str}\nSummary: "
        ans = self.annotator_model.answer_question(prompt, "", None)
        return ans

    ### For identifying low accuracy skills
    def acc_by_skill(self, dsetname, correct_per_q, sim_thresh=0.95, min_qs_in_cluster=8, cluster_info=None):
        """ correct_per_q should be a df with question_index as the index. You can pass cluster_info if you've already annotated skills. """
        qs_by_cluster, skills_by_cluster, skills_df = cluster_info if cluster_info else self.annotate_skills(dsetname, sim_thresh)
        
        qs_by_cluster = dict({cluster:qs for cluster, qs in qs_by_cluster.items() if len(qs) >= min_qs_in_cluster})

        correct_by_cluster = dict({
            rep_skill : np.array([correct_per_q.loc[q_id] for q_id in cluster_qs if q_id in correct_per_q.index])
            for rep_skill, cluster_qs in qs_by_cluster.items() 
        })

        acc_by_cluster = dict({
            rep_skill: (np.nanmean(correct_arr), len(correct_arr)) for rep_skill, correct_arr in correct_by_cluster.items()
            if len(correct_arr) > min_qs_in_cluster
        })

        acc_by_cluster = dict(sorted(acc_by_cluster.items(), key= lambda x:x[1][0]))
        return acc_by_cluster, qs_by_cluster, skills_by_cluster, skills_df
        
    #####################################################################
    ### Skill probing (using skill localization to retrieve spans, generate probe qs, and check consistency of responses)
    #####################################################################
    def get_spans_for_skill(self, skill, qs_by_cluster, skills_by_cluster, skills_df):
        spans = []
        for q_id in qs_by_cluster[skill]:
            sub_df = skills_df[skills_df.id == q_id]
            rows = sub_df[sub_df.skill.isin(skills_by_cluster[skill])]
            for _, r in rows.iterrows():
                spans.append([q_id, r['span']])
        
        return spans

    def gen_probe_qs_for_skill(self, skill, qs_by_cluster, skills_by_cluster, skills_df, max_to_gen=10):
        spans = self.get_spans_for_skill(skill, qs_by_cluster, skills_by_cluster, skills_df)
        skills_df = skills_df.set_index('id')
        df = pd.DataFrame(columns=['q_id', 'skill', 'span', 'probe_qs'])
        for q_id, span in spans[:max_to_gen]:
            ans = self.probe_model.answer_question(_SYS_MSGS["probe_template"].format(skill=skill, span=span), _SYS_MSGS["probe_sys_msg"], None)
            df.loc[len(df)] = [q_id, skill, span, ans]

        fname = os.path.join(_CACHE_ROOT, 'probe_qs_new', self.probe_model_name + '_generated_probe_qs', skill.replace(' ','_').replace('/','_')+'.csv')
        os.makedirs('/'.join(fname.split('/')[:-1]), exist_ok=True)

        df.to_csv(fname, index=False)
        
    def answer_probe_qs(self, skill, modelname_to_probe):#probes_df_fname):
        model_to_probe = _MODELS_DICT[modelname_to_probe]()
        fname = os.path.join(_CACHE_ROOT, 'probe_qs_new', self.probe_model_name + '_generated_probe_qs', skill.replace(' ','_').replace('/','_')+'.csv')
        probes_df = pd.read_csv(fname)
        answers, consistencies = [], []
        for _, row in tqdm(probes_df.iterrows(), total=len(probes_df)):
            try:
                dsetname, q_ind = row['q_id'].split('__')
                if dsetname not in self.loaded_dsets:
                    self.loaded_dsets[dsetname] = _DSET_DICT[dsetname]()
                dset = self.loaded_dsets[dsetname]
                q = dset[int(q_ind)]

                pq_dict = eval(row['probe_qs'].replace('```python','').replace('```',''))
                probe_qs = [pq_dict[f"Q{i}"] for i in range(1,4)]
                probe_qs = probe_qs + [pq_dict["Q3"]] * 2 # let's ask the last question 3 times

                probe_answers = [model_to_probe.answer_question(_SYS_MSGS['answer_probe_q'].format(question=q['prompt'], probe=p), "", q['image']) for p in probe_qs]
                answers.append(probe_answers)
                qs_and_as_str = '\n'.join([f"QUESTION: {q}. ANSWER: {a}" for q,a in zip(probe_qs, probe_answers)])
                consistencies.append(self.probe_model.answer_question(_SYS_MSGS['check_consistency'].format(qs_and_as=qs_and_as_str), "", q['image']))
            except Exception as e:
                answers.append(None)
                consistencies.append(np.nan)
                print(e)
                print(row['probe_qs'])

        probes_df['probe_answers'] = answers
        probes_df['consistency_response'] = consistencies
        out_fname = fname.replace(self.probe_model_name + '_generated_probe_qs', modelname_to_probe)
        os.makedirs('/'.join(out_fname.split('/')[:-1]), exist_ok=True)
        print(f'Saving answers from model {modelname_to_probe} to probing questions about {skill} skill to {out_fname}')
        probes_df.to_csv(out_fname, index=False)

    #####################################################################
    ### Skill-based retrieval
    #####################################################################
    def retrieve_by_query_skill(self, query_skill, num_to_return=100, avg_over_topk_tags=3):        
        skills_df, skills_list, skills_vecs = self.build_skillset()
        df = skills_df.groupby('id').apply(lambda sub_df: list(set(sub_df.skill)), include_groups=False)
        df = df.reset_index().rename(columns={0: 'skills_for_q'})
        
        query_vec = self.embed_strs([query_skill])[0]
        sims = skills_vecs @ query_vec
        skill_to_ind = {skill:ind for ind,skill in enumerate(skills_list)}
        df['query_sim'] = df['skills_for_q'].apply(
            lambda x: torch.topk(sims[[skill_to_ind[tag] for tag in x]], k=min(avg_over_topk_tags, len(x))).values.mean().item() 
            if len(x) > 0 else 0
        )
        
        return list(df['id'][list(df.query_sim.nlargest(num_to_return).index)])

if __name__ == '__main__':
    ### Example usage
    # annotator = SkillAnnotator()
    # qs_by_cluster, skills_by_cluster, skills_df = annotator.annotate_skills()
    pass