from analysis import *
import numpy as np
import random
from sklearn.metrics import confusion_matrix
import matplotlib.patches as mpatches
from pipeline import SkillAnnotator
from tqdm import tqdm
from models import _MODELS_DICT
from dsets import _DSET_DICT
from constants import _CACHE_ROOT, _PLOTS_ROOT
from PIL import Image

import textwrap
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
sns.set_theme(style="darkgrid")
plt.rcParams['text.usetex'] = True
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "cmss10",
    "axes.formatter.use_mathtext": "False"
})


#####################################################################
### Post-hoc verification of skills (direct validation)
#####################################################################
def build_skill_verification_set(dsetname, corpus_vecs, corpus_skills, annotator, max_sim=0.825, num_per_q=4, sample_size=100):
    """
    Here we build our 'verification sets', consisting of listed skills + randomly sampled negative skills per instance. 
    Randomly sampled negative skills are drawn from skills seen over the entire corpus, filtered by any skill that is 
    too similar to a listed skill for the current instance. See Sec 2 of the paper for details.

    We build verification sets in a separate step to running verification, so that multiple distinct verifier models can 
    be ran for the same task, allowing us to compute inter-verifier agreement. (results in appendix)
    """
    path = os.path.join(_CACHE_ROOT, 'verification', 'skill_verification_sets', annotator.annotator_model_name+'_annotator', dsetname+'.csv')

    if not os.path.exists(path):
        skills_df, sl, sv = annotator.get_skill_embeddings(dsetname)
        skills_by_q = skills_df.groupby('question_ind').apply(lambda sub_df: list(set(sub_df.skill)), include_groups=False)

        if sample_size:
            skills_by_q = skills_by_q.sample(min(sample_size, len(skills_by_q)))

        sims = (sv.cuda() @ corpus_vecs.cuda().T).detach().cpu()
        idx = np.arange(len(corpus_skills))
        ind_by_skill = {s:i for i,s in enumerate(corpus_skills)}
        ind_by_skill_curr_dset = {s:i for i,s in enumerate(sl)}

        all_sampled_pos_skills, all_neg_skills, all_neg_skill_sims = [], [], []
        for q_ind in skills_by_q.index:
            skills_for_q = skills_by_q.loc[q_ind]
            # we'll only include a subset of the skills per question, so to reduce how long it will take to verify
            sampled_pos_skills = random.sample(skills_for_q, min(len(skills_for_q), num_per_q))
            all_sampled_pos_skills.append(sampled_pos_skills)
            # Now we sample an equal number of negative skills
            keep_idx = torch.ones(len(idx))
            pos_idx = [ind_by_skill_curr_dset[s] for s in skills_for_q]
            # Remove any skills from the corpus that are too similar to a listed skill
            for pos_ind in pos_idx:
                keep_idx *= (sims[pos_ind] < max_sim)

            neg_idx = np.random.choice(idx[keep_idx == 1], len(sampled_pos_skills))
            neg_sims = [max(sims[pos_idx, neg_ind]).item() for neg_ind in neg_idx]
            neg_skills = [corpus_skills[x] for x in neg_idx]

            all_neg_skills.append(neg_skills)
            all_neg_skill_sims.append(neg_sims)

        # Now we add in the negative skills along with the positive ones
        pos_and_neg_skills = pd.DataFrame(zip(list(skills_by_q.index), all_sampled_pos_skills, all_neg_skills, all_neg_skill_sims), 
                                            columns = ['question_ind', 'pos_skills', 'neg_skills', 'neg_skill_sims'])

        pos_and_neg_skills.to_csv(path)

    pos_and_neg_skills = pd.read_csv(path)    
    return pos_and_neg_skills


def verify_skill_relevance(verifier_model_name, pos_and_neg_skills, dsetname, annotator_model_name='gpt-4o'):
    # By default, we verify skills annotated by 'gpt-4o'
    verifier = _MODELS_DICT[verifier_model_name]()
    dset = _DSET_DICT[dsetname]()

    temp = "QUESTION:\n{question}\n\nLIST OF SKILLS: -{skill_list}\n\nRELEVANCE OF EACH SKILL:\n"

    s = """I will present you a question and a list of skills. Your job is to assess if each listed skill is relevant to the question or not.
    Structure your response as a bulleted list where each item has the form '-{SKILL}: {relevant OR not relevant}'. Only answer with the list."""

    results = []
    path = os.path.join(_CACHE_ROOT, 'verification', 'results', verifier_model_name+'_verifier', annotator_model_name+'_annotator', dsetname+'.csv')
    os.makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)
    for _, row in tqdm(pos_and_neg_skills.iterrows(), total=len(pos_and_neg_skills)):
        q = dset[row['question_ind']]
        pos_skills, neg_skills = [eval(row[x]) for x in ['pos_skills', 'neg_skills']]
        k = len(pos_skills)
        both_skills = pos_skills + neg_skills
        rand_idx = random.sample(range(2*k), 2*k)
        skill_list = '\n-'.join([both_skills[i] for i in rand_idx])
        try:
            verifier_response = verifier.answer_question(
                temp.format(question=q['prompt'], skill_list=skill_list), s, q['image']
            )
            # let's parse the response a bit while we're here
            preds, gts = [], []
            for j, l in enumerate(verifier_response.split('\n')):
                if j >= 2*k:
                    break        
                # pred = Does verifier think skill is relevant?
                preds.append(('not relevant' not in l.lower()))
                # gt = Was skill originally 'relevant' according to GPT-4o (i.e. not a rand. sampled negative)
                gts.append((rand_idx[j] < k))
            

            results.append([row[x] for x in ['question_ind', 'pos_skills', 'neg_skills', 'neg_skill_sims']] + [verifier_response, gts, preds])
            results_df = pd.DataFrame(results, columns=['question_ind', 'pos_skills', 'neg_skills', 'neg_skill_sims', 'verifier_response', 'gts', 'preds'])
            results_df.to_csv(path, index=False)
        except Exception as e:
            print(e)

    return pos_and_neg_skills

def run_verification(dsetname, corpus_dsetnames, verifier_model_name='claude-sonnet', annotator_model_name='gpt-4o'):
    annotator = SkillAnnotator() # we want to use GPT-4o annotator for our corpus skills, just bc we have the most skills in this case 
    _, corpus_skills, corpus_vecs = annotator.build_skillset(corpus_dsetnames)  
    annotator = SkillAnnotator(annotator_model_name=annotator_model_name)
    pos_and_neg_skills = build_skill_verification_set(dsetname, corpus_vecs, corpus_skills, annotator)
    verify_skill_relevance(verifier_model_name, pos_and_neg_skills, dsetname, annotator_model_name)

def check_verification_consistency(verifier_model_names, dsetnames, corpus_dsetnames):
    annotator = SkillAnnotator()
    _, corpus_skills, corpus_vecs = annotator.build_skillset(corpus_dsetnames) 
    for dsetname in dsetnames:
        # It is crucial that we verify the same set of pos_and_neg_skills, 
        # as we want a direct comparison of verification outputs for our consistency assessment
        # Update: now pos_and_neg_skills is cached after being computed once and reused
        pos_and_neg_skills = build_skill_verification_set(dsetname, corpus_vecs, corpus_skills, annotator)
        for verifier_model_name in verifier_model_names:
            verify_skill_relevance(verifier_model_name, pos_and_neg_skills, dsetname)
        
        print(f'\nDone with {dsetname}\n'+'-'*25)

#####################################################################
### Plotting code for post-hoc verification (over all confusion matrix + breakdown of CC vs MC)
#####################################################################
def compute_conf_mats(df):
    preds, gts = [], []
    for p, gt in zip(df.preds, df.gts):
        preds.extend(eval(p))
        gts.extend(eval(gt))
    
    cm = confusion_matrix(gts, preds, normalize='true')
    return cm*100

def plot_single_conf_mat(verifiers =['gpt-4v', 'claude-sonnet', 'gemini-1.5-pro'], dsetnames=list(_DSET_DICT.keys())):
    f, ax = plt.subplots(1,1, figsize=(4,4))
    cm = np.zeros((2,2))

    for dsetname in dsetnames:
        for verifier in verifiers:
            df = pd.read_csv(os.path.join(_CACHE_ROOT, 'verification', verifier+'_verifier', 'gpt-4o_annotator', dsetname+'.csv')).set_index('question_ind')
            cm += compute_conf_mats(df)
    cm /= len(dsetnames) * len(verifiers)
    df_cm = pd.DataFrame(cm, 
        index=['Rand. Sampled\nNegative Skills', 'Listed Skills'], 
        columns=['Not relevant', 'Relevant']
    )
    sns.heatmap(df_cm, annot=True, cbar=False, ax=ax, fmt='.2f')
    ax.set_xlabel('Verifier Output')
    ax.set_ylabel('"Groundtruth"')
    f.tight_layout(); f.savefig(os.path.join(_PLOTS_ROOT, 'skill_verification_confmats_all.jpg'), dpi=300, bbox_inches='tight')

def cc_vs_not():
    dsetnames = ['mmbench', 'mmtbench', 'mmlu_pro', 'seedbench', 'mmmu', 'realworld_qa', 'mmvp', 'mme', 'mmc', 'mmvet', 'mathvista', 'reka_vibe']

    results = []
    for dsetname in tqdm(dsetnames):
        print(dsetname)
        df = pd.read_csv(os.path.join(_CACHE_ROOT, 'verification', 'results', 'claude-sonnet_verifier', 'gpt-4o_annotator', dsetname +'.csv'))
        correct_per_q = add_q_id_col(score_df_manually(dsetname, 'gpt-4o'), dsetname)
        # df['correct'] = df.question_ind.apply(lambda x: correct_per_q.loc[f'{dsetname}__{x}'] if f'{dsetname}__{x}' in correct_per_q.index else np.nan)
        correct = []
        for q in df.question_ind:  # I hate this but here we are
            key = f'{dsetname}__{q}'
            try:
                correct.append(correct_per_q.loc[key].correct)
            except:
                correct.append(np.nan)
        df['correct'] = correct
        df = df.dropna()
        # df['correct'] = df.question_ind.apply(lambda x: correct_per_q.get(f'{dsetname}__{x}', np.nan) if f'{dsetname}__{x}' in correct_per_q.index else np.nan)
        # df['correct'] = df.question_ind.apply(lookup_correct)

        ccs, preds, gts = [], [], []
        for i, row in df.iterrows():
            curr_preds, curr_gts = [eval(row[x]) for x in ['preds', 'gts']]
            preds.extend(curr_preds)
            gts.extend(curr_gts)
            ccs.extend([row.correct] * len(curr_preds))

        df2 = pd.DataFrame(zip(ccs, preds, gts), columns=['ccs', 'preds', 'gts'])
        df2['verifier_cc'] = (df2.preds == df2.gts)
        # overall accs
        acc_mc, acc_cc = list(df2.groupby('ccs').mean()['verifier_cc'])

        results.append([dsetname, acc_mc, acc_cc])
    
    df = pd.DataFrame(results, columns=['dataset', 'acc_mc', 'acc_cc'])
    df['avg'] = 0.5 * (df.acc_mc + df.acc_cc)
    df = df.sort_values('avg')
    f, ax = plt.subplots(1,1, figsize=(4,3))
    yticklabels, xs, cc_to_plot, mc_to_plot = [], [], [], []
    ctr = 0
    for i, row in df.iterrows():
        yticklabels.append(_DSET_PRETTY_NAMES[row['dataset']])
        acc_mc, acc_cc = [row[f"acc_{x}"] for x in ['mc', 'cc']]
        ax.plot([acc_mc, acc_cc], [ctr, ctr], ls ='-', color='deepskyblue' if acc_cc > acc_mc else 'coral')
        xs.append(ctr)
        cc_to_plot.append(acc_cc)
        mc_to_plot.append(acc_mc)
        ctr += 1
    ax.scatter(cc_to_plot, xs, marker='o', color='deepskyblue', label='Correctly\nAnswered')
    ax.scatter(mc_to_plot, xs, marker='^', color='coral', label='Incorrectly\nAnswered')
    ax.set_yticks(range(len(yticklabels)))
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel('Skill Relevance')
    ax.set_xlim([0.8, 1])
    ax.set_ylim([-1,12])
    ax.legend()
    f.tight_layout(); f.savefig(os.path.join(_PLOTS_ROOT, 'cc_vs_mc.jpg'), dpi=300, bbox_inches='tight')


def compare_cc_to_not_old():
    f, axs = plt.subplots(1,2, figsize=(8,4))
    for ax, dsetname in zip(axs, ['MMMU', 'MMC']):
        df = pd.read_csv(f'{dsetname.lower()}_skill_verification.csv')

        # Here, cc refers to if the original question was correctly answered
        # preds and gts refer to the skill verifier (which verifies if a listed skill is relevant or not)
        ccs, preds, gts = [], [], []
        for i, row in df.iterrows(): 
            curr_preds, curr_gts = [eval(row[x]) for x in ['preds', 'gts']]
            preds.extend(curr_preds)
            gts.extend(curr_gts)
            ccs.extend([row.correct] * len(curr_preds))
        
        df2 = pd.DataFrame(zip(ccs, preds, gts), columns=['ccs', 'preds', 'gts'])
        df2['verifier_cc'] = (df2.preds == df2.gts)
        # overall accs
        acc_mc, acc_cc = list(df2.groupby('ccs').mean()['verifier_cc'])
        acc_mc_std, acc_cc_std = list(df2.groupby('ccs').std()['verifier_cc'])
    
        # Breakdown over whether or not skill was a randomly sampled negative
        tnr_mc, tnr_cc, tpr_mc, tpr_cc = list(df2.groupby(['gts','ccs']).mean()['verifier_cc'])
        tnr_mc_std, tnr_cc_std, tpr_mc_std, tpr_cc_std = list(df2.groupby(['gts','ccs']).std()['verifier_cc'])

        ax.bar(0.3, acc_cc, width=0.4, color='deepskyblue', yerr=acc_cc_std)
        ax.bar(0.7, acc_mc, width=0.4, color='coral', yerr=acc_mc_std)

        ax.bar(1.3, tpr_cc, width=0.4, color='deepskyblue', yerr=tpr_cc_std)
        ax.bar(1.7, tpr_mc, width=0.4, color='coral', yerr=tpr_mc_std)
        
        ax.bar(2.3, tnr_cc, width=0.4, color='deepskyblue', yerr=tnr_cc_std)
        ax.bar(2.7, tnr_mc, width=0.4, color='coral', yerr=tnr_mc_std)

        ax.set_xticks([0.5,1.5,2.5])
        ax.set_xticklabels(['Accuracy', 'TPR', 'TNR'])
        ax.set_xlabel('Skill Verification Performance')
        handles = [mpatches.Patch(color=c, label=l) for c,l in zip(['deepskyblue','coral'],['correctly answered', 'incorrectly answered'])]
        ax.legend(handles=handles,title='Original question was:', loc='lower left')
        ax.set_ylabel('Performance (see xticklabels for metric)')
        ax.set_title(f'Dataset: {dsetname}')
    f.tight_layout(); f.savefig('../plots/skill_verification_cc_vs_not.jpg', dpi=300, bbox_inches='tight')


#####################################################################
### Plotting code for annotator consistency (across verifiers and skill-annotators)
#####################################################################
def check_annotator_consistency(ann_model1, ann_model2, dsetname):
    """
    I'm using a lot of abbreviations out of convenience. Here is a key:
        's' --> skill, 'sd' --> skills_df, 'sl' --> skills_list, 'sv' --> skill_vecs
    """
    a1, a2 = [SkillAnnotator(annotator_model_name=x) for x in [ann_model1, ann_model2]]
    sd1, sl1, sv1 = a1.get_skill_embeddings(dsetname)
    sd2, sl2, sv2 = a2.get_skill_embeddings(dsetname)
    s_to_ind1, s_to_ind2 = [{s:i for i,s in enumerate(sl)} for sl in [sl1, sl2]]

    sd = pd.concat([sd1, sd2])
    sd['annotator'] = [ann_model1] * len(sd1) + [ann_model2] * len(sd2)

    results = []
    for q_ind, g in sd.groupby('question_ind'):
        if len(list(set(g.annotator))) != 2: # skip any questions that had a parsing error for either annotator
            continue
        g1, g2 = [list(g[g['annotator'] == ann_model].skill) for ann_model in [ann_model1, ann_model2]]

        g1_idx, g2_idx = [[s_to_ind[s] for s in g] for s_to_ind, g in [(s_to_ind1, g1), (s_to_ind2, g2)]]
        sims = sv1[g1_idx] @ sv2[g2_idx].T

        max_sim_per_g1_skill, max_sim_per_g2_skill = [sims.max(i) for i in [1,0]]
        for s, ind, sim in zip(g1, max_sim_per_g1_skill.indices, max_sim_per_g1_skill.values):
            results.append([q_ind, ann_model1, s, g2[ind], sim.item()])

        for s, ind, sim in zip(g2, max_sim_per_g2_skill.indices, max_sim_per_g2_skill.values):
            results.append([q_ind, ann_model2, s, g1[ind], sim.item()])

    results_df = pd.DataFrame(results, columns=['question_ind', 'annotator', 'skill', 'matched_skill', 'sim'])
    return results_df

def plot_annotator_consistency_one_df(df, axs, min_x=1, dsetname=''):
    sns.histplot(df.sim, ax=axs[0], stat='probability', alpha=0.5, label=dsetname)
    xs = np.linspace(df.sim.min(), 1, 20)
    ys = [(df.sim > x).mean() for x in xs]
    i = max([i for i,y in enumerate(ys) if y > 0.995])
    axs[1].plot(xs[i:], ys[i:], label=dsetname)
    min_x = min(xs[i], min_x)

    axs[0].set_xlabel('Cross-Annotator Similarity\n(between skills listed by GPT-4o and -4v)')
    axs[0].set_ylabel('Frequency (over all annotated skills)')

    axs[1].set_ylabel('Dice Coefficient (\% of Matched Skills)')
    axs[1].set_xlabel('Match Threshold\n(min. similarity deemed a match)')

    _ = [ax.set_xlim(min_x-0.01, 1) for ax in axs]
    if dsetname != '':
        _ = [ax.legend() for ax in axs]
    return min_x

### Verifier consistency
def compute_verifier_consistency():
    model_pairs = [('gpt-4v', 'claude-sonnet'), ('gpt-4v', 'gemini-1.5-pro'), ('claude-sonnet', 'gemini-1.5-pro')]
    model_pairs = [('gpt-4v', 'gpt-4o'), ('gemini-1.5-pro', 'gpt-4o'), ('claude-sonnet', 'gpt-4o')]
    dsetnames = [x for x in _DSET_DICT if x not in ['mme', 'mmvet']] # haven't added those two in yet

    consistencies = pd.DataFrame(columns=['Verifier Model Pair', 'Dataset', 'Agreement'])
    # for (m1, m2) in model_pairs:
    m2 = 'gpt-4o'
    for m1 in modelnames:
        for dsetname in dsetnames:
            df1, df2 = [pd.read_csv(os.path.join(_CACHE_ROOT, 'verification', 'results', m+'_verifier', 'gpt-4o_annotator', dsetname+'.csv')) for m in [m1,m2]]
            df1, df2 = [df.set_index('question_ind') for df in [df1, df2]]

            cc, ctr = 0, 0
            for i in df1.index:
                ### Need to do this for now since skill orders are not consistent across verifiers
                row1, row2 = [df.loc[i] for df in [df1, df2]]
                r1, r2 = [row['verifier_response'] for row in [row1, row2]]
                p1s, p2s = [eval(row['preds']) for row in [row1, row2]]

                try:
                    skills1, skills2 = [[x.split(':')[0].replace('-', '').replace('.','').strip() for x in r.split('\n')] for r in [r1, r2]]
                    mapping_1to2 = {s:skills2.index(s) for s in skills1 if s in skills2} # Note that all s in s1 should be in s2
                    cc += sum([p1 == p2s[mapping_1to2[s]] for p1, s in zip(p1s, skills1) if s in mapping_1to2])
                    ctr += len(mapping_1to2)
                except: # parsing errors
                    print('uh oh')
                    print(p2s, skills2, mapping_1to2)
            
            consistencies.loc[len(consistencies)] = [m1, dsetname, cc / ctr]
            # consistencies.loc[len(consistencies)] = [f"{m1} \& {m2}", dsetname, cc / ctr]
    f, ax = plt.subplots(1,1, figsize=(9, 2.5))
    sns.barplot(consistencies, x="Dataset", y="Agreement", hue="Verifier Model Pair", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=9)
    ax.set_ylim([0.5, 1])
    ax.legend(ncol=3, loc='lower center', framealpha=0.95)
    f.tight_layout(); f.savefig(os.path.join(_PLOTS_ROOT, 'consistency', 'verifiers.jpg'), dpi=300, bbox_inches='tight')

def plot_consistency_verifier_and_annotator(ann_match_thresh=0.875):
    modelnames = ['claude-sonnet', 'gemini-1.5-pro', 'gpt-4v']
    dsetnames = [x for x in _DSET_DICT if x not in ['mme', 'mmvet']] # haven't added those two in yet

    verifier_agreement = pd.DataFrame(columns=['Second Verifier Model', 'Dataset', 'Skill Verification\nAgreement with GPT-4o'])
    annotator_agreement_dfs = []
    for dsetname in dsetnames:
        for m1 in modelnames:
            df1, df2 = [pd.read_csv(os.path.join(_CACHE_ROOT, 'verification', 'results', m+'_verifier', 'gpt-4o_annotator', dsetname+'.csv')) for m in [m1,'gpt-4o']]
            df1, df2 = [df.set_index('question_ind') for df in [df1, df2]]

            cc, ctr = 0, 0
            for i in df1.index:
                ### Need to do this for now since skill orders are not consistent across verifiers
                row1, row2 = [df.loc[i] for df in [df1, df2]]
                r1, r2 = [row['verifier_response'] for row in [row1, row2]]
                p1s, p2s = [eval(row['preds']) for row in [row1, row2]]

                try:
                    skills1, skills2 = [[x.split(':')[0].replace('-', '').replace('.','').strip() for x in r.split('\n')] for r in [r1, r2]]
                    mapping_1to2 = {s:skills2.index(s) for s in skills1 if s in skills2} # Note that all s in s1 should be in s2
                    cc += sum([p1 == p2s[mapping_1to2[s]] for p1, s in zip(p1s, skills1) if s in mapping_1to2])
                    ctr += len(mapping_1to2)
                except: # parsing errors
                    print('uh oh')
                    print(p2s, skills2, mapping_1to2)
            
            verifier_agreement.loc[len(verifier_agreement)] = [m1, dsetname, cc / ctr]
    
            annotator_df = check_annotator_consistency(m1, 'gpt-4o', dsetname)
            annotator_df['Dataset'] = dsetname
            annotator_df['Second Annotator Model'] = m1
            annotator_agreement_dfs.append(annotator_df)
    
    f, (ax1, ax2) = plt.subplots(2,1, figsize=(9,4.5))
    sns.barplot(verifier_agreement, x="Dataset", y="Skill Verification\nAgreement with GPT-4o", hue="Second Verifier Model", hue_order=modelnames, ax=ax1)

    annotator_agreement = pd.concat(annotator_agreement_dfs)
    annotator_agreement = annotator_agreement[annotator_agreement.annotator == 'gpt-4o']
    annotator_agreement['Skill Annotation\nAgreement with GPT-4o'] = (annotator_agreement.sim >= ann_match_thresh)
    sns.barplot(annotator_agreement, x="Dataset", y="Skill Annotation\nAgreement with GPT-4o", hue="Second Annotator Model", hue_order=modelnames, ax=ax2, errorbar=None)

    for ax, title in zip([ax1, ax2], ['Second Verifier Model', 'Second Annotator Model']):
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=9)
        ax.legend(ncol=3, loc='lower center', framealpha=0.95, title=title)

    f.tight_layout(); f.savefig(os.path.join(_PLOTS_ROOT, 'consistency', 'verifier_and_annotator.jpg'), dpi=300, bbox_inches='tight')

#####################################################################
### Human validation
#####################################################################
def save_question_as_img(image, prompt, fname):
    f, ax = plt.subplots(1,1, figsize=(4,6))
    prompt = prompt.replace('&', '\&')
    wrapper = textwrap.TextWrapper(width=80)
    text = ''
    for t in prompt.split('\n'):
        text += '\n'.join(wrapper.wrap(t)) +'\n'
    if image:
        ax.set_axis_off()
        img = image.resize((200,200))
        ax.imshow(img)
        ax.text(0, 210, text, va='top', ha='left', fontsize=12)
    else:
        ax.text(0, 0, text, va='top', ha='left', fontsize=12)

    f.tight_layout(); f.savefig(fname, dpi=300, bbox_inches='tight')

def setup_human_eval(num_per_dset=10):
    # For now, we will just move all images to one folder and create a csv
    plt.rcParams['text.usetex'] = False
    new_df = pd.DataFrame(columns=["img_path", "skill", "relevant", "gt"])
    root = os.path.join(_CACHE_ROOT, "human_eval", "skill_verification")
    img_ctr = 0
    dsetnames = ['mmlu_pro', 'mmbench','mmtbench','reka_vibe','mathvista']
    for dsetname in dsetnames:
        df = pd.read_csv(os.path.join(_CACHE_ROOT, 'verification', 'skill_verification_sets', 'gpt-4o_annotator', dsetname+'.csv'))
        dset = _DSET_DICT[dsetname]()
        sub_df = df[:num_per_dset]

        for i, row in tqdm(sub_df.iterrows(), total=len(sub_df)):
            q = dset[row['question_ind']]
            image = q['image'] if q['image'] else Image.new('RGB', (224, 224), color='white')
            prompt = q['prompt']
            img_fname = f"{img_ctr}__{dsetname}__{row['question_ind']}.jpg"
            img_path = os.path.join(root, "images", img_fname)
            save_question_as_img(image, prompt, img_path)

            pos_skills, neg_skills = [eval(row[x]) for x in ['pos_skills', 'neg_skills']]
            both_skills = pos_skills + neg_skills
            rand_idx = random.sample(range(len(both_skills)), len(both_skills))
            for j in rand_idx:
                new_df.loc[len(new_df)] = [img_fname, both_skills[j], '', j < len(pos_skills)]
            img_ctr += 1
    new_df.to_csv(os.path.join(root, 'to_annotate.csv'))
    plt.rcParams['text.usetex'] = True

if __name__ == '__main__':
    # corpus_dsetnames = ['mmmu', 'realworld_qa', 'mmbench', 'mmc', 'mmlu_pro']
    verifier_model_names = ['phi3']#['claude-sonnet', 'gemini-1.5pro', 'gpt-4v', 'gpt-4o']
    dsetnames = corpus_dsetnames = list(_DSET_DICT.keys())

    # dsetnames = dsetnames[5:]

    # run_verification('mmlu_pro', corpus_dsetnames, 'gpt-4o')

    check_verification_consistency(verifier_model_names, dsetnames, corpus_dsetnames)
    # compare_probe_qs_to_ogs()