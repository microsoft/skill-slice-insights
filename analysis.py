from pipeline import SkillAnnotator
import pandas as pd
import numpy as np
import os
import json
import torch
from scipy.stats import pearsonr
from tqdm import tqdm
from constants import _CACHE_ROOT, _PLOTS_ROOT, _SYS_MSGS, _PRETTY_NAMES
from dsets import _DSET_DICT
from models import _MODELS_DICT
import pickle
import random
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
### Score individual instances
#####################################################################
def score_df_w_llm_judge(dsetname, modelname, grader_model, prompt_key='standard_prompt', format_as_series=True):
    path = os.path.join(_CACHE_ROOT, 'model_outputs', prompt_key, modelname, dsetname+'.csv')
    df = pd.read_csv(path)
    if 'correct' not in df.columns or modelname == 'gemini-1.5-pro' and dsetname == 'mathvista':
        print(f"Grading {modelname} outputs for {dsetname} using {grader_model.modelname} as the grader model.")
        grades = []
        dset = _DSET_DICT[dsetname]()
        for i, row in tqdm(df.iterrows(), total = len(df)):
            q = dset[row['index']]
            response = str(row.response)
            response = response.replace('ANSWER:','') if 'ANSWER:' in response else response
            if 'options' in q:
                grades.append(grader_model.answer_question(_SYS_MSGS['grade_sid'].format(options=q['options'].replace('\n', ' '), gt_answer=row.answer, response=response), '', None))
            else:
                grades.append(grader_model.answer_question(_SYS_MSGS['grade_no_options'].format(question=row.question, gt_answer=row.answer, response=response), '', None))
        df['correct'] = [int(x) for x in grades]
        df.to_csv(path, index=False)
    
    output = add_q_id_col(df, dsetname) if format_as_series else df
    return df

def default_parse_ans(ans):
    ans = str(ans)
    if 'ANSWER:' in ans:
        ans = ans.split('ANSWER:')[1]
    return ans.strip()[:1]    

def score_df_manually(dsetname, modelname, prompt_key='standard_prompt', format_as_series=True, subsample=False):
    fname = 'SUBSAMPLE2__'+dsetname if subsample else dsetname
    path = os.path.join(_CACHE_ROOT, 'model_outputs', prompt_key, modelname, fname+'.csv')
    df = pd.read_csv(path)

    # Default parser
    parse_ans = default_parse_ans
    if dsetname == 'mmc':
        df['correct'] = df.response.apply(parse_ans) == df.answer.apply(lambda x: 'A' if x else 'B')
    elif dsetname in ['mmbench', 'mmtbench', 'mmlu_pro', 'mmlu_pro__lite', 'seedbench', 'mmmu']:
        df['correct'] = df.response.apply(parse_ans) == df.answer
    elif dsetname == 'mmvp':
        no_parens = lambda x: x.replace('(','').replace(')','')
        parse_ans = lambda x: no_parens(x.split('ANSWER:')[1].strip())[0] if 'ANSWER' in str(x) else None
        df['correct'] = df.response.apply(parse_ans) == df.answer.apply(no_parens)
    elif dsetname == 'mme':
        df['correct'] = df.response.apply(parse_ans) == df.answer.apply(lambda x: 'A' if x == 'Yes' else 'B')
    elif dsetname == 'realworld_qa':
        df['parsed_response'] = df.response.apply(lambda x: x.split('\n')[0] if '\n' in x else x)
        df['correct'] = df.apply(lambda x: x['answer'].lower() in x['parsed_response'].lower(), axis=1)
    
    output = add_q_id_col(df, dsetname) if format_as_series else df
    return df

def get_all_preds():
    dsetnames = ['realworld_qa', 'mmc', 'mme', 'mmbench', 'mmtbench', 'mmmu', 'mmlu_pro', 'seedbench', 'mathvista', 'mmvp', 'reka_vibe', 'mmvet']
    modelnames = ['gpt-4o', 'gemini-1.5-pro', 'claude-sonnet']
    dfs = []
    for modelname in modelnames:
        curr_dfs = []
        for dsetname in dsetnames:
            curr_dfs.append(add_q_id_col(manually_score_df(dsetname, modelname), dsetname))
            curr_dfs[-1]['dsetname'] = dsetname

        df = pd.concat(curr_dfs)
        df['model'] = modelname
        dfs.append(df)

    full_df = pd.concat(dfs)
    return full_df

def add_q_id_col(grades_df, dsetname):
    grades_df = grades_df[['index', 'correct']]
    grades_df = grades_df.reset_index().rename(columns={'index':'q_ind'})
    grades_df['id'] = grades_df.q_ind.apply(lambda x:f"{dsetname}__{x}")
    grades_df = grades_df.set_index('id')
    grades_df = grades_df.drop(columns=['q_ind', 'level_0'])
    return grades_df

def get_cluster_info(dsets='all'):
    an = SkillAnnotator()

    with open(os.path.join(_CACHE_ROOT, f'cluster_infos/{dsets}/qs_by_cluster.json'), 'r') as f:
        qs_by_cluster = json.load(f)
    with open(os.path.join(_CACHE_ROOT, f'cluster_infos/{dsets}/skills_by_cluster.json'), 'r') as f:
        skills_by_cluster = json.load(f)

    skills_df, _, _ = an.build_skillset()
    return qs_by_cluster, skills_by_cluster, skills_df

#####################################################################
### Model comparisons
#####################################################################
def strengths_and_weaknesses(modelnames = ['gpt-4o', 'gemini-1.5-pro', 'claude-sonnet']):
    cluster_info = get_cluster_info()
    df = get_all_preds()

    accs ={m: an.acc_by_skill(None, df[df.model == m][['correct']], cluster_info=cluster_info, min_qs_in_cluster=100)[0] for m in modelnames}
    df2 = pd.DataFrame({m:{k:tup[0]*100 for k,tup in v.items()} for m, v in accs.items()})
    df2 = df2.dropna()

    for m in modelnames:
        df2['gains__'+m] = df2[m] - df2[[not_m for not_m in modelnames if not_m!= m]].mean(1)

    df2['avgs'] = df2[modelnames].mean(1)
    df2['stds'] = df2[modelnames].std(1)
    df2['winner'] = df2.apply(lambda row: modelnames[np.argmax([np.abs(row[f"gains__{x}"]) for x in modelnames])], axis=1)

    wrapper = textwrap.TextWrapper(width=30)
    wrap = lambda x: '\n'.join(wrapper.wrap(x))

    cp = sns.color_palette()
    for mode, sign in zip(['weaknesses', 'strengths'], ['-', '+']):
        f, axs = plt.subplots(1,3, figsize=(10.5,5))
        for ax, modelname in zip(axs, modelnames):
            sub_df = df2[df2.winner == modelname]
            sub_df = sub_df.sort_values(f'gains__{modelname}').reset_index().rename(columns={'index':'skill'})

            sub_df.skill = sub_df.apply(lambda row: row.skill + " ({}{:.1f}\%)".format(sign, np.abs(row[f'gains__{modelname}'])), axis=1)
            ax.set_title(f"{mode.title()} of\n{_PRETTY_NAMES[modelname]}")
            to_plot = sub_df[:10] if mode == 'weaknesses' else sub_df[-10:]
            to_plot = pd.melt(to_plot[['skill']+modelnames], id_vars=['skill'], var_name='Model', value_name='Accuracy')
            to_plot['skill'] =  to_plot.skill.apply(wrap)
            sns.pointplot(data=to_plot, y='skill', x='Accuracy', hue='Model', hue_order=modelnames, legend=False, 
                        linewidth=0, markersize=5.5, ax=ax, order=to_plot.groupby('skill').mean('Accuracy').sort_values('Accuracy').index)
            
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=11)

        f.tight_layout(); f.savefig(os.path.join(_PLOTS_ROOT, f'model_comp/{mode}.jpg'), dpi=300, bbox_inches='tight')
    # let's save a legend too
    from matplotlib.lines import Line2D
    f, ax = plt.subplots(1,1, figsize=(8,1))
    handles = [Line2D([], [], marker='o', color='white', markerfacecolor=cp[i], markersize=15, label=_PRETTY_NAMES[m], linestyle=None, linewidth=0) for i, m in enumerate(modelnames)]
    f.legend(handles=handles, loc='center', fontsize=18, ncol=len(modelnames))
    ax.set_axis_off()
    f.tight_layout(); f.savefig(os.path.join(_PLOTS_ROOT, 'model_comp/legend.jpg'), bbox_inches='tight', dpi=300)

def model_evolution():
    # with open(os.path.join(_CACHE_ROOT, 'cluster_infos/mmlu_pro/qs_by_cluster.json'), 'r') as f:
    #     qs_by_cluster = json.load(f)
    # cluster_info = (qs_by_cluster, None, None)
    cluster_info = get_cluster_info('mmlu_pro')

    f, axs = plt.subplots(1,3, figsize=(12,5), sharey=True)
    for model_pair, ax in zip([('gpt-4v', 'gpt-4o'), ('gemini-1.0-pro', 'gemini-1.5-pro'), ('claude-opus', 'claude-sonnet')], axs):
        dfs = []
        for modelname in model_pair:
            df = add_q_id_col(manually_score_df('mmlu_pro', modelname), 'mmlu_pro')
            df['model'] = modelname
            dfs.append(df)

        df = pd.concat(dfs)        
        accs ={m: an.acc_by_skill(None, df[df.model == m][['correct']], cluster_info=cluster_info, min_qs_in_cluster=100)[0] for m in model_pair}
        df2 = pd.DataFrame({m:{k:tup[0]*100 for k,tup in v.items()} for m, v in accs.items()})

        df2['improvement'] = df2[model_pair[1]] - df2[model_pair[0]]
        df2 = df2.dropna().sort_values('improvement')['improvement'].reset_index().rename(columns={'index':'skill'})
        to_plot = pd.concat([df2[:5], pd.DataFrame([['', 0]], columns=df2.columns), df2[-5:]]) # we add a gap in the middle
         
        sns.barplot(to_plot, x="skill", y="improvement", ax=ax, hue="improvement", palette='coolwarm_r', hue_norm=(-60,60))
        ax.set_title(f'{_PRETTY_NAMES[model_pair[0]]} to {_PRETTY_NAMES[model_pair[1]]}', fontsize=14)
        ax.set_xticklabels(ax.get_xticklabels(), rotation='vertical', fontsize=12)

        avg_gain = 100*(df[df.model == model_pair[1]]['correct'].mean() - df[df.model == model_pair[0]]['correct'].mean())
        x_min, x_max = ax.get_xlim()
        h = ax.axhline(avg_gain, x_min, x_max, label=f'{avg_gain:.1f}\% Average\nImprovement', ls='--', color='black')
        ax.legend(handles=[h], loc='upper left')

    f.tight_layout(); f.savefig(os.path.join(_PLOTS_ROOT, 'model_comp2/improvements.jpg'), dpi=300, bbox_inches='tight')

def plot_skill_diffs(dsetname='mmc', num_to_show=5):
    phi3_correct, gpt_correct = grade_phi_and_4o(dsetname)
    annotator = SkillAnnotator()
    acc_phi3, acc_gpt = [annotator.acc_by_skill(dsetname, c) for c in [phi3_correct, gpt_correct]]
    diffs = {k:acc_gpt[k][0] - acc_phi3[k][0] for k in acc_gpt}
    sorted_diffs = sorted(diffs.items(), key=lambda x:x[1])

    # f, ax = plt.subplots(1,1, figsize=(max(14, int(17*len(diffs)/100)), 6))
    f, ax = plt.subplots(1,1, figsize=(26, 6))
    colors = ['darkred', 'deepskyblue']
    shapes = ['^', 'o']

    gpt_accs_sorted = dict(sorted(acc_gpt.items(), key=lambda x:x[1]))
    k_order = list(gpt_accs_sorted.keys())
    gs, ps = [[d[k][0] for k in gpt_accs_sorted] for d in [acc_gpt, acc_phi3]]

    for i, (g,p) in enumerate(zip(gs, ps)):
        ax.plot([i,i],[g,p], color=colors[1] if g > p else colors[0])
        ax.plot(i,g, shapes[1], color=colors[1], markersize=4)
        ax.plot(i,p, shapes[0], color=colors[0], markersize=4)

    xticks, xticklabels, xtickcolors = [], [], []     
    for to_plot, color in zip([sorted_diffs[:num_to_show], sorted_diffs[-1*num_to_show:]], colors): 
        for k,v in to_plot:
            i = k_order.index(k)
            xticks.append(i)
            xticklabels.append(f'{k.title()} ({"+" if v>0 else ""}{int(v*100)}%)')
            xtickcolors.append(color if v<0 or color==colors[1] else 'gray')
            # xtickcolors.append(colors[1] if v>0 else colors[0])

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation='vertical', fontsize=10)
    for tick_label, color in zip(ax.get_xticklabels(), xtickcolors):
        tick_label.set_color(color)
    
    ax.set_ylabel('Accuracy Per Skill')
    for color, shape, label in zip(colors, shapes, [f'Phi3 Acc; {phi3_correct.correct.mean()*100:.1f}% overall', f'GPT-4o Acc; {gpt_correct.correct.mean()*100:.1f}% overall']):
        ax.plot([],[], shape, color=color, label = label)

    ax.legend()
    ax.set_title(f'Skill-wise Comparison on {dsetname.upper().replace("_"," ")}. Blue color signifies GPT-4o $>$ Phi-3.')
    f.tight_layout();f.savefig(f'/home/t-mmoayeri/plots/phi3_vs_gpt4o/{dsetname}__lines.jpg', dpi=300, bbox_inches='tight')



#####################################################################
### Generalization of slice accuracies (scatter + routing experiments)
#####################################################################

def mmlu_scatter():
    modelnames = ['gpt-4o', 'gemini-1.5-pro', 'claude-sonnet']
    an = SkillAnnotator()
    corpus_cluster_info = get_cluster_info()

    min_qs_in_cluster = 100
    full_df = get_all_preds()
    df_by_model = {modelname:sub_df for modelname, sub_df in full_df.groupby('model')}
    accs = {}
    for m in modelnames:
        accs[f'other_{m}'] = an.acc_by_skill(None, df_by_model[m][df_by_model[m].dsetname !='mmlu_pro']['correct'], cluster_info=corpus_cluster_info, min_qs_in_cluster=min_qs_in_cluster)[0]
        accs[f'mmlupro_{m}'] = an.acc_by_skill(None, df_by_model[m][df_by_model[m].dsetname == 'mmlu_pro']['correct'], cluster_info=corpus_cluster_info, min_qs_in_cluster=min_qs_in_cluster)[0]

    f, ax = plt.subplots(1,1,figsize=(5,3.5))
    for m in modelnames:
        df2 = pd.DataFrame({key:{k:tup[0] for k,tup in accs[key].items()} for key in [f'other_{m}', 'mmlupro_'+m]})
        df3 = df2.dropna()
        r,p = pearsonr(df3['other_'+m], df3['mmlupro_'+m])
        ax.scatter(df3['other_'+m], df3['mmlupro_'+m], label=f"{_PRETTY_NAMES[m]}\n$r={r:.2f},p={p:.0e}$", alpha=0.75)
    ax.set_xlabel('Skill-wise Accuracies on\nMultimodal Benchmarks', fontsize=14)
    ax.set_ylabel('Skill-wise Accuracies on\nMMLU Pro (Language-only)', fontsize=14)
    ax.legend()
    f.tight_layout();f.savefig(_PLOTS_ROOT +'mmlu_pro_skill_acc_gen_scatter.jpg', bbox_inches='tight',dpi=300)


def adjust_acc(acc, modelname, qs_by_cluster, skill, q_id, full_df_by_model):
    correct = full_df_by_model[modelname].correct.loc[q_id]
    og_num_qs = len(qs_by_cluster[skill])
    new_acc = (acc * og_num_qs - correct) / (og_num_qs - 1)
    return new_acc

def route_per_instance_w_id_dset(min_qs_in_cluster=100):
    dsetnames = ['realworld_qa', 'mmc', 'mme', 'mmbench', 'mmtbench', 'mmmu', 'mmlu_pro', 'seedbench', 'mathvista', 'mmvp', 'mmvet', 'reka_vibe']
    modelnames = ['gpt-4o', 'gemini-1.5-pro', 'claude-sonnet']
    full_df = get_all_preds()
    full_df_by_model = {m:sub_df for m, sub_df in full_df.groupby('model')}

    corpus_cluster_info = get_cluster_info()
    qs_by_cluster = corpus_cluster_info[0]

    cnts_by_skill = {skill:1/len(qs) for skill, qs in qs_by_cluster.items() if len(qs) >= min_qs_in_cluster}
    cnts_df = pd.DataFrame.from_dict(cnts_by_skill, orient='index', columns=['cnt'])

    corpus_skill_accs_by_model_df, corpus_wo_dset_skill_accs_by_model = None, None
    corpus_skill_accs_by_model_df = get_corpus_skill_accs(min_qs_in_cluster, full_df, corpus_cluster_info)

    all_accs = []
    all_out = []
    for dsetname in dsetnames:
        sd, _, _ = an.get_skill_embeddings(dsetname)
        skills_by_q = sd.groupby('question_ind').apply(lambda sub_df: list(set(sub_df.skill)), include_groups=False)
        dsetname_dfs = {m:sub_df2 for m, sub_df2 in full_df[full_df.dsetname==dsetname].groupby('model')}

        out = []
        ctr = 0
        for q_id in tqdm(dsetname_dfs['gpt-4o'].index):
            ind = int(q_id.split('__')[-1])
            try:
                if ind in skills_by_q.index:
                    tmp = corpus_skill_accs_by_model_df.loc[[skill_to_cluster[s] for s in skills_by_q.loc[ind] if skill_to_cluster[s] in corpus_skill_accs_by_model_df.index]]

                    if len(tmp) == 0:
                        selected_model = 'claude-sonnet'
                        ctr += 1
                    else:
                        for m in tmp.columns:
                            tmp[m] = tmp.apply(lambda row: adjust_acc(row[m], m, qs_by_cluster, row.name, q_id, full_df_by_model), axis=1)
                        ### ADDING THIS FOR LOGS2: inverse weight by uniqueness
                        tmp = pd.concat([cnts_df.loc[tmp.index]['cnt'] * tmp[m] for m in modelnames], axis=1)
                        tmp.columns = modelnames
                        
                        tmp = tmp.mean(0).dropna()
                        selected_model = tmp.index[tmp.argmax()]
                        # print(tmp, selected_model)
                else:
                    selected_model = 'claude-sonnet'
                out.append([q_id, selected_model, dsetname_dfs[selected_model].loc[q_id].correct, dsetname])
            except Exception as e:
                print(e)
                break
                # pass
        out_df = pd.DataFrame(out, columns=['q_ind', 'selected_model', 'correct', 'dataset'])
        all_out.append(out_df)
        our_acc = out_df.correct.mean()
        curr_accs = full_df[full_df.dsetname == dsetname].groupby('model').mean('correct')
        curr_accs.loc['ours'] = our_acc
        print(f"Chose default {ctr/len(dsetname_dfs['gpt-4o'].index)*100:.2f}% of the time")
        print('\n'.join([f"{dsetname:<30} accuracy: {acc*100:.2f}, Model: {m:<15}" for m, acc in zip(curr_accs.index, curr_accs['correct'])]))
        curr_accs['dataset'] = dsetname
        all_accs.append(curr_accs.reset_index())
    all_accs_df = pd.concat(all_accs)
    all_out_df = pd.concat(all_out)
    print(all_accs_df.groupby('model').mean('correct'))
    print(all_out_df.correct.mean())

    return all_out_df, all_accs_df

def routing_bars():
    df = pd.read_csv('logs/routing_100__all_dset_accs.csv')
    mmlupro_results = df[df.dataset == 'mmlu_pro']
    mmlupro_results['correct'] = mmlupro_results['correct'] * 100
    
    df2 = pd.read_csv('logs/routing_100__all_outs.csv')
    full_df = get_all_preds()
    tot = full_df.groupby('model').mean('correct')
    tot.loc['ours'] = df2.correct.mean()
    tot = tot * 100

    f, axs = plt.subplots(1,2, figsize=(5,3.5), sharey=True)
    sns.barplot(mmlupro_results, x='correct', y='model', ax=axs[1], hue='model')
    sns.barplot(tot, x='correct', y='model', ax=axs[0], hue='model')

    for ax in axs:
        for c in ax.containers:
            ax.bar_label(c, fmt="%0.2f", label_type='center', color='white')

    axs[0].set_yticklabels(['Claude\nSonnet 3.5', 'Gemini\n1.5 Pro', 'GPT-4o', 'Skill\nRouting'])
    axs[0].set_xlabel('Accuracy over 12 Datasets'); axs[1].set_xlabel('MMLU Pro Accuracy')
    axs[0].set_ylabel('')
    axs[0].set_xlim([60,75]); axs[1].set_xlim([50, 65])
    f.tight_layout();f.savefig(_PLOTS_ROOT +'routing_bars.jpg', dpi=300, bbox_inches='tight')

def routing_table():
    full_df = get_all_preds()
    rand_choice = full_df.groupby(['id', 'dsetname']).mean('correct').groupby('dsetname').mean().reset_index().rename(columns={'dsetname':'dataset'})
    rand_choice['model'] = 'random'

    ensembled = full_df.groupby(['id', 'dsetname']).mean('correct').round().reset_index().groupby('dsetname').mean('correct').reset_index().rename(columns={'dsetname':'dataset'})
    ensembled['model'] = 'ensemble'

    df = pd.read_csv('logs/routing_100__all_dset_accs.csv')
    df3 = pd.concat([df, rand_choice, ensembled])

    dset_cnts = {dsetname:len(sub_df) for dsetname, sub_df in full_df[full_df.model=='gpt-4o'].groupby('dsetname')}
    dset_cnts = dict(sorted(dset_cnts.items(), key=lambda k:-1*k[1]))
    table_str = df3.pivot(index='model', columns=['dataset'])['correct'].loc[modelnames+['random', 'ensemble', 'ours']][list(dset_cnts.keys())].to_latex(float_format="%.2f")

#####################################################################
#### Probing questions analysis
#####################################################################

def gen_probe_qs2():
    ### Takes roughly 1 min to gen 20 (i.e. max_to_gen arg) sets of probe qs for one skill
    modelnames = ['gpt-4o', 'claude-sonnet', 'gemini-1.5-pro']
    if mode == 'multimodal':
        dsetnames = ['mathvista', 'mmc', 'mmtbench', 'mmmu', 'realworld_qa', 'reka_vibe', 'seedbench', 'mmvet','mme'] # 'mmbench', 

    dfs = []
    for modelname in modelnames:
        curr_dfs = []
        for dsetname in dsetnames:
            curr_dfs.append(add_q_id_col(manually_score_df(dsetname, modelname), dsetname))
            curr_dfs[-1]['dsetname'] = dsetname

        df = pd.concat(curr_dfs)
        df['model'] = modelname
        dfs.append(df)

    df = pd.concat(dfs)

    cluster_info = get_cluster_info()
    accs ={m: an.acc_by_skill(None, df[df.model == m][['correct']], cluster_info=cluster_info, min_qs_in_cluster=25)[0] for m in modelnames}
    df2 = pd.DataFrame({m:{k:tup[0] for k,tup in v.items()} for m, v in accs.items()})
    df2['avgs'] = df2.mean(1)
    df2 = df2.sort_values('avgs')

    skills_by_level = {
        'lowest accuracy':list(df2[:20].index),
        'average accuracy':list(df2[len(df2) // 2 - 10 : len(df2) // 2 + 10].index),
        'highest accuracy':list(df2[-20:].index),
    }

    with open('../cached/logs/probed_skills_multimodal.json', 'w') as f:
        json.dump(skills_by_level, f)

    for skills in skills_by_level.values():
        for skill in tqdm(skills):
            annotator.gen_probe_qs_for_skill(skill, qs_by_cluster, skills_by_cluster, skills_df, max_to_gen=20)    

def answer_probe_qs_over_skill_levels(modelname_to_probe):
    with open(_CACHE_ROOT + 'logs/probed_skills_multimodal.json', 'r') as f:
        skills_by_level = json.load(f)

    annotator = SkillAnnotator()
    for level, skills in skills_by_level.items():
        for skill in tqdm(skills):
            try:
                annotator.answer_probe_qs(skill, modelname_to_probe)
            except Exception as e:
                print('\n\n')
                print(f"Failed for skill {skill} with this exception:\n{e}")
            
        print(f'Done with {level} skills')

def parse_consistency_response(x):
    x = str(x).lower()
    x = x.replace('\n', ' ').strip()
    x = x.split(' ')[0] if ' ' in x else x
    if 'true' in x:
        return True
    else:
        return False

def plot_consistency_by_skill_level():
    with open(_CACHE_ROOT+'logs/probed_skills_multimodal.json', 'r') as f:
        skills_by_level = json.load(f)
    modelnames = ['gpt-4o', 'gemini-1.5-pro', 'claude-sonnet']

    accs_df = pd.DataFrame(columns=['level', 'skill', 'model', 'consistency'])
    for level, skills in skills_by_level.items():
        for skill in skills:
            for modelname in modelnames:
                fname = os.path.join(_CACHE_ROOT, 'probe_qs_new', modelname, skill.replace(' ','_').replace('/','_')+'.csv')
                probes_df = pd.read_csv(fname)
                probes_df['consistent'] = probes_df.consistency_response.apply(parse_consistency_response)
                accs_df.loc[len(accs_df)] = [level, skill, modelname, probes_df.consistent.mean()]
    accs_df['inconsistency'] = 1 - accs_df['consistency']

    f, ax = plt.subplots(1,1, figsize=(4,3.75))
    accs_df.level = accs_df.level.apply(lambda x: 'median accuracy' if 'average' in x else x)
    sns.swarmplot(data=accs_df, x='level', y='inconsistency', hue='level', ax=ax, size=3.75)
    avg_rates = accs_df.groupby('level').mean('inconsistency')['inconsistency']
    for i, level in enumerate(['Lowest\nAccuracy', 'Median\nAccuracy', 'Highest\nAccuracy']):
        ax.scatter([],[], color = sns.color_palette()[i], label=f"{avg_rates.loc[level]:.2f}%")

    legend = ax.legend(title='Average Rate of\nInconsistency', fontsize=12, title_fontsize=12)
    legend.get_title().set_ha('center')
    ax.set_xlabel('')
    ax.set_ylabel('Rate of Inconsistency in Probe Answers')
    f.tight_layout(); f.savefig(os.path.join(_PLOTS_ROOT, 'probe_q_inconsistency.jpg'), dpi=300, bbox_inches='tight')

    f, axs = plt.subplots(1,2, figsize=(7,4))
    accs_df.level = accs_df.level.apply(lambda x: x.replace(' ', '\n').title())
    sns.barplot(data=accs_df, x='level', y='consistency', hue='level', ax=axs[0])
    for c in axs[0].containers:
        axs[0].bar_label(c, fmt="%0.2f", label_type='center', color='white')

    sns.swarmplot(data=accs_df, x='level', y='consistency', hue='level', ax=axs[1])
    for ax in axs:
        ax.set_ylabel('Rate of Consistency in Probe Answers')
    f.tight_layout(); f.savefig('test.jpg',dpi=300, bbox_inches='tight')

def plot_inconsistency_slice_acc_scatter():
    with open(_CACHE_ROOT+'logs/probed_skills_multimodal.json', 'r') as f:
        skills_by_level = json.load(f)
    modelnames = ['gpt-4o', 'gemini-1.5-pro', 'claude-sonnet']

    # compute slice accs
    full_df = get_all_preds()
    qs_by_cluster, skills_by_cluster, sd = get_cluster_info()
    sub_qs = {clust:qs for clust, qs in qs_by_cluster.items() if clust in list(accs_df.skill)}
    accs = {m: an.acc_by_skill(None, full_df[full_df.model == m][['correct']], cluster_info=(sub_qs, skills_by_cluster, sd), min_qs_in_cluster=20)[0] 
                for m in modelnames}

    # compute inconsistencies
    accs_df = pd.DataFrame(columns=['level', 'skill', 'model', 'consistency', 'slice_acc'])
    for level, skills in skills_by_level.items():
        for skill in skills:
            for modelname in modelnames:
                fname = os.path.join(_CACHE_ROOT, 'probe_qs_new', modelname, skill.replace(' ','_').replace('/','_')+'.csv')
                probes_df = pd.read_csv(fname)
                probes_df['consistent'] = probes_df.consistency_response.apply(parse_consistency_response)
                accs_df.loc[len(accs_df)] = [level, skill, modelname, probes_df.consistent.mean(), accs[modelname][skill][0]]
    accs_df['inconsistency'] = 1 - accs_df['consistency']

    # plot scatter
    f, axs = plt.subplots(1,3, figsize=(9.5,3), sharey=True)
    for ax, m, color in zip(axs, modelnames, sns.color_palette()[:3]):
        model_accs = accs_df[accs_df.model == m]
        sns.scatterplot(model_accs, x='slice_accs', y='inconsistency', style='level', color=color, ax=ax, legend=False)
        r, p = pearsonr(model_accs['inconsistency'], model_accs['slice_accs'])
        ax.set_title(f"{_PRETTY_NAMES[m]}\n$r={r:.3f}$, $p={p:.1e}$")
        ax.set_xlabel('Skill slice accuracy')
        ax.set_ylabel('Rate of Inconsistency\nover Probe Answers')

    f.tight_layout(); f.savefig(os.path.join(_PLOTS_ROOT, 'consistency_full_scatter.jpg'), dpi=300, bbox_inches='tight')

    ### Save a legend
    f, ax = plt.subplots(1,1, figsize=(6,1))
    handles = [ax.plot([],[], marker=m, label=label, linewidth=0, color='black')[0] for m,label  in zip(['o','x', 's'], ['lowest accuracy', 'average accuracy', 'highest accuracy'])]
    ax.legend(handles=handles, loc='center', ncol=3)
    ax.set_axis_off()
    f.tight_layout; f.savefig(os.path.join(_PLOTS_ROOT, 'consistency_scatter_legend.jpg'), dpi=300, bbox_inches='tight')



#####################################################################
### Functions to compare datasets
#####################################################################
def find_unique_skills(dsetname1, dsetname2, thresh=0.85):
    annotator = SkillAnnotator()

    cnts_by_skill = dict()
    for dsetname in [dsetname1, dsetname2]:
        qs_by_cluster, _, _ = annotator.annotate_skills(dsetname)
        cnts_by_skill[dsetname] = {skill:len(qs) for skill, qs in qs_by_cluster.items()}

    skills1, skills2 = [list(cnts_by_skill[dsetname].keys()) for dsetname in [dsetname1, dsetname2]]
    vecs1, vecs2 = [annotator.embed_strs(skills) for skills in [skills1, skills2]]

    sims = vecs1 @ vecs2.T

    unique_skills = dict()
    for skills, dim, dsetname in zip([skills1, skills2], [1,0], [dsetname1, dsetname2]):
        # We check the similarity of the 
        max_sims, _ = sims.max(dim=dim)
        unique_skills[dsetname] = [skills[i] for i,s in enumerate(max_sims) if s < thresh]
    
    return unique_skills 

def viz_dset_skills():
    sd, sl, sv = an.build_skillset()
    embedding = umap.UMAP().fit_transform(sv)
    skill_to_ind = {s:i for i,s in enumerate(sl)}

    sd['dim1'] = sd.skill.apply(lambda s: embedding[skill_to_ind[s],0])
    sd['dim2'] = sd.skill.apply(lambda s: embedding[skill_to_ind[s],1])

    f, ax = plt.subplots(1,1, figsize=(7,7))
    sns.kdeplot(data=sd2, x='dim1', y='dim2', hue='dsetname', alpha=0.7, ax=ax, fill=True, palette='inferno')

def get_skill_specificity(min_count=5):
    qs_by_cluster, skills_by_cluster, skills_df = get_cluster_info()
    dsetnames = list(set(skills_df.dsetname))
    dsetname_to_ind = {dsetname:ind for ind, dsetname in enumerate(dsetnames)}
    # we count instances for each skill, overall and per dataset
    counts = []
    for skill, qs in qs_by_cluster.items():
        curr_counts = [0]*len(dsetname_to_ind)
        for q in qs:
            curr_counts[dsetname_to_ind[q.split('__')[0]]] += 1
        counts.append([skill, len(qs)] + curr_counts)
    cnts_by_dset = [len(set(skills_df[skills_df.dsetname==dsetname]['id'])) for dsetname in dsetnames]
    all_cnts = [sum(cnts_by_dset)]+cnts_by_dset

    # Now we begin to compute frequencies
    freqs = pd.DataFrame(counts, columns=['skill', 'corpus']+dsetnames)
    freqs = freqs.set_index('skill')
    freqs = freqs[freqs.corpus >= min_count] # only keep sufficiently popular skills
    freqs = freqs / all_cnts # turn them into frequencies within each column
    tfidf_dict = {col: freqs[col] / freqs['corpus'] for col in freqs.columns} # normalize by corpus freq
    tfidf = pd.DataFrame(tfidf_dict)

    skill_to_cluster = {}
    for cluster, skills in skills_by_cluster.items():
        for skill in skills:
            skill_to_cluster[skill] = cluster

    return tfidf, skill_to_cluster


#####################################################################
### Prompt ablation -- Appendix analysis
#####################################################################
def prompt_ablation():
    fs = glob.glob(os.path.join(_CACHE_ROOT, 'model_outputs/simple/gpt-4o/*'))
    dfs = []
    for f in fs:
        df = pd.read_csv(f)
        dsetname = f.split('/')[-1][:-4].replace('SUBSAMPLE__', '')
        df['q_id'] = df['index'].apply(lambda x : f"{dsetname}__{x}")
        dfs.append(df)
    df = pd.concat(dfs)
    df['num_skills'] = df.response.apply(lambda x: len(x.split('\n-')) if '\n-' in x else 0)

    sd, sl, sv = an.build_skillset()
    qs_by_cluster, skills_by_cluster, _ = get_cluster_info()
    qs_by_skill = sd.groupby('skill').apply(lambda sub_df: list(set(sub_df['id'])), include_groups=False)
    skills_by_q = sd.groupby('id').apply(lambda sub_df: list(set(sub_df.skill)), include_groups=False)
    in_both = [q_id for q_id in df.q_id if q_id in list(skills_by_q.index)]
    df = df.set_index('q_id')

    rows = []
    for q_id in in_both:
        rows.append([df.loc[q_id]['num_skills'], 'Direct Prompting'])
        rows.append([len(skills_by_q.loc[q_id]), 'Rationale Parsing'])
    df3 = pd.DataFrame(rows, columns=['\# of Skills per Instance', 'Method'])

    ### Now let's get the # of instances w/ a sufficiently similar skill 
    instance_count_for_skill = lambda sims: len(set(chain(*[qs_by_skill[sl[i]] for i in torch.where(sims >= 0.95)[0]])))

    save_dir = os.path.join(_CACHE_ROOT, 'prompt_ablation')
    simple_relevant_skills_dict_path = os.path.join(save_dir, 'simpl_prompt_cnts_per_skill.pkl')
    if not os.path.exists(simple_relevant_skills_dict_path):
        df['skills'] = df.response.apply(lambda x: [y.replace('- ', '').strip().lower() for y in x.split('\n-')] if '\n- ' in x else [])
        df2 = df.set_index('q_id').loc[in_both]
        df2_sl = []
        for s in df2.skills:
            df2_sl.extend(s)
        df2_sl = list(set(df2_sl))
        df2_sv = an.embed_strs(df2_sl, batch_size=16)
        df2_cnts = [instance_count_for_skill(sv @ vec) for vec in tqdm(df2_sv)]
        with open(simple_relevant_skills_dict_path, 'wb') as f:
            pickle.dump({'sl':df2_sl, 'sv': df2_sv, 'cnts': df2_cnts, 'df': df2}, f)

    our_relevant_skills_dict_path = os.path.join(save_dir, 'our_prompt_cnts_per_skill.pkl')
    if not os.path.exists(our_relevant_skills_dict_path):
        sub_sl = list(set(chain(*list(skills_by_q.loc[in_both]))))
        sub_sv = sv[np.array([sl.index(s) for s in sub_sl])]
    with open(our_relevant_skills_dict_path, 'rb') as f:
        our_relevant_skills_dict = pickle.load(f)
        cnts2 = our_relevant_skills_dict['cnts']

    log_cnts, log_cnts2 = [np.log(x)/np.log(10) for x in [cnts+1, cnts2]]
    df4 = pd.DataFrame([list(log_cnts) + list(log_cnts2), ['Direct Prompting']*len(cnts) + ['Rationale Parsing']*len(cnts2)]).T
    df4.columns = ['Log Counts', 'Method']

    f, axs = plt.subplots(1,2, figsize=(10,3))
    sns.histplot(df3, x="\# of Skills per Instance", hue='Method', bins=30, stat='probability', ax=axs[0], fill=True, kde=True); axs[0].set_ylabel('Fraction of Instances')
    sns.histplot(df4, x="Log Counts", hue="Method", ax=axs[1], bins=25, kde=True, stat='probability', fill=True)
    axs[1].set_xlabel('\# of Relevant Instances per Inferred Skill (log scale)'); axs[1].set_xticklabels(['', 0, 10, 100, 1000, 10000]); axs[1].set_ylabel('Fraction of Inferred Skills')
    f.tight_layout(); f.savefig(os.path.join(_PLOTS_ROOT, 'prompt_ablation.jpg'), bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    # gen_probe_qs_over_skill_levels()
    # answer_probe_qs_over_skill_levels()

    grader_model = _MODELS_DICT['gpt-4o']()
    for dsetname in ['mathvista', 'mmvet', 'reka_vibe']:
        for modelname in ['gpt-4o', 'claude-sonnet', 'gemini-1.5-pro', 'gpt-4o', 'claude-opus']:
            _ = score_df(dsetname, modelname, grader_model)

    # To do my in context experiments, I need to implement functionality to take in multiple images for each of my models
    # infer_w_in_context_egs(modelname='phi3')
    # infer_w_in_context_egs(modelname='phi3', mode='random')

    # import submitit
    # log_folder = '../logs/%j'
    # executor = submitit.AutoExecutor(folder=log_folder)
    # executor.update_parameters(timeout_min=1200)

    # jobs = []
    # with executor.batch():
    #     for modelname_to_probe in ['gpt-4o', 'claude-sonnet', 'gemini-1.5-pro']:
    #         jobs.append(executor.submit(answer_probe_qs_over_skill_levels, modelname_to_probe))