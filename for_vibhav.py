from validation import verify_skill_relevance, compute_conf_mats
import pandas as pd
from constants import _CACHE_ROOT
from dsets import _DSET_DICT


for dsetname in ['mme', 'mmvet']:
    try:
        pos_and_neg_skills = pd.read_csv(f'pos_neg_skills_{dsetname}.csv')
    except Exception as e:
        print('Failed with exception: ', e)
        print("MAKE SURE TO DOWNLOAD THE FILES MAZDA EMAILED, AND PLACE IT IN THIS DIRECTORY. If you want to place it in another directory, change the path in line 6 accordingly.")
    _ = verify_skill_relevance(
        verifier_model_name='claude-sonnet', pos_and_neg_skills=pos_and_neg_skills,
        dsetname='mmlu_pro', annotator_model_name='gpt-4o'
    )

    df = pd.read_csv(os.path.join(_CACHE_ROOT, 'verification', 'results', 'claude-sonnet_verifier', 'gpt-4o_annotator', 'mmlu_pro.csv'))
    cm = compute_conf_mats(df)
    print(f"Confusion matrix for GPT-4o inferred skills on {dsetname} (higher vals on diagonal = better):\n", cm)



# fs = glob.glob('direct_prompting__verification_sets/*')
# for f in fs:
#     try:
#         pos_and_neg_skills = pd.read_csv(f)
#     except Exception as e:
#         print('Failed with exception: ', e)
#         print("MAKE SURE TO DOWNLOAD and UNIZP THE FOLDER MAZDA EMAILED, AND PLACE IT IN THIS DIRECTORY.")
#         print("Here, you will also need to UNZIP THE FOLDER MAZDA SENT.")
#         print("If you want to place it in another directory, change the path in line 6 accordingly.")

#     dsetname = f.split('/')[-1].split('.csv')[0]
#     if dsetname not in _DSET_DICT:
#         continue
#     _ = verify_skill_relevance(
#         verifier_model_name='gpt-4o', pos_and_neg_skills=pos_and_neg_skills,
#         dsetname=dsetname, annotator_model_name='direct_prompting_gpt-4o'
#     )
#     out_df = pd.read_csv(os.path.join(_CACHE_ROOT, 'verification', 'results', 'gpt-4o_verifier', 'direct_prompting_gpt-4o_annotator', dsetname+'.csv'))
#     cm = compute_conf_mats(out_df)
#     print(f"Dataset: {dsetname}, Confusion matrix:")
#     print(cm)

#     print("USING CLAUDE AS VERIFIER")
#     _ = verify_skill_relevance(
#         verifier_model_name='claude-sonnet', pos_and_neg_skills=pos_and_neg_skills,
#         dsetname=dsetname, annotator_model_name='direct_prompting_gpt-4o'
#     )
#     out_df = pd.read_csv(os.path.join(_CACHE_ROOT, 'verification', 'results', 'gpt-4o_verifier', 'direct_prompting_gpt-4o_annotator', dsetname+'.csv'))
#     cm = compute_conf_mats(out_df)
#     print(f"Dataset: {dsetname}, Confusion matrix:")
#     print(cm)
