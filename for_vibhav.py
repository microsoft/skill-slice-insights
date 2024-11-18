from validation import verify_skill_relevance, compute_conf_mats
import pandas as pd
from constants import _CACHE_ROOT

try:
    pos_and_neg_skills = pd.read_csv('pos_neg_skills_mmlu_pro_llama32v90b.csv')
except Exception as e:
    print('Failed with exception: ', e)
    print("MAKE SURE TO DOWNLOAD THE FILE MAZDA EMAILED, AND PLACE IT IN THIS DIRECTORY. If you want to place it in another directory, change the path in line 6 accordingly.")
_ = verify_skill_relevance(
    verifier_model_name='gpt-4o', pos_and_neg_skills=pos_and_neg_skills,
    dsetname='mmlu_pro', annotator_model_name='llama32v-chat-90b'
)

df = pd.read_csv(os.path.join(_CACHE_ROOT, 'verification', 'results', 'gpt-4o_verifier', 'llama32v-chat-90b_annotator', 'mmlu_pro.csv'))
cm = compute_conf_mats(df)
print("Confusion matrix for Llama 3.2V 90B inferred skills on mmlu_pro (higher vals on diagonal = better):\n", cm)



fs = glob.glob('direct_prompting__verification_sets/*')
for f in fs:
    try:
        pos_and_neg_skills = pd.read_csv(f)
    except Exception as e:
        print('Failed with exception: ', e)
        print("MAKE SURE TO DOWNLOAD and UNIZP THE FOLDER MAZDA EMAILED, AND PLACE IT IN THIS DIRECTORY.")
        print("Here, you will also need to UNZIP THE FOLDER MAZDA SENT.")
        print("If you want to place it in another directory, change the path in line 6 accordingly.")

    dsetname = f.split('/')[-1].split('.csv')[0]
    _ = verify_skill_relevance(
        verifier_model_name='gpt-4o', pos_and_neg_skills=pos_and_neg_skills,
        dsetname='mmlu_pro', annotator_model_name='direct_prompting_gpt-4o'
    )
    out_df = pd.read_csv(os.path.join(_CACHE_ROOT, 'verification', 'results', 'gpt-4o_verifier', 'direct_prompting_gpt-4o_annotator', dsetname+'.csv'))
    cm = compute_conf_mats(out_df)
    print(f"Dataset: {dsetname}, Confusion matrix:")
    print(cm)
