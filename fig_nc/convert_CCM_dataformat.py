import pandas as pd
import pickle as pkl
from pathlib import Path
root_path = Path(__file__).parents[1]
from sklearn.metrics import roc_auc_score, roc_curve
recon_dict = {}
method = 'FDCCM'
for file in (root_path / 'results/outputfile').glob(f'N10subnet_noise3*_{method:s}.csv'):
    key = file.stem.split('_')[-2]
    print(key)
    data = pd.read_csv(file)
    data['pre_id'] = data['pre_id'] - 1
    data['post_id'] = data['post_id'] - 1
    data.rename(columns={'ste': method, 'log-ste': 'log-'+method}, inplace=True)
    data[method] = data[method].fillna(1e-12)
    data['log-'+method] = data['log-'+method].fillna(-12)
    recon_dict[key] = data
    # print(data.isna().sum())

key_order = ['HHEE', 'HHII', 'HHEI', 'HHconEE', 'HHconII', 'HHconEI', 'Lorenz', 'Lcon', 'Logistic', 'Gaussian']
recon_dict = {key:recon_dict[key] for key in key_order}
#%
auc_list = {
    key: [roc_auc_score(recon_df['connection'], recon_df[method]),]
    for key, recon_df in recon_dict.items()
}
#%
sfx = '_noisy3'
sfx = '' if sfx is None else sfx
save_path = root_path / 'results' / 'N10_subnet_noisy' / method
save_path.mkdir(parents=True, exist_ok=True)
auc_df = pd.DataFrame(auc_list, index=[method]).T
auc_df.to_pickle(save_path / f'auc_list{sfx:s}.pkl')
with open(save_path / f'recon_list{sfx:s}.pkl', 'wb') as f:
    pkl.dump(recon_dict, f)

auc_df