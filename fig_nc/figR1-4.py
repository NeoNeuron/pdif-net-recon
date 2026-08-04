# statistical analysis for the causal reconstruction results for all Visual Coding data 
#%%
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from figrc import *
# set True to postprocess the shuffle-ablation results instead of the real-data results
shuffle_toggle = [False, True]
fig_sfxs = [
    "ref=5-gap=250-sfx=0-K=1_5-bin=1.00",
    "ref=5-gap=250-sfx=0-K=1_5-bin=1.00-delay=0.00-shuffle",
]
#%%
dfs = []
for shuffle_toggle_, fig_sfxs_ in zip(shuffle_toggle, fig_sfxs):
    df = {'session_id':[],
        'consistency':[],
        'average_auc':[],
        }
    for out_dir in Path(root/'visualcoding').iterdir():
        if not out_dir.is_dir():
            continue
        session_id = int(out_dir.stem)
        df['session_id'].append(session_id)
        units = pd.read_pickle(out_dir / f"units.pkl")
        n_unit = len(units)
        # with open(out_dir/f'allen-data-{fig_suffix}-delay=0.00.pkl', 'rb') as f:
        with open(out_dir/f'allen-data-{fig_sfxs_}.pkl', 'rb') as f:
            data_fig_all = pickle.load(f)

        df['consistency'].append(data_fig_all['consistency']['TE'].min())
        df['average_auc'].append(
            (data_fig_all['drifting_gratings']['auc_gauss']['TE']
            + data_fig_all['static_gratings']['auc_gauss']['TE']
            + data_fig_all['natural_scenes']['auc_gauss']['TE']
            + data_fig_all['natural_movie']['auc_gauss']['TE']
            )/4)
    df = pd.DataFrame(df)
    df['shuffle_toggle'] = shuffle_toggle_
    dfs.append(df)
df_visualcoding = pd.concat(dfs)
df_visualcoding.head()
#%%
# set True to postprocess the shuffle-ablation results instead of the real-data results
shuffle_toggle = [False, True]
fig_sfxs = ["ref=5-gap=250-sfx=250-K=1_5-bin=1.00",
             "ref=5-gap=250-sfx=250-K=1_5-bin=1.00-delay=0.00-shuffle"]
#%%
dfs = []
for shuffle_toggle_, fig_sfxs_ in zip(shuffle_toggle, fig_sfxs):
    df = {'session_id':[], 'consistency':[], 'average_auc':[]}
    for out_dir in Path(root/'visualbehavior').iterdir():
        if not out_dir.is_dir():
            continue
        session_id = int(out_dir.stem)
        df['session_id'].append(session_id)
        units = pd.read_pickle(out_dir / f"units.pkl")
        n_unit = len(units)
        # results using fixed delay 0.00ms
        # with open(out_dir/f'allen-data-{fig_suffix}-delay=0.00.pkl', 'rb') as f:
        # results using optimal delay
        with open(out_dir/f'allen-data-{fig_sfxs_}.pkl', 'rb') as f:
            data_fig_all = pickle.load(f)

        df['consistency'].append(data_fig_all['consistency']['TE'][1,0])
        # df['consistency_binary'].append(data_fig_all['consistency_binary']['TE'][1,0])
        # df['inconsistent_ratio'].append(data_fig_all['active']['hist_inconsist']['TE'].sum()*(np.diff(data_fig_all['active']['edges']['TE'])[0]))
        df['average_auc'].append(
            (data_fig_all['active']['auc_gauss']['TE']
            + data_fig_all['passive']['auc_gauss']['TE'])/2)
    df = pd.DataFrame(df)
    df['shuffle_toggle'] = shuffle_toggle_
    dfs.append(df)
df_visualbehavior = pd.concat(dfs)
df_visualbehavior.head()
# %%
fig, ax = plt.subplots(figsize=(3,6))
sns.violinplot(df_visualcoding, x=1, y='consistency', ax=ax, cut=0, hue='shuffle_toggle', split=True, legend=False)
sns.violinplot(df_visualbehavior, x=2, y='consistency', ax=ax, cut=0, hue='shuffle_toggle', split=True, legend=False)

for label, values in [
    ('Visual Coding (shuffle=False)', df_visualcoding.loc[df_visualcoding['shuffle_toggle'] == False, 'consistency'].dropna()),
    ('Visual Coding (shuffle=True)', df_visualcoding.loc[df_visualcoding['shuffle_toggle'] == True, 'consistency'].dropna()),
    ('Visual Behavior (shuffle=False)', df_visualbehavior.loc[df_visualbehavior['shuffle_toggle'] == False, 'consistency'].dropna()),
    ('Visual Behavior (shuffle=True)', df_visualbehavior.loc[df_visualbehavior['shuffle_toggle'] == True, 'consistency'].dropna()),
]:
    print(f'{label}: mean={values.mean():.4f}')

ax.set_ylabel('Minimum Consistency', fontsize=20)
ax.set_ylim(0.3,1)
ax.set_xticks([0,1])
ax.set_xticklabels(['Visual\nCoding', 'Visual\nBehavior'], fontsize=16)
ax.tick_params(axis='y', labelsize=14)

# statistical tests between shuffle and non-shuffle for each dataset (visualcoding at x=1, visualbehavior at x=2)
from scipy.stats import permutation_test

def significance_label(p):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'

# compute p-values
vc_false = df_visualcoding[df_visualcoding['shuffle_toggle']==False]['consistency']
vc_true = df_visualcoding[df_visualcoding['shuffle_toggle']==True]['consistency']
vb_false = df_visualbehavior[df_visualbehavior['shuffle_toggle']==False]['consistency']
vb_true = df_visualbehavior[df_visualbehavior['shuffle_toggle']==True]['consistency']

# remove NaNs for permutation test
vc_false = vc_false.dropna()
vc_true = vc_true.dropna()
vb_false = vb_false.dropna()
vb_true = vb_true.dropna()

# permutation tests (difference in means)
pv_vc = permutation_test(
    (vc_false, vc_true),
    statistic=lambda x, y: x.mean() - y.mean(),
    permutation_type='independent',
    n_resamples=10000,
    alternative='two-sided'
).pvalue
pv_vb = permutation_test(
    (vb_false, vb_true),
    statistic=lambda x, y: x.mean() - y.mean(),
    permutation_type='independent',
    n_resamples=10000,
    alternative='two-sided'
).pvalue

# positions of the two categorical violins
xt = ax.get_xticks()
if len(xt) < 2:
    xt = [0,1]

# draw significance bars
def draw_sig(ax, x, y, text):
    dx = 0.12
    ax.plot([x-dx, x+dx], [y, y], color='k', linewidth=1.2)
    ax.plot([x-dx, x-dx], [y, y-0.02*(ax.get_ylim()[1]-ax.get_ylim()[0])], color='k', linewidth=1.2)
    ax.plot([x+dx, x+dx], [y, y-0.02*(ax.get_ylim()[1]-ax.get_ylim()[0])], color='k', linewidth=1.2)
    ax.text(x, y+0.01*(ax.get_ylim()[1]-ax.get_ylim()[0]), text, ha='center', va='bottom', fontsize=12)

ymax = ax.get_ylim()[1]
draw_sig(ax, xt[0], ymax - 0.02*(ymax-ax.get_ylim()[0]), significance_label(pv_vc))
draw_sig(ax, xt[1], ymax - 0.12*(ymax-ax.get_ylim()[0]), significance_label(pv_vb))

fig.savefig(root / 'fig_nc/pdf' / 'figR1-4.pdf')
# %%