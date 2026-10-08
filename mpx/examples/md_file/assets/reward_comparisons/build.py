"""Compare all available saved reward series; no simulator or training imports."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(__file__).resolve().parent
examples=out.parents[2];root=examples/'checkpoints/TitaJoystickFlatTerrain';analysis=examples/'analysis_tita'
known=[
 ('Flat','PPO single-left best sul flat',root/'20261006_215302/evaluation_flat/evaluation_plots/rollout_info.csv','tab:orange'),
 ('Singolo 1,5 cm','PPO flat best',root/'20261006_211344/evaluation_plots/rollout_info.csv','tab:blue'),
 ('Singolo 1,5 cm','PPO single-left best',root/'20261006_215302/evaluation_single_left/evaluation_plots/rollout_info.csv','tab:orange'),
 ('Doppio 2 cm','PPO single-left best',root/'20261006_215302/evaluation_plots/rollout_info.csv','tab:orange'),
 ('Rough','baseline',analysis/'runs/20261007_rough_baseline_full/rollout.csv','tab:blue'),
 ('Rough','PPO db=0.02',analysis/'runs/20261007_rough_singleleft_d02/rollout.csv','tab:orange'),
 ('Rough','PPO db=0.03',analysis/'runs/20261007_rough_singleleft_d03/rollout.csv','tab:green')]
records=[];terms=set();inventory=[]
def read(f):
 d=pd.read_csv(f);prefix='reward_terms/' if any(c.startswith('reward_terms/') for c in d) else 'reward/'
 ks=[c for c in d if c.startswith(prefix)]
 t=d.time_s.to_numpy() if 'time_s' in d else d.step.to_numpy()*.01
 return d,t,{k[len(prefix):]:d[k].to_numpy() for k in ks}
for group,label,path,color in known:
 d,t,rs=read(path);terms.update(rs);records.append((group,label,path,color,d,t,rs))
 inventory.append(dict(group=group,label=label,path=str(path),t_start=t[0],t_end=t[-1],components=len(rs)))
groups=['Flat','Singolo 1,5 cm','Doppio 2 cm','Rough']
for term in sorted(terms):
 fig,axs=plt.subplots(2,2,figsize=(14,8))
 for ax,group in zip(axs.flat,groups):
  for g,l,p,c,d,t,rs in records:
   if g==group and term in rs:ax.plot(t,rs[term],label=l,color=c)
  ax.set(title=group,xlabel='Time [s]',ylabel=term+' (weighted, before dt)');ax.grid(alpha=.3);ax.legend(fontsize=8)
 fig.tight_layout();fig.savefig(out/(term+'.png'),dpi=135);plt.close(fig)
# Contributions per episode: useful but episode durations differ.
contrib=[]
for g,l,p,c,d,t,rs in records:
 row={'scenario':g,'policy':l,'duration_s':t[-1]}
 # Initial state components at t=0 are diagnostics, not a transition reward.
 mask=t>0
 row.update({k:.01*np.sum(v[mask]) for k,v in rs.items()});contrib.append(row)
pd.DataFrame(contrib).to_csv(out/'episode_components.csv',index=False)
pd.DataFrame(inventory).to_csv(out/'rollout_inventory.csv',index=False)
# All training histories, one panel each; do not compare differently scaled returns blindly.
histories=[]
for p in sorted(root.glob('20261006_*')):
 f=p/'metrics_log.csv'
 if f.exists():histories.append((p.name,pd.read_csv(f)))
fig,axs=plt.subplots((len(histories)+2)//3,3,figsize=(16,3.2*((len(histories)+2)//3)))
for ax,(name,d) in zip(axs.flat,histories):
 ax.plot(d.num_steps/1e6,d.episode_reward,'o-',ms=3);ax.set(title=name,xlabel='Transitions [million]',ylabel='Eval return');ax.grid(alpha=.3)
for ax in list(axs.flat)[len(histories):]:ax.set_visible(False)
fig.tight_layout();fig.savefig(out/'all_training_returns.png',dpi=140);plt.close(fig)
# Every older available analysis CSV and unidentified older checkpoint eval gets an individual grid.
archive=list(analysis.glob('runs/20261005*/**/rollout.csv'))+list(analysis.glob('policy_rollouts/**/rollout.csv'))+[root/'20261006_195219/evaluation_plots/rollout_info.csv']
archive_lines=[]
for ix,f in enumerate(sorted(archive)):
 d,t,rs=read(f)
 if not rs:continue
 fig,axs=plt.subplots((len(rs)+2)//3,3,figsize=(15,2.8*((len(rs)+2)//3)))
 for ax,(term,v) in zip(axs.flat,sorted(rs.items())):ax.plot(t,v);ax.set(title=term,xlabel='Time [s]');ax.grid(alpha=.3)
 for ax in list(axs.flat)[len(rs):]:ax.set_visible(False)
 label=str(f.relative_to(examples));fig.suptitle(label,fontsize=9);fig.tight_layout();name=f'archive_{ix:02d}.png';fig.savefig(out/name,dpi=120);plt.close(fig)
 archive_lines.append(f'### {label}\n\nT={t[0]:.2f}–{t[-1]:.2f} s. Componenti pesate originali; configurazioni storiche non necessariamente confrontabili.\n\n![Reward]({name})\n')
(out/'archive.md').write_text('# Tutti gli altri rollout salvati: reward originali\n\n'+ '\n'.join(archive_lines))
run_labels={'20261006_205235':'Flat, db 0,01','20261006_211344':'Flat, db 0,02','20261006_214632':'Singolo, resume con warmup','20261006_215302':'Singolo, resume senza warmup','20261006_221648':'Doppio 1,5 cm PPO','20261006_223038':'Doppio 2 cm PPO','20261006_224230':'Doppio 2 cm SAC'}
section='''## Grafici delle singole reward — confronto delle prove salvate

Aggiornato il 7 ottobre 2026, solo dai dati esistenti: nessun nuovo training o rollout. Questi sono i **contributi realmente registrati con gli scaling di ogni prova**, non reward attivate retroattivamente.

Ogni figura confronta lo stesso termine in quattro pannelli: flat, singolo, doppio e rough. Le reward disattivate sono correttamente piatte a zero: non vengono nascoste. Il confronto dei nuovi scaling balance ricalcolati sugli stessi stati è separato nello [studio rough](../analysis_tita/rough_reward_study.md).

**Limiti dei dati:** il best del training flat non ha qui un rollout flat identificato: il suo CSV di evaluation è stato sovrascritto dalla valutazione sul singolo. Il flat disponibile è il best single-left rivalutato senza ostacoli. Sul doppio il CSV disponibile è la policy single-left su 2 cm, non il best dei successivi fine-tuning PPO/SAC. Per questi run sono conservati i totali delle eval, non le componenti temporali. Nessuna curva mancante è ricostruita o attribuita a un altro checkpoint.

I CSV `rollout_info.csv` iniziano al primo step, **t=0,01 s**; i CSV rough includono t=0. Non aggiungo uno zero artificiale. Ordinata: componente pesata prima di dt=0,01; il costo terminale −500 equivale a −5 nel ritorno del passo. Durate diverse non implicano qualità migliore: sul rough tutte le prove cadono.

### Dati effettivamente confrontati

| Scenario | Policy / controllo | Return somma componenti | Durata |
|---|---|---:|---:|
'''
for row in contrib:
 total=sum(v for k,v in row.items() if k not in ['scenario','policy','duration_s'])
 section+=f"| {row['scenario']} | {row['policy']} | {total:.6f} | {row['duration_s']:.2f} s |\n"
section+='\n[CSV con contributi cumulati per termine](assets/reward_comparisons/episode_components.csv) · [Inventario delle sorgenti](assets/reward_comparisons/rollout_inventory.csv)\n\n'
for term in sorted(terms):section+=f'### {term}\n\n![{term}](assets/reward_comparisons/{term}.png)\n\n'
section+='''### Tutti i run: andamento della reward totale nelle eval

I valori sono quelli originali. I run iniziali avevano guadagni e/o reward differenti; non confrontare valori assoluti di configurazioni diverse come se misurassero lo stesso obiettivo.

![Tutti i training](assets/reward_comparisons/all_training_returns.png)

| Run | Scenario documentato | Eval salvate | Best return | Ultimo return |
|---|---|---:|---:|---:|
'''
for name,d in histories:section+=f"| {name} | {run_labels.get(name,'Setup precedente: vedi log del run')} | {len(d)} | {d.episode_reward.max():.6f} | {d.episode_reward.iloc[-1]:.6f} |\n"
section+='\n### Archivio completo delle altre prove\n\n[Grafici di tutte le componenti degli altri '+str(len(archive_lines))+' rollout salvati](assets/reward_comparisons/archive.md), inclusi vecchia baseline con/senza ostacoli e prime policy SAC. Sono tenuti separati perché cambiano controller, scaling e durata.\n\n'
p=examples/'md_file/residual_training_setup_report.md'
s=p.read_text();marker='<!-- SAVED_REWARD_COMPARISON -->';end='<!-- END_SAVED_REWARD_COMPARISON -->'
if marker in s:s=s[:s.index(marker)]+s[s.index(end)+len(end):]
first=s.index('\n')+1;s=s[:first]+'\n'+marker+'\n'+section+end+'\n'+s[first:];p.write_text(s)
print('Updated',p,'reward terms',len(terms),'training runs',len(histories),'archive rollouts',len(archive_lines))
