"""Run the paper-style XGBoost/history comparison on April 2016 NYC DOT data."""
from pathlib import Path
import json, zipfile, hashlib, importlib.metadata, sys

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT/'data'/'april2016.csv.zip'
OUT = ROOT/'april2016_results'
CHUNK = 100_000
PERIOD_DAYS = 28
N_ROADS = 10
SEED = 42
BASE = ['road','weekday','hour']
HISTORY = BASE + ['congestion_5m_ago']
MULTI = BASE + ['congestion_5m_ago','congestion_15m_ago','congestion_30m_ago']
TRENDS = MULTI + ['speed_5m_ago','speed_15m_ago','speed_30m_ago','speed_mean_30m','speed_std_30m']


def read_chunks(stream, chunksize=CHUNK):
    return pd.read_csv(stream, chunksize=chunksize, dtype=str, usecols=['Speed','TravelTime','Status','DataAsOf','linkId'])


def clean(chunk):
    d = pd.DataFrame({'road':chunk['linkId'].astype('string'),
        'time':pd.to_datetime(chunk['DataAsOf'], format='mixed', errors='coerce'),
        'speed':pd.to_numeric(chunk['Speed'],errors='coerce'),
        'travel_time':pd.to_numeric(chunk['TravelTime'],errors='coerce'),
        'status':pd.to_numeric(chunk['Status'],errors='coerce')})
    return d.loc[d.road.notna() & d.time.notna() & d.speed.gt(0) & d.travel_time.gt(0) & d.status.ge(0)]


def aggregate_archive():
    if not ARCHIVE.exists():
        raise FileNotFoundError(ARCHIVE)
    with zipfile.ZipFile(ARCHIVE) as z:
        csv_name = next(n for n in z.namelist() if n.lower().endswith('.csv'))
        with z.open(csv_name) as stream:
            first = next(read_chunks(stream))
            first = clean(first)
            if first.empty:
                raise ValueError('No valid traffic rows in first CSV chunk.')
            # Use frequent sensor links in the initial sample, avoiding use of test targets for selection.
            roads = first.road.value_counts().head(N_ROADS).index.astype(str).tolist()
            start = first.time.min().normalize()
            end = start + pd.Timedelta(days=PERIOD_DAYS)
        parts=[]
        scanned=0
        with z.open(csv_name) as stream:
            for chunk in read_chunks(stream):
                scanned += len(chunk)
                d=clean(chunk)
                d=d.loc[d.road.isin(roads) & d.time.ge(start) & d.time.lt(end)].copy()
                if not d.empty:
                    d['time']=d.time.dt.ceil('5min')
                    parts.append(d.groupby(['road','time'],as_index=False).speed.agg(['sum','count']).reset_index())
        if not parts:
            raise ValueError('No valid rows for selected roads and 28-day period.')
    agg=pd.concat(parts).groupby(['road','time'],as_index=False)[['sum','count']].sum()
    raw=agg.assign(speed=agg['sum']/agg['count'])[['road','time','speed']]
    print(f'Scanned {scanned:,} CSV rows; kept {len(raw):,} five-minute link observations across {len(roads)} links.')
    print(f'Window: {start} through {end}; selected links: {", ".join(roads)}')
    return raw, roads, start, end, scanned


def build_observations(raw):
    # Reindex to a complete five-minute clock per link so shift(1) really
    # means five minutes ago, even when a sensor missed one or more updates.
    grid=pd.date_range(raw.time.min(),raw.time.max(),freq='5min')
    blocks=[]
    for road,g in raw.groupby('road',sort=False):
        g=g.set_index('time').reindex(grid)
        g.index.name='time'
        g['road']=road
        g['weekday']=g.index.dayofweek
        g['hour']=g.index.hour
        for m in [5,15,30]:
            lag=m//5
            g[f'speed_{m}m_ago']=g.speed.shift(lag)
            g[f'congestion_{m}m_ago']=g.Congestion_Level.shift(lag)
        prior=g.speed.shift(1)
        g['speed_mean_30m']=prior.rolling(6,min_periods=4).mean()
        g['speed_std_30m']=prior.rolling(6,min_periods=4).std()
        blocks.append(g.reset_index())
    return pd.concat(blocks,ignore_index=True)


def make_labels(raw, train_end):
    refs=raw.loc[raw.time < train_end].groupby('road').speed.agg(['mean','std'])
    refs['std']=refs['std'].replace(0,np.nan)
    d=raw.copy()
    z=(d.speed-d.road.map(refs['mean']))/d.road.map(refs['std'])
    # Project label definition: z >= 2 clear, [1,2) light, [0,1) moderate,
    # [-1,0) heavy, and below -1 severe. Reference statistics use train only.
    d['Congestion_Level']=np.select([z>=2,z>=1,z>=0,z>=-1],[0,1,2,3],default=4).astype(int)
    return d,refs


def encode(d, cols, road_vocab):
    out=pd.DataFrame({f'road_{i}':(d.road==road).astype(float) for i,road in enumerate(road_vocab)},index=d.index)
    return pd.concat([out,d[[c for c in cols if c!='road']].astype(float)],axis=1)


def accuracy(y,p): return float(np.mean(np.asarray(y)==np.asarray(p)))


def macro_f1(y,p,classes):
    scores=[]
    for c in classes:
        tp=np.sum((np.asarray(y)==c)&(np.asarray(p)==c))
        fp=np.sum((np.asarray(y)!=c)&(np.asarray(p)==c))
        fn=np.sum((np.asarray(y)==c)&(np.asarray(p)!=c))
        pr=tp/(tp+fp) if tp+fp else 0
        re=tp/(tp+fn) if tp+fn else 0
        scores.append(2*pr*re/(pr+re) if pr+re else 0)
    return float(np.mean(scores))


def fit_predict(train, frame, cols, config, classes, roads):
    mapping={int(c):i for i,c in enumerate(classes)}
    y=train.Congestion_Level.map(mapping)
    params=dict(objective='multi:softprob',num_class=len(classes),max_depth=config['max_depth'],
      eta=config['learning_rate'],min_child_weight=config['min_child_weight'],reg_lambda=2,
      subsample=.8,colsample_bytree=.9,tree_method='hist',nthread=2,seed=SEED)
    model=xgb.train(params,xgb.DMatrix(encode(train,cols,roads),label=y),num_boost_round=config['n_estimators'])
    pred=model.predict(xgb.DMatrix(encode(frame,cols,roads))).argmax(axis=1)
    return classes[pred],model


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    raw,roads,start,end,scanned=aggregate_archive()
    distinct=np.sort(raw.time.unique())
    cut1=pd.Timestamp(distinct[int(len(distinct)*.70)])
    cut2=pd.Timestamp(distinct[int(len(distinct)*.85)])
    labelled,refs=make_labels(raw,cut1)
    # Add past label features only after labels have been defined from training-only references.
    features=build_observations(labelled)
    # Require exact previous 5/15/30-minute bins so the variants share rows.
    features=features.dropna(subset=['Congestion_Level','congestion_5m_ago','congestion_15m_ago','congestion_30m_ago',
      'speed_5m_ago','speed_15m_ago','speed_30m_ago','speed_mean_30m','speed_std_30m']).copy()
    train=features[features.time<cut1].copy()
    valid=features[(features.time>=cut1)&(features.time<cut2)].copy()
    test=features[features.time>=cut2].copy()
    classes=np.sort(train.Congestion_Level.unique())
    if len(classes)<2 or min(len(train),len(valid),len(test))<50:
        raise ValueError(f'Insufficient usable data: train/validation/test={len(train)}/{len(valid)}/{len(test)}; classes={classes.tolist()}')
    assert train.time.max()<valid.time.min() and valid.time.max()<test.time.min()
    road_vocab=sorted(train.road.unique())
    configs=[dict(n_estimators=n,max_depth=d,learning_rate=lr,min_child_weight=w)
       for n,d,lr in [(80,2,.05),(160,3,.03),(120,4,.1),(240,2,.03)] for w in [1,3,5]]
    variants=[('XGBoost baseline',BASE),('XGBoost + previous congestion',HISTORY),
      ('XGBoost + 5/15/30m congestion',MULTI),('XGBoost + congestion and speed trends',TRENDS)]
    search=[]; best_models={}
    for name,cols in variants:
        best=None
        for cfg in configs:
            pred,_=fit_predict(train,valid,cols,cfg,classes,road_vocab)
            acc=accuracy(valid.Congestion_Level,pred); f1=macro_f1(valid.Congestion_Level,pred,classes)
            search.append(dict(model=name,**cfg,validation_accuracy=acc,validation_macro_f1=f1))
            if best is None or (acc,f1)>best[:2]: best=(acc,f1,cfg)
        pred,model=fit_predict(train,test,cols,best[2],classes,road_vocab)
        best_models[name]=(pred,model,cols,best)
        print(f'{name}: validation {best[0]*100:.2f}%, test {accuracy(test.Congestion_Level,pred)*100:.2f}%, {best[2]}')
    predmap={name:v[0] for name,v in best_models.items()}
    predmap['Persistence'] = test.congestion_5m_ago.to_numpy(dtype=int)
    predmap['Majority class'] = np.full(len(test),int(train.Congestion_Level.mode().iloc[0]))
    rows=[]
    for name,p in predmap.items():
        rows.append(dict(model=name,accuracy_percent=100*accuracy(test.Congestion_Level,p),
          macro_f1=macro_f1(test.Congestion_Level,p,classes),test_rows=len(test),
          validation_accuracy_percent=100*best_models[name][3][0] if name in best_models else np.nan))
    metrics=pd.DataFrame(rows)
    metrics.to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(search).to_csv(OUT/'validation_search.csv',index=False)
    outputs=test[['road','time','weekday','hour','speed','Congestion_Level','congestion_5m_ago',
       'congestion_15m_ago','congestion_30m_ago']].copy()
    for n,p in predmap.items(): outputs[n]=p
    outputs.to_csv(OUT/'test_predictions.csv',index=False)
    raw.to_csv(OUT/'sampled_traffic.csv',index=False)
    refs.to_csv(OUT/'training_speed_reference.csv')
    meta=dict(source_archive=str(ARCHIVE.name),archive_sha256=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
      scanned_rows=scanned,period_start=str(start),period_end_exclusive=str(end),roads=roads,
      aggregated_observations=len(raw),usable_observations=len(features),
      split_sizes=dict(train=len(train),validation=len(valid),test=len(test)),
      split_ranges={n:[str(x.time.min()),str(x.time.max())] for n,x in [('train',train),('validation',valid),('test',test)]},
      labels={int(c):['clear','light','moderate','heavy','severe'][int(c)] for c in range(5)},
      label_rule='road-specific training mean and standard deviation: z>=2 clear, >=1 light, >=0 moderate, >=-1 heavy, else severe',
      features={n:c for n,c in variants},selected_params={n:v[2] for n,v in best_models.items()},
      selected_by='validation accuracy, macro-F1 tie break',versions={k:importlib.metadata.version(k) for k in ['numpy','pandas','xgboost']},
      note='Exploratory 28-day, 10-link sample. NYC feed fields are speed, status, travel time, timestamp and link ID. Labels are project-derived, not supplied by DOT.')
    (OUT/'run_metadata.json').write_text(json.dumps(meta,indent=2))
    for name,(_,model,_,_) in best_models.items(): model.save_model(OUT/(name.lower().replace(' ','_').replace('/','_')+'.ubj'))
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    bars=ax.barh(metrics.model,metrics.accuracy_percent,color=['#2563eb','#0d9488','#16a34a','#f59e0b','#94a3b8','#64748b'])
    ax.set_xlim(0,100); ax.set_xlabel('Chronological test accuracy (%)'); ax.set_title(f'April 2016 NYC speeds; {len(test):,} test windows')
    for b,v in zip(bars,metrics.accuracy_percent): ax.text(v+0.5,b.get_y()+b.get_height()/2,f'{v:.1f}%',va='center')
    fig.savefig(OUT/'results.png',dpi=160); plt.close(fig)
    print('\n',metrics.to_string(index=False))
    print(f'Usable windows {len(features):,}; splits {len(train):,}/{len(valid):,}/{len(test):,}; output {OUT}')


if __name__=='__main__': run()
