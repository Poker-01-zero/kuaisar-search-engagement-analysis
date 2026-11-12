from pathlib import Path
import json
import pandas as pd

root = Path(__file__).resolve().parent
for name in ['src_inter.csv','rec_inter.csv','user_features.csv','item_features.csv']:
    matches = list((root/'data/raw').rglob(name))
    if not matches:
        print('Pending:',name)
        continue
    path = matches[0]
    x = pd.read_csv(path,nrows=5)
    print('\nFILE',name,'BYTES',path.stat().st_size)
    print(json.dumps({'columns':list(x.columns),'dtypes':x.dtypes.astype(str).to_dict(),
                      'examples':x.drop(columns=[c for c in ['keyword','caption'] if c in x]).head(2).to_dict('records')},ensure_ascii=False))
