"""Independent numerical and artifact checks after the formal pipeline."""
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from .prepare import dump


def main():
    root=Path(__file__).resolve().parents[1];checks={}
    q=json.loads((root/'outputs/audit/quality_report.json').read_text(encoding='utf-8'))
    s=pd.read_parquet(root/'data/interim/search_sessions_all.parquet')
    assert not s.duplicated(['user_id','search_session_id']).any()
    assert s.result_rows.sum()+q['search_sessions']['missing_key_rows']==q['search_sessions']['input_rows']
    checks['session_keys_and_row_conservation']=True
    v=pd.read_parquet(root/'data/interim/search_sessions.parquet')
    cfg=json.loads((root/'config.json').read_text(encoding='utf-8'))
    complete=v.loc[v.date.between(cfg['complete_date_start'],cfg['complete_date_end'])]
    m=pd.read_csv(root/'outputs/tables/search_source_metrics.csv')
    assert int(m.sessions.sum())==len(complete)
    assert np.allclose(m.session_click_rate+m.no_click_rate,1)
    assert int(m.clicked_sessions.sum())==int(complete.has_click.sum())
    checks['search_table_denominators']=True
    ex=pd.read_csv(root/'outputs/tables/search_exclusions_by_source.csv')
    assert (ex.total-ex.valid).sum()==q['search_sessions']['invalid_sessions']
    joins=pd.read_csv(root/'outputs/audit/join_audit.csv')
    assert (joins.input_rows==joins.output_rows).all()
    assert (joins.users_before==joins.users_after).all()
    checks['join_rows_and_users']=True
    content=pd.read_csv(root/'outputs/tables/content_duration_metrics.csv')
    played=pd.read_csv(root/'outputs/tables/content_played_duration_metrics.csv')
    assert content.rows.sum()==q['content']['complete_valid_rows']
    assert content.played_rows.sum()==played.rows.sum()
    assert (content.capped_play_ratio<=content.mean_play_ratio+1e-12).all()
    checks['content_denominators_and_caps']=True
    ab=json.loads((root/'outputs/tables/synthetic_ab_summary.json').read_text(encoding='utf-8'))
    data=pd.read_csv(root/'data/interim/synthetic_ab_users.csv')
    assert data.user_id.is_unique and not data.isna().any().any()
    assert data.arm.isin([0,1]).all() and data.converted.isin([0,1]).all()
    g=data.groupby('arm').converted.agg(['sum','size'])
    observed=np.array([[g.loc[0,'sum'],g.loc[0,'size']-g.loc[0,'sum']],
                       [g.loc[1,'sum'],g.loc[1,'size']-g.loc[1,'sum']]])
    chi,p,_,expected=stats.chi2_contingency(observed,correction=False)
    assert expected.min()>5 and abs(p-ab['ztest_pvalue'])<1e-10
    d=g.loc[1,'sum']/g.loc[1,'size']-g.loc[0,'sum']/g.loc[0,'size']
    assert abs(d-ab['difference'])<1e-12
    # Newcombe independent Wilson-score interval, without calling statsmodels.
    def wilson(k,n):
        z=stats.norm.ppf(.975);r=k/n;den=1+z*z/n
        center=(r+z*z/(2*n))/den;half=z*math.sqrt(r*(1-r)/n+z*z/(4*n*n))/den
        return center-half,center+half
    p1=g.loc[1,'sum']/g.loc[1,'size'];p0=g.loc[0,'sum']/g.loc[0,'size']
    l1,h1=wilson(g.loc[1,'sum'],g.loc[1,'size']);l0,h0=wilson(g.loc[0,'sum'],g.loc[0,'size'])
    ci=[d-math.sqrt((p1-l1)**2+(h0-p0)**2),d+math.sqrt((h1-p1)**2+(p0-l0)**2)]
    assert np.allclose(ci,ab['ci95_newcombe'],atol=1e-12)
    a=data.loc[data.arm.eq(1),'synthetic_revenue'];b=data.loc[data.arm.eq(0),'synthetic_revenue']
    va=a.var(ddof=1)/len(a);vb=b.var(ddof=1)/len(b)
    t=(a.mean()-b.mean())/math.sqrt(va+vb)
    df=(va+vb)**2/(va*va/(len(a)-1)+vb*vb/(len(b)-1))
    wp=2*stats.t.sf(abs(t),df)
    assert abs(wp-ab['welch_revenue_pvalue_exploratory'])<1e-10
    checks['synthetic_independent_statistics']={'pearson_p':p,'newcombe_ci':ci,'welch_p':wp,'passed':True}
    # Execute each exported Notebook with this interpreter and save rendered outputs.
    import os, tempfile, asyncio, sys
    if sys.platform=='win32':
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',DeprecationWarning)
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    runtime=root/'data/interim/runtime';runtime.mkdir(parents=True,exist_ok=True)
    os.environ['TEMP']=str(runtime);os.environ['TMP']=str(runtime);tempfile.tempdir=str(runtime)
    os.environ['JUPYTER_RUNTIME_DIR']=str(runtime)
    os.environ['IPYTHONDIR']=str(runtime/'ipython')
    import nbformat
    from nbclient import NotebookClient
    import sys
    from jupyter_client.kernelspec import KernelSpecManager
    kernel_dir=root/'data/interim/kernels/python3';kernel_dir.mkdir(parents=True,exist_ok=True)
    (kernel_dir/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Python 3','language':'python'}),encoding='utf-8')
    manager=KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)])
    notebooks=[]
    for path in sorted((root/'notebooks').glob('*.ipynb')):
        nb=nbformat.read(path,as_version=4)
        client=NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(root)}})
        from jupyter_client import AsyncKernelManager
        client.km=AsyncKernelManager(kernel_name='python3',kernel_spec_manager=manager,connection_file=str(runtime/(path.stem+'-kernel.json')))
        client.execute();nbformat.write(nb,path)
        notebooks.append({'file':path.name,'executed':True,'code_cells':sum(c.cell_type=='code' for c in nb.cells)})
    checks['notebooks']=notebooks
    evidence=pd.read_csv(root/'reports/成果证据表.csv')
    assert all((root/f).exists() for f in evidence.file)
    assert all((root/f).exists() for f in evidence.code)
    assert len(list((root/'outputs/figures').glob('*.png')))==4
    checks['evidence_paths_and_four_figures']=True
    checks['passed']=True
    dump(root/'outputs/audit/validation.json',checks)
    print(json.dumps(checks,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
