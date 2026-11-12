"""Generate figures and evidence-linked Chinese delivery documents from actual outputs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def md_table(x):
    # Avoid optional tabulate dependency.
    x=x.copy().fillna('NA')
    def fmt(v):return f'{v:.6f}' if isinstance(v,float) else str(v).replace('|','/')
    return '| '+' | '.join(map(str,x.columns))+' |\n| '+' | '.join(['---']*len(x.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in x.itertuples(index=False,name=None))


def deliver(root,cfg,q,manifest,st,ct,ab):
    reports=root/'reports';figs=root/'outputs/figures'
    def write(name,text): (reports/name).write_text(text.strip()+'\n',encoding='utf-8')
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'figure.dpi':130,'savefig.dpi':160})
    m=st['search_source_metrics']; top=m.query('display_eligible').head(2); a,b=top.iloc[0],top.iloc[1]
    gap=(a.session_click_rate-b.session_click_rate)*100
    std=st['search_standardized'].set_index('source');sgap=(std.loc[a.source,'standardized_click_rate']-std.loc[b.source,'standardized_click_rate'])*100
    daily=st['search_daily'].pivot(index='date',columns='source',values='session_click_rate')
    complete_daily=daily.loc[cfg['complete_date_start']:cfg['complete_date_end']]
    same=int(((complete_daily[a.source]-complete_daily[b.source])*gap>0).sum())
    n=sessions=int(q['search_sessions']['total_sessions']-q['search_sessions']['invalid_sessions'])
    users=int(q['search_users']['valid_search_users']); raw=sum(z['rows'] for z in manifest['files'] if z['file'] in ['src_inter.csv','rec_inter.csv'])
    c=ct['content_duration_metrics']; cmin=c.loc[c.reaches_duration_rate.idxmin()];cmax=c.loc[c.reaches_duration_rate.idxmax()]
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    axes[0].barh(m.source,m.sessions,color='#3c83a5');axes[0].set_xlabel('完整日有效会话数（次）');axes[0].invert_yaxis()
    axes[1].barh(m.source,m.session_click_rate,color='#348a75');axes[1].xaxis.set_major_formatter(PercentFormatter(1));axes[1].set_xlabel('至少一条点击结果的会话占比');axes[1].invert_yaxis()
    fig.suptitle(f'KuaiSAR-small 样本内搜索入口｜{cfg["complete_date_start"]}—{cfg["complete_date_end"]}\n分母：{int(m.sessions.sum()):,} 个合法会话；各来源用户可能重叠',fontsize=11)
    fig.savefig(figs/'01_search_sources.png');plt.close(fig)
    mix=st['search_source_itemtype']; mix=mix.pivot(index='source',columns='type_mix',values='sessions').reindex(m.source).fillna(0)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    mix.div(mix.sum(axis=1),axis=0).plot.barh(stacked=True,ax=axes[0],color=['#9bb7d0','#efbb70','#448c82','#a29ec0'][:len(mix.columns)])
    axes[0].xaxis.set_major_formatter(PercentFormatter(1));axes[0].set_xlabel('互斥结果构成的会话占比');axes[0].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.16),ncol=2);axes[0].set_ylabel('')
    for col,label in [('session_click_rate','会话加权'),('user_day_equal_click_rate','用户日等权'),('user_equal_click_rate','用户等权')]:axes[1].plot(m[col],m.source,'o',label=label)
    axes[1].xaxis.set_major_formatter(PercentFormatter(1));axes[1].legend(fontsize=8);axes[1].set_xlabel('会话点击率（按标示权重）')
    fig.suptitle(f'样本内结构与权重对照｜完整日 n={int(m.sessions.sum()):,} 会话',fontsize=11)
    fig.savefig(figs/'02_search_structure.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained')
    x=np.arange(len(c));axes[0].bar(x,c.rows,color='#3c83a5');axes[0].set_ylabel('合法时长分发记录数（含零播放）')
    axes[1].plot(x,c.reaches_duration_rate,'o-',label='记录加权');axes[1].plot(x,c.user_day_equal_reaches_rate,'s--',label='用户日等权');axes[1].yaxis.set_major_formatter(PercentFormatter(1));axes[1].set_ylabel('累计播放达时长比例');axes[1].legend()
    for ax in axes:ax.set_xticks(x,c.duration_band);ax.set_xlabel('内容时长')
    fig.suptitle(f'合法时长内容记录，未完全识别媒体类型｜完整日 n={int(c.rows.sum()):,}\n推荐发布样本存在反馈采样；不解释为平台完播率',fontsize=11)
    fig.savefig(figs/'03_content_duration.png');plt.close(fig)
    cat=ct['content_category_duration'];names=cat.groupby('category').rows.sum().nlargest(8).index
    heat=cat.loc[cat.category.isin(names)].pivot(index='category',columns='duration_band',values='reaches_duration_rate').reindex(names)
    counts=cat.loc[cat.category.isin(names)].pivot(index='category',columns='duration_band',values='rows').reindex_like(heat)
    fig,ax=plt.subplots(figsize=(10,6),layout='constrained');im=ax.imshow(heat.to_numpy(),vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
    ax.set_yticks(range(len(heat)),heat.index);ax.set_xticks(range(len(heat.columns)),heat.columns)
    for i in range(len(heat)):
        for j in range(len(heat.columns)):
            if pd.notna(heat.iloc[i,j]):ax.text(j,i,f'{heat.iloc[i,j]:.1%}\nn={int(counts.iloc[i,j]):,}',ha='center',va='center',fontsize=8,color='white' if heat.iloc[i,j]>.55 else 'black')
    fig.colorbar(im,ax=ax,format=PercentFormatter(1),label='累计播放达时长比例')
    ax.set_title('样本记录量前8类目 × 时长｜合法时长记录（含零播放）\n完整日；每格显示记录分母；类目英文沿用发布文件',fontsize=11)
    fig.savefig(figs/'04_content_categories.png');plt.close(fig)
    finding=f'完整日中，按会话量选择的 {a.source} 与 {b.source} 分别有 {int(a.sessions):,}、{int(b.sessions):,} 个会话，点击率 {a.session_click_rate:.2%}、{b.session_click_rate:.2%}，前者减后者为 {gap:.2f} 个百分点。共同日期×互斥结果类型×公开搜索活跃层级标准化后，差异为 {sgap:.2f} 个百分点。'
    sensitivity=st['search_positive_click_sensitivity'].set_index('source')
    altgap=(sensitivity.loc[a.source,'session_click_rate']-sensitivity.loc[b.source,'session_click_rate'])*100
    finding+=f" 严格口径因click_cnt非二元而隔离 {q['click_contract']['strict_excluded_sessions']:,} 个会话；若仅在独立敏感性情形中假定正整数表示有点击，两来源差异为 {altgap:.2f} 个百分点。主结果仅适用于符合官方二元字段约定的会话，排除会话可能带来选择偏差。"
    content_finding=f'合法时长内容记录中，{cmax.duration_band} 组累计播放达时长比例为 {cmax.reaches_duration_rate:.2%}，{cmin.duration_band} 组为 {cmin.reaches_duration_rate:.2%}。时长与结果明显相关，类目对照必须放在同一时长层内。'
    limits='本项目是公开选择样本的观察分析；无点击不等于不满意，自动播放可能满足需求。来源编码仅沿用原值；用户意图、查询难度与排序位置未充分观测。推荐点击正负样本经过构建，不估算平台CTR或增长空间。内容媒体类型未完整识别，累计播放达时长也不等于一次连续完播。匿名特征不解读成人口属性。'
    intro=f'''# KuaiSAR 短视频搜索入口与内容消费分析

基于公开行为日志，对比搜索入口点击率，并分析相近时长内容的观看表现。项目使用 AI 辅助实现。采用 KuaiSAR-small 正式ZIP，MD5 `{manifest['archive_md5']}`。全量扫描和分析两张行为表共 **{raw:,} 条记录**；用户维表 **{manifest['selected_users']:,} 人**；合法搜索会话 **{n:,} 个**，来自 **{users:,} 人**。没有用户抽样。完整日为 {cfg['complete_date_start']} 至 {cfg['complete_date_end']}，首尾采集日另列。

{finding}

{content_finding}

## 运行

```powershell
python -m pip install -r requirements.txt
python download_data.py
python -m unittest discover -s tests -v
python -m src.run --config config.json
python -m src.validate
```

从本目录运行。首次下载支持校验HTTP Range续传；已有ZIP每次也检查MD5。统一入口分块读取原始字段并审计，然后合并必要列后聚合会话；没有自称流式会话聚合。原始数据和用户级中间表不入Git。配置参数与运行版本见 `data/manifest.json`；生产处理耗时与进程峰值见 `outputs/audit/quality_report.json`，未测性能提升。

## 交付导航

- `reports/项目总结.md`：证据、结构对照、内容专题、限制与行动。
- `reports/决策摘要.md`：一页决策表。
- `reports/实验分析练习.md`：明确标注模拟的A/B案例，未实施产品实验。
- `reports/数据字典与指标口径.md`、`reports/成果证据表.csv`：数字与来源。
- [分析流程与代码导读](reports/代码导读.md)。
- `notebooks/01_real_analysis.ipynb` 与 `04_ab_statistics_practice.ipynb`：读取正式结果的展示Notebook。

## 边界

{limits}

## 来源与许可

数据：[Zenodo](https://zenodo.org/records/8181109)；[官网](https://kuaisar.github.io/)；[字段说明](https://ethan00si.github.io/KuaiSAR/detailed_statistics.html)。引用 Sun et al., KuaiSAR: A Unified Search And Recommendation Dataset, CIKM 2023，[论文](https://arxiv.org/abs/2306.07705)。中文类目部分在源文件内损坏，使用ID与英文标签。原始数据许可遵循发布者 [使用说明](https://kuaisar.github.io/)（CC BY-NC-SA 4.0，署名、非商业、相同方式共享）；不重新分发原始数据、用户级明细或将数据许可套用为代码许可。代码未另行授予开源许可证。个人简历放在项目目录之外，不纳入公开成果。
'''
    (root/'README.md').write_text(intro,encoding='utf-8')
    write('决策摘要.md',f'''# 决策摘要（样本内）

业务问题：优先核查哪些有规模的搜索来源差异？按会话量选前两大合格来源，避免追逐小样本极端值。

|现象与差异（百分点）|涉及规模|核验后是否保持|替代解释|下一步|
|---|---|---|---|---|
|{a.source}−{b.source}：{gap:.2f}pp|{int(a.sessions):,} / {int(b.sessions):,} 会话；{int(a.users):,} / {int(b.users):,} 用户（可重叠）|共同层标准化 {sgap:.2f}pp；完整日 {same}/{len(complete_daily)} 天同方向|意图、自动播放、结果构成、排序位置与用户选择|先核对入口编码及自动播放、满意反馈埋点；保留点击与满意度独立指标，再决定是否实验|
|时长组达时长比例存在差异|合法时长完整日 {int(c.rows.sum()):,} 记录|已输出用户日等权与同类目同长度组对照|长内容阈值更难达到、重复播放、样本反馈构建|核查分发与播放标记，不能直接按比例调配流量|

{limits}
''')
    write('项目总结.md',f'''# 项目总结

## 一、问题、范围与主要结论

目标是在样本内筛出值得核查的搜索差异，补充内容消费描述，并形成可复现的求职作品。{finding}

{content_finding}

两张行为表共 {raw:,} 行；用户维表 {manifest['selected_users']:,} 人。合法搜索会话 {n:,} 个，{q['search_sessions']['invalid_sessions']:,} 个会话被排除；缺键行 {q['search_sessions']['missing_key_rows']:,}。搜索完整日会话 {int(m.sessions.sum()):,} 个。维表人数、两张日志去重人数、合法搜索人数是不同口径；未在该阶段获得合法搜索的维表用户数为 {q['search_users']['no_valid_search_users']:,}。

实际文件规模、原始字段缺失、重复哈希、时间核对分别见 `outputs/audit/data_inventory.csv`、`field_profile.csv`、`quality_report.json`。哈希诊断保留完全相同记录，不据此删除。用户维度真实字段 `reco_active_level` 与网页拼写不同；脱敏特征的取值也不强套网页示例范围。

## 二、搜索粒度与入口概览

搜索结果行不是搜索次数。会话键为 `(user_id, search_session_id)`，有至少一条合法点击即计入点击会话。任一行点击不合法、来源缺失／冲突或时间缺失／冲突时隔离整个会话；排除会话保存在审计表。合法会话保存在Parquet，用户日表单独保存。

{md_table(m[['source','users','sessions','session_share','session_click_rate','no_click_rate']])}

![入口规模与反馈](../outputs/figures/01_search_sources.png)

正式比较门槛固定为100会话和30用户，仅用于显示规模判断，不代表显著性。小来源全部保留；分组人数不可求和当总人数。完整日以外的数据见 `search_source_all_dates.csv` 和 `search_daily.csv`。

## 三、结构、权重与替代解释

{finding} 选择两大来源是因为规模充分，且避免事后挑最好最差的极端值。

会话互斥分为纯VIDEO、含VIDEO混合、不含VIDEO和未知，未知类型不强制归入视频。行级类型关联率另存审计表，不与会话率混用。

{md_table(st['search_weighting_sensitivity'])}

用户日等权先算每个来源×用户×日的点击会话比例，再平均。活跃天多的用户仍可贡献多次，因此另列用户等权；三者目标统计对象不同，不能挑一个有利口径。完整日 {same}/{len(complete_daily)} 天两大来源差异同方向，日期对照不构成随机对照。

{md_table(st['search_standardized'])}

标准化只保留两来源都有会话的日期×互斥结果类型×公开搜索活跃度层，以两来源在共同层的合计会话数作为同一权重；报告覆盖比例及共同层原始率，避免把失去支持的会话静默纳入比较。稀疏层和未观测意图仍限制解释。活跃层级是发布者按过去一个月活跃天数得到的公开标签，非本项目预测特征；公开说明未给出逐用户计算时点。

![结构与权重](../outputs/figures/02_search_structure.png)

`click_cnt`真实含大于1的值，与官方是否点击描述不一致，不能直接求和当点击次数。严格主分析保留0/1完整会话；独立正值敏感性情形不是已确认修正规则。各入口排除比例见 `search_exclusions_by_source.csv`，敏感性比例见 `search_positive_click_sensitivity.csv`。不把严格样本率当全体搜索率。

在相同可观测结构权重下，原始来源优势没有保持；这支持先核查结果组成与样本结构，而非据原始率决定入口改版。仍不能排除未观测意图或稀疏层的影响。

## 四、内容消费专题（合法时长内容）

无法完整识别视频和图文，按规划降级为合法时长内容记录。毫秒单位依据官方字段及人类时间逐行核对。合法时长要求duration有限且>0、playing_time有限且≥0，零播放进入分发分母；另存实际播放>0的分母。内容表同名类型重命名为content_item_type，不将NORMAL当视频。少量类型属性混入文本，标为未核实，不据此推断媒体；记录数和政策见quality_report的item_attribute_warning。类目ID与英文名保留。

完整日合法记录 {int(c.rows.sum()):,}，原始全期非法时长记录 {q['content']['invalid_duration_rows']:,}；零时长 {q['content']['zero_duration_rows']:,}、负播放 {q['content']['negative_play_rows']:,}（这些诊断可能重叠，不相加）。关联未知类目保留为UNMATCHED。关联前后行数、用户数及播放、时长、点赞、关注合计通过核对。

{md_table(c)}

![时长对照](../outputs/figures/03_content_duration.png)

{content_finding} 原始比例保留大于1的值，可能包含重复播放；封顶比例只是辅助统计。均值、中位数、P90与全局分位数分别见内容汇总和 `playback_distribution.csv`，不可将均值等同典型用户。

![类目与时长](../outputs/figures/04_content_categories.png)

同一宽时长箱内仍有时长分布残差，`content_category_duration.csv` 保留时长中位数用于核查。类目表全部保留；图只按记录量选择前8类，不按表现挑选。对照是观察性描述，不推荐据此增加某类目流量。

## 五、核验与下一步

SQLite在会话表上独立计算每来源人数、会话数、点击会话数、结果行数、点击行数及点击率，与Pandas一致。200名固定用户完整日志的整表筛选与分块筛选后统一聚合一致。边界测试覆盖点击缺失、来源冲突、缺键、重复维度、时长边界；正式运行逐次校验维度唯一性、关联合计和会话行数守恒。见 `outputs/audit/quality_report.json`。

优先核对两大来源的产品语义、自动播放与反馈测量。新增可信满意度指标、曝光位置与查询难度代理后，才讨论策略实验。后续方案见《后续实验设计》，没有真实上线或收益估算。模拟案例见《实验分析练习》，不能解释本节真实日志差异。

{limits}

## 六、数据来源

[官方字段说明](https://ethan00si.github.io/KuaiSAR/detailed_statistics.html)、[官网](https://kuaisar.github.io/)、[Zenodo正式small包](https://zenodo.org/records/8181109)。版本、摘要、范围和配置见 `data/manifest.json`。未追加P2网站、模型或在线实验。
''')
    write('数据字典与指标口径.md',f'''# 数据字典与指标口径

## 表与粒度

|表|键与单位|说明|
|---|---|---|
|src_inter|一条搜索结果记录|item_type是返回对象类型；click_cnt仅0/1合法|
|search_sessions|user_id + search_session_id|result_rows、distinct_items、clicked_rows、has_click、source、date/hour、type_mix、valid|
|search_user_day|user_id + date + source|会话数、点击会话数，供用户日等权核对|
|rec_inter|一条推荐分发记录|相同用户内容时间不自动去重；playing_time/duration_ms均毫秒|
|user_features|唯一非空user_id|实际为search_active_level、reco_active_level；匿名onehot不解释|
|item_features|唯一非空item_id|content_item_type与搜索item_type不同；类目用ID和英文名|

## 指标

- 会话点击率=has_click求和/合法会话数；无点击率=1−会话点击率。
- 结果点击记录比例=clicked_rows总和/result_rows总和，非会话率。
- 用户日等权=先在用户×日×来源内算会话率，再对来源内用户日平均；用户等权先按用户聚合。
- 标准化=共同日期×互斥类型×公开活跃层上的来源会话率，按共同层两来源合计会话数等权重加权。
- 播放比例=playing_time/duration_ms，仅有限duration>0且playing_time≥0；封顶比例=min(比例,1)。
- 累计播放达时长=playing_time≥duration_ms；超时长严格>；均在合法时长记录内计算。
- 记录点赞比例=like=1的合法记录/合法记录；关注同理。没有把点击、点赞、关注拼成严格漏斗。
- 用户点赞比例=对应时长组内至少一次点赞的用户/该组用户，见content_duration_metrics.csv。
- 分发侧含零播放；播放侧只含playing_time>0，分别保存在content_duration_metrics与content_played_duration_metrics。
- 时长箱：(0,15]、(15,30]、(30,60]、(60,+∞)秒，右闭；阈值为分析选择。

## 时间与排除

UTC解析毫秒再转{cfg['timezone']}；人类时间精度到秒，按floor秒核验。首尾日2023-05-22、2023-05-31不完整；两表共用完整日2023-05-23至2023-05-30。全期仍单列。缺键行单独保存；坏行所在整个已知会话隔离，不保留其余行冒充完整会话。完全相同行仅哈希诊断并保留；哈希碰撞有理论可能，不用于删除。

维度关联均left many_to_one，关联前后行数、用户数及关键合计守恒；搜索按返回类型另报覆盖率。未关联保留并抽样审计。完整字段全量缺失及数值范围见field_profile；数值范围不会替代业务合法性检查。原始匿名文本不做语义解释。
''')
    write('简历候选条目.md',f'''# 简历候选条目（已运行的P1成果）

**短视频用户行为与搜索体验分析｜Python（Pandas / NumPy / Matplotlib）、SQLite**

- 基于 KuaiSAR-small 公开日志，处理 {raw:,} 条搜索与推荐记录，构建 {n:,} 个合法搜索会话及用户日层，完成入口诊断和合法时长内容专题。
- 对比 {a.source} 与 {b.source}，样本会话点击率差异 {gap:.2f} 个百分点；通过日期、结果构成、公开活跃度共同层标准化及用户等权核查，标准化差异为 {sgap:.2f} 个百分点，提出反馈测量与产品语义核查建议。
- 交付可复现流程、4张核心图及质量审计；SQLite独立复核入口指标。内容专题区分分发／播放分母与用户日权重，明确公开样本和媒体类型限制。

实现由AI辅助；以上为项目完成事实，不声明本人独立熟练掌握全部代码。模拟A/B仅为独立补充案例，没有上线或增长效果。个人联系方式不放在此目录。
''')
    write('后续实验设计.md',f'''# 后续实验设计（方案，尚未实施）

依据：{finding}

先核实原始来源编码的产品含义与满意度测量，不把现有点击差当生产基线。替代解释包括用户意图、结果组成、自动播放、排序位置。若诊断确认某入口结果卡片存在可改进的可读性问题，可只改变该卡片摘要呈现方式；具体方案须产品核查后锁定。

资格在人群进入实验前定义；首次符合条件请求且展示前按用户持久随机1:1分流，同一用户持续同组，按初始分配组分析，不能按实验后的点击选择分析人群。主指标拟为首次合格搜索的满意度成功比例，须新增可靠反馈或随机问卷，预先定义抽样、未响应与缺失处理。测量不可得时先补埋点；若改用点击，只能对点击作结论。

辅助：点击、重复查询、后续搜索；护栏：延迟、负反馈、退出及满意度下降。先检查用户唯一性、分流、SRM、埋点与问卷响应差异。生产基线、MDE、alpha、功效、预计流量和周期均待真实业务测量，不借用公开样本；周期需覆盖完整周并满足预先样本量。固定分析窗口和停止规则，不每天看显著即停止。主指标达到统计与实际价值门槛且护栏无不可接受恶化才考虑上线；证据不足则延长至既定窗口或不决策，不事后换指标。
''')
    lo,hi=ab['ci95_newcombe']
    write('实验分析练习.md',f'''# 模拟实验分析（无真实用户、无上线策略）

原假设H0：处理组与控制组转化概率相同；双侧备择H1：不同。教学预设：用户随机1:1分组，固定一次观察窗口，每人一行二元转化，控制概率10%、处理概率11%，MDE=1个百分点，双侧alpha=0.05，目标功效80%。正态近似两比例功效规划每组 {ab['planned_users_per_arm']:,} 人。种子固定20261111，不因未显著而换种子。样本量规划为等组，实际Bernoulli分流略不等组。

{md_table(pd.read_csv(root/'outputs/tables/synthetic_ab_metrics.csv'))}

观测差异 **{ab['difference']*100:.3f}个百分点**，相对变化 {ab['relative_change']:.2%}；Newcombe差异95%区间 [{lo*100:.3f}, {hi*100:.3f}] 个百分点；双侧合并比例z检验p={ab['ztest_pvalue']:.4f}，SRM卡方p={ab['srm_pvalue']:.4f}。该次模拟证据不足以拒绝零差异；区间包含0，不能建议据此上线。教学预设真实差异不保证每次抽样显著，也不能把预设效果当观测结果。

质量：唯一用户={ab['quality']['unique_user']}，缺失单元={ab['quality']['missing_cells']}，跨组用户={ab['quality']['cross_group_users']}；分组与二元结果合法性在验证入口再次检查。SRM通过不证明其他埋点无问题。

二比例检验假设用户独立且两组成功／失败计数足够大。Newcombe为两独立比例Wilson区间组合。用户级模拟收入为二元转化×对数正态值，含大量零与右偏长尾，Welch t检验（非等方差）探索p={ab['welch_revenue_pvalue_exploratory']:.4f}；分组分位数、零值比例与极端1%收入贡献见 synthetic_ab_revenue_diagnostics.csv。大样本有助均值近似，但重尾和异常值仍应检查；收入检验不是主指标，不因它显著改写主结论，真实研究需预先约定稳健性与多重比较处理。

{md_table(pd.DataFrame(ab['simulations']))}

各2000次等组重复模拟：零效果拒绝比例估计误报率，有预设效果拒绝比例估计功效，均有Monte Carlo波动，不是业务转化提升。重复模拟采用固定等组人数；与单次随机人数分配机制有细微差别，已明确区分。独立复核用合并2×2表Pearson卡方（不作连续性校正），应与z检验平方等价；连续指标均值方差另算Welch统计量及自由度。结果保存validation.json。

实际产品实验另需真实基线、满意度测量、日志检查、护栏和固定停止规则；历史入口差异不是A/B结果。源代码src/ab_analysis.py，生成数据仅data/interim/synthetic_ab_users.csv，与KuaiSAR不混用。
''')
    write('代码导读.md','''# 代码导读

入口 `python -m src.run --config config.json`。

1. `download_data.py`：可校验Range续传，完整MD5正确才改名并解压，检查路径。
2. `src/prepare.py`：分块读完整原始字段做质量统计，只保留计算列；UTC毫秒转换；维表唯一性与关联审计。
3. `src/metrics.py`：结果行聚合会话；全会话合法性、互斥类型、不同权重；播放派生字段。
4. `src/search_analysis.py`：完整日与全日期、来源×类型×活跃度、共同层标准化、用户日表；SQLite复核。
5. `src/content_analysis.py`：合法时长限定、关联未知保留、分发与播放分母、时长和类目汇总。
6. `src/ab_analysis.py`：独立模拟、规划、分流检查、区间与检验；固定种子。
7. `src/deliver.py`：只从已算表生成图表、报告和Notebook；不另写指标。
8. `src/validate.py`：从输出与模拟明细做守恒、统计独立计算、证据表与Notebook执行核验。

学习顺序：先用tests/test_metrics.py手算7行，指出行、会话、用户日的差别；再读source_metrics的两级groupby；修改一个展示门槛并解释不会改变原始会话。内容练习比较含零播放和已播放分母。最后解释为何模拟p>0.05不能证明无效果。练习不影响已交付项目。
''')
    write('项目事实卡与面试提纲.md',f'''# 项目事实卡与面试提纲

事实：KuaiSAR-small全量；{raw:,} 行行为；{n:,} 合法搜索会话；完整日2023-05-23—30。结果文件是数字来源；AI辅助实现，不能把未掌握部分说成独立熟练。

## 90秒讲解

我用公开短视频搜索和推荐日志分析搜索来源的反馈差异。先区分结果行和会话，按用户加会话ID构建合法会话，再查看有规模的来源。{finding} 我也比较用户日及用户等权，避免高频行为主导。因为用户意图和自动播放等因素未被充分观测，结论是优先核查反馈测量和入口语义，并非策略效果。另做了限定合法时长的内容分析，使用SQLite核对入口结果，交付可重跑代码、审计和图表。模拟A/B是独立练习，没有线上实验。

## 5分钟展开

- 0—1分钟：业务问题、公开选择样本、两张行为表与实际范围。
- 1—2分钟：会话键、合法性、结果构成互斥及分母。
- 2—3分钟：两大入口差异、日期／活跃度共同层及三种权重；参照search_standardized和weighting表。
- 3—4分钟：合法时长内容、含零播放和已播放分母，不能说成完播率。
- 4—5分钟：SQLite核验、局限、补埋点及实验设计；模拟主指标未显著的解释。

## 主要追问

为什么不是行级CTR？同会话可有多个结果。为什么缺失点击不填0？未知不是无点击。为什么不用入口差证明因果？无随机分流且存在未观测混杂。为什么用户日等权还需用户等权？活跃天数不同。共同支持是什么？两来源同层都可观测才比较。为什么不按内容达时长率推荐流量？时长、分发采样和媒体类型均影响观察。模拟没显著怎么办？报告区间与不确定性，不换种子。

复现定位：成果证据表→CSV→计算模块；先展示两张搜索图，再展示限定内容范围。讲不清的实现如实说明AI辅助并回到手算例子。
''')
    evidence=pd.DataFrame([
      {'claim':'行为分析行数','value':raw,'file':'outputs/audit/data_inventory.csv','code':'src/prepare.py'},
      {'claim':'合法搜索会话','value':n,'file':'outputs/audit/quality_report.json','code':'src/metrics.py'},
      {'claim':'两大来源点击率差百分点','value':gap,'file':'outputs/tables/search_source_metrics.csv','code':'src/search_analysis.py'},
      {'claim':'共同层标准化差百分点','value':sgap,'file':'outputs/tables/search_standardized.csv','code':'src/search_analysis.py'},
      {'claim':'完整日合法时长内容记录','value':int(c.rows.sum()),'file':'outputs/tables/content_duration_metrics.csv','code':'src/content_analysis.py'},
      {'claim':'模拟差异百分点','value':ab['difference']*100,'file':'outputs/tables/synthetic_ab_summary.json','code':'src/ab_analysis.py'}])
    evidence.to_csv(reports/'成果证据表.csv',index=False,encoding='utf-8-sig')
    # Plain notebook JSON keeps generation dependency-free; validation executes cells.
    def notebook(name,title,cells):
        doc={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':[]}
        for i,(kind,source) in enumerate([('markdown',title)]+cells):
            cell={'cell_type':kind,'id':f'cell-{i}','metadata':{},'source':source.splitlines(True)}
            if kind=='code':cell.update(execution_count=None,outputs=[])
            doc['cells'].append(cell)
        (root/'notebooks'/name).write_text(json.dumps(doc,ensure_ascii=False,indent=1),encoding='utf-8')
    setup="from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\nroot=Path.cwd()\nif root.name=='notebooks': root=root.parent\nassert (root/'config.json').exists()\n"
    notebook('01_real_analysis.ipynb','# 真实KuaiSAR-small分析\n只读取统一入口的正式结果，不复制计算逻辑。\n'+finding,[('code',setup),('code',"display(pd.read_csv(root/'outputs/tables/search_source_metrics.csv'))\ndisplay(pd.read_csv(root/'outputs/tables/search_standardized.csv'))\ndisplay(pd.read_csv(root/'outputs/tables/content_duration_metrics.csv'))"),('code',"display(Image(filename=str(root/'outputs/figures/01_search_sources.png')))\ndisplay(Image(filename=str(root/'outputs/figures/03_content_duration.png')))"),('markdown',limits)])
    notebook('04_ab_statistics_practice.ipynb','# 模拟A/B实验练习\n非KuaiSAR真实用户，不含产品上线效果。固定种子，不选择显著结果。',[('code',setup),('code',"import json\nsummary=json.loads((root/'outputs/tables/synthetic_ab_summary.json').read_text(encoding='utf-8'))\ndisplay(summary)\ndisplay(pd.read_csv(root/'outputs/tables/synthetic_ab_metrics.csv'))\ndisplay(pd.read_csv(root/'outputs/tables/synthetic_ab_simulation.csv'))"),('markdown','主指标p>0.05且区间含0，证据不足。不显著不等于证明无效果。独立复核见outputs/audit/validation.json。')])
