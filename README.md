# KuaiSAR 短视频搜索入口与内容消费分析

基于公开行为日志，对比搜索入口点击率，并分析相近时长内容的观看表现。项目使用 AI 辅助实现。采用 KuaiSAR-small 正式ZIP，MD5 `daea8cbf605db6bd5841740f0e4a12d9`。全量扫描和分析两张行为表共 **10,664,332 条记录**；用户维表 **25,877 人**；合法搜索会话 **251,336 个**，来自 **25,722 人**。没有用户抽样。完整日为 2023-05-23 至 2023-05-30，首尾采集日另列。

完整日中，按会话量选择的 SUGGESTION 与 USER_INPUT 分别有 73,156、52,500 个会话，点击率 64.15%、60.26%，前者减后者为 3.89 个百分点。共同日期×互斥结果类型×公开搜索活跃层级标准化后，差异为 -0.18 个百分点。 严格口径因click_cnt非二元而隔离 16,272 个会话；若仅在独立敏感性情形中假定正整数表示有点击，两来源差异为 3.65 个百分点。主结果仅适用于符合官方二元字段约定的会话，排除会话可能带来选择偏差。

合法时长内容记录中，0–15s 组累计播放达时长比例为 35.95%，>60s 组为 10.66%。时长与结果明显相关，类目对照必须放在同一时长层内。

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

本项目是公开选择样本的观察分析；无点击不等于不满意，自动播放可能满足需求。来源编码仅沿用原值；用户意图、查询难度与排序位置未充分观测。推荐点击正负样本经过构建，不估算平台CTR或增长空间。内容媒体类型未完整识别，累计播放达时长也不等于一次连续完播。匿名特征不解读成人口属性。

## 来源与许可

数据：[Zenodo](https://zenodo.org/records/8181109)；[官网](https://kuaisar.github.io/)；[字段说明](https://ethan00si.github.io/KuaiSAR/detailed_statistics.html)。引用 Sun et al., KuaiSAR: A Unified Search And Recommendation Dataset, CIKM 2023，[论文](https://arxiv.org/abs/2306.07705)。中文类目部分在源文件内损坏，使用ID与英文标签。原始数据许可遵循发布者 [使用说明](https://kuaisar.github.io/)（CC BY-NC-SA 4.0，署名、非商业、相同方式共享）；不重新分发原始数据、用户级明细或将数据许可套用为代码许可。代码未另行授予开源许可证。个人简历放在项目目录之外，不纳入公开成果。
