"""图结构消融数据集生成器（Q4 阻塞项落地）—— 放大版 v2。

与 v1 的区别（**S1 修复：让实验具备判别力**）
-------------------------------------------
v1 每个模块只有 4 条记忆 → 候选集 ≈ kept 条数（实测 candidates 5.08 / kept 3.81），
"保持条数"而非"排序质量"成了主导变量，G0/G1 与 G3/G4 的差异无法归因到算法。
本版把每模块扩到 **20 条记忆**（14×20 = 280 条），使 K 真正具备选择性。

其余设计约束（沿用 v1，来自 POC 两轮教训 + 实测坑）
--------------------------------------------------
1. **四类 typed 边齐全**：上一版真实数据 7 条边全 `depends_on` → 强度零方差 → G3≡G4。
   本生成器强制混合边型，并**故意留噪**（真实依赖中 1 条落弱边、干扰源中 1 条落强边）。
2. **不构造"最强边恰好指向最相关节点"**：相关源由任务语义人工标注，与边强度**独立**。
3. **ground truth 必须语义化**：逐查询手工标注相关记忆 id，不用前缀截取。
4. **任务标签是任务级、不是节点级**：否则跨源记忆被任务头判"离题"，recall 被系统性清零。
5. **报告天花板**：`ceiling = min(1, K/|relevant|)`。
6. 完全确定性（固定 seed + 固定时间基准）。

用法
----
    python -m analysis.gen_graph_dataset
输出
----
    data/graph_dataset.jsonl   (data/ 在 .gitignore 内，按需重生成)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TIME_BASE = 1_700_000_000.0
HOUR = 3600.0
NODE_SPAN_H = 6.0
#: 任务标签是**任务级**（同一项目任务），**不是**节点级
TASK_TAG = "research"

# ── 模块表：14 模块 × 20 条记忆 ─────────────────────────────────────────────
MODULES: list[dict] = [
    {"id": "n_config", "topic": "配置与参数管理", "mems": [
        "调度阈值 threshold 默认取 0.6",
        "共享池上限 max_shared 固定为 10 条",
        "候选集 candidate_override 统一 50",
        "配置文件 srtp.yaml 是唯一手写真源",
        "插件按 _impl 后缀字段装配",
        "selector.topk 用 top_k 参数控制保留条数",
        "embedding 维度锁定 1024 维",
        "融合系数 alpha 固定 0.6",
        "融合系数 beta 固定 0.4",
        "条件键 condition_key 形如 u_tu|research|srtp",
        "任务标签 task_tag 默认 main_task",
        "先验采样次数 llm_prior_samples 设为 1",
        "检索时机 retrieval_mode 走 pre_schedule",
        "埋点目录 log_dir 指向 data/metrics",
        "实验登记号 run_id 写入 run_manifest",
        "embedding 模型名 text-embedding-v4",
        "embedding_cache 开关默认打开",
        "缓存目录 embedding_cache_dir 指向 data/embed_cache",
        "条件化维度 conditioning_dims 只含语义与任务",
        "条件向量维度 condition_emb_dim 取 32",
    ]},
    {"id": "n_embed", "topic": "文本向量化与落盘缓存", "mems": [
        "向量化后端用 DashScope 的 text-embedding-v4",
        "输出向量维度是 1024 维",
        "首次计算把向量写入 embed_cache 目录",
        "重启后直接从落盘缓存复用向量",
        "无 API key 时降级为确定性占位向量",
        "占位向量按整串文本哈希生成、不含主题信息",
        "占位向量不落盘，避免补 key 后取到陈旧值",
        "真实向量落盘保证实验可复现",
        "缓存以文本原文为键做精确命中",
        "查询向量与记忆向量共用同一后端",
        "节点主题向量在建节点时一并缓存",
        "embedding 调用失败会记录 last_error 供告警",
        "缓存未命中的调用才计入成本埋点",
        "缺少 key 时管线不中断只降级为占位",
        "向量归一化后用于余弦相似度",
        "embedding 是语义头的输入来源",
        "缓存文件为 JSONL 每行一条文本与向量",
        "向量维度与注意力维度保持一致",
        "文本为空时返回全零向量",
        "对比不同后端需保持同源向量",
    ]},
    {"id": "n_retrieval", "topic": "检索基座与召回原语", "mems": [
        "检索器共五种且全部可插拔",
        "五种是 full、bm25、vector、rrf、reme",
        "vector 检索器优先走 faiss 内积索引",
        "faiss 缺失时退化为线性余弦",
        "候选集统一返回 50 条保证可比",
        "BM25 采用字符级中文分词",
        "未引入 jieba 以保证确定性",
        "rrf 用双路名次融合且常数取 60",
        "reme 检索器仅用于 baseline 对照",
        "recall 接口签名带 query 与 top_k",
        "full 检索器按写入序取前 top_k 条",
        "bm25 打分写回候选的 bm25_score 字段",
        "vector 打分写回候选的 vector_score 字段",
        "检索在常驻池上执行而非 ReMe 原生检索",
        "限定范围检索需要先把池收窄",
        "全池 faiss 索引不适合限定范围场景",
        "范围内条目少时用精确线性余弦更准",
        "检索器不参与排序决策只做召回",
        "检索原语保持中性不叠加融合分",
        "五种检索器共享同一接口契约",
    ]},
    {"id": "n_attention", "topic": "四维注意力打分器", "mems": [
        "四个打分头是时间、语义、频率、任务",
        "默认权重依次是 0.25/0.35/0.15/0.25",
        "时间头用指数衰减 exp(-0.01t)",
        "时间参数 t 是记忆年龄、单位为小时",
        "频率头用 1-exp(-0.1n) 的饱和增长",
        "频率头输入是记忆的被访问次数 n",
        "任务头匹配得 1.0、不匹配得 0.3",
        "任务头双方标签为空时给中性 0.5",
        "语义头内部融合余弦 0.5、向量分 0.3",
        "语义头内部 BM25 分占 0.2",
        "语义头三个系数与全局 alpha beta 无关",
        "enabled_dims 掩码可关闭指定维度",
        "关闭某维后其余维权重重新归一化",
        "首轮调度时频率头恒为 0 因为访问次数为 0",
        "同一任务标签下任务头退化为常数",
        "时间头反映的是记忆绝对年龄而非相关性",
        "多个维度量纲不一致会稀释主导维度",
        "批内标准化可消除量纲与常数偏置",
        "无区分度的维度应被门控掉并再分配权重",
        "打分器只影响排序不影响召回范围",
    ]},
    {"id": "n_weights", "topic": "混合权重与条件化", "mems": [
        "混合权重等于 0.6 乘先验加 0.4 乘可学习",
        "可学习侧是条件化多层感知机",
        "感知机输入是查询向量拼 32 维条件向量",
        "隐藏层 64 维后接 ReLU 再接 4 维输出",
        "输出经 Softmax 得到四维权重",
        "只有语义维与任务维随用户场景漂移",
        "时间维与频率维保持全局不个性化",
        "alpha 与 beta 固定不参与学习",
        "先验权重由轻量模型单次采样给出",
        "先验提示词要求输出四维 JSON 且和为一",
        "解析失败或超时回落默认权重",
        "离线蒸馏的损失函数是均方误差",
        "蒸馏目标取多次采样的先验平均值",
        "在线更新用 bandit 式一步微调",
        "学习率默认 0.01 且带权重衰减",
        "更新后向先验回归防止权重学偏",
        "先验结果做 LRU 缓存避免重复调用",
        "先验门控在候选过少时跳过调用",
        "条件存储用 32 维嵌入表实现",
        "权重融合只改变维度配额不改变公式",
    ]},
    {"id": "n_feedback", "topic": "反馈闭环与在线学习", "mems": [
        "反馈奖励取显式点赞加一与点踩减一",
        "更新方式是 bandit 式一步",
        "在线更新只改条件对应的参数切片",
        "不动融合系数 alpha 与先验来源",
        "更新后向先验回归防止权重学偏",
        "训练集来自实际交互与模拟生成两条来源",
        "模拟数据用多种典型场景模板生成",
        "样本需人工或自动标注目标权重",
        "标注规范要求四维权重和为一",
        "训练前把样本文本向量化并做 L2 归一化",
        "增量训练采用批量 DataLoader",
        "验证集切分比例可配置",
        "正则化约束权重不偏离先验过大",
        "反馈步长取 0.1 的同阶小量",
        "偏好漂移通过条件切片局部生效",
        "冷启动阶段先验权重占主导",
        "反馈数据与调度埋点同源落盘",
        "奖励信号仅在用户显式操作时触发",
        "更新频率受限以避免震荡",
        "闭环目标是越用越贴合该用户与该场景",
    ]},
    {"id": "n_pool", "topic": "双池与共享池拓扑", "mems": [
        "常驻完整池负责全量记忆存储",
        "共享池只承载被选中要注入的记忆",
        "共享池按分数有序插入",
        "共享池超过 10 条时淘汰最低分",
        "共享池不做二次检索",
        "不做二次检索是为让排序成为唯一变量",
        "记忆元数据直接写进 resident_pool.jsonl",
        "不设独立元数据侧表以避免不一致",
        "常驻池用 faiss 索引支持近似最近邻",
        "常驻池落盘为 JSONL 每行一条记忆",
        "记忆主键是 uuid 形式的 memory_id",
        "访问计数在写入共享池时同步加一",
        "常驻池按用户与会话做隔离过滤",
        "共享池快照按用户会话分文件保存",
        "大记忆池方案下节点专属记忆靠标记过滤",
        "单一正文存储避免拷贝带来的不一致",
        "标记倒排索引可在启动时从池重建",
        "不新增侧表以保持存储简洁",
        "共享池历史留存点与新调度点需区分展示",
        "池容量与压缩比是验收指标之一",
    ]},
    {"id": "n_graph", "topic": "图结构记忆调度", "mems": [
        "图的节点是一次主题会话而不是一条记忆",
        "这与知识图谱以实体为节点有根本区别",
        "边强度是静态量不在每次查询时重算",
        "边强度只在建边、反馈、摘要重算三时机变化",
        "选源阶段先于选记忆阶段执行",
        "先选源再选记忆形成多对多筛选",
        "边强度与查询相关度按 0.7 与 0.3 融合",
        "查询相关度项复用语义头已算的查询向量",
        "邻域沿入边展开取的是上游被依赖节点",
        "邻域扩散深度默认封顶为一跳",
        "建边前做可达性检查拒绝成环",
        "成环的边降级为弱引用且不入邻域",
        "节点主题向量的余弦作为相关性度量",
        "源相关度方差过低时退化为纯静态",
        "软偏置把源分加进记忆总分而不设硬配额",
        "硬配额只在源池大于配额时才有区分度",
        "typed 边只用于初始化默认强度",
        "摘要只用于建边与源打分不用于召回",
        "节点专属记忆由所有者标记过滤得到",
        "图结构本身是常量、调度组件才是消融变量",
    ]},
    {"id": "n_ablation", "topic": "消融实验设计", "mems": [
        "五组消融分别是 G0 到 G4",
        "G0 是邻域全量不做调度",
        "G1 是图加均匀调度",
        "G2 是图加单源即主副线等价物",
        "G3 是多源加四维打分配额均匀",
        "G4 是多源加边强度配额即完整方案",
        "主副线架构等价于 G2 单源对照组",
        "五组检索基座统一为向量检索",
        "统一检索基座是为修正此前的归因污染",
        "图结构由数据集固定注入不作为变量",
        "消融必须报告相关集规模与召回天花板",
        "天花板公式是注入上限除以相关集大小",
        "保持条数需在组间可比否则归因不纯",
        "固定 K 选择器与自适应阈值选择器不可混用",
        "源召回命中率衡量选源阶段的质量",
        "多源覆盖率衡量注入集是否真的来自多个源",
        "跨源相关性防止为凑覆盖率拉入无关记忆",
        "去重率在单值标记下结构性为零",
        "实验需登记 run_manifest 便于审计",
        "延迟口径需在组外预热查询向量后再测",
    ]},
    {"id": "n_dataset", "topic": "数据集与场景设计", "mems": [
        "数据来自多模块项目会话",
        "节点天然形成有向无环图",
        "要求至少十个节点十五条边",
        "图中需含至少一个分叉结构",
        "typed 边必须四类齐全",
        "边强度零方差会让强度配额无法检验",
        "相关源需人工标注且与边强度独立",
        "相关记忆需逐条标注不能按前缀截取",
        "任务标签取任务级而非节点级",
        "固定时间基准以保证结果可复现",
        "记忆按所有者标记归属到节点",
        "每模块记忆数要远大于保持条数",
        "候选集必须显著大于 K 才有判别力",
        "标注源必须在该节点的上游否则题目矛盾",
        "生成器需内置自检拒绝非法标注",
        "场景需覆盖单源与多源两类查询",
        "多源查询是检验多对多优于一对一的关键样本",
        "数据集文件在 gitignore 内按需重生成",
        "种子固定后同命令多次运行结果一致",
        "数据集规模不足会让消融不可辨识",
    ]},
    {"id": "n_demo", "topic": "演示壳与会话画布", "mems": [
        "会话画布实现一回合一节点",
        "点击节点跳转到节点控制台",
        "主线可以创建副线",
        "副线调度按第一问执行",
        "副线创建后展示从主线继承的记忆清单",
        "画布内嵌脚本必须做语法检查",
        "一次多余括号就会导致整页空白",
        "画布支持节点拖拽并落盘坐标",
        "画布泳道标签需与首节点对齐",
        "双池视图用降维投影到二维展示",
        "虚线表示记忆继承语义",
        "控制台左侧是节点上下文历史",
        "控制台右侧是调度流水线明细",
        "创建副线有全流程过渡动画",
        "画布需屏蔽节点点击与拖拽的冲突",
        "抽屉改为浮层覆盖避免隔断画布",
        "rpx 适配在宽屏下放大系数过大会错位",
        "画布节点标题需截断避免溢出",
        "服务端接口需返回权重与入选记忆编号",
        "无模型密钥时自动切换备用后端",
    ]},
    {"id": "n_deploy", "topic": "服务器部署与运维", "mems": [
        "演示服务部署在腾讯云的 8787 端口",
        "服务器地址是 150.158.26.158",
        "用 systemd 单元托管服务",
        "单元名为 srtp-demo",
        "服务绑定 0.0.0.0 并随开机启动",
        "公网端口不通优先检查云厂商安全组",
        "系统防火墙通常不是端口不通的主因",
        "前台安装大体积依赖会超时截断",
        "必须后台安装再轮询结果",
        "本机访问公网需绕开代理",
        "部署目录在 opt 下的子目录",
        "运行环境是 3.10 的虚拟环境",
        "依赖包含数值库与向量索引库",
        "内存上限通过单元配置限制",
        "服务异常会自动重启",
        "服务器上无智能体框架故走无头模式",
        "真实大模型调用不受无头模式影响",
        "更新流程为改本地再打包上传再重启",
        "未使用容器直接跑在系统上",
        "演示入口是根路径与会话画布两个页面",
    ]},
    {"id": "n_metrics", "topic": "指标与可复现性", "mems": [
        "核心指标含召回准确率与压缩比",
        "效率指标是平均响应时间",
        "经济性指标是 token 降低率",
        "多源场景需新增源召回命中率",
        "还需多源覆盖率与跨源相关性",
        "去重率衡量多路径冗余程度",
        "全链路延迟要求小于一秒",
        "单条打分耗时要求小于十毫秒",
        "压缩比验收目标是小于等于 0.3",
        "每次实验登记 run_manifest",
        "埋点记录候选数与保留数与压缩比",
        "埋点还记录权重与延迟分项",
        "实验模式分离线与真实两种",
        "缓存命中率反映成本控制效果",
        "查询向量预热后延迟口径才可比",
        "指标必须同时报告天花板避免误读",
        "指标口径变化需在文档中显式说明",
        "不可辨识与判据未通过要区分对待",
        "登记表由脚本生成禁止手改",
        "摘要文件由 YAML 自动生成",
    ]},
    {"id": "n_paper", "topic": "论文叙事与投稿", "mems": [
        "创新点是多对多筛选而非主副线结构",
        "论文需写明与知识图谱的边界区别",
        "需报告图结构贡献与算法贡献的归因",
        "归因靠图加均匀调度的下界对照",
        "主副线降级为图加单源的特例",
        "双池架构表述需演进为单实体池加标记视图",
        "四维注意力是申报书的核心创新点之一",
        "混合权重与条件化需说明固定系数理由",
        "反馈闭环需说明只更新条件切片",
        "摘要边界需声明不用于记忆召回",
        "实验需报告天花板与相关集规模",
        "投稿窗口集中在十月至次年三月",
        "目标为 EI 会议或中文核心期刊",
        "论文需给出消融变量控制表",
        "需说明静态边强度的设计动机",
        "需承认打分器在真实数据上的退化并给出修复",
        "成本效果速度三角需以取点与曲线呈现",
        "不应主张综合目标函数是可优化目标",
        "局限性章节需写明数据集为自建自证",
        "引用需覆盖多智能体通信与记忆机制两类综述",
    ]},
]

# ── 边表：(src, dst, type)。混合 typed；含多个分叉；故意留噪 ──────────────────
EDGES: list[tuple[str, str, str]] = [
    ("n_config", "n_embed", "depends_on"),
    ("n_embed", "n_retrieval", "depends_on"),
    ("n_retrieval", "n_attention", "depends_on"),
    ("n_attention", "n_weights", "depends_on"),
    ("n_weights", "n_feedback", "derives_from"),
    ("n_embed", "n_pool", "depends_on"),
    ("n_retrieval", "n_graph", "depends_on"),
    ("n_pool", "n_graph", "depends_on"),
    ("n_attention", "n_ablation", "depends_on"),
    ("n_weights", "n_ablation", "depends_on"),
    ("n_graph", "n_demo", "depends_on"),
    ("n_pool", "n_demo", "depends_on"),
    ("n_demo", "n_deploy", "derives_from"),
    ("n_ablation", "n_dataset", "depends_on"),
    ("n_graph", "n_paper", "depends_on"),
    ("n_dataset", "n_paper", "depends_on"),
    ("n_metrics", "n_paper", "references"),
    # 干扰源（弱边）—— 故意留噪
    ("n_config", "n_deploy", "similar_to"),
    ("n_feedback", "n_metrics", "references"),
    ("n_deploy", "n_paper", "similar_to"),
    ("n_feedback", "n_demo", "similar_to"),
]

# ── 语义干扰项（S2'）────────────────────────────────────────────────────────
# 设计意图：v2 数据集的相关标注只看"语义是否命中查询主题"，没有"语义相似但应被排除"
# 的样本 → time/task 维无用武之地，打分器不可能显著超过"不调度"的 G0（实测确认）。
# 本节补两类干扰项，让 time / task 维有发挥空间：
#
#   A. **过时版本**（考 time 维）：与某条"当前版"记忆语义高度重叠（查询词都能命中），
#      但时间戳**早于**所在模块的普通记忆 → time 头应压低它。
#      文本刻意保持中性（用"v1 方案 / 早期设计 / 曾计划"），不带"已废弃"等显式标记，
#      避免语义检索器靠字面就识破 —— 这样只有 time 维能区分，干扰才有效。
#      标注理由：v1 设计已被对应"当前版"取代，人工判定为不相关（可辩护，见 replaced_by）。
#
#   B. **跨任务口径**（考 task 维）：task_tag ≠ 任务级标签 research（如运维/教学/评审口径），
#      句式与某些查询高度重叠但属于另一类任务口径 → task 头应压低它。
#      attention.noop（G0/G1）完全不看 task_tag → 会被这类干扰拖累；
#      attention.full/normalized 会压它 → 形成可检验的对照。
#
# 干扰项**不进入任何查询的 relevant 集**；它们进候选集后会挤占 K 个名额，
# 从而使"无算法"的 G0/G1 recall 下降、而"有算法"的 G3/G4 保持 —— 这正是要检验的。

#: A. 过时版本：(owner_node, 文本, 被哪条取代)。时间戳自动取所在模块起点前 2 小时。
STALE_VERSIONS: list[tuple[str, str, str]] = [
    ("n_config", "调度阈值 threshold 在 v1 方案中默认 0.7", "n_config_m1"),
    ("n_config", "共享池上限 max_shared 早期设计为 8 条", "n_config_m2"),
    ("n_config", "候选集 candidate_override 曾计划 30 条", "n_config_m3"),
    ("n_embed", "文本向量化曾计划采用 text-embedding-v3", "n_embed_m1"),
    ("n_embed", "向量维度早期方案是 768 维", "n_embed_m2"),
    ("n_retrieval", "候选集曾统一返回 30 条", "n_retrieval_m5"),
    ("n_attention", "四维默认权重早期为 0.3/0.3/0.2/0.2", "n_attention_m2"),
    ("n_weights", "混合权重早期设计为 0.5 乘先验加 0.5 乘可学习", "n_weights_m1"),
    ("n_pool", "共享池早期上限设计为 8 条", "n_pool_m2"),
    ("n_graph", "边强度与相关度早期按 0.5 与 0.5 融合", "n_graph_m7"),
    ("n_graph", "邻域扩散深度曾计划两跳", "n_graph_m10"),
    ("n_metrics", "全链路延迟要求早期写为 2 秒", "n_metrics_m7"),
]

#: B. 跨任务口径：(owner_node, 文本, task_tag)。时间戳取所在模块的普通时间。
CROSS_TASK_ITEMS: list[tuple[str, str, str]] = [
    ("n_deploy", "运维口径：共享池告警阈值设为 12 条", "ops"),
    ("n_deploy", "运维口径：演示服务内存上限 1500M", "ops"),
    ("n_config", "教学演示口径：阈值示例取 0.5", "teaching"),
    ("n_pool", "容量规划口径：共享池预留 15 条", "planning"),
    ("n_embed", "接口兼容口径：维度可回退 768", "compat"),
    ("n_attention", "调参口径：语义头系数曾试 0.4/0.4/0.2", "tuning"),
]

# ── S-A 时序演进场景：版本链（考 time 维，E1）────────────────────────────────
# 设计：同一事实的 3 个版本，文本仅版本词与取值不同（语义几乎不可分），
#       时间戳按版本递增、跨度 96 小时级（LAMBDA_TIME=0.01/h 下 time 才有区分度）。
# 查询不带任何版本词（避免语义偏向当前版）→ 语义检索无法区分版本，
#       旧版（干扰）与当前版（相关）的余弦几乎相同 → 排序由 time 维决定。
# 旧版 task_tag 仍为 research → task 头不参与，场景纯净（只考 time）。
SA_CHAINS: list[dict] = [
    {"owner": "n_config", "subject": "共享池的容量上限",
     "versions": [("第一版", "八条"), ("第二版", "九条"), ("当前版", "十条")], "cur": 3,
     "cur_node": "n_embed"},
    {"owner": "n_config", "subject": "调度阈值的默认值",
     "versions": [("第一版", "0.7"), ("第二版", "0.65"), ("当前版", "0.6")], "cur": 3,
     "cur_node": "n_embed"},
    {"owner": "n_embed", "subject": "向量化输出的维度",
     "versions": [("第一版", "768 维"), ("第二版", "512 维"), ("当前版", "1024 维")], "cur": 3,
     "cur_node": "n_retrieval"},
    {"owner": "n_weights", "subject": "先验与可学习的融合比例",
     "versions": [("第一版", "各占一半"), ("第二版", "六四开"), ("当前版", "0.6 与 0.4")], "cur": 3,
     "cur_node": "n_feedback"},
    {"owner": "n_metrics", "subject": "全链路延迟的验收要求",
     "versions": [("第一版", "两秒"), ("第二版", "一点五秒"), ("当前版", "一秒")], "cur": 3,
     "cur_node": "n_paper"},
    {"owner": "n_graph", "subject": "边强度与相关度的融合比例",
     "versions": [("第一版", "各占一半"), ("第二版", "六四开"), ("当前版", "0.7 与 0.3")], "cur": 3,
     "cur_node": "n_demo"},
]

# ── S-B 口径隔离场景：同一参数多口径（考 task 维，E1）────────────────────────
# 设计：同一主体在不同任务口径下取值不同，文本除口径词与取值外几乎相同、
#       时间戳相同（time 不参与，场景纯净只考 task）。
# 查询带口径标签（episode.task_tag = q_task），relevant = 该口径的版本，
#       其余口径版本为干扰（task 头应压低）。
SB_PARAMS: list[dict] = [
    {"owner": "n_pool", "subject": "共享池的容量上限",
     "values": {"ops": "十二条", "research": "十条", "demo": "六条"}, "q_task": "ops",
     "cur_node": "n_graph"},
    {"owner": "n_config", "subject": "调度阈值的默认值",
     "values": {"ops": "0.5", "research": "0.6", "demo": "0.4"}, "q_task": "ops",
     "cur_node": "n_embed"},
    {"owner": "n_metrics", "subject": "单条打分的耗时要求",
     "values": {"ops": "五毫秒", "research": "十毫秒", "demo": "二十毫秒"}, "q_task": "ops",
     "cur_node": "n_paper"},
    {"owner": "n_embed", "subject": "向量化输出的维度",
     "values": {"ops": "512 维", "research": "1024 维", "demo": "768 维"}, "q_task": "ops",
     "cur_node": "n_retrieval"},
]

# ── 查询表（人工标注 ground truth，与边强度独立）─────────────────────────────
# (当前节点, 查询, [相关记忆 id])。多源查询 = 相关记忆跨 >=2 个上游源。
QUERIES: list[tuple[str, str, list[str]]] = [
    # ——— 单源 ———
    ("n_embed", "调度阈值和共享池上限的默认值是多少", ["n_config_m1", "n_config_m2"]),
    ("n_embed", "配置怎么驱动插件装配、向量维度锁在多少", ["n_config_m4", "n_config_m5", "n_config_m7"]),
    ("n_embed", "融合系数 alpha 和 beta 取多少", ["n_config_m8", "n_config_m9"]),
    ("n_retrieval", "文本向量化用哪个模型、输出多少维", ["n_embed_m1", "n_embed_m2"]),
    ("n_retrieval", "embedding 缓存怎么落盘复用", ["n_embed_m3", "n_embed_m4", "n_embed_m8"]),
    ("n_retrieval", "没有密钥时向量化怎么降级、占位向量有什么特点", ["n_embed_m5", "n_embed_m6", "n_embed_m7"]),
    ("n_attention", "检索基座一共有哪几种召回原语", ["n_retrieval_m1", "n_retrieval_m2"]),
    ("n_attention", "向量检索底层用什么索引、候选集多大", ["n_retrieval_m3", "n_retrieval_m4", "n_retrieval_m5"]),
    ("n_attention", "BM25 用什么分词、RRF 常数取多少", ["n_retrieval_m6", "n_retrieval_m7", "n_retrieval_m8"]),
    ("n_weights", "四维注意力都有哪四个打分头、默认权重是多少", ["n_attention_m1", "n_attention_m2"]),
    ("n_weights", "时间衰减和频率饱和的公式分别是什么", ["n_attention_m3", "n_attention_m4", "n_attention_m5"]),
    ("n_weights", "任务头匹配与不匹配分别给多少分", ["n_attention_m7", "n_attention_m8"]),
    ("n_feedback", "混合权重的公式是什么", ["n_weights_m1", "n_weights_m2"]),
    ("n_feedback", "条件化作用在哪几维、alpha 会不会被学习", ["n_weights_m6", "n_weights_m7", "n_weights_m8"]),
    ("n_feedback", "离线蒸馏用什么损失、在线更新怎么走", ["n_weights_m12", "n_weights_m13", "n_weights_m14"]),
    ("n_pool", "向量化后端和维度是多少", ["n_embed_m1", "n_embed_m2"]),
    ("n_pool", "占位向量存不存盘、为什么", ["n_embed_m7", "n_embed_m12"]),
    ("n_deploy", "会话画布上节点和连线分别表示什么", ["n_demo_m1", "n_demo_m11", "n_demo_m12"]),
    ("n_deploy", "画布内嵌脚本踩过什么坑", ["n_demo_m6", "n_demo_m7"]),
    ("n_dataset", "五组消融分别是什么、主副线对应哪一组", ["n_ablation_m1", "n_ablation_m2", "n_ablation_m4"]),
    ("n_dataset", "为什么必须报告相关集规模与召回天花板", ["n_ablation_m11", "n_ablation_m12"]),
    ("n_metrics", "用户反馈怎么更新权重、奖励怎么取", ["n_feedback_m1", "n_feedback_m2"]),
    ("n_metrics", "在线更新会不会把权重学偏、怎么防", ["n_feedback_m5", "n_feedback_m13"]),
    # ——— 多源（G2 单源必然漏源，核心对照样本）———
    ("n_graph", "检索候选集多大、共享池容量上限是多少", ["n_retrieval_m5", "n_pool_m2"]),
    ("n_graph", "向量检索用什么索引、共享池为什么不做二次检索", ["n_retrieval_m3", "n_pool_m5", "n_pool_m6"]),
    ("n_graph", "记忆元数据存在哪里、检索在哪个池上执行", ["n_pool_m7", "n_pool_m8", "n_retrieval_m14"]),
    ("n_ablation", "四维打分和混合权重是怎么串起来的", ["n_attention_m1", "n_weights_m1"]),
    ("n_ablation", "时间衰减与频率饱和的公式是什么、alpha 能不能学",
     ["n_attention_m3", "n_weights_m8"]),
    ("n_ablation", "打分器有哪些已知退化、权重怎么更新", ["n_attention_m14", "n_attention_m17", "n_weights_m14"]),
    ("n_demo", "图里节点指什么、共享池最多放几条", ["n_graph_m1", "n_pool_m2"]),
    ("n_demo", "边强度是不是静态的、共享池会不会二次检索", ["n_graph_m3", "n_graph_m4", "n_pool_m5"]),
    ("n_demo", "大记忆池怎么形成节点专属记忆", ["n_graph_m19", "n_pool_m15", "n_pool_m16"]),
    ("n_paper", "创新点落在哪里、核心指标有哪些", ["n_graph_m1", "n_metrics_m1"]),
    ("n_paper", "数据集对图的节点和边有什么硬性要求", ["n_dataset_m1", "n_dataset_m3", "n_dataset_m5"]),
    ("n_paper", "图结构贡献和算法贡献怎么归因", ["n_graph_m20", "n_dataset_m20"]),
]


def _check_dag(nodes: list[str], edges: list[tuple[str, str, str]]) -> None:
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    indeg: dict[str, int] = {n: 0 for n in nodes}
    for s, d, _t in edges:
        adj[s].append(d)
        indeg[d] += 1
    q = [n for n in nodes if indeg[n] == 0]
    seen = 0
    while q:
        cur = q.pop()
        seen += 1
        for nxt in adj[cur]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                q.append(nxt)
    if seen != len(nodes):
        raise ValueError("边表不是 DAG，请检查 EDGES")


def build_graph_payload() -> tuple[list[dict], list[dict]]:
    """构造全图 nodes / corpus（所有 episode 共用同一张图，含 S2' 语义干扰项）。"""
    corpus: list[dict] = []
    node_idx = {m["id"]: ni for ni, m in enumerate(MODULES)}
    for ni, m in enumerate(MODULES):
        for i, text in enumerate(m["mems"], 1):
            corpus.append({
                "memory_id": f"{m['id']}_m{i}",
                "text": text,
                "task_tag": TASK_TAG,
                "owner_node": m["id"],
                "timestamp": TIME_BASE + ni * NODE_SPAN_H * HOUR + (i - 1) * 0.25 * HOUR,
            })

    # A. 过时版本干扰项：时间戳取所在模块起点前 2 小时（必早于该模块所有普通记忆）
    #    → time 头会给它低分；语义上却与"当前版"高度重叠，检索器无法靠字面排除。
    for k, (owner, text, replaced_by) in enumerate(STALE_VERSIONS, 1):
        if owner not in node_idx:
            raise ValueError(f"过时版本干扰项的 owner 未知: {owner}")
        ni = node_idx[owner]
        corpus.append({
            "memory_id": f"{owner}_stale{k}",
            "text": text,
            "task_tag": TASK_TAG,
            "owner_node": owner,
            "timestamp": TIME_BASE + ni * NODE_SPAN_H * HOUR - 2 * HOUR,
            "kind": "stale_version",
            "replaced_by": replaced_by,
        })

    # B. 跨任务口径干扰项：task_tag != research（task 头应压低）；时间戳取所在模块正常时间。
    for k, (owner, text, tag) in enumerate(CROSS_TASK_ITEMS, 1):
        if owner not in node_idx:
            raise ValueError(f"跨任务干扰项的 owner 未知: {owner}")
        ni = node_idx[owner]
        corpus.append({
            "memory_id": f"{owner}_xtra{k}",
            "text": text,
            "task_tag": tag,
            "owner_node": owner,
            "timestamp": TIME_BASE + ni * NODE_SPAN_H * HOUR + 1.5 * HOUR,
            "kind": "cross_task",
        })

    # C. S-A 时序演进场景：版本链（考 time 维）。
    #    同一事实 3 个版本，文本仅版本词/取值不同（语义几乎不可分）；
    #    时间戳 = 场景基准时刻 now 往前 (n-k+1)*96h → 当前版最新、区分度足够
    #    （LAMBDA_TIME=0.01/h，96h 差 → time 分 0.38 vs 0.99）。旧版 task_tag 仍为 research。
    sa_now = TIME_BASE + len(MODULES) * NODE_SPAN_H * HOUR + HOUR
    for ci, ch in enumerate(SA_CHAINS, 1):
        if ch["owner"] not in node_idx:
            raise ValueError(f"S-A 版本链 owner 未知: {ch['owner']}")
        n_v = len(ch["versions"])
        if ch["cur"] != n_v:
            raise ValueError(f"S-A 链{ci} 当前版序号须为最后一版")
        for k, (ver_word, val) in enumerate(ch["versions"], 1):
            corpus.append({
                "memory_id": f"{ch['owner']}_sa{ci}v{k}",
                "text": f"{ch['subject']}：{ver_word}取值{val}",
                "task_tag": TASK_TAG,               # 与查询同口径 → task 不参与，纯考 time
                "owner_node": ch["owner"],
                "timestamp": sa_now - (n_v - k + 1) * 96 * HOUR,
                "kind": "sa_version",
                "chain": ci, "version": k,
                "is_current": (k == ch["cur"]),
            })

    # D. S-B 口径隔离场景：同一参数多口径（考 task 维）。
    #    文本除口径词/取值外几乎相同、时间戳相同（time 不参与，纯考 task）；
    #    相关版 task_tag = 查询口径，其余口径 = 干扰。
    for pi, pm in enumerate(SB_PARAMS, 1):
        if pm["owner"] not in node_idx:
            raise ValueError(f"S-B 参数 owner 未知: {pm['owner']}")
        ni = node_idx[pm["owner"]]
        for tag, val in pm["values"].items():
            corpus.append({
                "memory_id": f"{pm['owner']}_sb{pi}_{tag}",
                "text": f"{pm['subject']}：{tag}口径取值{val}",
                "task_tag": tag,
                "owner_node": pm["owner"],
                "timestamp": TIME_BASE + ni * NODE_SPAN_H * HOUR + 1.0 * HOUR,
                "kind": "sb_param",
                "param": pi,
            })

    nodes = [{"node_id": m["id"], "topic": m["topic"],
              "summary": f"{m['topic']}：" + "；".join(m["mems"][:8])} for m in MODULES]
    return nodes, corpus


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/graph_dataset.jsonl")
    ap.add_argument("--seed", type=int, default=20260922)
    args = ap.parse_args()
    _ = args.seed   # 数据集本身完全确定，seed 仅为接口一致性保留

    node_ids = [m["id"] for m in MODULES]
    _check_dag(node_ids, EDGES)
    nodes, corpus = build_graph_payload()
    by_id = {c["memory_id"]: c for c in corpus}
    per_node = {m["id"]: len(m["mems"]) for m in MODULES}

    episodes: list[dict] = []
    for idx, (cur_node, query, rel_mems) in enumerate(QUERIES):
        missing = [m for m in rel_mems if m not in by_id]
        if missing:
            raise ValueError(f"ep{idx} {cur_node} 标注引用不存在的记忆: {missing}")
        rel_sources = sorted({by_id[m]["owner_node"] for m in rel_mems})
        preds = {s for s, d, _t in EDGES if d == cur_node}
        bad = [s for s in rel_sources if s not in preds]
        if bad:
            raise ValueError(f"ep{idx} {cur_node} 标注源不在其上游: {bad}（上游={sorted(preds)}）")
        episodes.append({
            "episode_id": f"gep_{idx:04d}",
            "task_tag": TASK_TAG,
            "user_id": "u_tu",
            "scenario": "multi_module_project",
            "business": "srtp",
            "query": query,
            "cur_node": cur_node,
            "now": TIME_BASE + len(MODULES) * NODE_SPAN_H * HOUR + HOUR,
            "nodes": nodes,
            "edges": [{"src": s, "dst": d, "type": t} for s, d, t in EDGES],
            "corpus": corpus,
            "relevant": sorted(rel_mems),
            "relevant_sources": rel_sources,
            "scene": "fact",
        })

    # E1 场景 episodes：S-A 时序演进（考 time）/ S-B 口径隔离（考 task）
    ep_now = TIME_BASE + len(MODULES) * NODE_SPAN_H * HOUR + HOUR
    for ci, ch in enumerate(SA_CHAINS, 1):
        cur_id = f"{ch['owner']}_sa{ci}v{ch['cur']}"
        if cur_id not in by_id:
            raise ValueError(f"S-A 链{ci} 当前版记忆缺失: {cur_id}")
        episodes.append({
            "episode_id": f"sa_{ci:02d}",
            "task_tag": TASK_TAG,                # 与记忆同口径 → 纯考 time
            "user_id": "u_tu",
            "scenario": "multi_module_project",
            "business": "srtp",
            "query": f"{ch['subject']}是多少",
            "cur_node": ch["cur_node"],
            "now": ep_now,
            "nodes": nodes,
            "edges": [{"src": s, "dst": d, "type": t} for s, d, t in EDGES],
            "corpus": corpus,
            "relevant": [cur_id],                # 只认当前版；旧版 = 干扰
            "relevant_sources": [ch["owner"]],
            "scene": "SA_time",
        })
    for pi, pm in enumerate(SB_PARAMS, 1):
        cur_id = f"{pm['owner']}_sb{pi}_{pm['q_task']}"
        if cur_id not in by_id:
            raise ValueError(f"S-B 参数{pi} 口径记忆缺失: {cur_id}")
        episodes.append({
            "episode_id": f"sb_{pi:02d}",
            "task_tag": pm["q_task"],            # 查询口径 → task 头是唯一区分维
            "user_id": "u_tu",
            "scenario": "multi_module_project",
            "business": "srtp",
            "query": f"{pm['subject']}是多少",
            "cur_node": pm["cur_node"],
            "now": ep_now,
            "nodes": nodes,
            "edges": [{"src": s, "dst": d, "type": t} for s, d, t in EDGES],
            "corpus": corpus,
            "relevant": [cur_id],                # 只认该口径；异口径 = 干扰
            "relevant_sources": [pm["owner"]],
            "scene": "SB_task",
        })

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    types: dict[str, int] = {}
    for _s, _d, t in EDGES:
        types[t] = types.get(t, 0) + 1
    forked = [n for n in node_ids if sum(1 for s, _d, _t in EDGES if s == n) >= 2]
    multi = [e for e in episodes if len(e["relevant_sources"]) >= 2]
    n_stale = sum(1 for c in corpus if c.get("kind") == "stale_version")
    n_xtra = sum(1 for c in corpus if c.get("kind") == "cross_task")
    n_sa = sum(1 for c in corpus if c.get("kind") == "sa_version")
    n_sb = sum(1 for c in corpus if c.get("kind") == "sb_param")
    scenes: dict[str, int] = {}
    for e in episodes:
        scenes[e["scene"]] = scenes.get(e["scene"], 0) + 1
    print(f"OK {out}  ({len(episodes)} episodes, scenes={scenes})")
    print(f"   nodes={len(node_ids)} edges={len(EDGES)} memories={len(corpus)} "
          f"(core=280, stale={n_stale}, cross_task={n_xtra}, sa_versions={n_sa}, sb_params={n_sb})")
    print(f"   edge types: {types}")
    print(f"   forked(outdeg>=2): {len(forked)}")
    print(f"   |relevant| set: {sorted({len(e['relevant']) for e in episodes})}")
    print(f"   multi-source queries: {len(multi)}/{len(episodes)}")
    print("   NOTE: S-A/S-B 场景的主指标是 MRR/nDCG（|relevant|=1，recall@8 无区分度）；")
    print("         干扰项不进任何 relevant 集。")


if __name__ == "__main__":
    main()
