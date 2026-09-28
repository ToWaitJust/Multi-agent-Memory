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
    # ══════════════════════════════════════════════════════════════════
    # D-045 扩图：新增 8 个模块，目的是把"平均上游源数"从 1.6 提到 2.1，
    #   并造出多个**高上游汇多点**（4~5 个上游）。原因：D-044 证明
    #   "强度配额 vs 均匀配额"要在 `K > 源数` 且源数较多时才有倾斜空间，
    #   14 节点 / 21 边的图**天然测不出配额差异**（图规模局限）。
    # ══════════════════════════════════════════════════════════════════
    {"id": "n_api", "topic": "接口与调用层", "mems": [
        "调度入口是 schedule_once 单次调用", "应用层只依赖调度结果不感知图内部",
        "接口返回调度明细含四维分与最终分", "共享池内容随调度结果一并返回",
        "调用层不缓存调度结果避免陈旧", "批量查询走同一入口串行执行",
        "接口错误以异常抛出不做静默降级", "冷启动首次调用会触发建图",
        "调用参数含查询文本与条件键", "返回结构稳定不因插件更换而变",
        "调用层不直接读写记忆文件", "单次调用只做一次向量化",
        "接口版本号随不兼容变更递增", "调用日志记录耗时与命中条数",
        "调度失败时返回空池而不是旧池", "调用层不做业务相关过滤",
        "并发调用由上层串行化", "接口不含图结构的隐式约定",
        "参数校验在入口完成", "返回值不含中间态调试字段",
    ]},
    {"id": "n_ui", "topic": "可视化界面", "mems": [
        "控制台左侧是节点上下文历史", "右侧是调度流水线与双池视图",
        "双池视图用散点展示常驻池与共享池", "会话画布用泳道展示主线与副线",
        "图拓扑页用专业图库渲染", "图拓扑支持缩放平移与框选",
        "节点标签显示主题与节点标识", "边宽映射边强度",
        "虚线表示被降级的环边", "界面文案全部使用中文",
        "选源结果用橙色高亮源节点", "当前节点用光晕区分",
        "右侧面板分仪表与详情两态", "统计口径与后端保持一致",
        "界面不暴露内部字段名", "配色遵循统一设计令牌",
        "布局支持分层环形网格三种", "深链可直达某个节点的选源视图",
        "提示信息避免技术黑话", "空态给出明确的下一步引导",
    ]},
    {"id": "n_auth", "topic": "用户与场景切换", "mems": [
        "条件键由用户场景业务三段拼成", "切换用户会切换整套条件切片",
        "场景切换不重建图只换条件", "条件向量维度固定 32 维",
        "只有语义与任务维随条件漂移", "用户隔离靠条件键而不是多进程",
        "切换后退回默认任务标签", "场景定义放配置文件不硬编码",
        "未识别的条件键回落到默认", "切换过程不做全量重算",
        "条件键是记忆隔离的唯一依据", "同一用户多场景共用一份图",
        "跨场景记忆默认不共享", "切场景不清理常驻池",
        "条件切片只影响权重不改记忆", "用户标识不进入记忆正文",
        "权限只区分读写不区分节点", "会话标识与会话节点一一对应",
        "切换时保留当前会话上下文", "条件键解析失败要显式报错",
    ]},
    {"id": "n_prod", "topic": "生产运行与容量", "mems": [
        "生产环境与演示环境共用同一套代码", "容量按共享池上限推算",
        "单进程部署避免跨进程同步", "内存占用主要是向量缓存",
        "长跑需定期清理过期缓存", "生产日志分级落盘",
        "异常重启后图从磁盘恢复", "不依赖外部数据库只用文件",
        "容量不足时优先降候选集规模", "生产开关与实验开关分离",
        "生产默认关闭自适应档位", "服务以守护进程方式托管",
        "端口开放需在云厂商侧放行", "长任务必须后台执行避免超时",
        "生产数据不含个人身份信息", "容量评估按峰值查询数估算",
        "生产不开启调试字段输出", "运行指标按小时聚合",
        "故障恢复以重启为第一手段", "生产变更需有回滚路径",
    ]},
    {"id": "n_bench", "topic": "基准测试与指标", "mems": [
        "基准测试覆盖全部五个消融组", "每组跑五十条查询取平均",
        "指标含召回命中率与排序质量", "排序质量用平均倒数排名衡量",
        "还用归一化折损累计增益衡量", "候选命中率单独报告",
        "延迟取九十分位数", "结果落盘为可追溯的清单",
        "基准测试必须可一键复现", "数值波动超过阈值需显式说明",
        "天花板接近一说明标注偏松", "裁剪后召回需与天花板同时报告",
        "候选命中率与最终召回分开看", "基准分组按场景切分统计",
        "多源样本比例需一并报告", "压缩比按注入前后条数算",
        "指标计算不依赖人工干预", "基准数据与结论一并归档",
        "随机种子固定以保证复现", "基准不因提升单场景而调参",
    ]},
    {"id": "n_abtest", "topic": "A/B 对比实验", "mems": [
        "对比实验每次只改一个变量", "变量清单写进配置文件",
        "对照组与实验组共用数据集", "种子固定保证可复现",
        "差异需在两个以上场景同时成立", "单场景优势不作为结论",
        "结果同时报告均值与天花板", "天花板过近说明标注过松",
        "对比实验不做参数微调", "结论与原始数据一并归档",
        "对照组的选择本身要可解释", "变量名与配置文件字段一一对应",
        "对比前先确认基线可复现", "差异不到一个百分点不宣称有效",
        "负结果同样要归档", "场景设计要覆盖模式的假设条件",
        "评估指标要能分辨目标能力", "约束要真实否则测不出差异",
        "样本量不足时明确标注", "对比结论写明适用条件",
    ]},
    {"id": "n_regress", "topic": "回归验证", "mems": [
        "回归用例覆盖全部公开接口", "每次改动前先跑全量用例",
        "用例断言数值而非仅状态", "关键路径的数值逐位比对",
        "回归失败优先查数据而非代码", "新增功能必须补对应用例",
        "用例总数是质量下限", "不稳定用例需标注原因",
        "回归在本地跑不依赖外部服务", "通过标准是全绿而非部分通过",
        "回归覆盖配额与排序两条链路", "数值回归比对到小数点后四位",
        "接口变更需同步更新用例", "回归用例不做随机断言",
        "失败用例要能单独复现", "回归耗时控制在分钟级",
        "回归结果写入清单留档", "回归不覆盖性能只覆盖正确性",
        "回归用例命名体现被保护的行为", "回归是发布的前置条件",
    ]},
    {"id": "n_release", "topic": "发布验收", "mems": [
        "发布前确认回归全绿", "发布前确认基准测试可复现",
        "发布前确认文档与实现一致", "验收清单逐项打勾",
        "发布产物含运行清单", "发布说明写明变更与影响面",
        "回滚方案必须提前准备", "验收不通过不进入发布",
        "发布后观察一轮真实查询", "发布记录留存发布时刻与版本",
        "发布包与仓库提交一一对应", "验收标准以可复现为准",
        "发布不附带未验证的实验代码", "验收需覆盖配额模式差异",
        "发布说明不含内部实现细节", "验收以指标为准不以主观判断为准",
        "发布后指标回落需能定位", "验收清单可被第三方照着复跑",
        "发布不带调试开关", "验收通过才更新文档状态",
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
    # D-045 扩图：应用侧 + 评估侧两条新分支，形成多个高上游汇多点
    #   n_api(2) / n_ui(4) / n_auth(2) / n_prod(4) / n_bench(4) / n_abtest(2)
    #   / n_regress(2) / n_release(5)  —— 4~5 上游的节点共 4 个
    ("n_retrieval", "n_api", "depends_on"),
    ("n_pool", "n_api", "depends_on"),
    ("n_demo", "n_ui", "derives_from"),
    ("n_api", "n_ui", "depends_on"),
    ("n_pool", "n_ui", "depends_on"),
    ("n_attention", "n_ui", "similar_to"),      # 弱边（0.3）——S-F 干扰挂这里
    ("n_config", "n_auth", "depends_on"),
    ("n_weights", "n_auth", "similar_to"),
    ("n_deploy", "n_prod", "derives_from"),
    ("n_api", "n_prod", "depends_on"),
    ("n_auth", "n_prod", "depends_on"),
    ("n_config", "n_prod", "references"),
    ("n_metrics", "n_bench", "depends_on"),
    ("n_ablation", "n_bench", "depends_on"),
    ("n_graph", "n_bench", "depends_on"),
    ("n_retrieval", "n_bench", "references"),
    ("n_bench", "n_abtest", "depends_on"),
    ("n_demo", "n_abtest", "references"),
    ("n_bench", "n_regress", "depends_on"),
    ("n_paper", "n_regress", "references"),
    ("n_prod", "n_release", "derives_from"),
    ("n_regress", "n_release", "depends_on"),
    ("n_abtest", "n_release", "depends_on"),
    ("n_paper", "n_release", "references"),
    ("n_ui", "n_release", "similar_to"),        # 弱边（0.3）——S-F 干扰挂这里
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
    ("n_attention", "教学演示口径：阈值示例取 0.5", "teaching"),
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
# 主题选择原则（F1 实测后修正）：**链主题必须是该模块 20 条核心记忆里没有的参数**。
#   原因：第一版选了"共享池上限/调度阈值/维度"等核心参数，结果链内"当前版"与模块内
#   已有的同义记忆（如 n_config_m2）语义高度重叠、且后者时间戳更晚 →
#   版本过滤把"当前版"误判为旧版本而剔除，S-A 的 MRR 反而从 0.806 掉到 0.667。
#   换成模块内不存在的参数后，版本过滤只在链内 3 条之间生效，不会误伤。
SA_CHAINS: list[dict] = [
    {"owner": "n_config", "subject": "日志保留天数",
     "versions": [("第一版", "七天"), ("第二版", "十四天"), ("当前版", "三十天")], "cur": 3,
     "cur_node": "n_embed"},
    {"owner": "n_retrieval", "subject": "失败重试次数上限",
     "versions": [("第一版", "一次"), ("第二版", "两次"), ("当前版", "三次")], "cur": 3,
     "cur_node": "n_attention"},
    {"owner": "n_embed", "subject": "批量编码的批大小",
     "versions": [("第一版", "十六条"), ("第二版", "三十二条"), ("当前版", "六十四条")], "cur": 3,
     "cur_node": "n_retrieval"},
    {"owner": "n_weights", "subject": "先验缓存的容量",
     "versions": [("第一版", "一百二十八条"), ("第二版", "二百五十六条"), ("当前版", "五百一十二条")], "cur": 3,
     "cur_node": "n_feedback"},
    {"owner": "n_metrics", "subject": "多源覆盖率的验收线",
     "versions": [("第一版", "零点一"), ("第二版", "零点二"), ("当前版", "零点三")], "cur": 3,
     "cur_node": "n_paper"},
    {"owner": "n_graph", "subject": "源裁剪的软上限阈值",
     "versions": [("第一版", "两个"), ("第二版", "四个"), ("当前版", "五个")], "cur": 3,
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
    {"owner": "n_retrieval", "subject": "调度阈值的默认值",
     "values": {"ops": "0.5", "research": "0.6", "demo": "0.4"}, "q_task": "ops",
     "cur_node": "n_attention"},
    {"owner": "n_metrics", "subject": "单条打分的耗时要求",
     "values": {"ops": "五毫秒", "research": "十毫秒", "demo": "二十毫秒"}, "q_task": "ops",
     "cur_node": "n_paper"},
    {"owner": "n_embed", "subject": "向量化输出的维度",
     "values": {"ops": "512 维", "research": "1024 维", "demo": "768 维"}, "q_task": "ops",
     "cur_node": "n_retrieval"},
]

# ── S-C 链式依赖场景（考图选源 vs 语义检索，E2）─────────────────────────────
# 设计：查询只提"目标模块的任务"，relevant 分散在 **1 跳上游的多个源池**里，
#       且相关记忆**不含查询的核心词**（字面不相似）——语义检索即便全池也难命中；
#       单源（G2）只取最强边源 → 必漏其他源的相关记忆。
#       主指标：**cand_hit_rate（相关记忆进候选的比例）**——"多对多选源"的直接度量；
#       MRR/nDCG/recall 作参考（语义排序会把字面不相似的相关记忆排后）。
SC_QUERIES: list[tuple[str, str, list[str]]] = [
    # n_deploy 的 1 跳上游：n_demo(0.8 强)、n_config(0.3 弱) —— relevant 跨强弱两源
    ("n_deploy", "演示服务上线前需要确认哪些前置条件",
     ["n_config_m1", "n_config_m2", "n_config_m3", "n_demo_m11", "n_demo_m12"]),
    # n_paper 的 1 跳上游：n_graph(0.8)、n_dataset(0.8)、n_metrics(0.5) —— 三源
    # 改 cur=n_ablation（上游 n_attention(0.8)/n_weights(0.8)）——
    # 原先用 n_paper 与 S-F 查询2 共用源池，两个场景的注入项会互相污染（D-040 教训）
    ("n_ablation", "消融章节需要交代哪些机制设定",
     ["n_attention_m1", "n_attention_m2", "n_weights_m1", "n_weights_m2"]),
    # n_feedback 的 1 跳上游：仅 n_weights(0.8) —— 单源对照组（G2 不吃亏）
    ("n_feedback", "反馈闭环实现前要确认哪些机制约束",
     ["n_weights_m2", "n_weights_m8", "n_weights_m16"]),
]

# ── S-F 源池不均场景（考强度配额 vs 均匀配额，E2）───────────────────────────
# 设计原则（涂总：正确的模式应被坚持，须给它公平的检验场景）：
#   相关记忆集中在 **强边源**（depends_on 0.8），弱边源（similar_to 0.3）里塞入
#   **与查询高度相似**的干扰记忆（下方 SF_DISTRACTORS，task_tag=research）。
#   预测：均匀配额（G3）给弱边源同等名额 → 高相似干扰挤进 kept；
#         强度配额（G4）把名额让给强边源 → relevant 进 kept。
#   若 G4 在此场景仍不占优，才可下"强度配额无价值"的结论。
SF_DISTRACTORS: list[tuple[str, str]] = [
    # (owner=弱边源, 文本) —— 文本与 SF 查询高度相似，但内容是"示例/口径"而非机制
    ("n_config", "会话画布示例：共享池上限在示例环境取六条"),
    ("n_deploy", "论文素材示例：共享池容量在论文配图取十二条"),
]
SF_QUERIES: list[tuple[str, str, list[str], str]] = [
    # (cur_node, 查询, relevant, 干扰来源说明)
    # ⚠️ D-043：**双向**设计（一个测"强度配额该赢"、一个测它的"风险场景"）；
    # ⚠️ D-045：改用**高上游汇多点** + 配合 K > 源数，配额才有倾斜空间。
    # 【SF1 · 4 上游】cur=n_ui：
    #   n_demo(0.9 derives_from) / n_api(0.8) / n_pool(0.8) / n_attention(0.3 similar_to)
    #   relevant 全在强源；干扰挂最弱的 n_attention
    ("n_ui", "可视化界面从哪些模块取得数据",
     ["n_demo_m11", "n_api_m1", "n_pool_m1"], "4 上游汇多点；干扰挂最弱边源 n_attention(0.3)"),
    # 【SF2 · 5 上游 + 风险场景】cur=n_release：
    #   n_prod(0.9) / n_regress(0.8) / n_abtest(0.8) / n_paper(0.5) / n_ui(0.3 similar_to)
    #   relevant 含**中等强度源** n_paper(0.5)——若强度配额把它压到 0 席就会漏
    ("n_release", "发布验收前要确认哪些前置条件",
     ["n_prod_m1", "n_regress_m1", "n_paper_m1"], "5 上游汇多点；干扰挂最弱边源 n_ui(0.3)"),
]

# ── 查询表（人工标注 ground truth，与边强度独立）─────────────────────────────
# 注：S-C/S-F 场景查询见下方 SC_QUERIES / SF_QUERIES（relevant 须为 1 跳上游池内记忆）。
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

    # E. S-F 高相似干扰（挂在弱边源，考强度配额 vs 均匀配额）。
    for k, (owner, text) in enumerate(SF_DISTRACTORS, 1):
        if owner not in node_idx:
            raise ValueError(f"S-F 干扰项 owner 未知: {owner}")
        ni = node_idx[owner]
        corpus.append({
            "memory_id": f"{owner}_sf{k}",
            "text": text,
            "task_tag": TASK_TAG,
            "owner_node": owner,
            "timestamp": TIME_BASE + ni * NODE_SPAN_H * HOUR + 2.0 * HOUR,
            "kind": "sf_distractor",
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
            # 评估修正：|relevant| 1 -> 3。原来只有 1 条时 recall 仅 {0,1} 两档、
            # MRR 退化为"第一条是否命中"，对排序质量几乎没有分辨力。
            # 补入同模块另外 2 条**现行核心记忆**（真实场景：问某参数现值时，
            # 该模块的其他现行配置同样是回答所需上下文）。
            "relevant": [cur_id] + [f"{ch['owner']}_m{x}" for x in (1, 2)],
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
            # 评估修正：|relevant| 1 -> 3（理由同 S-A）
            "relevant": [cur_id] + [f"{pm['owner']}_m{x}" for x in (1, 2)],
            "relevant_sources": [pm["owner"]],
            "scene": "SB_task",
        })
    # E2 场景：S-C 链式依赖（考图选源 vs 语义检索，主指标 cand_hit_rate）
    for qi, (cur_node, query, rel_mems) in enumerate(SC_QUERIES, 1):
        missing = [m for m in rel_mems if m not in by_id]
        if missing:
            raise ValueError(f"S-C 查询{qi} 标注引用不存在的记忆: {missing}")
        rel_sources = sorted({by_id[m]["owner_node"] for m in rel_mems})
        preds = {s for s, d, _t in EDGES if d == cur_node}
        bad = [s for s in rel_sources if s not in preds]
        if bad:
            raise ValueError(f"S-C 查询{qi} {cur_node} 标注源不在其上游: {bad}")
        episodes.append({
            "episode_id": f"sc_{qi:02d}",
            "task_tag": TASK_TAG,
            "user_id": "u_tu",
            "scenario": "multi_module_project",
            "business": "srtp",
            "query": query,
            "cur_node": cur_node,
            "now": ep_now,
            "nodes": nodes,
            "edges": [{"src": s, "dst": d, "type": t} for s, d, t in EDGES],
            "corpus": corpus,
            "relevant": sorted(rel_mems),        # 跨多源、字面不相似
            "relevant_sources": rel_sources,
            "scene": "SC_chain",
        })
    # E2 场景：S-F 源池不均（考强度配额 vs 均匀配额）
    for qi, (cur_node, query, rel_mems, _note) in enumerate(SF_QUERIES, 1):
        missing = [m for m in rel_mems if m not in by_id]
        if missing:
            raise ValueError(f"S-F 查询{qi} 标注引用不存在的记忆: {missing}")
        rel_sources = sorted({by_id[m]["owner_node"] for m in rel_mems})
        preds = {s for s, d, _t in EDGES if d == cur_node}
        bad = [s for s in rel_sources if s not in preds]
        if bad:
            raise ValueError(f"S-F 查询{qi} {cur_node} 标注源不在其上游: {bad}")
        episodes.append({
            "episode_id": f"sf_{qi:02d}",
            "task_tag": TASK_TAG,
            "user_id": "u_tu",
            "scenario": "multi_module_project",
            "business": "srtp",
            "query": query,
            "cur_node": cur_node,
            "now": ep_now,
            "nodes": nodes,
            "edges": [{"src": s, "dst": d, "type": t} for s, d, t in EDGES],
            "corpus": corpus,
            "relevant": sorted(rel_mems),        # 集中在强边源；弱边源有高相似干扰
            "relevant_sources": rel_sources,
            "scene": "SF_skew",
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
    n_sf = sum(1 for c in corpus if c.get("kind") == "sf_distractor")
    scenes: dict[str, int] = {}
    for e in episodes:
        scenes[e["scene"]] = scenes.get(e["scene"], 0) + 1
    print(f"OK {out}  ({len(episodes)} episodes, scenes={scenes})")
    print(f"   nodes={len(node_ids)} edges={len(EDGES)} memories={len(corpus)} "
          f"(core=280, stale={n_stale}, cross_task={n_xtra}, "
          f"sa={n_sa}, sb={n_sb}, sf_distract={n_sf})")
    print(f"   edge types: {types}")
    print(f"   forked(outdeg>=2): {len(forked)}")
    print(f"   |relevant| set: {sorted({len(e['relevant']) for e in episodes})}")
    print(f"   multi-source queries: {len(multi)}/{len(episodes)}")
    print("   NOTE: S-A/S-B 主指标 MRR/nDCG；S-C 主指标 cand_hit_rate（图选源价值）；")
    print("         S-F 检验强度配额 vs 均匀配额（源池不均场景）。")


if __name__ == "__main__":
    main()
