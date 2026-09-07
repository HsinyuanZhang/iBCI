# 两条主线阶段审查：M2 小 Transformer 与 M1 P preflight

日期：2026-09-05  
职责：Astra规划/审核，不执行正式训练。  
总判决：**两条主线继续；M2既有配方的预注册确认未成立；M1 P保留但当前正式12epoch不批准，先修执行隔离和验证缺口。**

## 0. 固定的研究目标，不被单个null或工程问题替代

1. **Decoder的创新设计探索。** 研究怎样组织神经元集合、校准条件信息和时间动态，在强SPINT-like参照上得到可复制的收益。压缩、LR、EMA和选点是使结构比较可信的条件，不是第二个结构创新本身。当前B配方不成立不等于Transformer/SSM路线无效。
2. **T4 carrier跨数据集的通用解释和性能提升。** 统一的是activity signature、任务相关tuning profile与源训练consumer的作用关系，不是要求M2/688/H1/M1共用同一生理坐标或字面cosine四列。解释与改善都要产出，不能只剩FiLM救援或对null的叙述。

每次后继工单都应说明：属于哪条主线、改变哪个算子、同面强对照是谁、获得什么证据才可晋升。暂不再扩M2既有LR网格，不关闭decoder结构探索；暂停有问题的P正式执行，不撤销其校准感知估计假设。

## 1. 证据与本次实际核查

入口：`tfpd_exploration/results/m2_b_small_stability_v1/astra_pack_v1/HANDOFF_FOR_ASTRA.md`。

- 入口SHA：`e6dee4cf73488186fc34f682743b2ec930c7572fe7fdeb3c977f8f965deab8bc`。
- `matrix_2x2_both_seeds.csv` SHA：`4a67ce5a83988c698d10ca5b7e22812638f27af45709624879f2eeb890c8ad23`。
- 由两seed逐session CSV重新计算全部16行的source-pick、endpoint24、last4/8均值与总体std，全部一致（容差1e-10）。
- ext-4明细1536行，每(cell,seed,view,epoch)恰4session；window counts固定519/490/425/635，未发现缺失R²。
- 包内P的normalizer、consumer_tests、disposable_profile、profile_cost_estimate与`20260905_123100`原件字节一致。
- 阅读v2 seed43选点工单及完整2×2的用户授权说明；不把后来新增的N格/seed43错称未授权，亦不回溯修改其选择统计量。
- 本次未重跑decoder评分、未打开新NWB、未启动GPU、未修改模型代码或旧结果。另做了内存内factory隔离复现及实际parent的CPU权重加载/模式检查。

## 2. M2：预注册结论不改写

唯一REF：`R_REF_clean=0.3582396424175502`，M33-disjoint clean ext-4；禁止减历史0.360、不同query面或官方HO。

| S1-SMALL-COS / EMA | seed42 | seed43 | 判定 |
|---|---:|---:|---|
| v1 source-picked ext-4 | e2: 0.276861 | e10: 0.333615 | 两seed均未过原路由 |
| endpoint24 | 0.449450 | 0.337340 | 42为产生v2假设的探索证据；43的预注册v2确认未复制其高分 |
| last8均值 | 0.438982 | 0.333353 | 分裂不只是某一epoch偶然取值 |
| last8总体std | 0.010149 | 0.006272 | 两条都相对平稳，但平稳中心相差0.105629 |

seed43的endpoint24低REF约0.020900。不得用两个seed平均后为正、事后改选S0/N0或改median来宣布原S1主张成立。合法visible-development选点继续显示，但不是独立确认。

### 2.1 分裂至少包含两个不同现象

**选择面错位（seed42）**：S1/EMA的source-minival e2→e24等session均值从0.268101降至0.195959；同期间ext-4从0.276861升到0.449450。source 10-20两session变化对7session均值贡献−0.137810，其余5session贡献+0.065668。median从0.274551升至0.344494是描述性敏感性，不授权事后更换主选择器。

这不是选择器实现出错，也不能将10-20删掉称“清理数据”；它是当前source风险与outer风险排序不一致。

**训练轨迹与调度的交互（两seed）**：固定last8、EMA权重视图：

| 对照 | seed42 Δ | seed43 Δ |
|---|---:|---:|
| cosine−legacy，3e-4 | +0.059943 | −0.060530 |
| cosine−legacy，1e-4 | +0.041342 | −0.057010 |

两个LR档均反向，不能继续写“cosine全面改善迁移”。可能是初始化/随机训练路径与衰减时机产生不同解；这只是与数据相容的解释，未证明具体损失盆地或唯一根因。

e24的S1 seed42−43在4个ext-session上均为正，分别+0.161785/+0.026677/+0.068172/+0.191807。因此也不只是11-19一个日期造成整体差异。不要把seed43叫异常值或丢掉。

### 2.2 已获得的有用证据

- 在预先固定last8口径下，8个cell/seed组合的EMA−RAW均值均为正，范围+0.004153到+0.021944；EMA/RAW的std比为0.235–0.702。支持本包中的轨迹平滑/平均收益，非8个独立统计复制，也非新的结构机制。
- 小模型3.543M并未表现出必然的容量不足。历史seed42、同legacy制度、RAW last8：小模型0.358354，旧大Transformer0.340155。是单seed历史缩窄参照，不是全面的等容量结构胜出。
- last-k均值是多个模型分数的均值，不是某个可部署模型的分数或预测集成结果。

### 2.3 不要被均值掩盖的安全门

| EMA、source-picked | ext-4均值 | Δ对REF | worst-session Δ |
|---|---:|---:|---:|
| N0 seed42 | 0.381298 | +0.023059 | −0.121495 |
| N0 seed43 | 0.412441 | +0.054202 | −0.072412 |
| N1 seed42 | 0.404665 | +0.046425 | −0.063623 |
| N1 seed43 | 0.363152 | +0.004912 | −0.059887 |

原B安全线为无session Δ<−0.05。N格不能仅凭平均为正晋升。11-19的“洞”是性能低，不是缺评分。例如N0 seed42 e24在11-19为−0.125544，对REF的该session 0.153059差−0.278603。

S1 seed42那个0.449450端点也不是所有session更好：10-30 Run2=0.391420，对REF 0.492630差−0.101210。若继续沿用原安全线，它同样不通过；不因总体最高而隐藏这项限制。

### 2.4 主线一的资源决定

当前8次小模型训练及权重视图信息已足够，不自动再加seed44、48epoch、LR档或事后selector来“找回0.45”。S0/N0稳定的较高末段可保留为下一项结构研究的候选训练参照，不能现在改立为原主臂。

**保留decoder创新主线，结束这一张配方搜索表。** 下一份decoder工单应提出一个新的、明确的结构假设，并与SPINT-like及冻结的小Transformer参照在相同数据/预算/选点上比较；不能把EMA、压缩或重新取epoch当创新。无自动新Mamba训练，本结论也不排除未来有依据的SSM设计。

## 3. P：方法保留，但“preflight三项全过”超出了测试覆盖

接受且不撤回：S-Fix/teacher命名、float64 ridge目标、active-set NNLS的分段梯度、D0共同source normalizer策略。ReLU锁8/48零元素是已知能力限制，不是停止整个路线的理由。

**当前不批准正式12epoch。主要原因不是7 GPU-hours太贵，而是以下执行正确性缺口。**

### P1. 两臂不是独立consumer初始化【阻塞】

`model_adapter.py:171`逐臂调用`build_p_pair`，但`:609`通过全局`_PAIR`返回同一对象；`del pair`不清全局引用。按该提交代码路径，P-CA继承P-FIX更新100次后的student，而非重新从S-Fix开始。

本次用内存内dummy factory复现：第二次调用`a is b=True`，第一次对象的状态修改在第二次仍可见。无需打开数据即可证明该factory不具备所声称的隔离。构造时克隆的两份state dict相同，不能替代两次真实训练开始时的hash检查。

修复要求：共享只读D0/support允许缓存；**可训练student/basis/optimizer不共享全局缓存**。独立进程或真正fresh factory，两个真实step0记录相同consumer/D0 SHA；验证先训练P-FIX不会改变尚未训练的P-CA。

### P2. profile实际在eval模式做更新【阻塞】

`_load_sfix_student`调用`lit.eval()`；GPU循环未调用`student.train()`。本次实际CPU权重加载确认student/decoder/id_encoder全部`training=False`，原decoder的dynamic dropout=True、Transformer dropout=0.1但处于关闭状态。

eval不禁止反传，因此100steps确实可以更新权重；但它不是声明的完整训练制度。修复时显式train/eval切换，绑定实际dropout、bias/norm decay排除、LR和专用seed。profile代码中只把seed写入spec并不等于已设置该RNG。

### P3. “resume通过”没有执行恢复【阻塞】

`:562 new_stage_resume_roundtrip`只是把payload字段重新装到一个dict，再返回`roundtrip_ok=True`。没有重新构建模型、load optimizer、恢复RNG或比较下一步。

同时`:537 new_stage_payload`的sampler index/epoch固定0、没有CUDA RNG/真实global step，normalizer只有receipt而没有可直接恢复的数组。不能叫full new-stage resume通过。

修复要求：真实磁盘save/load、带非空AdamW状态、真实sampler游标、CPU/CUDA RNG与固定normalizer数组/字典；恢复后下一batch、dropout、LR、loss、梯度和参数更新与未中断路径匹配。

### P4. parent carrier parity证据不独立【需补】

`normalizer.py:173`验证的是新旧μ/σ；`:184 parent_normalized`却用**新拟合的source_raw**配旧μ/σ生成。随后用该对象做“parent”预测比较，主要证明相同新raw在两套近似相等normalizer下相等，未独立验证旧D0/旧逐unit carrier。

D0 SHA不同不自动意味着错误，可能只是数值末位变化；但均值/std相同也不证明每个unit行相同。需要独立旧字典/逐unit参考或绑定原环境的重建与数值parity；不可因当前预测差0就宣称该问题已证明解决。保持两个新臂相同初始化，同时如实区分原parent算子回放与重物化初值。

### P5. 测试名称比实际断言更强【收紧】

- `source_query_loss_dictionary_grad`使用随机neural与全零synthetic target；证明live consumer链可导，但不是真实source query EMG loss的证据。profile应记录真实loss到D的梯度。
- `query_history_disjointness`只把一个显式忽略的`query_labels`参数改值，未审计真正loader窗口边界。
- `target_fit_no_backward`在no_grad下构建carrier是正确的，但函数本身未比较前后source状态hash；不能只用固定返回字段代替测试。
- `passed=[]/true`与44 tests通过不抵消上述断言覆盖不足；应增加会在当前错误实现上失败的测试。

这些发现只针对当前P preflight，不回溯撤销M2矩阵或早先纯ridge/NNLS gradcheck。

## 4. P成本与下一步的范围

`placeholder_bank=true`**不是伪造数据**：它是既有loader第五项carrier的占位，实际profile使用真实source神经/EMG并在每step重算carrier。不要仅因这个字段判NO-GO。

但7.0378 GPU-h目前只能作粗估：100step profile为eval模式、有跨臂继承、只取loader前100个batch、updates/epoch由旧59412/12推算；计时未分别可靠同步CUDA，`active_set_solve=0`实际被包入encode，不能说明无该成本。未绑定正式source选择/训练manifest也不能承诺正式4951 updates。

有一个不改方法的直接降本点：**P-FIX的D/support/μ/σ全固定，可每source session预计算z及carrier**；仅P-CA需随当前D重算。它不减少标签或训练窗口曝光，也不妨碍两臂匹配consumer更新。不要为“计算量一样”重复执行P-FIX的确定性NNLS。P-CA可按active-set分组批量solve，必须保留原解与梯度parity，不能detach或改估计器。

下一步仅建议执行者完成：

1. P1–P3的fresh初始化、train模式与真实resume修复，P4独立参考和P5真实数据断言补齐；不扩展科学方法。
2. 输出正式source训练/选择/outer报告window-ID manifest。source选择若被parent/字典历史接触应披露；不能用outer查询调基础对象。
3. 新根进行一次修正后的100step配对profile，覆盖3source session，分别记录成本/VRAM/实际可训参数和step0/step100状态；旧root只读。
4. 同时量出P-FIX安全缓存后的总成本。若仍需约7 GPU-h，不能仅凭费用否定：是否投入取决于上面门是否真实通过、配方是否闭合。当前尚不能批准该正式对。

本审核不启动修复、profile或正式GPU训练；也不将“等Astra”当作反复请求系统权限的理由。执行者可按已有preflight工程整改范围准备修正，提交真实通过的有限证据后再做正式阶段决定。E2 lag不进入第一格。

## 5. 主线二的科学读数与未来晋升

H1的backward H-C对群体组成敏感，是估计器方向的解释证据，不自动导出某个增强就有效；M1 rSyn3稳定而未胜Zero4，说明可重复的系数不保证对强consumer有增量。

P应检验：**通过短calibration估计器和真实source query目标联合优化的行为坐标，能否改善强activity+carrier consumer？** 不是再次用FiLM去证明一切，也不是只在弱ridge上取正数。

若P以后跑通，至少分开报告P-CA−P-FIX、两者相对同面S-Fix parent，以及现有强activity-only Z-Fix上下文。超过P-FIX但仍低于强无carrier参照，只能称“改进carrier系统”，不能说证明了任务标签净增量；内容机制与更广跨session复制另行做。

始终保留两份产出：跨数据集作用/估计器差异的可解释框架，以及至少一项匹配强基线的性能检验。没有一项正数时不硬写普适性能；有一项正数时也不外推同一四维生理轴适用于所有任务。

## 6. 最终阶段决定

| 对象 | 决定 |
|---|---|
| Decoder创新主线 | 保留；下一阶段须是明确结构假设与固定参照，不继续无界配方搜索 |
| M2现有S1预注册性能主张 | v1未过；v2 seed43未复制；全部历史数字保留 |
| M2 EMA/压缩证据 | 有用的优化/工程证据，不是独立结构创新 |
| T4/tuning profile跨数据集主线 | 保留；解释与匹配性能都要做 |
| M1 P named estimator | 保留，不因8个锁零元素否决 |
| 当前P preflight“全部通过” | REVISE；执行隔离、模式和验证覆盖不足 |
| P正式12epoch | **当前不批准**；不是全路线NO-GO |
| H1已关闭carrier/FiLM格、M1 dual-axis、E2 lag | 不自动重开/并入 |

所有旧receipt、矩阵、v1/v2选择权利和Stage0终端保持不可变。
