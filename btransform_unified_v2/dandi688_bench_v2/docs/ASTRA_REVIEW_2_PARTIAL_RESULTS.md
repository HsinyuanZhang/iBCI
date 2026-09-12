# Astra 第二轮：CPU 部分结果与 B3S 合同审核

日期：2026-09-11。**CPU 正式 dev 结果：PASS；新 B3S 神经实现：尚未实现、没有可评分结果；整项实验尚未完成。**

本轮严格使用CPU只读检查，未启动GPU，未重新拟合候选，未打开真实final NWB或final缓存。用户已经终止当前GPU实验；首轮关于开跑及补充seed的批准不能覆盖这一后续停止指令。旧任务不得自动恢复，新B3S训练须等实现和后续执行安排明确后再审。

## CPU 产物独立核验

审核对象：`results/baselines_dev_formal/{selection.json,receipt.json,*.pkl,*.dev_predictions.npz}` 与 `results/prepared_2015_m33_v2`。预测与模型文件仅读取/计算hash；未执行pickle内模型或读取final。

完成以下独立检查，全部通过：

- `selection.json` 自身canonical SHA与receipt中的selection文件SHA一致。selection文件SHA为 `14d3f17df3eb177f42b3e2b21a40cd683ed032fbfa18c4cb2022ca8b3a8b0230`；status为`DEV_SELECTED`，`final_loaded=false`。
- 18source/6dev PMUA binding与prepared的split、array SHA及缓存metadata中的raw NWB SHA逐项一致。prepared的48个source/dev配对NPZ文件SHA全部吻合。
- 独立构造预定网格并核对：7个方法共144个唯一配置，未漏配置或重复计数。120个eligible候选全部有完整dev-6、有限分数、等日期均值。每方法选中项都等于原候选顺序的earliest-max。
- 7个模型文件及7个预测文件，共14个artifact SHA吻合。每个预测NPZ恰含6个dev日期。两表示dev的query物理速度逐行相同；6日期合计166,137个query的完整评价面得到保留。
- 用缓存物理速度和保存预测转float64，依照`common.py:172`独立重算42个每日期variance-weighted R²，均与receipt一致（绝对误差小于1e-12），再核对等日期均值。没有以汇总数字替代预测核验。
- selection记录的执行源码SHA除`run_campaign.py`之外均与当前源码一致；baseline、runner、data、common和共享numerics均无漂移。campaign启动管理文件的后续改变不在CPU baseline数值路径中，不要求重跑baseline；首次final仍须解决完整源码封存问题。

当前baseline selection的统一source/dev binding只记录PMUA；WF-FSS SUA实际只在本目标session的M33拟合，不消费source SUA监督。本轮额外核验SUA dev缓存文件、配对标签和保存预测，补足了这次审核的表示关联；自动seal仍应按所用表示校验缓存，不能仅靠名单相同。

## 已选 CPU dev 分数

以下均为用于选参的完整dev-6分数，不是最终泛化分数。每日期先按物理速度计算variance-weighted R²，再等日期平均。

| 方法 | 均值 R² | 已选配置 | 合格/总候选 |
| --- | ---: | --- | ---: |
| WF h0 PMUA | 0.075829 | alpha=1e5，H1，causal240ms | 4/4 |
| diag-z WF PMUA | 0.078199 | alpha=1e5，H10，causal240ms | 4/4 |
| CORAL WF PMUA | 0.078198 | alpha=1e5，H10，shrinkage=1，causal240ms | 16/16 |
| aligned-FA all PMUA | 0.115463 | alpha=1e5，K10，stable fraction=.75，H10 | 24/36 |
| aligned-FA stable PMUA | 0.132465 | alpha=1e5，K10，stable fraction=1，H10 | 24/36 |
| WF-FSS SUA | 0.220556 | alpha=1e4，H10，无平滑 | 24/24 |
| WF-FSS PMUA | 0.214321 | alpha=1e4，H10，无平滑 | 24/24 |

两个FA方法各12个不合格候选均为K20，覆盖3个stable fraction×4个alpha。记录原因均为`too few common loading rows pass the threshold for latent alignment`。这些是既定资格门拒绝整个候选，不是删除不利日期再计算均值；所选K10候选均有完整6日期。无需放宽loading阈值、替换日期或扩大K/alpha搜索。错误消息没有具体session字段，降低失败定位便利性，但不影响本次全日期资格判断。

## 日期敏感性与能说的结论

| 日期 | WF-FSS SUA | WF-FSS PMUA | SUA−PMUA | FA stable | FA all |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2015-11-03 | .233852 | .220620 | +.013231 | .076258 | −.068248 |
| 2015-11-04 | .241787 | .214972 | +.026815 | .132809 | .134244 |
| 2015-11-06 | .070354 | .086951 | −.016597 | .071678 | .079512 |
| 2015-11-09 | .191778 | .180975 | +.010804 | .128769 | .110778 |
| 2015-11-10 | .309887 | .320222 | −.010335 | .191028 | .211838 |
| 2015-11-12 | .275675 | .262184 | +.013492 | .194245 | .224653 |

**WF-FSS 的 SUA−PMUA dev均值差为+.006235，6日期中4日为正、2日为负。** 固定已选配置和预测，仅逐一去掉某日期做描述性敏感性检查，剩余5日期均值差范围为+.002119至+.010801，未因去掉任一单日反转。可称“此dev评价面上有小幅平均差异”，不能称显著优势、普遍跨日优势或神经Full/ACT/Raw的表示结论。该检查未重新选参，也不是独立验证集或置信区间。

**FA stable相对all的+.017002均值优势主要由11月3日驱动。** stable只在2/6日期更好；11月3日差为+.144507，去掉该日后平均差为−.008499。因此不能称stable posterior稳定胜过all-observed posterior。此外两个方法分别选择不同stable fraction（1与.75），本比较也不是只改变posterior的纯机制消融。

FA stable相对diag-z在6/6日期为正，均值差+.054266，去掉任一单日后仍为+.041434至+.064772；WF-FSS PMUA相对FA stable在6/6日期为正，均值差+.081856。这些描述不依赖单个日期反转，但后者的目标dense velocity标签预算更大，不构成等信息预算优势。

CORAL选中shrinkage=1，与diag-z均值只差约−5.06e-7；此dev网格未给出支持非对角协方差对齐带来收益的证据，不能把近似持平表述为CORAL有效提升。多个PMUA方法选择alpha网格上界1e5，按预先网格保留即可；不基于此次结果扩大搜索。

## strict M33 修复复核

首轮R1继续保持已闭合。当前`baselines.py:354`在平滑前把support外raw置零、平滑后再次置零；`:417` target adapter和`:453` reference统计均使用该校准stream；`:501` WF-FSS在标准化后再次置零再构造历史。query推理仍使用完整因果stream。目标速度标签只在carrier/MOVE端点读取。

正式结果绑定的`baselines.py` SHA为 `db59a5a28d23b1a21fae4cd153af7b670c6b894472666b82f9eca9ee978d0569`，与当前已修复实现相同；不是旧宽支持实现跑出的分数。第一轮的±1e6支持外扰动测试与真实source smoke证据仍适用。**无需重cache，也不要求重跑已完成的CPU144网格。**

## B3S 结构判断与旧 GPU 产物资格

新合同的结构在DANDI上直接可行：`pre_pool Linear(100,64)+ReLU → M33 trial mean → post_pool`，输出每unit E0=50；Full在mean后拼接4维MOVE-T4，ACT不读side，Raw无encoder。输入和cache的T100、M33、可变unit+mask已经满足，不需要重建缓存。M1的`SideFeatureEarlyPoolEncoder`构造函数可参数化T/E；`streaming_calibration_exp/src/models/components/streaming_encoders.py:387`–`:399`实现该结构和side列零初始化，`:410`–`:445`实现trial mean后拼接。M1 wrapper `btransform_unified_v1/src/btransform_unified_v1/m1_b3s_joint.py:70`将同一计算写成批量方式，但其硬编码T1024/N64几何不能原样用作DANDI wrapper。

必须保留的初始化细节是`post_pool[0].weight[:, 64:] = 0`，这些列在source预训练中仍可训练。它保证**初始化时即使side非零，E0也不受side影响**；不是永久抹去carrier信息。若声称与某个独立B3/ACT网络在初始化时逐字节等价，还须复制相同activity权重和bias；仅同seed构造不同fan-in的Linear再清零side列，并不自动满足这一更强说法。M1 wrapper在`:56`显式复制B3主列/bias再清零side；DANDI无匹配预训练B3可移植，直接从头初始化即可，不必新增一个B3预训练阶段。

上述“初始无side影响”只指encoder E0；Full既定decoder仍直接接收T4，不能由此声称整个Full预测初始化时不依赖carrier。新实现应消除FiLM参数和路径，保持Full训练阶段冻结表示匹配encoder、ACT联合训练及Raw零校准合同，并以新的architecture/provenance标识拒绝旧FiLM权重。

当前`model.py:61`仍有`contrast_context`/`contrast_film`，`:83`仍执行FiLM；post-pool side列没有新合同要求的zero-init。**因此新B3S只是已决定的合同，尚不能称实现完成。** 实施后需先做CPU结构、side-zero初始化与梯度、mask/permutation、旧checkpoint拒绝及导出/冻结交接检查；在用户暂停GPU期间不得以此审核启动GPU。

`TERMINATED_BY_USER.json`明确`automatic_gpu_resume_allowed=false`、`old_weights_eligible_for_b3s=false`、`final_sessions_opened=0`。独立读取两套pretrain的`progress.json`，各4segments、12,660updates，development为null；两目录都没有最终`encoder.pt`或completed `receipt.json`。这些是**用户终止的旧FiLM任务片段**，既未完成75,960-step预算，也不匹配新B3S。不能恢复成新B3S任务、删除FiLM权重后冒充新训练，或用于B3S dev/final选择。未来若执行新B3S，SUA/PMUA encoder及其依赖Full decoder均需按新定义从头训练。

## 尚未闭合的事项

首轮R2（完整执行代码与artifact绑定）及R3（自动seal核验完整候选、全dev资格、earliest-max和prepared binding）仍未闭合。本轮人工独立核验确认这份CPU selection实际正确，但不能据此宣称自动封存逻辑已修复。二者不否定现有CPUdev结果，首次final前必须修复。

首轮R4已看到`finalize.py:350`新增逐表示source canonical keys与全部final表比较的实现；本轮没有重新执行其合成测试，不将其余final入口标为整体通过。最终15格主表尚不完整，两个Raw-PMUA静态神经adapter也没有合格新神经checkpoint可评分。

文档小修已闭合：`B3S_ENCODER_CONTRACT.md` 中两个 M1 实现链接已改用正确的 `../../../` 相对位置；文档负责人逐一检查了本轮三份文档的全部20个本地链接。这不影响结构判断，不要求GPU或重新跑CPU网格。

当前可保留并报告的成果是通过复核的7个CPU方法完整dev结果。GPU已终止、B3S尚待实施、神经结果尚不存在，真实final访问仍为0；本审核没有授权恢复GPU、改动矩阵或提前打开final。
