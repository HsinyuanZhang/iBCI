# Cross-dataset P operator：Astra 接受与有限 preflight 冻结

日期：2026-09-05  
判决：**ACCEPT_NAMED_OPERATOR_REVISION__GO_NORMALIZER_CONSUMER_TESTS_AND_DISPOSABLE_PROFILE__NO_FORMAL_12EP_AUTHORITY**  
本轮作者动作：只读代码/既有文件、CPU测试、写本审核文档。未打开新NWB、未改模型实现、未创建result root、未启动GPU。

## 1. 接受的范围

接受[REVISION_CROSS_DATASET_P_OPERATOR_V1_20260905.md](REVISION_CROSS_DATASET_P_OPERATOR_V1_20260905.md)，绑定SHA256：

`81ef30c9b08f3a5d36ce9bb7400973aefa344816c5ba27b312a62847e1f90638`。

三项选择不再是UNRESOLVED：

1. **共同parent**：S-Fix `pilot_r3/s_fix/epoch_011.pt`，SHA `7976e0b064fc4d92396b38a8e385aaa78379f0330c52ba7244bb45831b72178a`。P-FIX/P-CA是从该权重新开优化阶段，不加载旧Adam optimizer为AdamW续训。
2. **行为坐标估计器**：`row_normalized_nnmf_nnls_v1`，3×16字典，源RMS固定、ReLU与行L2、NNLS。P-CA训练该字典；P-FIX冻结同一初始字典。它是本轮选定的可行参数化，不宣称数学上唯一能维持零扰动parity的参数化。
3. **ridge**：`(XᵀX/n + diag(0,1,1,1))β = Xᵀrates/n`，float64，截距不惩罚，输出顺序`[w1,w2,w3,b]`。

**明确放行执行者继续**：共同normalizer重算、consumer连接与原工单§6测试、一个disposable source-only性能profile。通过这里的技术检查即可运行这次profile，不需再次请求日常系统权限或因等待B结果而暂停。

**仍不放行**：P-FIX/P-CA正式12epoch、E2 lag加入、换rank/换基函数/新decoder/FiLM、outer-session新R²扫描、EvalAI。完成profile后交付Stage1完整spec，下一次审查决定正式pilot。

## 2. 亲自核查的证据

- CPU复跑四个`test_cross_dataset_functional_calibration*`测试文件：**25 passed in 1.94s**。关闭CUDA、Python bytecode与pytest缓存；不是转述执行者测试数。
- `parent_audit.audit`实际核对S-Fix/teacher等checkpoint字节、S-Fix的`student.carrier_projection_weight [1024,4]`、source-session列表及query范围。
- 复核teacher当前`.hydra/config.yaml`与`source_only_decoder_manifest.json`：训练仅26/27/28、outer排除24；其resume来源的初始配置`ckpt_path:null`，同source列表。两个源配置路径见§7。
- 接受窄标签`CLEAN_OUTER_SESSION_FILE_EXCLUSION`，不是“整个研究从未查看过20120924表现”，也不是完整resume或新鲜outer-test证据。teacher source query从0开始的暴露仍披露。
- Stage0 `terminal.json` SHA仍为`ea4c3227ad09d09610a8d78a38dbe400bf03ada0893f6058f35594073b36f875`。Stage0与`20260905_122000` revision根均未由作者改写。
- 额外CPU检查：seed591、严格正3×16字典/8条synthetic EMG/3unit，整条`row-normalize → active-set NNLS → ridge`对字典执行float64 gradcheck，通过（eps1e-6、atol1e-7、rtol1e-5）。它验证活动集内部的链式梯度，不证明边界可微或live consumer已正确接线。
- 额外常数偏移检查：rates整体加3后，截距偏移误差`4.44e-16`，slope变化`5.27e-16`；支持截距未受罚。既有同名测试主要验证unit列置换，不能仅靠测试名称声称覆盖了截距性质。

## 3. 现在冻结的共同normalizer法则

采用 **`SOURCE_INITIAL_DICTIONARY_FROZEN_NORMALIZER_V1`**：

1. 以既有source-frozen初始字典`D0`、RMS scale和相同M10 source-support配对/交集mask，分别物化26/27/28的原始carrier。
2. 精确复用既有bank的source-pooled逐列mean/std与floor法则，得到`μ0,σ0`，绑定实际数组/输入数据/代码SHA。
3. P-FIX与P-CA共用同一套`μ0,σ0`字节；**只在初始化重算一次**，不每step/epoch更新、不在target refit。为重现parent而重建同语义对象，不引入一条额外自适应路径。
4. 学习期P-CA使用`T(D)=(carrier(D)-μ0)/σ0`。D更新后原始carrier必须在当前forward重算，梯度穿过NNLS与ridge；不得把旧D下的carrier跨optimizer step缓存。
5. 这不保证训练后carrier仍均值0/方差1；它是固定坐标尺度，consumer共同训练来适应。若后续要动态normalizer，须新named revision，不能塞入本轮。
6. 初始P-FIX/P-CA carrier与parent bank做数值parity，consumer输出也要parity。NNLS/ridge CPU float64：`atol=1e-8,rtol=1e-8`；相同设备eval的FP32 consumer输出：`atol=1e-6,rtol=1e-5`。记录最大误差与数组SHA；文件SHA不同不等于算子错误。

若重算值实质改变parent的归一化/输入，不许默默修改P或decoder来吸收差异；先核对D0、RMS、source数据范围、support配对及dtype。两臂不能一边沿用旧normalizer、一边重拟。

## 4. 接线前必须补充的局部检查与披露

这些是执行preflight的一部分，不是要求重新发散设计。

### 4.1 NNLS与数值边界

- 当前实现以SciPy NNLS决定active set，再以torch solve重算活动系数；这是分段可微解算，不是对NNLS迭代程序求导。活动集切换处不保证普通导数。
- 冻结当前support判据`coef>1e-12`。核对重算值与SciPy前向、非负性和KKT残差；对零EMG/空active set、接近边界、退化字典给出显式行为，不能静默加ridge修补NNLS。
- scale输入须有限且≥既有`1e-8` floor；字典、EMG、rates须有限。当前`encode`直接除scale而不自行floor：绑定正确scale并增加入口断言，不能只相信文档写了floor。
- `solve_ridge`随输入dtype运算，本身未强制float64；consumer adapter必须显式在autocast外使用float64 solve，并测试FP32/BF16不会悄悄替代它。
- 保留source真实D0及局部扰动的前向parity；整链梯度测试选择严格活动集内部点，并单独报告边界情况，不能以光滑点gradcheck证明全局光滑。

### 4.2 ReLU字典的可学习范围

`D=row_l2(relu(D_raw))`会使初始零项在当前ReLU导数下无法被一阶梯度激活，正项一旦跨入负区也可能锁死。NNMF通常可能包含零项，但本次审核没有重新拟合/读取真实D0，不能虚报零元素个数。

执行时记录D0的正/零元素数、每step可动元素与梯度比例、每行范数/Gram条件性。接受这是**初始稀疏支撑受限的字典微调**，不宣称可探索任意非负3×16基。首轮不临时换softplus、加ε或重新排序字典行，因为会改变parity/坐标合同。

### 4.3 完整consumer路径

- P-FIX/P-CA初始consumer参数逐字节一致，decoder/B3 identity/P的trainable allowlist一致；candidate仅额外开放字典。保留原生16维输出能力，不新造“输出残差头”或hard projection。
- 验证source query loss确实给字典非零有限梯度，不止证明`carrier.sum()`可导。
- 对support/query整窗隔离、共同unit置换、dropout同步、target-fit零backward、native输出单位、full new-stage resume执行原工单§6合同。target标签只在原有合法M10 calibration使用，不参与source normalizer。
- 旧S-Fix仅有epoch11不阻塞新阶段存档；旧optimizer缺完整RNG/normalizer也不阻塞权重初始化。新的RAW/basis/optimizer/RNG/sampler/normalizer必须完整保存。

## 5. 一个有界disposable profile

以上CPU/接线门通过后，用新preflight根运行**一次100个配对训练step的profile**（P-FIX/P-CA各100次update，同source batch manifest、有效batch32）。不把smoke参数继承为正式初始化，结束后丢弃其训练状态作为候选的资格。

- 仅源26/27/28的合法post-M10 query，零outer评分、零调参选点。可记录训练loss用于finite/梯度检查，不以loss好坏选择新的D/LR/窗口。
- 同当前强consumer、float64 basis/ridge、原工单FP32消费路径；不附加M2 EMA/cosine或E2 lag。
- 最多一张已lease的3090；若B占两卡先完成CPU检查等待释放，不抢占。GPU≤20GiB，host MemAvailable≥12GiB；一次profile含加载/计时先限30分钟，超时记录已完成step与瓶颈，不延长为正式训练。
- 分别计时：支持读取、SciPy NNLS/CPU↔GPU传输、active-set torch solve、ridge、consumer fwd/bwd、checkpoint；报告每个epoch真实step数对应的12epoch预计成本。
- **重点风险**：当前NNLS逐support-bin Python/SciPy循环，并在训练中`.cpu().numpy()`同步；“只有48个字典参数”不代表它便宜。可缓存原始support/RMS，不能缓存旧D的z/carrier或detach整条路径。
- 如需优化，可按rank3的至多8种active set分组、在torch中批量解线性系统；必须保持当前NNLS解/梯度合同与parity，不能为速度换近似估计器或减少support曝光。优化另出工程receipt。
- profiler失败允许一次同配置工程修复尝试，保留失败证据；不触发新LR/rank/schedule搜索。

## 6. profile之后仍需交付，不自动训练

提供新的`STAGE1_IMPLEMENTATION_SPEC`和短交接包：

1. D0/RMS/共同normalizer实际SHA和parity、字典可动元素统计。
2. live consumer allowlist、整链梯度、完整§6测试与new-stage resume结果。
3. source训练/选点/outer报告的**具体文件与window IDs**；选择数据不得与当前P阶段训练query混用而不披露。历史source teacher或dictionary已接触的source数据不能叫“从未见过”。
4. profile成本、每epoch曝光、12epoch两臂总预算和正式全新初始化证明。
5. 保持原12epoch/AdamW1e-4/固定rank与静态profile合同；不加入E2 lag。缺少source选择面不能因有outer分数就偷换路由。

判定只到`PREFLIGHT_READY`或具体技术失败；正式12epoch需下一份明确冻结记录。现有通用`--execute-gpu`继续拒绝正式pilot是正确的；执行者可增加**仅允许本节disposable profile**的独立入口/许可标记，不直接移除全部保护。

## 7. 核查文件

- `tfpd_exploration/src/cross_dataset_functional_calibration_v1/{basis.py,carrier_solver.py,parent_audit.py,plan.py}`。
- `tfpd_exploration/tests/test_cross_dataset_functional_calibration_{v1,e1,e2,p_operator}.py`（文件实际名称分别以`_v1.py`、`_e1.py`、`_e2.py`、`_p_operator.py`结束）。
- `tfpd_exploration/results/cross_dataset_functional_calibration_v1/20260905_122000/{parent_audit.json,HANDOFF_FOR_ASTRA_REVIEW.md,STAGE1_IMPLEMENTATION_SPEC.md}`。
- teacher run `streaming_calibration_exp/logs/m1_afc4_source_decoder_fold0/runs/2026-08-06-16-15-55-070150_rid-m1_afc4_source_decoder_fold0_dev20_resume_e1r1_fNone_s42/{.hydra/config.yaml,source_only_decoder_manifest.json}`。
- teacher initial run `.../2026-08-06-16-10-23-783819_rid-m1_afc4_source_decoder_fold0_e1r1_fNone_s42/.hydra/config.yaml`。

这份接受记录补充原revision，不改写其绑定hash或旧root。M2小模型中间工单独立执行；两条线只协调硬件，不共享科学结论或互相替代实验。
