# 688 bench 状态交接（2026-09-10）

## 正在跑
- GPU0（PID 427264）：`run_688_bench.py --stage train`，当前臂 **exp2015_full f_labelfree e10/12**（~12min 出分）
- CUDA_VISIBLE_DEVICES=0 钉死；GPU1 是外来进程（18GB/99%），从未触碰

## 已完成结果（全 seal）

### EXP-1 narrow（train 9-session 2015/06-07，exam 0713-0716 隔 3-6 天）
| 臂 | exam R² |
|---|---|
| t4 (FULL) | 0.8729 |
| f_labelfree (ACTIVITY) | 0.8257 |
| floor (NONE) | 0.3247 |
| f0 | 0.8726 |
| z0 | 0.2791 |
| ts4 | 0.8656 |
| equiv_zero | 0.3454 |
| vstate | 0.8681 |
| vstate_concat | 0.8688 |
| u1_m10 | 0.8826 |
| u1_m30 | 0.8739 |

**嵌套三分解**：activity +0.5010 / carrier +0.0472 / total +0.5483

### 五条件 carrier 增量
| 条件 | carrier 增量 | vs 基线 |
|---|---|---|
| 基线 (9sess, W50, D4) | +0.0472 | — |
| 贫穷 (3sess) | +0.1385 | ×2.9 |
| 短窗 (W=10) | +0.1515 | ×3.2 |
| 浅核 (D1) | +0.0364 | −0.011 |
| dir16 | +0.0491 | +0.002（已证退化） |

### EXP-2 类（远期考卷）
| 条件 | FULL | ACT | NONE |
|---|---|---|---|
| exp2015_full t4 (train 18-sess 2015/03-07, exam 2015/11 隔 ~4 月) | **0.5120** | 跑着 | 排队 |

## 队列（GPU0 串行，~6min/epoch，~1.1h/臂）
1. exp2015_full f_labelfree（e10/12）
2. exp2015_full floor
3. PMUA 阶梯 t4 / f_labelfree / floor（cache_pmua_t4, cache_pmua_f_labelfree 已建好）
4. exp2016 t4（2015 训练 → 2016 考卷，cache_exp2016 已建好，first-91 法则 + 诊断已入 contract）

## 关键文件
- plan.py：PROTOCOLS（exp1_narrow / exp1_poverty / exp2_full / exp2015_full / exp2016）、ARMS（11 臂）、COMPONENT_ARMS_ON_FROZEN_CACHE、EXP1_NARROW_BASELINE_R2、component_ablation_report()
- run_688_bench.py：--arm / --protocol / --prepared-cache / --model-override
- caches：cache_vstate_exp1_narrow / cache_f_labelfree / cache_equiv_zero / cache_t4_dir16 / cache_pmua_t4 / cache_pmua_f_labelfree / cache_exp2016
- condition_comparison_20260910/condition_comparison.json（五条件对比 sealed）

## 待办
- exp2015_full 三分解（等 f_labelfree + floor）
- PMUA vs SUA 对比
- exp2016（NoMAD 对齐面）
- 论文：成本表 + NoMAD/Gallego 性能参考表
