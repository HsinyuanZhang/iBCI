# M1 B3S+BT 联合训（P16 D2）

日期：2026-09-07。授权：在 M1 上联合训会看 rSyn3 的 B3S 与 B-transformer。不冻 decoder。不抢 RIFT（GPU0）。不 pack、不交 EvalAI。

## 唯一变量

相对 582045（P16 CausalPE D2 BT-EORT，官方 HO 0.533 / 本地 HO-calib 0.613）：

- 身份编码器从冻结 B3（`side_features=None`）换成 **可训 B3S**，rSyn3 进 `post_pool`
- transformer 与 B3S **同一 AdamW 1e-4**，一起更新
- rSyn3 仍是闭式 carrier（concat 到 token 的 T4 路径不变）
- 同一 fullsession 4 日、seed 42、EMA 0.9995、24 ep
- 选模：可见 HO-calib trio，ext6 e18–24。`register:false`

不是 581727（那是冻 578244）。不是 581982 V3 current-query。

过线：先打 581982 官方 0.574。0.64 是另一套消费者，不当成功线。

## 命令

```bash
ROOT=btransform_unified_v1/results/m1_b3s_joint/fullsession_<UTC>
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= \
  python btransform_unified_v1/scripts/m1_b3s_joint_series.py --stage probe --face fullsession --root "$ROOT"
python btransform_unified_v1/scripts/launch_m1_b3s_joint_when_free.py "$ROOT"
```
