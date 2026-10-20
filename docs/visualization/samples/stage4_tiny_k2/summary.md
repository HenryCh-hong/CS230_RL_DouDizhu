# Predictor run stage4_tiny_k2

- **Predictor:** K=2 (next and other hand), one per seat, for M2.
- **Data:** `artifacts/stage4_tiny_data` (manifest sha256 6d3f7310b8c60f09), from DouZero-ADP from baselines/douzero_ADP, ε = 0.01.
- **Samples per seat** (train / val / test): landlord 758 / 156 / 176; landlord_down 738 / 148 / 171; landlord_up 716 / 143 / 168
- **Training:** batch_size=64, eval_every=30, lr=0.001, max_grad_norm=10.0, max_steps=300, optimizer=Adam, patience=4, seed=0; 6 s wall-clock.
- **Selection rule:** for each seat separately, the checkpoint with the lowest validation CE (the mean over the predicted hands and the 15 card kinds), evaluated on the whole validation split every eval_every steps; the earliest on ties; training stops after patience evaluations without improvement or at max_steps. The test split is scored once, on the selected checkpoints, after training; it never influences the choice.
- **Phases:** decision progress p = t / (game_length - 1), where t counts every decision of the game from 0: early if p < 1/3, mid if 1/3 <= p < 2/3, late otherwise; fixed before any predictor was trained

## Test split (scored once, after selection)

Differences are baseline − model (positive: the model is better), with 95% intervals from a bootstrap over whole test games, because decisions of one game are correlated.

| seat | games | selected step | CE model | CE hypergeometric | CE legal-uniform | hypergeometric − model [95% CI] | uniform − model [95% CI] | MAE |
|---|---|---|---|---|---|---|---|---|
| landlord | 15 | 30 | 0.8052 | 0.8004 | 0.8022 | -0.0049 [-0.0663, 0.0531] | -0.0030 [-0.0770, 0.0720] | 0.5096 |
| landlord_down | 15 | 30 | 0.8022 | 0.7866 | 0.7521 | -0.0157 [-0.0863, 0.0503] | -0.0501 [-0.1137, 0.0095] | 0.5300 |
| landlord_up | 15 | 30 | 0.7792 | 0.8165 | 0.7948 | 0.0373 [-0.0186, 0.0988] | 0.0156 [-0.0119, 0.0401] | 0.5063 |

## Test CE by game phase (model / hypergeometric / legal-uniform)

| seat | early | mid | late |
|---|---|---|---|
| landlord | 1.0643 / 1.0110 / 1.0182 | 0.7947 / 0.8290 / 0.7953 | 0.5515 / 0.5599 / 0.5890 |
| landlord_down | 1.0789 / 1.0056 / 1.0111 | 0.8143 / 0.8330 / 0.7542 | 0.5135 / 0.5211 / 0.4911 |
| landlord_up | 0.9681 / 0.9752 / 0.9787 | 0.8074 / 0.8720 / 0.8136 | 0.5827 / 0.6212 / 0.6119 |

## Two hands (test split)

| seat | CE_next | CE_other | conservation violation | reflect TV |
|---|---|---|---|---|
| landlord | 0.8191 | 0.7914 | 0.1722 | 0.1533 |
| landlord_down | 0.8151 | 0.7894 | 0.1168 | 0.1048 |
| landlord_up | 0.7811 | 0.7773 | 0.0879 | 0.0816 |

- **Warnings:** tiny dataset (fewer than 10,000 training samples per seat): a pipeline check, not a result; trained from a git tree with uncommitted changes (code sha256 affb365ab7ba6aec86dc0854e9d1d5a9ab92bce0a25ca22d6dc2e44479f5c862)
- **Files:** `checkpoints/<seat>.ckpt` (the selected predictors), `metrics.csv`, `baselines.json`, `test_metrics.json`, `plots/index.html`, `plots/examples.html`.
