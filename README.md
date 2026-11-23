# 類神經網路 ANN：糖尿病病程進展預測

以 R 比較 Linear Regression 與單隱藏層 ANN（`nnet`）預測糖尿病一年病程進展指標的樣本外表現。

| 檔案 | 內容 |
|---|---|
| `01_analysis_LM_ANN.R` | 作業主程式：EDA、Train/Validation/Test 切割、ANN size 選模、Test 最終評估、作圖 |
| `00_install_packages.R` | 安裝 `nnet` |
| `data/diabetes_progression.csv` | 442 位個案、10 個解釋變數、Y = `progression`（scikit-learn `load_diabetes(scaled = FALSE)`） |
| `results/` | 選模表、Test 比較表、預測值、三張圖、完整文字輸出 |

## 執行

```r
source("00_install_packages.R")   # 第一次
source("01_analysis_LM_ANN.R")    # 結果寫入 results/
```

## 分析流程

1. **切割**：Train 60% / Validation 20% / Test 20%（seed = 20261123）。
2. **前處理**：ANN 的連續 X 與 Y 只用訓練資料的平均數和標準差標準化；`sex` 轉成 0/1，不標準化。
3. **選模**：ANN（`linout = TRUE`、`decay = 0.01`、`maxit = 900`）比較 size = 3、6、10，以 Validation RMSE 選擇。
4. **最終評估**：Train + Validation 合併後重新擬合，Test 只評估一次。

## 結果

Validation 選模（n = 88）

| Model | Train RMSE | Val. RMSE | Val. MAE | Val. R² |
|---|---|---|---|---|
| Linear Regression（基準） | 53.25 | 56.43 | 47.09 | 0.478 |
| **ANN-3（選定）** | 46.30 | 60.29 | 49.56 | 0.405 |
| ANN-6 | 39.78 | 71.04 | 56.84 | 0.173 |
| ANN-10 | 30.67 | 95.09 | 77.49 | −0.481 |

Test 最終比較（n = 89）

| Model | RMSE | MAE | Test R² |
|---|---|---|---|
| **Linear Regression** | **53.15** | **42.30** | **0.504** |
| ANN-3 | 62.09 | 48.29 | 0.323 |

- 在這份資料上，ANN 沒有勝過線性迴歸：主要變數與 Y 的關係接近線性，而且樣本小。
- size 越大，Train RMSE 越低、Validation RMSE 越高，呈現典型的過度配適。
- 本分析僅供統計學習練習，不可用於臨床診斷或醫療決策。
