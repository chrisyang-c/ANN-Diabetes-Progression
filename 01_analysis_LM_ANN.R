# ===========================================================================
# HW02｜糖尿病病程進展預測：Linear Regression vs. ANN Regression
# 資料：scikit-learn load_diabetes(scaled = FALSE)，442 筆、10 個解釋變數
# Y = progression：基線後一年的病程進展數值指標（不是機率，也不是血糖）
# 依賴：nnet（第一次請先 Source 00_install_packages.R）
# 執行：開啟 HW02.Rproj 後 Source 本檔；結果寫入 results/
# 資料角色：Train 60% / Validation 20% / Test 20%
#   Train      估計候選模型與縮放參數
#   Validation 只用來在 size = 3、6、10 之間選 ANN 架構
#   Test       選模結束後只評估一次
# ===========================================================================

script_path <- local({
  frames <- sys.frames()
  paths <- unlist(lapply(frames, function(f) f$ofile))
  if (length(paths)) tail(paths, 1) else {
    a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
    if (length(a)) sub("^--file=", "", a[1]) else NA_character_
  }
})
project_dir <- if (!is.na(script_path)) dirname(normalizePath(script_path)) else getwd()

if (!requireNamespace("nnet", quietly = TRUE))
  stop("找不到 nnet 套件，請先執行 00_install_packages.R。")

options(width = 120)
SEED <- 20261123
ANN_SIZES <- c(3, 6, 10)
ANN_DECAY <- 0.01
ANN_MAXIT <- 900

input <- file.path(project_dir, "data", "diabetes_progression.csv")
out <- file.path(project_dir, "results")
dir.create(out, showWarnings = FALSE, recursive = TRUE)
while (sink.number() > 0) sink()  # 清掉上次中斷時可能殘留的 sink
sink(file.path(out, "analysis_report.txt"), split = TRUE)
save_csv <- function(x, name) write.csv(x, file.path(out, name), row.names = FALSE,
                                        fileEncoding = "UTF-8")

# ---------------------------------------------------------------------------
# 1. 讀取與檢查資料
# ---------------------------------------------------------------------------
raw <- read.csv(input, fileEncoding = "UTF-8-BOM", check.names = FALSE)
cont_vars <- c("age", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6")
stopifnot(nrow(raw) == 442L,
          all(c("sample_id", cont_vars, "sex", "progression") %in% names(raw)),
          !anyNA(raw), all(raw$sex %in% c(1, 2)))

# sex 是 1/2 的類別編碼：LM 用 factor，ANN 用 0/1 虛擬變數（不標準化）
raw$sex2 <- as.integer(raw$sex == 2)

cat("糖尿病病程進展預測：Linear Regression vs. ANN\n")
cat("執行日期：", format(Sys.Date()), "；R ", R.version$major, ".", R.version$minor,
    "；nnet ", as.character(packageVersion("nnet")), "\n", sep = "")
cat("資料：", nrow(raw), " 筆，10 個解釋變數，缺失值 ", sum(is.na(raw)), "\n\n", sep = "")

# ---------------------------------------------------------------------------
# TODO 1：探索 y 分布與 BMI 散點圖（完整資料的描述性 EDA，不用於選模）
# ---------------------------------------------------------------------------
y_all <- raw$progression
skew <- mean((y_all - mean(y_all))^3) / sd(y_all)^3
cat("[TODO 1] progression 的描述統計\n")
print(summary(y_all))
cat(sprintf("標準差 %.2f；偏態係數 %.3f（> 0 表示右偏）\n", sd(y_all), skew))
cat(sprintf("BMI 與 progression 的 Pearson 相關 %.3f\n",
            cor(raw$bmi, raw$progression)))
cor_tab <- data.frame(variable = c(cont_vars, "sex"),
                      cor_with_progression = round(vapply(c(cont_vars, "sex"), function(v)
                        cor(raw[[v]], raw$progression), numeric(1)), 3))
cor_tab <- cor_tab[order(-abs(cor_tab$cor_with_progression)), ]
print(cor_tab, row.names = FALSE)
save_csv(cor_tab, "01_eda_correlations.csv")

png(file.path(out, "01_eda_y_and_bmi.png"), width = 1500, height = 620, res = 140)
par(mfrow = c(1, 2), mar = c(4.3, 4.3, 3.2, 1))
hist(y_all, breaks = 25, col = "#CFE3EA", border = "white",
     main = "Distribution of progression", xlab = "progression (1-year index)",
     ylab = "Count")
abline(v = c(mean(y_all), median(y_all)), col = c("#C0504D", "#2D7185"),
       lwd = 2, lty = c(1, 2))
legend("topright", c("Mean", "Median"), col = c("#C0504D", "#2D7185"),
       lwd = 2, lty = c(1, 2), bty = "n")
plot(raw$bmi, raw$progression, pch = 16, cex = .7, col = "#2D718599",
     main = "BMI vs progression", xlab = "BMI", ylab = "progression")
abline(lm(progression ~ bmi, data = raw), col = "#C0504D", lwd = 2)
lines(lowess(raw$bmi, raw$progression), col = "#333333", lwd = 2, lty = 2)
legend("topleft", c("Linear fit", "LOWESS"), col = c("#C0504D", "#333333"),
       lwd = 2, lty = c(1, 2), bty = "n")
invisible(dev.off())

# ---------------------------------------------------------------------------
# 2. 資料切割：Train 60% / Validation 20% / Test 20%（固定 seed 的簡單隨機切割）
# ---------------------------------------------------------------------------
set.seed(SEED)
n <- nrow(raw)
perm <- sample.int(n)
n_train <- floor(.6 * n); n_valid <- floor(.2 * n)
role <- character(n)
role[perm[seq_len(n_train)]] <- "Train"
role[perm[n_train + seq_len(n_valid)]] <- "Validation"
role[perm[(n_train + n_valid + 1):n]] <- "Test"
save_csv(data.frame(sample_id = raw$sample_id, role = role), "02_data_split.csv")
train <- raw[role == "Train", ]
valid <- raw[role == "Validation", ]
test  <- raw[role == "Test", ]
cat("\n[切割] Train ", nrow(train), "；Validation ", nrow(valid), "；Test ",
    nrow(test), "\n", sep = "")

# ---------------------------------------------------------------------------
# 共用函式
# ---------------------------------------------------------------------------
# 縮放參數只由「當下的訓練資料」估計（選模階段 = Train；最終模型 = Train+Val）
fit_scaler <- function(d) {
  list(mu = vapply(d[cont_vars], mean, numeric(1)),
       sd = vapply(d[cont_vars], sd, numeric(1)),
       mu_y = mean(d$progression), sd_y = sd(d$progression))
}
# TODO 2：Y 也只用訓練資料的平均數與標準差標準化
nn_frame <- function(d, s, with_y = TRUE) {
  z <- as.data.frame(scale(as.matrix(d[cont_vars]), center = s$mu, scale = s$sd))
  z$sex2 <- d$sex2
  if (with_y) z$y <- (d$progression - s$mu_y) / s$sd_y
  z
}
lm_formula <- progression ~ age + factor(sex) + bmi + bp + s1 + s2 + s3 + s4 + s5 + s6
fit_ann <- function(d, s, size) {
  set.seed(SEED)  # 每個 size 使用相同的初始權重亂數來源，可重現
  nnet::nnet(y ~ ., data = nn_frame(d, s), size = size, linout = TRUE,
             decay = ANN_DECAY, maxit = ANN_MAXIT, trace = FALSE)
}
predict_ann <- function(model, newdata, s) {
  as.numeric(predict(model, newdata = nn_frame(newdata, s, with_y = FALSE))) * s$sd_y + s$mu_y
}
# R² 的 SST 使用「被評估資料集」自己的觀察值平均數
metrics <- function(y, p) {
  c(RMSE = sqrt(mean((y - p)^2)), MAE = mean(abs(y - p)),
    R2 = 1 - sum((y - p)^2) / sum((y - mean(y))^2))
}

# ---------------------------------------------------------------------------
# TODO 3：在 Train 上擬合候選模型，以 Validation RMSE 比較 ANN 的 size
# ---------------------------------------------------------------------------
sc_train <- fit_scaler(train)
save_csv(data.frame(variable = c(cont_vars, "progression"),
                    mean_train = c(sc_train$mu, sc_train$mu_y),
                    sd_train = c(sc_train$sd, sc_train$sd_y)), "03_train_scaler.csv")

lm_train <- lm(lm_formula, data = train)
# Train_RMSE 只用來觀察 Train 與 Validation 的差距（過度配適），不參與選模
val_rows <- list(c(Model = "Linear Regression (baseline)", hidden_units = NA,
                   Train_RMSE = metrics(train$progression, predict(lm_train, train))[["RMSE"]],
                   metrics(valid$progression, predict(lm_train, valid)),
                   convergence = NA))
for (k in ANN_SIZES) {
  m <- fit_ann(train, sc_train, k)
  val_rows[[length(val_rows) + 1]] <- c(Model = paste0("ANN-", k), hidden_units = k,
                                        Train_RMSE = metrics(train$progression,
                                                             predict_ann(m, train, sc_train))[["RMSE"]],
                                        metrics(valid$progression, predict_ann(m, valid, sc_train)),
                                        convergence = m$convergence)
}
val_tab <- as.data.frame(do.call(rbind, val_rows), stringsAsFactors = FALSE)
num_cols <- c("hidden_units", "Train_RMSE", "RMSE", "MAE", "R2", "convergence")
val_tab[num_cols] <- lapply(val_tab[num_cols], as.numeric)

# TODO 4：只在 ANN 候選中，以 Validation RMSE 最小者為最終架構
ann_rows <- !is.na(val_tab$hidden_units)
best_size <- val_tab$hidden_units[ann_rows][which.min(val_tab$RMSE[ann_rows])]
val_tab$selected <- ifelse(!is.na(val_tab$hidden_units) & val_tab$hidden_units == best_size,
                           "yes", "")
cat("\n[TODO 3-4] Validation 選模表（R² 的 SST 以 Validation 平均數計算）\n")
print(transform(val_tab, Train_RMSE = round(Train_RMSE, 3), RMSE = round(RMSE, 3),
                MAE = round(MAE, 3), R2 = round(R2, 4)), row.names = FALSE)
cat("RMSE/MAE/R2 為 Validation 指標；Train_RMSE 為同一模型在 Train 上的 RMSE\n")
cat("convergence：0 = 達到 nnet 收斂條件；1 = 跑滿 maxit\n")
cat("選定的隱藏層神經元數 size =", best_size, "\n")
save_csv(val_tab, "04_validation_model_selection.csv")

# ---------------------------------------------------------------------------
# TODO 5：Train + Validation 合併，重新估計縮放參數並重新擬合最終模型
# ---------------------------------------------------------------------------
final_train <- rbind(train, valid)
sc_final <- fit_scaler(final_train)
lm_final <- lm(lm_formula, data = final_train)
ann_final <- fit_ann(final_train, sc_final, best_size)
cat("\n[TODO 5] 最終訓練集 n =", nrow(final_train),
    "；ANN 收斂代碼 =", ann_final$convergence, "\n")

# ---------------------------------------------------------------------------
# TODO 6：Test 只評估一次（之後不再回頭修改任何設定）
# ---------------------------------------------------------------------------
pred_lm <- as.numeric(predict(lm_final, test))
pred_ann <- predict_ann(ann_final, test, sc_final)
test_tab <- data.frame(Model = c("Linear Regression", paste0("ANN-", best_size)),
                       rbind(metrics(test$progression, pred_lm),
                             metrics(test$progression, pred_ann)))
cat("\n[TODO 6] Test 最終評估（n = ", nrow(test),
    "；R² 的 SST 以 Test 觀察值平均數計算）\n", sep = "")
print(transform(test_tab, RMSE = round(RMSE, 3), MAE = round(MAE, 3), R2 = round(R2, 4)),
      row.names = FALSE)
save_csv(test_tab, "05_test_model_comparison.csv")
save_csv(data.frame(sample_id = test$sample_id, observed = test$progression,
                    pred_lm = pred_lm, pred_ann = pred_ann,
                    resid_lm = test$progression - pred_lm,
                    resid_ann = test$progression - pred_ann), "06_test_predictions.csv")

# ---------------------------------------------------------------------------
# TODO 7：Observed vs Predicted 與 Residual vs Fitted（兩模型並排、座標一致）
# ---------------------------------------------------------------------------
preds <- list(pred_lm, pred_ann)
titles <- c("Linear Regression", paste0("ANN (size = ", best_size, ")"))
lims <- range(c(test$progression, unlist(preds)))
png(file.path(out, "07_observed_vs_predicted.png"), width = 1500, height = 680, res = 140)
par(mfrow = c(1, 2), mar = c(4.3, 4.3, 3.2, 1))
for (i in 1:2) {
  plot(test$progression, preds[[i]], pch = 16, cex = .8, col = "#2D7185AA",
       xlim = lims, ylim = lims, xlab = "Observed progression",
       ylab = "Predicted progression",
       main = sprintf("%s\nTest RMSE = %.1f", titles[i], test_tab$RMSE[i]))
  abline(0, 1, col = "#C0504D", lwd = 2)
}
invisible(dev.off())

resids <- lapply(preds, function(p) test$progression - p)
rlim <- range(unlist(resids)); flim <- range(unlist(preds))
png(file.path(out, "08_residual_vs_fitted.png"), width = 1500, height = 680, res = 140)
par(mfrow = c(1, 2), mar = c(4.3, 4.3, 3.2, 1))
for (i in 1:2) {
  plot(preds[[i]], resids[[i]], pch = 16, cex = .8, col = "#2D7185AA",
       xlim = flim, ylim = rlim, xlab = "Fitted (predicted) progression",
       ylab = "Residual (observed - predicted)", main = titles[i])
  abline(h = 0, col = "#C0504D", lwd = 2)
  lines(lowess(preds[[i]], resids[[i]]), col = "#333333", lwd = 2, lty = 2)
}
invisible(dev.off())

# 模型物件只存本機（不上傳）
saveRDS(list(lm_final = lm_final, ann_final = ann_final, best_size = best_size,
             scaler_final = sc_final, seed = SEED),
        file.path(out, "09_final_models.rds"))
# ---------------------------------------------------------------------------
# 附錄：穩健性檢查（報告「討論與限制」第 1 點）
# 以 30 組亂數種子重複「切割 → Validation 選 size → Train+Val 重估 → Test」整個流程，
# 只用來觀察結論會不會隨切割改變；不回頭修改上面的正式分析。
# ---------------------------------------------------------------------------
cat("\n[附錄] 30 組切割的穩健性檢查（約需 1 分鐘）\n")
run_split <- function(s) {
  set.seed(s)
  pm <- sample.int(n)
  tr <- raw[pm[seq_len(n_train)], ]
  va <- raw[pm[n_train + seq_len(n_valid)], ]
  te <- raw[pm[(n_train + n_valid + 1):n], ]
  fit_s <- function(d, sc, k) {
    set.seed(s)
    nnet::nnet(y ~ ., data = nn_frame(d, sc), size = k, linout = TRUE,
               decay = ANN_DECAY, maxit = ANN_MAXIT, trace = FALSE)
  }
  sc <- fit_scaler(tr)
  v_rmse <- vapply(ANN_SIZES, function(k)
    metrics(va$progression, predict_ann(fit_s(tr, sc, k), va, sc))[["RMSE"]], numeric(1))
  k_best <- ANN_SIZES[which.min(v_rmse)]
  ft <- rbind(tr, va); sc_f <- fit_scaler(ft)
  m_lm <- metrics(te$progression, as.numeric(predict(lm(lm_formula, data = ft), te)))
  m_nn <- metrics(te$progression, predict_ann(fit_s(ft, sc_f, k_best), te, sc_f))
  c(seed = s, best_size = k_best, LM_RMSE = m_lm[["RMSE"]], ANN_RMSE = m_nn[["RMSE"]],
    LM_R2 = m_lm[["R2"]], ANN_R2 = m_nn[["R2"]])
}
rob <- as.data.frame(t(vapply(1:30, run_split, numeric(6))))
save_csv(rob, "10_robustness_30_splits.csv")
print(table(選出的size = rob$best_size))
print(round(apply(rob[c("LM_RMSE", "ANN_RMSE", "LM_R2", "ANN_R2")], 2, quantile,
                  c(0, .25, .5, .75, 1)), 3))
cat("ANN 的 Test RMSE 低於線性迴歸的次數：", sum(rob$ANN_RMSE < rob$LM_RMSE), "/ 30\n")

cat("\n完成。所有結果已寫入：", normalizePath(out), "\n")
sink()
# TODO 8：研究報告另行繳交（docx / PDF）
