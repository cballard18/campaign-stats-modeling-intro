library(tidyverse)
library(readxl)
library(xgboost)
library(caret)

# --- Load & Clean ---
raw <- read_xlsx("datasets/issue 1 survey.xlsx", sheet = "All Data")

clean_label <- function(x) str_remove(x, "^\\d+\\.\\t")

base <- raw |>
  select(
    trump_image  = `Trump Image`,
    issue1_vote  = `Issue 1 Ballot Test Recall`,
    party        = `Party Affiliation`,
    gender       = `Gender`,
    age          = `Age`,
    education    = `Education`,
    income       = `Household Income`,
    religion     = `Religious Affiliation`,
    area         = `Area Description`,
    ethnicity    = `Ethnicity`
  ) |>
  mutate(across(where(is.character), clean_label))

# --- Shared Feature Matrix ---
make_data <- function(df, target_col, keep_values) {
  df |>
    filter(.data[[target_col]] %in% keep_values) |>
    mutate(y = as.integer(.data[[target_col]] == keep_values[1])) |>
    drop_na(party, gender, age, education, income, religion, area, ethnicity, y)
}

encode_features <- function(df) {
  df |>
    select(party, gender, age, education, income, religion, area, ethnicity) |>
    mutate(across(everything(), as.factor)) |>
    (\(x) model.matrix(~ . - 1, data = x))()
}

# --- Train & Evaluate ---
run_xgb <- function(X, y, label) {
  set.seed(42)
  train_idx <- createDataPartition(y, p = 0.8, list = FALSE)

  dtrain <- xgb.DMatrix(X[train_idx, ],  label = y[train_idx])
  dtest  <- xgb.DMatrix(X[-train_idx, ], label = y[-train_idx])

  model <- xgb.train(
    params = list(
      objective   = "binary:logistic",
      eval_metric = "logloss",
      max_depth   = 4,
      eta         = 0.1,
      subsample   = 0.8
    ),
    data      = dtrain,
    nrounds   = 100,
    watchlist = list(train = dtrain, test = dtest),
    verbose   = 0
  )

  preds      <- predict(model, dtest)
  pred_class <- factor(as.integer(preds > 0.5))
  actual     <- factor(y[-train_idx])

  cat("\n---", label, "---\n")
  print(confusionMatrix(pred_class, actual, positive = "1"))

  list(model = model, feature_names = colnames(X), label = label)
}

# --- Model 1: Trump Favorability ---
d1  <- make_data(base, "trump_image", c("Favorable", "Unfavorable"))
X1  <- encode_features(d1)
m1  <- run_xgb(X1, d1$y, "Trump Favorability")

# --- Model 2: Issue 1 Vote (Yes vs No) ---
d2  <- make_data(base, "issue1_vote", c("Voted Yes", "Voted No"))
X2  <- encode_features(d2)
m2  <- run_xgb(X2, d2$y, "Issue 1 Vote")

# --- Feature Importance Plots ---
plot_importance <- function(result) {
  xgb.importance(feature_names = result$feature_names, model = result$model) |>
    as_tibble() |>
    slice_max(Gain, n = 15) |>
    mutate(Feature = fct_reorder(Feature, Gain)) |>
    ggplot(aes(Gain, Feature)) +
    geom_col(fill = "#1B7FA0", width = 0.7) +
    labs(x = "Gain", y = NULL,
         title = "XGBoost Feature Importance",
         subtitle = result$label) +
    theme_minimal(base_size = 12) +
    theme(
      axis.line          = element_line(color = "black", linewidth = 0.5),
      axis.text          = element_text(color = "black"),
      panel.grid.minor   = element_blank(),
      panel.grid.major.y = element_blank(),
      panel.grid.major   = element_line(color = "grey90"),
      plot.background    = element_rect(fill = "white", color = NA)
    )
}

imp1 <- plot_importance(m1)
imp1
imp2 <- plot_importance(m2)
imp2

ggsave("plots/xgb_importance_trump.png",  imp1, width = 9, height = 6, dpi = 300)
ggsave("plots/xgb_importance_issue1.png", imp2, width = 9, height = 6, dpi = 300)

# --- Score Full Dataset ---
# Encode full base, then align columns to each model's expected features
score_dataset <- function(df, result) {
  X_full <- df |>
    drop_na(party, gender, age, education, income, religion, area, ethnicity) |>
    encode_features()

  # Add any missing columns (unseen levels in training), remove extras
  missing_cols <- setdiff(result$feature_names, colnames(X_full))
  extra_cols   <- setdiff(colnames(X_full), result$feature_names)

  for (col in missing_cols) X_full <- cbind(X_full, setNames(data.frame(0), col))
  X_full <- X_full[, result$feature_names, drop = FALSE]

  predict(result$model, xgb.DMatrix(X_full))
}

scored_rows <- base |>
  drop_na(party, gender, age, education, income, religion, area, ethnicity)

base_scored <- scored_rows |>
  mutate(
    trump_fav_score = score_dataset(scored_rows, m1),
    issue1_score    = score_dataset(scored_rows, m2)
  )

glimpse(base_scored)
