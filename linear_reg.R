library(tidyverse)
library(ggtext)
library(gt)

col_yes <- "#1B7FA0"
col_no  <- "#C0392B"

data <- read.csv("datasets/issue 1.csv") |>
  select(Side, Creative, Spending, Recall) |>
  mutate(
    Recall = Recall / 100,
    Side   = ifelse(Side == "YES", "Yes", "No")
  )

spend_fmt <- function(x) {
  case_when(
    x == 0   ~ "0",
    x >= 1e6 ~ paste0(x / 1e6, "M"),
    TRUE     ~ paste0(x / 1e3, "K")
  )
}

chart_theme <- theme_minimal(base_size = 12) +
  theme(
    panel.grid.minor     = element_blank(),
    panel.grid.major     = element_line(color = "grey90"),
    axis.line            = element_line(color = "black", linewidth = 0.5),
    axis.text            = element_text(color = "black"),
    axis.title.x         = element_text(hjust = 1, color = "grey50", size = 11),
    axis.title.y         = element_text(hjust = 1, angle = 0, vjust = 1.05, color = "grey50", size = 11),
    legend.position      = "top",
    legend.justification = "left",
    legend.title         = element_text(face = "bold"),
    legend.text          = element_markdown(),
    legend.background    = element_rect(color = "black", linewidth = 0.4, fill = NA),
    legend.margin        = margin(4, 8, 4, 8),
    plot.background      = element_rect(fill = "white", color = NA)
  )

# --- Models ---
model_spend  <- lm(Spending ~ Recall - 1, data = data)
summary(model_spend)

model_recall <- lm(Recall ~ Spending - 1, data = data)
summary(model_recall)

data$pred_spend  <- predict(model_spend,  newdata = data)
data$spend_surplus <- data$pred_spend - data$Spending

data$pred_recall <- predict(model_recall, newdata = data)
data$recall_surplus <- data$Recall - data$pred_recall

# --- Chart 1: Spending vs Recall ---
chart <- data |>
  ggplot(aes(Spending, Recall, color = Side)) +
  geom_smooth(method = "lm", formula = y ~ x - 1, se = FALSE,
              color = "grey70", linewidth = 0.7) +
  geom_text(aes(label = Creative), size = 5.52) +
  scale_color_manual(
    name   = "Side",
    values = c("Yes" = col_yes, "No" = col_no),
    labels = c(
      "Yes" = glue::glue("<span style='color:{col_yes}'>Yes</span>"),
      "No"  = glue::glue("<span style='color:{col_no}'>No</span>")
    )
  ) +
  guides(color = guide_legend(override.aes = list(size = 0, alpha = 0))) +
  scale_x_continuous(labels = spend_fmt, limits = c(0, 4000000),
                     breaks = seq(0, 4000000, by = 500000)) +
  scale_y_continuous(labels = scales::percent_format(), limits = c(0, 0.85),
                     breaks = seq(0, 0.8, by = 0.1)) +
  labs(x = "Spending", y = "Recall") +
  chart_theme

ggsave("plots/spending_vs_recall.png", chart, width = 14, height = 7, dpi = 300)

# --- Table: Final Results ---
tbl <- data |>
  arrange(Side, desc(Spending)) |>
  select(Side, Creative, Spending, pred_spend, spend_surplus, Recall, pred_recall, recall_surplus) |>
  gt() |>
  cols_label(
    Creative       = "Creative",
    Side           = "Side",
    Spending       = "Actual",
    pred_spend     = "Predicted",
    spend_surplus  = "Surplus",
    Recall         = "Actual",
    pred_recall    = "Predicted",
    recall_surplus = "Surplus"
  ) |>
  tab_spanner(label = "Spending", id = "spanner_spending", columns = c(Spending, pred_spend, spend_surplus)) |>
  tab_spanner(label = "Recall",   id = "spanner_recall",   columns = c(Recall, pred_recall, recall_surplus)) |>
  fmt_currency(columns = c(Spending, pred_spend, spend_surplus), decimals = 0) |>
  fmt_percent(columns = c(Recall, pred_recall, recall_surplus), decimals = 1) |>
  tab_style(
    style = cell_text(color = col_yes, weight = "bold"),
    locations = cells_body(columns = Side, rows = Side == "Yes")
  ) |>
  tab_style(
    style = cell_text(color = col_no, weight = "bold"),
    locations = cells_body(columns = Side, rows = Side == "No")
  ) |>
  tab_style(
    style = cell_text(color = "#2E7D32"),
    locations = cells_body(columns = spend_surplus,  rows = spend_surplus > 0)
  ) |>
  tab_style(
    style = cell_text(color = col_no),
    locations = cells_body(columns = spend_surplus,  rows = spend_surplus < 0)
  ) |>
  tab_style(
    style = cell_text(color = "#2E7D32"),
    locations = cells_body(columns = recall_surplus, rows = recall_surplus > 0)
  ) |>
  tab_style(
    style = cell_text(color = col_no),
    locations = cells_body(columns = recall_surplus, rows = recall_surplus < 0)
  ) |>
  tab_style(
    style = cell_fill(color = "grey96"),
    locations = cells_body(rows = seq(1, nrow(data), by = 2))
  ) |>
  cols_align(align = "right",  columns = c(Spending, pred_spend, spend_surplus, Recall, pred_recall, recall_surplus)) |>
  cols_align(align = "left",   columns = c(Side, Creative)) |>
  tab_options(
    table.border.top.color            = "black",
    table.border.bottom.color         = "black",
    column_labels.border.bottom.color = "black",
    column_labels.border.top.color    = "black",
    column_labels.font.weight         = "bold",
    table.width                       = pct(100)
  )

gtsave(tbl, "plots/results_table.png")
