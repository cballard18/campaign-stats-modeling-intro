library(tidyverse)
library(readxl)

# --- Load & Clean ---
raw <- read_xlsx("datasets/oh results 2024.xlsx", sheet = "Master", col_names = FALSE)

# Row 2 = headers, rows 5:92 = one row per county (88 Ohio counties)
# Cols: 1=county, 6=turnout, 14=Harris(D), 19=Trump(R)
data <- raw[5:nrow(raw), c(1, 6, 14, 19)] |>
  setNames(c("county", "turnout", "harris", "trump")) |>
  mutate(across(c(turnout, harris, trump), as.numeric)) |>
  mutate(
    trump_pct        = trump / (trump + harris),
    turnout          = percent_rank(as.numeric(turnout)) * 100,
    trump_percentile = percent_rank(trump_pct) * 100
  ) |>
  select(county, turnout, trump_percentile)

# --- K-Means ---
set.seed(42)
features <- data |> select(turnout, trump_percentile) |> scale()

# Elbow plot to inform k choice
wss <- map_dbl(1:8, ~ kmeans(features, centers = .x, nstart = 25)$tot.withinss)

elbow_plot <- tibble(k = 1:8, wss = wss) |>
  ggplot(aes(k, wss)) +
  geom_line(color = "grey60") +
  geom_point(size = 3, color = "#1B7FA0") +
  scale_x_continuous(breaks = 1:8) +
  labs(x = "Number of Clusters (k)", y = "Total Within-Cluster SS", title = "Elbow Plot") +
  theme_minimal(base_size = 12) +
  theme(axis.line = element_line(color = "black", linewidth = 0.5),
        panel.grid.minor = element_blank())

ggsave("plots/elbow_plot.png", elbow_plot, width = 7, height = 5, dpi = 300)

# Fit with k = 4
km <- kmeans(features, centers = 4, nstart = 25)
data$cluster <- factor(km$cluster)

# --- Plot: Cluster Profiles ---
profile_refs <- tibble(
  metric = c("Trump Percentile", "Turnout"),
  ref    = c(mean(data$trump_percentile), mean(data$turnout))
)

profile_plot <- data |>
  group_by(cluster) |>
  summarise(
    `Trump Percentile` = mean(trump_percentile),
    `Turnout`          = mean(turnout)
  ) |>
  pivot_longer(-cluster, names_to = "metric", values_to = "value") |>
  ggplot(aes(cluster, value, fill = cluster)) +
  geom_col(width = 0.6, show.legend = FALSE) +
  geom_hline(data = profile_refs, aes(yintercept = ref),
             linetype = "dashed", color = "grey30", linewidth = 0.6) +
  facet_wrap(~ metric, scales = "free_y") +
  labs(x = "Cluster", y = NULL, title = "Cluster Profiles: Ohio Counties (2024)") +
  theme_minimal(base_size = 12) +
  theme(
    axis.line        = element_line(color = "black", linewidth = 0.5),
    axis.text        = element_text(color = "black"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major = element_line(color = "grey90"),
    strip.text       = element_text(face = "bold"),
    plot.background  = element_rect(fill = "white", color = NA)
  )

ggsave("plots/cluster_profiles.png", profile_plot, width = 10, height = 6, dpi = 300)

# --- Plot: Trump % vs Turnout, colored by cluster ---
scatter_plot <- data |>
  ggplot(aes(turnout, trump_percentile, color = cluster)) +
  geom_text(aes(label = county), size = 2.8) +
  scale_x_continuous(labels = function(x) paste0(x, "%")) +
  scale_y_continuous(labels = function(x) paste0(x, "%")) +
  labs(x = "Turnout", y = "Trump Percentile", color = "Cluster",
       title = "K-Means Clusters: Ohio Counties (2024)") +
  theme_minimal(base_size = 12) +
  theme(
    axis.line        = element_line(color = "black", linewidth = 0.5),
    axis.text        = element_text(color = "black"),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "grey90"),
    legend.position  = "top",
    plot.background  = element_rect(fill = "white", color = NA)
  )

ggsave("plots/cluster_scatter.png", scatter_plot, width = 10, height = 7, dpi = 300)
