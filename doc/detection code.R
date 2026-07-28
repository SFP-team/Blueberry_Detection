
########################################################

library(readxl)
library(dplyr)
library(ggplot2)
library(ggpubr)
library(patchwork)
library(tidyr)
library(purrr)
library(writexl)
library(MASS)
library(randomForest)
library(glmnet)
library(pls)
library(e1071)
library(Hmisc)
library(broom)

base_path = getwd()

#========================================================
# Load data and create maturity variables for cv & gt
#========================================================

vdt_mean <- read_excel(file.path(base_path, "validation-counts.xlsx"))

merged <- vdt_mean %>%
  dplyr::select(
    Genotype,
    berry.immature,berry.mature,total_detections,
    Immature,Mature,Total
  )

merged$maturity_gt = merged$Mature/merged$Total
merged$maturity_cv = merged$berry.mature/merged$total_detections

#========================================================
# Correlations
#========================================================

cor_df <- tibble(
  Trait = c("Immature", "Mature", "Total", "maturity_gt"),
  r = c(
    cor(merged$Immature, merged$berry.immature, use = "complete.obs"),
    cor(merged$Mature, merged$berry.mature, use = "complete.obs"),
    cor(merged$Total, merged$total_detections, use = "complete.obs"),
    cor(merged$maturity_gt, merged$maturity_cv, use = "complete.obs")
  )
)

cor_df

################################################################################

#========================================================
# OLS plots
#========================================================

make_plot <- function(df, xvar, yvar, xlab, ylab, title, color = "#2c7fb8"){
  
  dat <- df[, c(xvar, yvar)]
  dat <- na.omit(dat)
  names(dat) <- c("x", "y")
  
  r <- cor(dat$x, dat$y)
  mod <- lm(y ~ x, data = dat)
  coefs <- coef(mod)
  r2 <- summary(mod)$r.squared
  rmse <- sqrt(mean((dat$y - predict(mod))^2))
  
  eq <- paste0("y = ", round(coefs[1], 2), " + ", round(coefs[2], 4), "x")
  
  axis_max <- max(c(dat$x, dat$y), na.rm = TRUE) * 1.05
  axis_min <- min(0, min(c(dat$x, dat$y), na.rm = TRUE))
  
  ggplot(dat, aes(x = x, y = y)) +
    geom_point(size = 3.2, alpha = 0.75, color = color) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red", linewidth = 0.9) +
    geom_smooth(method = "lm", formula = y ~ x, color = "black", linewidth = 1.1, se = TRUE) +
    coord_equal(xlim = c(axis_min, axis_max), ylim = c(axis_min, axis_max), expand = TRUE) +
    #annotate("text", x = Inf, y = -Inf, label = eq, hjust = 1.05, vjust = -1.1, size = 4.5, fontface = "bold") +
    theme_bw(base_size = 15) +
    theme(
      plot.title = element_text(face = "bold", size = 18, hjust = 0.5),
      plot.subtitle = element_text(size = 18, hjust = 0.5, face = "italic"), 
      axis.title = element_text(face = "bold", size = 18),                  
      axis.text = element_text(size = 14, color = "black"),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "grey90"),
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white")
    ) +
    labs(
      title = title,
      subtitle = paste0("r = ", round(r, 3), " | R² = ", round(r2, 3), " | RMSE = ", round(rmse, 2)),
      x = xlab,
      y = ylab
    )
}

p1 <- make_plot(
  merged,
  xvar = "Immature",
  yvar = "berry.immature",
  xlab = "Observed immature berry count",
  ylab = "Detected immature berry count",
  title = "", #"Immature berries",
  color = "#1B9E77"
)

p2 <- make_plot(
  merged,
  xvar = "Mature",
  yvar = "berry.mature",
  xlab = "Observed mature berry count",
  ylab = "Detected mature berry count",
  title = "", #"Mature berries",
  color = "#D95F02"
)

p3 <- make_plot(
  merged,
  xvar = "Total",
  yvar = "total_detections",
  xlab = "Observed total berry count",
  ylab = "Detected total berry count",
  title = "", #"Total detections",
  color = "#7570B3"
)

p4 <- make_plot(
  merged,
  xvar = "maturity_gt",
  yvar = "maturity_cv",
  xlab = "Observed maturity (%)",
  ylab = "Detected maturity (%)",
  title = "", #"Maturity",
  color = "#E7298A"
)

combined_plot <- p1 + p2 + p3 + p4 +
  plot_layout(ncol = 4) +
  plot_annotation(
    title = "", #"Validation of image-based berry count and maturity estimates",
    theme = theme(
      plot.title = element_text(face = "bold", size = 20, hjust = 0.5)
    )
  )

combined_plot

ggsave(
  filename = paste0(base_path, "/plot2/validation_panel_4col.png"),
  plot = combined_plot,
  width = 22,
  height = 6,
  dpi = 300
)

################################################################################

#========================================================
# Occlusion estimate
#========================================================

occlusion_df <- merged %>%
  group_by(Genotype) %>%
  summarise(
    GroundTruth = mean(Total, na.rm = TRUE),
    Predicted   = mean(total_detections, na.rm = TRUE),
    
    Detection_Rate = Predicted / GroundTruth,
    
    Occlusion_Rate = 1 - Detection_Rate
  ) %>%
  arrange(desc(Occlusion_Rate))

head(occlusion_df)
occlusion_df$genotype <- paste0("G", seq_len(nrow(occlusion_df)))

# save data

write_xlsx(
  occlusion_df,
  path = paste0(base_path, "/plot2", "/occlusion.xlsx"),
  col_names = TRUE,
  format_headers = TRUE,
  use_zip64 = FALSE
)

# plot

p = ggplot(
  occlusion_df,
  aes(
    x = reorder(Genotype, Occlusion_Rate),
    y = Occlusion_Rate
  )
) +
  
  geom_col(fill = "#b2182b") +
  
  #coord_flip() +
  
  labs(
    #title = "Estimated Berry Occlusion by Genotype",
    #subtitle = "Higher values indicate more berries hidden from camera view",
    x = "Genotype",
    y = "Occlusion Rate"
  ) +
  
  theme_bw(base_size = 14) +
  
  theme(
    plot.title = element_text(face = "bold"),
    panel.grid.minor = element_blank(), panel.grid.major.y = element_blank(),
    axis.text.x = element_text(face = "bold", angle = 45, hjust = 1, color = "black")
  )

p

ggsave(
  filename = paste0(base_path, "/plot2", "/berry_occulution v2.png"),
  plot = p,
  width = 10,
  height = 4,
  dpi = 600
)

#========================================================
# Include Occlusion estimate in data
#========================================================

merged1 <- vdt_mean
merged1$maturity_gt = merged1$Mature/merged1$Total
merged1$maturity_cv = merged1$berry.mature/merged1$total_detections

merged1 <- merged1 %>%
  mutate(
    occlusion = Total - total_detections,
    
    occlusion_rate =
      1 - (total_detections / Total)
  )

# save

write_xlsx(
  merged1,
  path = paste0(base_path, "/plot2", "/process_data.xlsx"),
  col_names = TRUE,
  format_headers = TRUE,
  use_zip64 = FALSE
)

################################################################################

#========================================================
# Detection metrics in validation images & Counting error
#========================================================

# load data
det <- read_csv(file.path(base_path, "mask_filtered_err_counts.csv"), show_col_types = FALSE)
det$ID= det$image_id

sel <- vdt_mean %>%
  dplyr::select(
    Genotype,ID,
    berry.immature,berry.mature,total_detections,
    Immature,Mature,Total
  )

# merge data
sel1 = left_join(sel,det[,-1],"ID")

###################
# Detection metrics
###################

sel1_metrics <- sel1 %>%
  mutate(
    maturity = berry.mature / total_detections * 100,
    precision_mature = berry.mature / (berry.mature + FG) * 100,
    recall_mature    = berry.mature / (berry.mature + MR) * 100,
    f1_mature = 2 * (precision_mature * recall_mature) / (precision_mature + recall_mature),
    
    precision_immature = berry.immature / (berry.immature + FR) * 100,
    recall_immature    = berry.immature / (berry.immature + MG) * 100,
    f1_immature = 2 * (precision_immature * recall_immature) / (precision_immature + recall_immature),
    
    precision_total_detections = total_detections / (total_detections + FR + FG) * 100,
    recall_total_detections    = total_detections / (total_detections + MR +  MG) * 100,
    f1_total_detections = 2 * (precision_total_detections * recall_total_detections) / (precision_total_detections + recall_total_detections),
    
    counting_error = abs(total_detections - (total_detections + MR + MG)) / (total_detections + MR + MG) * 100
  )

sel1_metrics

###################################
# Counting error vs Occlusion rate
###################################

sel1_metrics <- sel1_metrics %>%
  mutate(
    occlusion = Total - total_detections,
    
    Occlusion_Rate =
      1 - (total_detections / Total)
  )

# mean across genotypes

genotype_means <- sel1_metrics %>%
  group_by(Genotype) %>%
  summarise(
    Occlusion_Rate = mean(Occlusion_Rate, na.rm = TRUE),
    counting_error = mean(counting_error, na.rm = TRUE),
    .groups = "drop"
  )

# plot

p1 <- ggplot(
  genotype_means,
  aes(
    x = counting_error,
    y = Occlusion_Rate
  )
) +
  geom_point(
    size = 2.5,
    alpha = 0.8,
    color = "#2166ac"
  ) +
  geom_smooth(
    method = "lm",
    formula = y ~ x,
    se = FALSE,
    linewidth = 0.7,
    color = "black"
  ) +
  labs(
    x = "Counting error (%)",
    y = "Occlusion rate"
  ) +
  theme_bw(base_size = 9) +
  theme(
    axis.title = element_text(size = 20),
    axis.text = element_text(size = 20, color = "black"),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(
      color = "grey90",
      linewidth = 0.3
    ),
    plot.background = element_rect(
      fill = "white",
      color = "black",
      linewidth = 0.5
    ),
    plot.margin = ggplot2::margin(4, 4, 4, 4)
  )

inset_r <- cor(
  genotype_means$Occlusion_Rate,
  genotype_means$counting_error,
  use = "complete.obs"
)

p1 <- p1 +
  labs(
    title = paste0("r = ", round(inset_r, 2))
  ) +
  theme(
    plot.title = element_text(
      size = 20,
      face = "bold",
      hjust = 0.5
    )
  )
p1

# save plot
ggsave(
  filename = paste0(base_path, "/plot2/occl_vs_err.png"),
  plot = p1,
  width = 8,
  height = 8,
  dpi = 300
)

################################################################################

#========================================================
# Feature selection with stepwise regression
#========================================================

# full model
m_full <- lm(
  occlusion_rate ~ 
    `LDI.Whole.(%)` +
    `LDI.Bounded.(%)` +
    `LDI.Hull/Solidity.(%)` +
    `Canopy.Width.(px)` +
    `Canopy.Height.(px)` +
    `Canopy.Area.(px^2)` +
    `Canopy.Perimeter.(px)` +
    `Convex.Hull.Area.(px^2)` +
    Solidity +
    `Bounding.Box.Area.(px^2)` +
    `Width-Height.Ratio` +
    Circularity +
    `Fractal.Dimension` +
    Orientation +
    `Major.Axis.Length` +
    `Minor.Axis.Length` +
    `Mean.GLI` +
    `Mean.VARI` +
    `Mean.Hue` +
    `Std.Dev.Hue` +
    `Mean.Saturation` +
    `Std.Dev.Saturation` +
    `Mean.Value` +
    `Std.Dev.Value` +
    `Yellow.(%)` +
    `Brown.(%)` +
    `Texture.Contrast` +
    `Texture.Dissimilarity` +
    `Texture.Homogeneity` +
    `Texture.Energy` +
    `Texture.Correlation` +
    `Texture.ASM`,
  data = merged1
)

# summary(m_full)

# Feature selection

m_step <- stepAIC(m_full, direction = "both", trace = FALSE)
summary(m_step)

best_vars <- names(coef(m_step))[-1]
best_vars
length(best_vars)

top_var = best_vars
#top_var <- best_vars[1:min(15, length(best_vars))]

form <- as.formula(
  paste("Total ~ total_detections +", paste(top_var, collapse = " + "))
)

m_top <- lm(form, data = merged1)
summary(m_top)

coefs <- summary(m_full)$coefficients
coefs <- coefs[rownames(coefs) != "(Intercept)", , drop = FALSE]
top_var <- rownames(coefs)[order(abs(coefs[, "t value"]), decreasing = TRUE)][1:length(best_vars)]
top_var

################################################################################

# Visualize Feature Selection

# ---------------------------------------------------------
# Panel A: standardized coefficients of selected variables
# ---------------------------------------------------------

# Extract the data actually used in the final stepwise model
# Data used in the final stepwise model
step_dat <- model.frame(m_step)

response_var <- names(step_dat)[1]
selected_vars <- names(step_dat)[-1]

# Confirm variables are numeric
non_numeric <- names(step_dat)[
  !vapply(step_dat, is.numeric, logical(1))
]

if (length(non_numeric) > 0) {
  stop(
    paste(
      "The following variables are not numeric:",
      paste(non_numeric, collapse = ", ")
    )
  )
}

# Standardize response and predictor variables
step_dat_std <- step_dat %>%
  dplyr::mutate(
    dplyr::across(
      dplyr::everything(),
      ~ as.numeric(scale(.x))
    )
  )

# Reuse the formula from the selected model
standardized_formula <- formula(m_step)

# Fit the standardized selected model
m_step_std <- lm(
  formula = standardized_formula,
  data = step_dat_std
)

summary(m_step_std)

coef_df <- broom::tidy(
  m_step_std,
  conf.int = TRUE,
  conf.level = 0.95
) %>%
  dplyr::filter(term != "(Intercept)") %>%
  dplyr::mutate(
    term = gsub("`", "", term),
    direction = dplyr::if_else(
      estimate >= 0,
      "Positive",
      "Negative"
    ),
    term = reorder(term, estimate)
  )

p_coef <- ggplot(
  coef_df,
  aes(
    x = estimate,
    y = term
  )
) +
  geom_vline(
    xintercept = 0,
    linetype = "dashed",
    color = "grey45",
    linewidth = 0.7
  ) +
  geom_errorbarh(
    aes(
      xmin = conf.low,
      xmax = conf.high
    ),
    height = 0.18,
    linewidth = 0.8,
    color = "grey30"
  ) +
  geom_point(
    aes(fill = direction),
    shape = 21,
    size = 3.5,
    color = "black",
    stroke = 0.5
  ) +
  scale_fill_manual(
    values = c(
      "Positive" = "#2166ac",
      "Negative" = "#b2182b"
    )
  ) +
  labs(
    title = "A. ",#Variables retained by stepwise selection
    x = "Standardized regression coefficient",
    y = NULL,
    fill = "Association"
  ) +
  theme_bw(base_size = 13) +
  theme(
    plot.title = element_text(
      face = "bold",
      size = 14
    ),
    axis.text = element_text(
      color = "black"
    ),
    axis.text.y = element_text(
      face = "bold"
    ),
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    legend.position = "top"
  )

# ---------------------------------------------------------
# Panel B: AIC change during stepwise selection
# ---------------------------------------------------------

aic_df <- as.data.frame(m_step$anova) %>%
  mutate(
    selection_step = seq_len(n()) - 1,
    Step = as.character(Step),
    Step = if_else(
      is.na(Step) | Step == "",
      "Full model",
      Step
    )
  )

p_aic <- ggplot(
  aic_df,
  aes(
    x = selection_step,
    y = AIC
  )
) +
  geom_line(
    linewidth = 0.9,
    color = "black"
  ) +
  geom_point(
    size = 3,
    shape = 21,
    fill = "#4d4d4d",
    color = "black"
  ) +
  scale_x_continuous(
    breaks = aic_df$selection_step,
    labels = aic_df$Step
  ) +
  labs(
    title = "B. ", #AIC trajectory during stepwise selection
    x = "Model-selection step",
    y = "Akaike information criterion"
  ) +
  theme_bw(base_size = 13) +
  theme(
    plot.title = element_text(
      face = "bold",
      size = 14
    ),
    axis.text = element_text(
      color = "black"
    ),
    axis.text.x = element_text(
      angle = 45,
      hjust = 1
    ),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank()
  )

# join

stepwise_figure <- p_coef / p_aic +
  plot_layout(
    heights = c(1.7, 1)
  )

stepwise_figure

ggsave(paste0(base_path, "/plot2/stepwise_figure.png"), stepwise_figure, width = 9, height = 9, dpi = 600)

################################################################################

#========================================================
# Correlation of canopy architecture (CA) with occlusion
#========================================================

candidate_vars <- names(merged1)[11:42]
vars <- intersect(candidate_vars, gsub("`", "", best_vars))

cor_df <- data.frame(
  variable = vars,
  correlation = sapply(vars, function(v) cor(merged1[[v]], merged1$occlusion_rate, use = "complete.obs"))
)

cor_df <- cor_df[order(abs(cor_df$correlation), decreasing = TRUE), ]

# Select top 10 features for the correlation plot

top10 <- head(cor_df, 10)
top10$variable

traits <- merged1 %>%
  dplyr::select(
    occlusion_rate, Immature, Mature, maturity_gt, Total, 
    top10$variable
  )

tn <- c(
  "Occlusion rate", "Immature berry", "Mature berry", "Maturity ratio", "Total berry count", 
  "Mean value", "Canopy height (px)", "LDI Whole (%)", "Convex hull area (px2)",
  "Canopy width (px)", "Texture Energy", "Texture asm", "Capony perimeter (px)",
  "Std. dev. hue",  "Mean hue"
)

cor_res <- rcorr(as.matrix(traits), type = "pearson")

cor_mat <- cor_res$r

rownames(cor_mat) <- colnames(cor_mat) <- tn

cor_mat1 <- cor_mat[6:15, 1:5]

plot_df <- as.data.frame(as.table(cor_mat1)) %>%
  rename(Canopy_trait = Var1, Berry_trait = Var2, r = Freq)

# plot

p_cor <- ggplot(plot_df, aes(x = Canopy_trait, y = Berry_trait, fill = round(r,2)#r_plot
)) +
  geom_tile(color = "white", linewidth = 0.8) +
  geom_text(aes(label = round(r,2)#label
  ), size = 4.2, fontface = "bold", color = "black") +
  scale_fill_gradient2(
    low = "#1B5E20",
    mid = "white",
    high = "#4B0082",
    midpoint = 0,
    limits = c(-1, 1),
    na.value = "white",
    name = "r"
  ) +
  coord_fixed() +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0),
    axis.title = element_blank(),
    axis.text.x = element_text(face = "bold", angle = 45, hjust = 1, color = "black"),
    axis.text.y = element_text(face = "bold", color = "black"),
    panel.grid = element_blank(),
    legend.title = element_text(face = "bold"),
    legend.position = "right"
  )

p_cor

ggsave(paste0(base_path, "/plot2/canopy_correlation_plot.png"), p_cor, width = 9, height = 6, dpi = 600)

################################################################################

#========================================================
# Cross-validation of ground-truth with CA & CV traits
#========================================================

set.seed(123)

# ----------------------------
# Define response/predictor sets
# ----------------------------
responses <- list(
  Total = "Total",
  Immature = "Immature",
  Mature = "Mature",
  maturity_gt = "maturity_gt"
)

merged2 <- merged1
names(merged2) <- make.names(names(merged2))

best_vars2 <- make.names(gsub("`", "", best_vars))
best_vars2 <- best_vars2[best_vars2 %in% names(merged2)]
best_vars2

predictors <- c(
  "total_detections",
  best_vars2
)

# for maturity_gt use maturity_cv instead of total_detections
predictors_maturity <- c(
  "maturity_cv",
  best_vars2
)

# ----------------------------
# Helpers
# ----------------------------
calc_metrics <- function(obs, pred) {
  ok <- complete.cases(obs, pred)
  obs <- obs[ok]
  pred <- pred[ok]
  r <- cor(obs, pred)
  tibble(
    R = r,
    R2 = r^2,
    RMSE = sqrt(mean((obs - pred)^2))
  )
}

fit_predict_model <- function(df, response, xvars, method, train_frac = 0.8) {
  df <- df %>% dplyr::select(all_of(c(response, xvars))) %>% drop_na()
  n <- nrow(df)
  idx <- sample(seq_len(n), size = floor(train_frac * n))
  train <- df[idx, , drop = FALSE]
  test  <- df[-idx, , drop = FALSE]
  
  form <- as.formula(paste(response, "~", paste(xvars, collapse = " + ")))
  
  if (method == "lm") {
    mod <- lm(form, data = train)
    pred <- predict(mod, newdata = test)
  } else if (method == "ridge") {
    x_train <- model.matrix(form, train)[, -1, drop = FALSE]
    y_train <- train[[response]]
    x_test  <- model.matrix(form, test)[, -1, drop = FALSE]
    cvfit <- cv.glmnet(x_train, y_train, alpha = 0)
    pred <- as.numeric(predict(cvfit, newx = x_test, s = "lambda.min"))
  } else if (method == "rf") {
    mod <- randomForest(form, data = train, importance = TRUE)
    pred <- predict(mod, newdata = test)
  } else if (method == "plsr") {
    mod <- plsr(form, data = train, scale = TRUE, validation = "none")
    ncomp <- min(10, ncol(train) - 1)
    pred <- as.numeric(predict(mod, newdata = test, ncomp = ncomp))
  } else if (method == "svm") {
    mod <- svm(form, data = train, scale = TRUE)
    pred <- predict(mod, newdata = test)
  } else {
    stop("Unknown method")
  }
  
  metrics <- calc_metrics(test[[response]], pred)
  metrics$model <- method
  metrics$response <- response
  metrics
}

run_all_models <- function(df, response, xvars, nrep = 100) {
  map_dfr(seq_len(nrep), function(r) {
    map_dfr(c("lm", "ridge", "rf", "plsr", "svm"), function(m) {
      fit_predict_model(df, response, xvars, m) %>%
        mutate(rep = r)
    })
  })
}

# ----------------------------
# Run analyses
# ----------------------------

res_total <- run_all_models(merged2, "Total", predictors, nrep = 100)
res_immature <- run_all_models(merged2, "Immature", predictors, nrep = 100)
res_mature <- run_all_models(merged2, "Mature", predictors, nrep = 100)
res_maturity_gt <- run_all_models(merged2, "maturity_gt", predictors_maturity, nrep = 100)

all_res <- bind_rows(
  res_total,
  res_immature,
  res_mature,
  res_maturity_gt
)

summary_tbl <- all_res %>%
  group_by(response, model) %>%
  summarise(
    mean_R = mean(R, na.rm = TRUE),
    sd_R = sd(R, na.rm = TRUE),
    mean_R2 = mean(R2, na.rm = TRUE),
    sd_R2 = sd(R2, na.rm = TRUE),
    mean_RMSE = mean(RMSE, na.rm = TRUE),
    sd_RMSE = sd(RMSE, na.rm = TRUE),
    .groups = "drop"
  )

plot_df <- summary_tbl %>%
  pivot_longer(c(mean_R, mean_R2, mean_RMSE),
               names_to = "metric", values_to = "value")

ggplot(plot_df, aes(x = model, y = value, fill = model)) +
  geom_col(width = 0.7) +
  facet_grid(metric ~ response, scales = "free_y") +
  theme_minimal(base_size = 13) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "none",
    strip.text = element_text(face = "bold")
  ) +
  labs(x = NULL, y = NULL, title = "Model performance across responses")

# save

write_xlsx(
  all_res,
  path = paste0(base_path, "/plot2", "/all_model_replicates.xlsx"),
  col_names = TRUE,
  format_headers = TRUE,
  use_zip64 = FALSE
)

write_xlsx(
  summary_tbl,
  path = paste0(base_path, "/plot2", "/model_performance_summary.xlsx"),
  col_names = TRUE,
  format_headers = TRUE,
  use_zip64 = FALSE
)

summary_tbl

################################################################################

#=================================================================
# Full ground-truth prediction with CA & CV traits with best model
#=================================================================

pred_vars_total <- c(
  "total_detections",
  best_vars2
)

pred_vars_immaturity <- c(
  "berry.immature",
  best_vars2
)

pred_vars_maturity <- c(
  "berry.mature",
  best_vars2
)

pred_vars_maturityp <- c(
  "maturity_cv",
  best_vars2
)

set.seed(123)

calc_rmse <- function(obs, pred) {
  sqrt(mean((obs - pred)^2, na.rm = TRUE))
}

fit_ridge_plot <- function(
    dat,
    response,
    xvars,
    plot_title = "",
    xlab,
    ylab,
    color = "forestgreen"
) {
  
  # Fixed ridge penalty
  ridge_lambda <- 0.02
  
  # Remove duplicated predictors
  xvars <- unique(xvars)
  
  # Keep complete observations only
  dat2 <- dat %>%
    dplyr::select(all_of(c(response, xvars))) %>%
    na.omit()
  
  if (nrow(dat2) < 3) {
    stop("Not enough complete observations to fit the ridge model.")
  }
  
  # Create predictor matrix
  form <- as.formula(
    paste("~", paste(xvars, collapse = " + "))
  )
  
  x <- model.matrix(form, data = dat2)[, -1, drop = FALSE]
  y <- dat2[[response]]
  
  # Fit ridge regression using the full dataset
  ridge_model <- glmnet::glmnet(
    x = x,
    y = y,
    alpha = 0,
    family = "gaussian",
    lambda = ridge_lambda,
    standardize = TRUE
  )
  
  # Fitted values for the full dataset
  pred <- as.numeric(
    predict(
      ridge_model,
      newx = x,
      s = ridge_lambda
    )
  )
  
  obs <- y
  
  # In-sample model statistics
  r <- cor(obs, pred, use = "complete.obs")
  r2 <- r^2
  rmse <- sqrt(mean((obs - pred)^2, na.rm = TRUE))
  
  dat2$pred <- pred
  
  # Plot limits
  min_axis <- min(c(obs, pred), na.rm = TRUE)
  max_axis <- max(c(obs, pred), na.rm = TRUE)
  
  axis_range <- max_axis - min_axis
  
  if (axis_range == 0) {
    axis_range <- 1
  }
  
  axis_lim <- c(
    min_axis - 0.05 * axis_range,
    max_axis + 0.05 * axis_range
  )
  
  if (min_axis >= 0) {
    axis_lim[1] <- 0
  }
  
  p <- ggplot(dat2, aes(x = .data[[response]], y = pred)) +
    geom_point(
      size = 3.2,
      alpha = 0.75,
      color = color
    ) +
    geom_abline(
      slope = 1,
      intercept = 0,
      linetype = "dashed",
      color = "red",
      linewidth = 0.9
    ) +
    geom_smooth(
      method = "lm",
      color = "black",
      se = FALSE,
      linewidth = 0.9
    ) +
    coord_equal(
      xlim = axis_lim,
      ylim = axis_lim,
      expand = TRUE
    ) +
    theme_bw(base_size = 15) +
    theme(
      plot.title = element_text(
        face = "bold",
        size = 18,
        hjust = 0.5
      ),
      plot.subtitle = element_text(
        size = 18,
        hjust = 0.5
      ),
      axis.title = element_text(
        face = "bold",
        size = 18
      ),
      axis.text = element_text(
        size = 14,
        color = "black"
      ),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "grey90")
    ) +
    labs(
      title = plot_title,
      subtitle = paste0(
        "r = ", round(r, 3),
        " | R² = ", round(r2, 3),
        " | RMSE = ", round(rmse, 2)
      ),
      x = xlab,
      y = ylab
    )
  
  list(
    model = ridge_model,
    data = dat2,
    plot = p,
    predictions = pred,
    r = r,
    r2 = r2,
    rmse = rmse,
    lambda = ridge_lambda,
    coefficients = as.matrix(
      coef(ridge_model, s = ridge_lambda)
    )
  )
}

# run

immature_fit <- fit_ridge_plot(
  dat = merged2,
  response = "Immature",
  xvars = pred_vars_immaturity,
  plot_title = "",
  xlab = "Observed immature berry count",
  ylab = "Fitted immature berry count",
  color = "#1B9E77"
)

mature_fit <- fit_ridge_plot(
  dat = merged2,
  response = "Mature",
  xvars = pred_vars_maturity,
  plot_title = "",
  xlab = "Observed mature berry count",
  ylab = "Fitted mature berry count",
  color = "#D95F02"
)

total_fit <- fit_ridge_plot(
  dat = merged2,
  response = "Total",
  xvars = pred_vars_total,
  plot_title = "",
  xlab = "Observed total berry count",
  ylab = "Fitted total berry count",
  color = "#7570B3"
)

maturity_fit <- fit_ridge_plot(
  dat = merged2,
  response = "maturity_gt",
  xvars = pred_vars_maturityp,
  plot_title = "",
  xlab = "Observed maturity (%)",
  ylab = "Fitted maturity (%)",
  color = "#E7298A"
)


combined_plot <- immature_fit$plot + mature_fit$plot + total_fit$plot + maturity_fit$plot +
  plot_layout(ncol = 4) +
  plot_annotation(
    title = "",# "Prediction of berry count and maturity using image-based detections and canopy architecture",
    theme = theme(
      plot.title = element_text(face = "bold", size = 20, hjust = 0.5)
    )
  )

combined_plot

ggsave(
  filename = paste0(base_path, "/plot2/prediction_panel_4col.png"),
  plot = combined_plot,
  width = 22,
  height = 6,
  dpi = 300
)
