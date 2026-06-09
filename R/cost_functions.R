# main cost function in the package
cost <- function(x, y, p = 2, tensorized = TRUE, cost_function = NULL, df = NULL) {
  if(missing(p) || is.null(p) || is.na(p)) p <- 2L
  cfm <- (missing(cost_function) || is.null(cost_function))
  if (inherits(cost_function, "character") ) {
    costObj <- costOnline$new(x, y, p = p, cost_function = cost_function)
    
  } else if (cfm && isTRUE(tensorized)) {
    costObj <- costTensor$new(x, y, p = p, cost_function = cost_function, df = df)
  } else if (isFALSE(tensorized) && cfm) {
    costObj <- costOnline$new(x, y, p = p, cost_function = cost_function)
  } else if(inherits(cost_function, "function") && isTRUE(tensorized) ) {
    costObj <- costTensor$new(x, y, p = p, cost_function = cost_function, df = df)
  } else if (is.function(cost_function) && isFALSE(tensorized) ){
    stop("cost_function must be either not given or a keops recognized character of functions.")
  } else {
    stop("cost options combination not accounted for! Please report this bug.")
  }
  return(costObj)
}

costParent <- R6::R6Class("cost",
            public = list(
              data = "torch_tensor",
              fun = "function",
              p   = "numeric",
              reduction = "function",
              algorithm = "character",
              df = "ANY"
            ))
costTensor <- R6::R6Class("costTensor",
         inherit = costParent,
         public = list(
           initialize = function(x, y, p = 2, cost_function = NULL, df = NULL) {
             self$p = as.numeric(p)
             self$df = df
             cfm <- (missing(cost_function) || is.null(cost_function))
             if (cfm) {
               self$fun <- function(x1, x2, p, df = NULL) {
                   if(!inherits(x1, "torch_tensor")) {
                     x1 <- torch::torch_tensor(x1, dtype = torch::torch_double())
                   }
                   if(!inherits(x2, "torch_tensor")) {
                     x2 <- torch::torch_tensor(x2, dtype = torch::torch_double())
                   }

                   
                   l <- judge(x1, x2, df)
                   df1 <- l$df_x1
                   df2 <- l$df_x2
                   
                   dt <- x1$dtype
                   dev <- x1$device
                   pforMatrix <- pfor(df1, df2, dt, dev)

                   ((1/p) * (torch::torch_cdist(x1 = x1, 
                                                x2 = x2, 
                                                p = p) + 0.001*pforMatrix)^p)$contiguous()
                 }
                 if( p  == 1) {
                   self$algorithm = "L1"
                 } else if (p == 2) {
                   self$algorithm = "squared.euclidean"
                 } else {
                   self$algorithm = "other"
                 }
             } else if(inherits(cost_function, "function") ) {
              # self$fun <- function(x, y, p) {
              #   (1/p) * cost_function(x, y, p)^p
              # }
               self$fun <- cost_function                          
               self$algorithm = "user"
             } else {
               stop("cost function not found. please report this bug")
             }
             self$data =  self$fun(x, y, p, df = df)
             
             if(!inherits(self$data, "torch_tensor")) {
               self$data <-torch::torch_tensor(self$data, dtype = torch::torch_double())$contiguous()
             }
           }
         ),
         active = list(
           to_device = function(device) {
             if(missing(device) || is.null(device) ){
               return(NULL)
             }
             self$data <- self$data$to(device = device)
             return(invisible(self))
           }
         )
)
costOnline <- R6::R6Class("costOnline",
         inherit = costParent,
         public = c(
           initialize = function(x, y, p = 2, cost_function = NULL) {
             self$p = p
             if (missing(cost_function) || is.null(cost_function) || is.na(cost_function)) {
               self$algorithm <- if (p == 2) {
                 "squared.euclidean"
               } else if (p == 1) {
                 "L1"
               } else {
                 "other"
               }
               cost_function <-  if (p == 2) {
                 "(SqDist(X,Y) / IntCst(2))"
               } else if (p == 1) {
                 "Sum(Abs(X-Y))"
               } else if (is.integer(p)) {
                 paste0("(Sum(Pow(Abs(X-Y),",p,")) /  IntCst(",p,"))")
               } else {
                 stop("'p' must be an integer for online cost functions.")
               }
             } else if (!is.character(cost_function)) {
               stop("cost_function must be not provided or a keops recognized character function.")
             } else {
               self$algorithm <-  "user"
             }
             
            self$data = list(x = x, 
                             y = y)
            self$fun = cost_function
            self$reduction = function(...){NULL}
           }
         ),
         active = list(
           to_device = function(device) {
             if (missing(device) || is.null(device)) {
               return(NULL)
             }
             self$data$x <- self$data$x$to(device = device)
             self$data$y <- self$data$y$to(device = device)
             return(invisible(self))
           }
      )
)


to_device <- function(cost, device) {UseMethod("to_device")}
# setGeneric("to_device", function(cost, device) standardGeneric("to_device"))

# setOldClass(c("costParent","R6"))
# setOldClass(c("costTensor","costParent"))
# setOldClass(c("costOnline", "costParent"))

to_device.costTensor <- function(cost, device) {
  function(cost, device) {
    cost$data <- cost$data$to(device = device)
    return(cost)
  }
}

# setMethod("to_device", signature(cost = "costTensor", device = "ANY"),
# function(cost, device) {
#   cost$data <- cost$data$to(device = device)
#   return(cost)
# }
# )

to_device.costOnline <- function(cost, device) {
  cost$data <- list(x = cost$data$x$to(device = device),
                    y = cost$data$y$to(device = device))
  return(cost)
}

# setMethod("to_device", signature(cost = "costOnline", device = "ANY"),
#           function(cost, device) {
#             cost$data <- list(x = cost$data$x$to(device = device),
#                               y = cost$data$y$to(device = device))
#             return(cost)
#           }
# )

update_cost <- function(cost, x, y) {UseMethod("update_cost")}
setGeneric("update_cost", function(cost, x, y) standardGeneric("update_cost"))

update_cost.costOnline <- function(cost, x, y) {
  n <- nrow(cost$data$x)
  m <- nrow(cost$data$y)
  stopifnot("data for cost rows has different number of rows" = (n == nrow(x)))
  stopifnot("data for cost columns has different number of rows" = (m == nrow(y)))
  stopifnot("data must have same number of columns" = ncol(x) == ncol(y))
  cost$data <- list(x = x, y = y)
}  
# setMethod("update_cost", signature(cost = "costOnline", x = "ANY", y = "ANY"),
# function(cost, x, y) {
#   n <- nrow(cost$data$x)
#   m <- nrow(cost$data$y)
#   stopifnot("data for cost rows has different number of rows" = (n == nrow(x)))
#   stopifnot("data for cost columns has different number of rows" = (m == nrow(y)))
#   stopifnot("data must have same number of columns" = ncol(x) == ncol(y))
#   cost$data <- list(x = x, y = y)
# }          
# )

update_cost.costTensor <- function(cost, x, y) {
  nm <- dim(cost$data)
  device <- cost$data$device
  dtype <- cost$data$dtype
  stopifnot("data for rows has different number of rows" = (nm[1] == nrow(x)))
  stopifnot("data for columns has different number of rows" = (nm[2] == nrow(y)))
  stopifnot("data must have same number of columns" = ncol(x) == ncol(y))
  cost$data <- cost$fun(x,y,cost$p, df = cost$df)$to(device = device, dtype = dtype)
} 
# setMethod("update_cost", signature(cost = "costTensor", x = "ANY", y = "ANY"),
# function(cost, x, y) {
#   nm <- dim(cost$data)
#   device <- cost$data$device
#   dtype <- cost$data$dtype
#   stopifnot("data for rows has different number of rows" = (nm[1] == nrow(x)))
#   stopifnot("data for columns has different number of rows" = (nm[2] == nrow(y)))
#   stopifnot("data must have same number of columns" = ncol(x) == ncol(y))
#   cost$data <- cost$fun(x,y,cost$p)$to(device = device, dtype = dtype)
# }          
# )


# update_cost_col <- function(cost, x, y_vec, j) {UseMethod("update_cost")}
# setGeneric("update_cost", function(cost, x, y) standardGeneric("update_cost"))
# 
# update_cost_col.costOnline <- function(cost, x, y_vec, j) {
#   stopifnot("data must have same number of columns" = ncol(x) == length(y))
#   cost$data$y[j,] <- y_vec
# }  
# 
# update_cost_col.costTensor <- function(cost, x, y_vec, j) {
#   nm <- dim(cost$data)
#   device <- cost$data$device
#   dtype <- cost$data$dtype
#   stopifnot("data for rows has different number of rows" = (nm[1] == nrow(x)))
#   stopifnot("data for columns has different number of rows" = (nm[2] == nrow(y)))
#   stopifnot("data must have same number of columns" = ncol(x) == length(y))
#   cost$data <- cost$fun(x,y,cost$p)$to(device = device, dtype = dtype)
# } 


judge <- function(x1, x2, df) {
  # Split df and build X1, X0, X
  df1 <- subset(df, z == 1)
  df1 <- df1[, colnames(df1) != "z"]
  df0 <- subset(df, z == 0)
  df0 <- df0[, colnames(df0) != "z"]
  df <- df[, colnames(df) != "z"]
  
  X1 <- as.matrix(df1[, colnames(df1) != "y"])
  X0 <- as.matrix(df0[, colnames(df0) != "y"])
  X  <- as.matrix(df[,  colnames(df)  != "y"])

  n1 <- nrow(df1)
  n0 <- nrow(df0)
  n  <- nrow(df)

  # Coerce x1, x2 to base matrices
  to_mat <- function(a) {
    if (inherits(a, "torch_tensor")) {
      return(as.matrix(torch::as_array(a$to(device = "cpu"))))
    }
    as.matrix(a)
  }
  x1m <- to_mat(x1)
  x2m <- to_mat(x2)

  n_x1 <- nrow(x1m)
  n_x2 <- nrow(x2m)

  pick_by_n <- function(nx) {
    if (nx == n1) return(df1)
    if (nx == n0) return(df0)
    if (nx == n)  return(df)
    if (nx == 2) {
      # Handle diameter calculation case where we have min/max values
      # Create a minimal df subset for diameter calculation
      # Use first 2 rows from the full df for consistency
      return(df[1:2, , drop = FALSE])
    }
    stop("No matching size found for x.")
  }

  if (n1 != n0) {
    df_x1 <- pick_by_n(n_x1)
    df_x2 <- pick_by_n(n_x2)
  } else {
    if (n_x1 != n_x2) {
      df_x1 <- pick_by_n(n_x1)
      df_x2 <- pick_by_n(n_x2)
    } else {
      # n1 == n0 and n_x1 == n_x2: decide by contents
      eq <- function(a, b) isTRUE(all.equal(a, b))
      # x1
      if (eq(x1m, X1)) {
        df_x1 <- df1
      } else if (eq(x1m, X0)) {
        df_x1 <- df0
      } else if (n_x1 == n) {
        df_x1 <- df
      } else {
        stop("Cannot decide df_x1 by contents.")
      }
      # x2
      if (eq(x2m, X1)) {
        df_x2 <- df1
      } else if (eq(x2m, X0)) {
        df_x2 <- df0
      } else if (n_x2 == n) {
        df_x2 <- df
      } else {
        stop("Cannot decide df_x2 by contents.")
      }
    }
  }

  list(df_x1 = df_x1, df_x2 = df_x2)
}

pfor <- function(df1, df2, dt, dev) {
  if (ncol(df1) == 6) {
    n_rows <- nrow(df1)
    n_cols <- nrow(df2)
    
    result <- matrix(0, nrow = n_rows, ncol = n_cols)
    pfor <- torch::torch_tensor(result, dtype = dt, device = dev)
    
  } else {
    X1 <- as.matrix(df1[, colnames(df1) != "y"])
    X2 <- as.matrix(df2[, colnames(df2) != "y"])
    
    y1 <- df1$y
    y2 <- df2$y
    
    standardize_with <- function(X, mu, sd) {
      sd_safe <- sd
      sd_safe[is.na(sd_safe) | sd_safe == 0] <- 1
      Xc <- sweep(X, 2, mu, "-")
      sweep(Xc, 2, sd_safe, "/")
    }
    
    mu1 <- colMeans(X1)
    sd1 <- apply(X1, 2, sd)
    mu2 <- colMeans(X2)
    sd2 <- apply(X2, 2, sd)
    
    X1_std <- standardize_with(X1, mu1, sd1)
    X2_std <- standardize_with(X2, mu2, sd2)
    
    X2_for_model1 <- standardize_with(X2, mu1, sd1)
    X1_for_model2 <- standardize_with(X1, mu2, sd2)
    
    # === 高速なランダムサーチによるチューニング関数 ===
    optimizer_tune_lgb_random <- function(X_train, y_train, X_val, y_val, n_trials = 10) { # n_trialsのデフォルト値を変更
      best_r2 <- -Inf
      best_params <- list()
      
      for (i in 1:n_trials) {
        # ランダムにパラメータを生成
        params <- list(
          objective = "regression",
          metric = "rmse",
          verbosity = -1,
          learning_rate = runif(1, 0.01, 0.2),
          num_leaves = sample(10:150, 1),
          feature_fraction = runif(1, 0.5, 1.0),
          bagging_fraction = runif(1, 0.5, 1.0),
          lambda_l1 = runif(1, 0, 5),
          lambda_l2 = runif(1, 0, 5),
          min_data_in_leaf = sample(5:50, 1),
          max_depth = sample(3:15, 1)
        )
        
        tryCatch({
          data_train <- lightgbm::lgb.Dataset(data = X_train, label = y_train)
          use_early_stopping <- length(y_val) >= 10
          
          if (use_early_stopping) {
            model <- lightgbm::lgb.train(
              params = params, data = data_train, nrounds = 200,
              early_stopping_rounds = 20,
              valid = list(val = lightgbm::lgb.Dataset(data = X_val, label = y_val))
            )
            pred <- predict(model, X_val)
            sse <- sum((pred - y_val)^2)
            sst <- sum((y_val - mean(y_val))^2)
          } else {
            model <- lightgbm::lgb.train(params = params, data = data_train, nrounds = 100)
            pred <- predict(model, X_train)
            sse <- sum((pred - y_train)^2)
            sst <- sum((y_train - mean(y_train))^2)
          }
          
          r2 <- if (sst > 0) 1 - sse / sst else 0
          
          if (r2 > best_r2) {
            best_r2 <- r2
            best_params <- params
          }
        }, error = function(e) {})
      }
      
      if (length(best_params) == 0) {
        return(list(params = list(objective="regression", metric="rmse", verbosity=-1, learning_rate=0.1, num_leaves=31), r2=NA))
      }
      return(list(params = best_params, r2 = best_r2))
    }
    
    # データ分割ロジック (変更なし)
    set.seed(42)
    min_val_size <- 10
    val_ratio <- 0.2
    
    if (length(y1) > min_val_size * 2) {
      train_idx1 <- sample(length(y1), floor((1 - val_ratio) * length(y1)))
      X1_train <- X1_std[train_idx1, , drop = FALSE]; y1_train <- y1[train_idx1]
      X1_val <- X1_std[-train_idx1, , drop = FALSE]; y1_val <- y1[-train_idx1]
    } else {
      X1_train <- X1_std; y1_train <- y1
      X1_val <- matrix(0, nrow=0, ncol=ncol(X1_std)); y1_val <- numeric(0)
    }
    
    if (length(y2) > min_val_size * 2) {
      train_idx2 <- sample(length(y2), floor((1 - val_ratio) * length(y2)))
      X2_train <- X2_std[train_idx2, , drop = FALSE]; y2_train <- y2[train_idx2]
      X2_val <- X2_std[-train_idx2, , drop = FALSE]; y2_val <- y2[-train_idx2]
    } else {
      X2_train <- X2_std; y2_train <- y2
      X2_val <- matrix(0, nrow=0, ncol=ncol(X2_std)); y2_val <- numeric(0)
    }
    
    # ★試行回数を10回に減らしてチューニング実行
    tuned1 <- optimizer_tune_lgb_random(X1_train, y1_train, X1_val, y1_val, n_trials = 10)
    tuned2 <- optimizer_tune_lgb_random(X2_train, y2_train, X2_val, y2_val, n_trials = 10)
    
    cat(sprintf("Model 1: Random Search completed with R^2 = %.6f\n", tuned1$r2))
    cat(sprintf("Model 2: Random Search completed with R^2 = %.6f\n", tuned2$r2))
    
    # 最終モデルの学習
    data1 <- lightgbm::lgb.Dataset(data = X1_std, label = y1)
    model1 <- lightgbm::lgb.train(params = tuned1$params, data = data1, nrounds = 200)
    y1Hat <- predict(model1, X2_for_model1)
    
    data2 <- lightgbm::lgb.Dataset(data = X2_std, label = y2)
    model2 <- lightgbm::lgb.train(params = tuned2$params, data = data2, nrounds = 200)
    y2Hat <- predict(model2, X1_for_model2)
    
    # === 高速なベクトル化計算 ===
    term1 <- outer(y1, y1Hat, "-")^2
    term2 <- outer(y2Hat, y2, "-")^2
    result <- term1 + term2
    
    pfor <- torch::torch_tensor(result, dtype = dt, device = dev)
  }
  return(pfor)
}