## Introduction

There is a `combine` function in the `randomForest` package that
combines two or more ensembles of trees into one. Therefore, we can
train ensembles in parallel and combine them! This is useful for large
datasets or for testing different parameters.

Install the required packages if missing and then load them.

``` r
.libPaths('/packages')
my_packages <- c('randomForest', 'foreach', 'doParallel', 'parallel', 'doRNG')

for (my_package in my_packages){
   if(!require(my_package, character.only = TRUE)){
      install.packages(my_package, '/packages')
      library(my_package, character.only = TRUE)
   }
}
```

## Load data

We will use the [Letter Recognition Data
Set](https://archive.ics.uci.edu/ml/datasets/Letter+Recognition) which
has 20,000 cases, 16 features, and 26 labels.

``` r
my_url <- 'https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data'
my_df <- read.table(file=url(my_url), header=FALSE, sep=',')
colnames(my_df) <- c('class', 'xbox', 'ybox', 'width', 'high', 'onpix', 'xbar', 'ybar', 'x2bar', 'y2bar', 'xybar', 'x2ybr', 'xy2br', 'xege', 'xegvy', 'yege', 'yegvx')
my_df$class <- factor(my_df$class, levels = LETTERS)

dim(my_df)
```

    ## [1] 20000    17

``` r
table(my_df$class)
```

    ## 
    ##   A   B   C   D   E   F   G   H   I   J   K   L   M   N   O   P   Q   R   S   T 
    ## 789 766 736 805 768 775 773 734 755 747 739 761 792 783 753 803 783 758 748 796 
    ##   U   V   W   X   Y   Z 
    ## 813 764 752 787 786 734

## Metrics

Write a function to calculate metrics.

``` r
metric <- function(rf, type = "accuracy"){
   # not used for now
   # tab <- table(rf$y, rf$predicted)
   # accuracy <- (TP + TN) / (TP + FN + FP + TN)
   # precision <- TP / (TP + FP)
   # recall <- TP / (TP + FN)
   # specificity <- TN / (TN + FP)
   if(type == "accuracy"){
      return(sum(rf$predicted == rf$y) / length(rf$y))
   } else {
      return(NULL)
   }
}
```

## Training

Train random forests model with 2,000 trees without parallelisation.

``` r
set.seed(1984)
my_time <- system.time(
   my_rf <- randomForest(
      class ~ .,
      data = my_df,
      ntree = 2000
   )
)

my_time
```

    ##    user  system elapsed 
    ##  80.772   6.143  87.121

``` r
metric(my_rf)
```

    ## [1] 0.96935

Train random forests with 2,000 trees in parallel. Note the line of code
`registerDoRNG(seed = 1984)`: this is to ensure that we train the same
model even with parallelisation.

``` r
cl <- makeCluster(10)
registerDoParallel(cl)
registerDoRNG(seed = 1984)

my_time_par <- system.time(
   my_rf_par <- foreach(
      ntree = rep(200, 10),
      .combine = combine,
      .packages = 'randomForest'
   ) %dopar% {
      randomForest(class ~ ., data = my_df, ntree=ntree)
   }
)
stopCluster(cl)

my_time_par
```

    ##    user  system elapsed 
    ##   5.357   4.252  19.719

``` r
metric(my_rf_par)
```

    ## [1] 0.9696

[Set
`.multicombine = TRUE`](https://stackoverflow.com/questions/14106010/parallel-execution-of-random-forest-in-r)
to further increase the speed up. As per the
[documentation](https://cran.r-project.org/web/packages/foreach/foreach.pdf),
the `.multicombine` argument is a:

> logical flag indicating whether the .combine function can accept more
> than two arguments. If an arbitrary .combine function is specified, by
> default, that function will always be called with two arguments. If it
> can take more than two arguments, then setting .multicombine to TRUE
> could improve the performance. The default value is FALSE unless the
> .combine function is cbind, rbind, or c, which are known to take more
> than two arguments.

``` r
cl <- makeCluster(10)
registerDoParallel(cl)
registerDoRNG(seed = 1984)

my_time_par_mc <- system.time(
   my_rf_par_mc <- foreach(
      ntree = rep(200, 10),
      .combine = combine,
      .multicombine = TRUE,
      .packages = 'randomForest'
   ) %dopar% {
      randomForest(class ~ ., data = my_df, ntree=ntree)
   }
)
stopCluster(cl)

my_time_par_mc
```

    ##    user  system elapsed 
    ##   1.551   1.084  12.873

``` r
metric(my_rf_par_mc)
```

    ## [1] 0.9696

As noted in the
[documentation](https://cran.r-project.org/web/packages/randomForest/randomForest.pdf)
in the `combine` section:

> The confusion, err.rate, mse and rsq components (as well as the
> corresponding components in the test compnent, if exist) of the
> combined object will be NULL.

But we can calculate those ourselves, if we want.

``` r
# confusion matrix, not run
# table(my_df$class, my_rf_par_mc$predicted)

setdiff(names(my_rf), names(my_rf_par_mc))
```

    ## [1] "err.rate"  "confusion"

Most predictions are the same between the model trained without and with
parallelisation.

``` r
table(my_rf$predicted == my_rf_par_mc$predicted)
```

    ## 
    ## FALSE  TRUE 
    ##    98 19902

## Tuning

Check accuracy as a function of the number of trees.

``` r
my_models <- list()
my_time_ntree <- system.time(
   for(n in seq(from = 500, to = 2000, by = 500)){
      set.seed(1984)
      my_rf <- randomForest(
         class ~ .,
         data = my_df,
         ntree = n
      )
      my_models[[paste0("ntree_", n)]] <- my_rf
   }
)

my_time_ntree
```

    ##    user  system elapsed 
    ## 199.319   7.435 207.239

``` r
sort(sapply(my_models, metric))
```

    ##  ntree_500 ntree_2000 ntree_1000 ntree_1500 
    ##    0.96925    0.96935    0.96945    0.96955

Check accuracy as a function of the number of trees in parallel.

``` r
cl <- makeCluster(4)
registerDoParallel(cl)
registerDoRNG(seed = 1984)

my_time_ntree_par <- system.time(
   my_models_par <- foreach(
      ntree = seq(from = 500, to = 2000, by = 500),
      .packages = 'randomForest'
   ) %dopar% {
      randomForest(class ~ ., data = my_df, ntree = ntree)
   }
)
stopCluster(cl)

my_time_ntree_par
```

    ##    user  system elapsed 
    ##   1.136   0.784  93.234

``` r
sapply(my_models_par, metric)
```

    ## [1] 0.96825 0.97000 0.97000 0.97040

Check accuracy as a function of the number of trees and features in
parallel.

``` r
cl <- makeCluster(32)
registerDoParallel(cl)
registerDoRNG(seed = 1984)

ntrees <- seq(from = 500, to = 2000, by = 500)
mtrys <- 1:ncol(my_df)
my_grid <- expand.grid(ntrees, mtrys)

my_time_ntree_mtry_par <- system.time(
   my_models_mtry_par <- foreach(ntree = my_grid$Var1, mtry = my_grid$Var2, .packages = 'randomForest') %dopar% {
      randomForest(class ~ ., data = my_df, ntree = ntree, mtry = mtry)
   }
)

stopCluster(cl)
my_time_ntree_mtry_par
```

    ##    user  system elapsed 
    ##  17.752  20.017 240.348

``` r
my_grid$accuracy <- sapply(my_models_mtry_par, metric)

head(my_grid[order(my_grid$accuracy, decreasing = TRUE), ])
```

    ##    Var1 Var2 accuracy
    ## 8  2000    2  0.97140
    ## 12 2000    3  0.97125
    ## 11 1500    3  0.97050
    ## 7  1500    2  0.97040
    ## 10 1000    3  0.97040
    ## 15 1500    4  0.96995

## Session info

Time built.

    ## [1] "2022-11-17 04:51:48 UTC"

Session info.

    ## R version 4.2.0 (2022-04-22)
    ## Platform: x86_64-pc-linux-gnu (64-bit)
    ## Running under: Ubuntu 20.04.4 LTS
    ## 
    ## Matrix products: default
    ## BLAS:   /usr/lib/x86_64-linux-gnu/openblas-pthread/libblas.so.3
    ## LAPACK: /usr/lib/x86_64-linux-gnu/openblas-pthread/liblapack.so.3
    ## 
    ## locale:
    ##  [1] LC_CTYPE=en_US.UTF-8       LC_NUMERIC=C              
    ##  [3] LC_TIME=en_US.UTF-8        LC_COLLATE=en_US.UTF-8    
    ##  [5] LC_MONETARY=en_US.UTF-8    LC_MESSAGES=en_US.UTF-8   
    ##  [7] LC_PAPER=en_US.UTF-8       LC_NAME=C                 
    ##  [9] LC_ADDRESS=C               LC_TELEPHONE=C            
    ## [11] LC_MEASUREMENT=en_US.UTF-8 LC_IDENTIFICATION=C       
    ## 
    ## attached base packages:
    ## [1] parallel  stats     graphics  grDevices utils     datasets  methods  
    ## [8] base     
    ## 
    ## other attached packages:
    ## [1] doRNG_1.8.2          rngtools_1.5.2       doParallel_1.0.17   
    ## [4] iterators_1.0.14     foreach_1.5.2        randomForest_4.7-1.1
    ## 
    ## loaded via a namespace (and not attached):
    ##  [1] codetools_0.2-18 digest_0.6.30    magrittr_2.0.3   evaluate_0.17   
    ##  [5] rlang_1.0.6      stringi_1.7.8    cli_3.4.1        rstudioapi_0.14 
    ##  [9] rmarkdown_2.17   tools_4.2.0      stringr_1.4.1    xfun_0.34       
    ## [13] yaml_2.3.6       fastmap_1.1.0    compiler_4.2.0   htmltools_0.5.3 
    ## [17] knitr_1.40
