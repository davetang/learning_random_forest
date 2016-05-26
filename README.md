# Introduction

The Random Forest method is based on [decision trees](https://en.wikipedia.org/wiki/Decision_tree); hundreds of them. A decision tree can be thought of as a flow chart that you follow through to classify a case; I have a [blog post](http://davetang.org/muse/2013/03/12/building-a-classification-tree-in-r/) on building a decision tree (a.k.a. classification tree) in R. To illustrate the process of building a Random Forest classifier, consider a two-dimensional dataset with _N_ cases (rows) that has _M_ variables (columns). The Random Forest algorithm will start building independent decision trees; the default number of trees is 500 in the randomForest R package. For each tree, a random subset of _n_ cases is sampled from all available _N_ cases; the cases not used in tree construction are the Out Of Bag (OOB) cases. However, unlike typical decision trees, at each node of a tree, a random number of _m_ variables is used to determine the decision at the node; _m_ is typically the square root of _M_ and the best split is calculated based on the _m_ variables. When classifying a new case, it is fed down each tree and is classified as a specific class or label. This procedure is carried out across all the trees in the ensemble and the majority class or label is assigned to the new case.

For more information read on...

# Terminology

From the glossary of this review: [Machine learning applications in genetics and genomics](http://www.ncbi.nlm.nih.gov/pubmed/25948244).

* Features/Predictors/Variables: Single measurements or descriptors of examples used in a machine learning task.
* Label: The target of a prediction task. In classification, the label is discrete (for example, 'expressed' or 'not expressed'); in regression, the label is of real value (for example, a gene expression value).
* Feature selection: The process of choosing a smaller set of features from a larger set, either before applying a machine learning method or as part of training.
* Sensitivity: (Also known as recall). The fraction of positive examples identified; it is given by the number of positive predictions that are correct divided by the total number of positive examples.
* Precision: The fraction of positive predictions that are correct; it is given by the number of positive predictions that are correct divided by the total number of positive predictions.
* Precision-recall curve: For a binary classifier applied to a given data set, a curve that plots precision (y axis) versus recall (x axis) for a variety of classification thresholds.

# More details

In the introduction, I have outlined the main concept behind the Random Forest algorithm; this section provides more information.

The name Random Forest kinda describes the method; a forest is made up of trees and these trees are randomly build. We have a dataset and a random subset (using bootstrap resampling) of it is used to build a decision tree; this sample is typically half of the dataset. This process is repeated again to create a second random subset that is used to build a second decision tree. Since these are random subsets, the predictions made by the second tree should be different from the first tree. At the end, we will have hundreds of trees (a forest) each built from a slightly different subsets of the dataset and each generating different predictions. To add more randomness to the trees, a subset of candidate features/predictors/variables are used to produce a split in a decision tree. For example, if there were 100 predictors, a random subset of 10 will be used at each node to define the best split, instead of the full set of 100. Note that a new random subset of predictors are used at each node; this is different from selecting a random subset of predictors and using that random subset to build the entire tree. The number of predictors to consider at each node is a key parameter and it is recommended that empirical tests be conducted to fidn the best value; the square root of the number of available predictors is usually recommended as a good starting point. Finally, averaging or majority voting are used to combine all the separate predictions made by the individual trees.

Each tree in a Random Forest was built using a random subset and thus we automatically have holdout data for that particular tree; this is known as "Out Of Bag" (OOB) data. Every case in the full dataset will be "in bag" for some trees and "out of bag" for other trees; this is used to evaluate the Random Forest classifier. For example, if a particular case, _x_, was used in 250 trees and not used in another 250 trees, we can apply _x_ to the trees that didn't use it for training. Since _x_ was never used to generate any of the 250 trees, the result provides an assessment of the reliability of the Random Forest classifier. This can be carried out across all the cases in the dataset. Due to this OOB feature in the Random Forest algorithm, we do not need to create an additional holdout or testing dataset.

The Random Forest method also provides a measure of how close each case is to another case in the dataset, which is known as the "proximity". The procedure for calculating the proximity of two cases is to drop a pair of records down each individual tree in a Random Forest, and counting the number of times the two cases end up at the same terminal node, i.e. the same classification, and dividing by the number of trees tested. By carrying out this step across all pairs of cases, a proximity can be constructed. The proximity matrix provides a measure of how similar any two cases are in a dataset. One can perform hierarchical clustering on the proximity matrix to examine the underlying structure of a dataset.

Another useful feature of the Random Forest method is its estimation of relative predictor importance. The method is based on measuring the effect of the classifier if one of the predictors was removed. This is performed by randomly scrambling the values associated to a given predictor; the scrambling is done by moving values from a specific row to another row. The scrambling is performed one predictor at a time and predictive accuracy is measured for each predictor to obtain an estimation of relative predictor importance; note that the data is re-scramble for each predictor being tested.

# An example

Refer to the R Markdown file, [random_forest.Rmd](https://github.com/davetang/learning_random_forest/blob/master/random_forest.Rmd), for more information.

* Using this [Wine Data Set](http://archive.ics.uci.edu/ml/datasets/Wine).
* There are 13 features/predictors/variables, which are the results of a chemical analysis of wines
* There are three labels, representing three different [cultivars](https://en.wikipedia.org/wiki/Cultivar)
* We can build a Random Forest classifier to classify wines based on their 13 features
* <http://davetang.org/muse/2012/12/20/random-forests-in-predicting-wines/>

# Classification and Regression Trees

* <http://www.stat.wisc.edu/~loh/treeprogs/guide/wires11.pdf>

# Random Forest and R

* <http://mkseo.pe.kr/stats/?p=220>

~~~~{.r}
install.packages("randomForest")
library(randomForest)

# preparing the data
data_url <- 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
df <- read.table(file=url(data_url), header=FALSE, sep=",")
header <- c('class',
            'alcohol',
            'malic_acid',
            'ash',
            'ash_alcalinity',
            'magnesium',
            'total_phenols',
            'flavanoids',
            'nonflavanoid_phenols',
            'proanthocyanins',
            'colour',
            'hue',
            'od280_od315',
            'proline')
names(df) <- header
df$class <- as.factor(df$class)

# analysis
# install if necessary
# install.packages("randomForest")
library(randomForest)

set.seed(31)
my_sample <- sort(sample(x = 1:nrow(df), replace = FALSE, size = nrow(df)/2))
my_sample_comp <- setdiff(1:nrow(df), my_sample)

test <- df[my_sample, ]
train <- df[my_sample_comp, ]

r <- randomForest(class ~ ., data=train, importance=TRUE, do.trace=100)

# plots
# install if necessary
# install.packages(ggplot2)
library(ggplot2)
class_1_importance <- data.frame(feature=names(r$importance[,1]), importance=r$importance[,1])
ggplot(class_1_importance, aes(x=feature, y=importance)) + geom_bar(stat="identity")

class_2_importance <- data.frame(feature=names(r$importance[,2]), importance=r$importance[,2])
ggplot(class_2_importance, aes(x=feature, y=importance)) + geom_bar(stat="identity")

class_3_importance <- data.frame(feature=names(r$importance[,3]), importance=r$importance[,3])
ggplot(class_2_importance, aes(x=feature, y=importance)) + geom_bar(stat="identity")

boxplot(df$colour ~ df$class, main="Colour by class")
boxplot(df$alcohol ~ df$class, main="Alcohol by class")

ggplot(df, aes(x=alcohol, y=colour, colour=class)) + geom_point()
~~~~

![Scatter plot of alcohol versus colour by class](image/alcohol_colour.png)

~~~~{.r}
varImpPlot(r)
~~~~

![Variance Importance Plot](image/var_imp_plot.png)

# Parallelisation

The `combine()` function in `randomForest` allows us to combine an ensemble of trees. Using the `doSNOW` and `foreach` packages, we can train several forests in parallel and combine them.

~~~~{.r}
library(randomForest)

# http://archive.ics.uci.edu/ml/datasets/Letter+Recognition
# 20,000 by 16 dataset
data_url <- 'http://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data'
df <- read.table(file=url(data_url), header=FALSE, sep=',')
colnames(df) <- c('class', 'xbox', 'ybox', 'width', 'high', 'onpix', 'xbar', 'ybar', 'x2bar', 'y2bar', 'xybar', 'x2ybr', 'xy2br', 'xege', 'xegvy', 'yege', 'yegvx')

system.time(r <- randomForest(class ~ ., data=df, importance=TRUE, do.trace=100, proximity=TRUE))
# ntree      OOB      1      2      3      4      5      6      7      8      9     10     11     12     13     14     15     16     17     18     19     20     21     22     23     24     25     26
#   100:   3.58%  0.63%  3.79%  3.67%  2.61%  4.17%  5.55%  4.40%  8.72%  4.90%  5.22%  7.04%  2.76%  1.26%  3.96%  3.85%  4.36%  3.45%  4.35%  2.94%  2.39%  1.48%  3.80%  2.13%  2.16%  2.04%  2.32%
#   200:   3.16%  0.25%  3.26%  3.12%  2.36%  3.78%  4.65%  3.75%  7.36%  4.90%  4.28%  6.09%  2.37%  1.39%  3.96%  3.45%  3.86%  3.19%  4.75%  2.94%  1.88%  1.35%  3.14%  0.80%  2.16%  1.53%  2.18%
#   300:   3.25%  0.38%  3.39%  3.40%  2.61%  4.17%  4.65%  3.49%  7.90%  4.64%  4.69%  5.82%  2.50%  1.14%  3.83%  3.45%  4.11%  2.94%  4.75%  2.81%  2.01%  1.60%  3.40%  1.33%  2.29%  1.53%  2.32%
#   400:   3.11%  0.51%  3.00%  2.58%  2.36%  4.17%  4.77%  3.49%  7.77%  4.90%  4.55%  5.82%  2.10%  1.14%  3.70%  3.19%  4.11%  3.19%  4.22%  2.67%  1.88%  1.60%  3.53%  0.80%  1.78%  1.53%  2.04%
#   500:   3.09%  0.51%  2.87%  2.85%  2.98%  3.91%  4.52%  3.49%  7.63%  5.03%  4.82%  5.55%  1.97%  1.14%  3.96%  3.05%  3.74%  2.94%  4.35%  2.67%  1.63%  1.72%  3.40%  0.93%  1.78%  1.53%  1.91%
   user  system elapsed
460.992  44.084 505.639

library(foreach)
library(doSNOW)
registerDoSNOW(makeCluster(5, type='SOCK'))

system.time(rf <- foreach(ntree = rep(100, 5), .combine = combine, .packages = "randomForest") %dopar%  randomForest(class ~ ., data=df, proximity=TRUE, importance=TRUE, ntree = ntree))
#    user  system elapsed
#  48.856  99.192 265.819

# faster parallelisation by using .multicombine=TRUE
# see http://stackoverflow.com/questions/14106010/parallel-execution-of-random-forest-in-r
system.time(rfm <- foreach(ntree = rep(100, 5), .combine = combine, .multicombine=TRUE, .packages = "randomForest") %dopar%  randomForest(class ~ ., data=df, proximity=TRUE, importance=TRUE, ntree = ntree))
#   user  system elapsed
# 30.404  49.052 200.325

length(names(r))
# [1] 19

length(names(rf))
# [1] 17

# as noted in the documentation of the Random Forest package under combine
# The confusion, err.rate, mse and rsq components (as well as the corresponding
# components in the test compnent, if exist) of the combined object will be NULL.
setdiff(names(r), names(rf))
# [1] "err.rate"  "confusion"
~~~~

# Further reading

* [Machine learning 101](http://www.astroml.org/sklearn_tutorial/general_concepts.html)
* [Some Things Every Biologist Should Know About Machine Learning](http://www.bioconductor.org/help/course-materials/2003/Milan/Lectures/MachineLearning.pdf)
* [An introduction to ROC analysis](https://ccrma.stanford.edu/workshops/mir2009/references/ROCintro.pdf)
* Decision tree learning on [Wikipedia](https://en.wikipedia.org/wiki/Decision_tree_learning)
* [Identifying Mendelian disease genes with the variant effect scoring tool](http://www.ncbi.nlm.nih.gov/pubmed/23819870)

