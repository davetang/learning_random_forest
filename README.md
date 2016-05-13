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

# Further reading

* [Machine learning 101](http://www.astroml.org/sklearn_tutorial/general_concepts.html)
* [Some Things Every Biologist Should Know About Machine Learning](http://www.bioconductor.org/help/course-materials/2003/Milan/Lectures/MachineLearning.pdf)
* [An introduction to ROC analysis](https://ccrma.stanford.edu/workshops/mir2009/references/ROCintro.pdf)
* Decision tree learning on [Wikipedia](https://en.wikipedia.org/wiki/Decision_tree_learning)
* [Identifying Mendelian disease genes with the variant effect scoring tool](http://www.ncbi.nlm.nih.gov/pubmed/23819870)

