# Random Forests

A random forest is constructed from many decision trees.

# Terminology

From the glossary of this review: [Machine learning applications in genetics and genomics](http://www.ncbi.nlm.nih.gov/pubmed/25948244).

* Features: Single measurements or descriptors of examples used in a machine learning task.
* Label: The target of a prediction task. In classification, the label is discrete (for example, 'expressed' or 'not expressed'); in regression, the label is of real value (for example, a gene expression value).
* Feature selection: The process of choosing a smaller set of features from a larger set, either before applying a machine learning method or as part of training.
* Sensitivity: (Also known as recall). The fraction of positive examples identified; it is given by the number of positive predictions that are correct divided by the total number of positive examples.
* Precision: The fraction of positive predictions that are correct; it is given by the number of positive predictions that are correct divided by the total number of positive predictions.
* Precision-recall curve: For a binary classifier applied to a given data set, a curve that plots precision (y axis) versus recall (x axis) for a variety of classification thresholds.

# An example

* Using this [Wine Data Set](http://archive.ics.uci.edu/ml/datasets/Wine).
* There are 13 features, which are the results of a chemical analysis of wines
* There are three labels, representing three different [cultivars](https://en.wikipedia.org/wiki/Cultivar)
* We can build a Random Forest classifier to classify wines based on their 13 features
* <http://davetang.org/muse/2012/12/20/random-forests-in-predicting-wines/>

# Decision trees

* <https://en.wikipedia.org/wiki/Decision_tree>
* <http://davetang.org/muse/2013/03/12/building-a-classification-tree-in-r/>

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

# Further reading

* [Machine learning 101](http://www.astroml.org/sklearn_tutorial/general_concepts.html)
* [Some Things Every Biologist Should Know About Machine Learning](http://www.bioconductor.org/help/course-materials/2003/Milan/Lectures/MachineLearning.pdf)
* [An introduction to ROC analysis](https://ccrma.stanford.edu/workshops/mir2009/references/ROCintro.pdf)
* Decision tree learning on [Wikipedia](https://en.wikipedia.org/wiki/Decision_tree_learning)
* [Identifying Mendelian disease genes with the variant effect scoring tool](http://www.ncbi.nlm.nih.gov/pubmed/23819870)

