Practical Machine Learning JHU Coursera Course Project
=======================================================
---
author: Yue Harriet Huang; date: December 2014
---

install.packages("RCurl")
install.packages("caret")
install.packages("randomForest")
install.packages("rmarkdown")
install.packages("e1071")




```r
library(RCurl)
library(rmarkdown)

train_url = getURL("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
test_url = getURL("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")

train = read.csv(text = train_url, na.strings = c("NA", "#DIV/0!", ""))
testing = read.csv(text = test_url, na.strings = c("NA", "#DIV/0!", ""))
```

Data Exploration for Missing Value
----------------------------------------
Firstly we import the data and take a rough look at the overall picture and run **summary(train)** command afterwards, we find that variables that have missing values are encoded as either **"#DIV/0!"**, or **"NA"**.

# Data Cleaning
----------------------------------------
# 1. Clear out variables that are almost constant

```r
library(caret)
var.constant = nearZeroVar(train)
print("Here are the variables that are almost constant:")
```

```
## [1] "Here are the variables that are almost constant:"
```

```r
colnames(train)[var.constant]
```

```
##  [1] "new_window"             "kurtosis_yaw_belt"     
##  [3] "skewness_yaw_belt"      "amplitude_yaw_belt"    
##  [5] "avg_roll_arm"           "stddev_roll_arm"       
##  [7] "var_roll_arm"           "avg_pitch_arm"         
##  [9] "stddev_pitch_arm"       "var_pitch_arm"         
## [11] "avg_yaw_arm"            "stddev_yaw_arm"        
## [13] "var_yaw_arm"            "max_roll_arm"          
## [15] "min_roll_arm"           "min_pitch_arm"         
## [17] "amplitude_roll_arm"     "amplitude_pitch_arm"   
## [19] "kurtosis_yaw_dumbbell"  "skewness_yaw_dumbbell" 
## [21] "amplitude_yaw_dumbbell" "kurtosis_yaw_forearm"  
## [23] "skewness_yaw_forearm"   "max_roll_forearm"      
## [25] "min_roll_forearm"       "amplitude_roll_forearm"
## [27] "amplitude_yaw_forearm"  "avg_roll_forearm"      
## [29] "stddev_roll_forearm"    "var_roll_forearm"      
## [31] "avg_pitch_forearm"      "stddev_pitch_forearm"  
## [33] "var_pitch_forearm"      "avg_yaw_forearm"       
## [35] "stddev_yaw_forearm"     "var_yaw_forearm"
```

```r
print("We pick those variables that have variances significantly bigger than 0 in training models:")
```

```
## [1] "We pick those variables that have variances significantly bigger than 0 in training models:"
```

```r
train.new = train[, -var.constant, drop=FALSE]

###############################################
# We do the same to dataset testing           #
###############################################

var.constant.t = nearZeroVar(testing)
# Remove variables almost constant from the predictors
testing.new = testing[, -var.constant.t, drop=FALSE]
```

# 2. We cut off the index variable denoted as **X** from **train.new** and from **testing.new**

```r
train.new = train.new[, -1, drop=FALSE]

testing.new = testing.new[, -1, drop=FALSE]
```


# 3. Response Variable Distribution

We identify that the **classe** variable is our response variable and we will take a look at its distribution first:


```r
print("Absolute Quantity")
```

```
## [1] "Absolute Quantity"
```

```r
table = table(train$classe)
table
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

```r
print("Probability Distribution")
```

```
## [1] "Probability Distribution"
```

```r
prop.table(table)
```

```
## 
##         A         B         C         D         E 
## 0.2843747 0.1935073 0.1743961 0.1638977 0.1838243
```
We consider that the probability distribution of the response variable is regarded relatively balanced and we noticed that there is no missing value in the response variable.

Missing Value Treatment
-------------------------------------------------------------------
We are going to delete those predictors that have missing values and at the same time we will also delete variables with missing values from the **testing.new** dataset.
Firstly we identify variables that contain missing values


```r
# return var index that contains missing values 
contain_NA = function(dataset){
  var = c()
  for (i in 1:ncol(dataset)){
    vec = is.na(dataset[, i])
    if (any(vec == TRUE)){ 
      var = c(var, i) 
    } 
  }
  return(var) 
}

# These are the variabls that have missing values
delete = contain_NA(train.new)

delete.t = contain_NA(testing.new)

# Delete the variables with missing values from the predictor list

train.new = train.new[, -delete, drop=FALSE]

# We find that there is no missing value contained in testing.new

delete.t
```

```
## NULL
```

```r
# Check if the dataset has any missing value now

check_NA = function(dataset){
  out = FALSE
  for (i in 1:ncol(dataset)){  
    vec = is.na(dataset[, 1])
    if (any(vec == TRUE)){ 
      out = TRUE
    }
  }
  return(out)
}

# Returns TRUE if the dataset still contains missing values and FALSE otherwise
check_NA(train.new)
```

```
## [1] FALSE
```

```r
check_NA(testing.new)
```

```
## [1] FALSE
```

How do I make Cross-Validation
--------------------------------------------------------------------
Split the training dataset as names **train.new** above into 60% as **train.split** and 40% as **test.split**


```r
library(caret)
train.ind = createDataPartition(y=train.new$classe, p=0.6, list=FALSE) # create index of training set
train.split = train.new[train.ind, ]
test.split = train.new[-train.ind, ]
```


Feature Selectiong
--------------------------------------------------
formula = classe~.
feature.model

Model Building: Random Forest
--------------------------------------------------
We will be using the package **"randomForest"**


```r
library(randomForest)
formula = classe~.
model.forest = randomForest(formula, data = train.split)
```

We make predictions on the dataset **test.imputed**


```r
pred.forest = predict(model.forest, test.split, type = "class")

# We can do a premilinary check on accuracy
table(pred.forest, test.split$classe)
```

```
##            
## pred.forest    A    B    C    D    E
##           A 2232    0    0    0    0
##           B    0 1518    2    0    0
##           C    0    0 1366    2    0
##           D    0    0    0 1284    1
##           E    0    0    0    0 1441
```

```r
sum(test.split$classe == pred.forest)/nrow(test.split)
```

```
## [1] 0.9993627
```
The Expected Out of Sample Error
---------------------------------------------------------------------


```r
install.packages("e1071")
```

```
## Error in contrib.url(repos, "source"): trying to use CRAN without setting a mirror
```

```r
library(e1071)
```

```
## Error in library(e1071): there is no package called 'e1071'
```

```r
confusionMatrix(pred.forest, test.split[, ncol(test.split)])
```

```
## Error in loadNamespace(name): there is no package called 'e1071'
```

Submission
---------------------------------------------------------------------


```r
pred.test = predict(model.forest, testing, type = "class")
```

```
## Error in predict.randomForest(model.forest, testing, type = "class"): Type of predictors in new data do not match that of the training data.
```

```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(pred.test)
```

```
## Error in pml_write_files(pred.test): object 'pred.test' not found
```
