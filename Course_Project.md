Practical Machine Learning JHU Coursera Course Project
=======================================================
---
author: Yue Harriet Huang
date: December 2014
---

<span style="font-size: 1em;">
This project needs the following R packages:
</span>
<span style="font-size: 2em;">
- install.packages("RCurl")
</span>
<span style="font-size: 2em;">
- install.packages("rmarkdown")
</span>
install.packages("caret")




```r
library(RCurl)
library(rmarkdown)

train_url = getURL("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
test_url = getURL("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")

train = read.csv(text = train_url, na.strings = c("NA", "#DIV/0!", ""))
summary(train)
```

Data Exploration for Missing Value
----------------------------------------
Firstly we import the data and take a rough look at the overall picture and run **summary(train)** command afterwards, we find that variables that have missing values are encoded as either **"#DIV/0!"**, or **"NA"**.

Data Cleaning
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
print("We can take a look at the remaining variables:")
```

```
## [1] "We can take a look at the remaining variables:"
```

```r
colnames(train.new)
```

```
##   [1] "X"                        "user_name"               
##   [3] "raw_timestamp_part_1"     "raw_timestamp_part_2"    
##   [5] "cvtd_timestamp"           "num_window"              
##   [7] "roll_belt"                "pitch_belt"              
##   [9] "yaw_belt"                 "total_accel_belt"        
##  [11] "kurtosis_roll_belt"       "kurtosis_picth_belt"     
##  [13] "skewness_roll_belt"       "skewness_roll_belt.1"    
##  [15] "max_roll_belt"            "max_picth_belt"          
##  [17] "max_yaw_belt"             "min_roll_belt"           
##  [19] "min_pitch_belt"           "min_yaw_belt"            
##  [21] "amplitude_roll_belt"      "amplitude_pitch_belt"    
##  [23] "var_total_accel_belt"     "avg_roll_belt"           
##  [25] "stddev_roll_belt"         "var_roll_belt"           
##  [27] "avg_pitch_belt"           "stddev_pitch_belt"       
##  [29] "var_pitch_belt"           "avg_yaw_belt"            
##  [31] "stddev_yaw_belt"          "var_yaw_belt"            
##  [33] "gyros_belt_x"             "gyros_belt_y"            
##  [35] "gyros_belt_z"             "accel_belt_x"            
##  [37] "accel_belt_y"             "accel_belt_z"            
##  [39] "magnet_belt_x"            "magnet_belt_y"           
##  [41] "magnet_belt_z"            "roll_arm"                
##  [43] "pitch_arm"                "yaw_arm"                 
##  [45] "total_accel_arm"          "var_accel_arm"           
##  [47] "gyros_arm_x"              "gyros_arm_y"             
##  [49] "gyros_arm_z"              "accel_arm_x"             
##  [51] "accel_arm_y"              "accel_arm_z"             
##  [53] "magnet_arm_x"             "magnet_arm_y"            
##  [55] "magnet_arm_z"             "kurtosis_roll_arm"       
##  [57] "kurtosis_picth_arm"       "kurtosis_yaw_arm"        
##  [59] "skewness_roll_arm"        "skewness_pitch_arm"      
##  [61] "skewness_yaw_arm"         "max_picth_arm"           
##  [63] "max_yaw_arm"              "min_yaw_arm"             
##  [65] "amplitude_yaw_arm"        "roll_dumbbell"           
##  [67] "pitch_dumbbell"           "yaw_dumbbell"            
##  [69] "kurtosis_roll_dumbbell"   "kurtosis_picth_dumbbell" 
##  [71] "skewness_roll_dumbbell"   "skewness_pitch_dumbbell" 
##  [73] "max_roll_dumbbell"        "max_picth_dumbbell"      
##  [75] "max_yaw_dumbbell"         "min_roll_dumbbell"       
##  [77] "min_pitch_dumbbell"       "min_yaw_dumbbell"        
##  [79] "amplitude_roll_dumbbell"  "amplitude_pitch_dumbbell"
##  [81] "total_accel_dumbbell"     "var_accel_dumbbell"      
##  [83] "avg_roll_dumbbell"        "stddev_roll_dumbbell"    
##  [85] "var_roll_dumbbell"        "avg_pitch_dumbbell"      
##  [87] "stddev_pitch_dumbbell"    "var_pitch_dumbbell"      
##  [89] "avg_yaw_dumbbell"         "stddev_yaw_dumbbell"     
##  [91] "var_yaw_dumbbell"         "gyros_dumbbell_x"        
##  [93] "gyros_dumbbell_y"         "gyros_dumbbell_z"        
##  [95] "accel_dumbbell_x"         "accel_dumbbell_y"        
##  [97] "accel_dumbbell_z"         "magnet_dumbbell_x"       
##  [99] "magnet_dumbbell_y"        "magnet_dumbbell_z"       
## [101] "roll_forearm"             "pitch_forearm"           
## [103] "yaw_forearm"              "kurtosis_roll_forearm"   
## [105] "kurtosis_picth_forearm"   "skewness_roll_forearm"   
## [107] "skewness_pitch_forearm"   "max_picth_forearm"       
## [109] "max_yaw_forearm"          "min_pitch_forearm"       
## [111] "min_yaw_forearm"          "amplitude_pitch_forearm" 
## [113] "total_accel_forearm"      "var_accel_forearm"       
## [115] "gyros_forearm_x"          "gyros_forearm_y"         
## [117] "gyros_forearm_z"          "accel_forearm_x"         
## [119] "accel_forearm_y"          "accel_forearm_z"         
## [121] "magnet_forearm_x"         "magnet_forearm_y"        
## [123] "magnet_forearm_z"         "classe"
```

# 2. We cut off the index variable denoted as **X** in **train.new**

```r
train.new = train.new[, -1, drop=FALSE]
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
prob = table/sum(table)
prob
```

```
## 
##         A         B         C         D         E 
## 0.2843747 0.1935073 0.1743961 0.1638977 0.1838243
```
We consider that the probability distribution of the response variable is considered relatively balanced and we noticed that there is no missing value in the response variable.

Missing Value Treatment
-------------------------------------------------------------------
We use **rfImpute** to handle missing values in predictors.


```r
formula = classe ~ .
train.imputed = rfImpute(formula, train.new)
```

```
## Error in eval(expr, envir, enclos): could not find function "rfImpute"
```

How do I make Cross-Validation
--------------------------------------------------------------------
Split the training dataset as names **train** above into 60% as **train.split** and 40% as **test.split**


```r
library(caret)
train.ind = createDataPartition(y=train$class, p=0.6, list=FALSE) # create index of training set
train.split = train[train.ind, ]
test.split = train[-train.ind, ]
```

The Expected Out of Sample Error
---------------------------------------------------------------------
