import random
import re
import os
import sys
import time

# Database of questions and answers extracted from the documents
questions_db = [
       {
        "question": "In the context of system security, the term 'system' refers to what?",
        "options": {
            "A": "Any combination of hardware, software, infrastructure, and processes that work together to perform specific functions or tasks",
            "B": "Only a computer's operating system",
            "C": "Only the software applications installed on a computer",
            "D": "Only the hardware components of a network"
        },
        "correct": "A"
    },
    {
        "question": "In supervised learning, what information is available during training that distinguishes it from unsupervised learning?",
        "options": {
            "A": "Unlabeled data points",
            "B": "Labeled examples with input data and associated true labels",
            "C": "Only feature vectors without any target information",
            "D": "Raw data without any preprocessing"
        },
        "correct": "B"
    },
    {
        "question": "Why is K-Nearest Neighbors (KNN) often described as a 'lazy learner'?",
        "options": {
            "A": "It requires extensive preprocessing of data",
            "B": "It builds complex models during training",
            "C": "It doesn't build a model during training, storing data and making predictions only at query time",
            "D": "It uses too much computational power during training"
        },
        "correct": "C"
    },
    {
        "question": "Are decision trees prone to overfitting, and why?",
        "options": {
            "A": "No, they always generalize well to new data",
            "B": "Yes, because they can grow very deep and fit noise or specific patterns that don't generalize",
            "C": "No, they are immune to overfitting due to their structure",
            "D": "Only when the dataset is too small"
        },
        "correct": "B"
    },
    {
        "question": "Why is it important to have a diverse set of models in an ensemble like Bagging or Random Forest?",
        "options": {
            "A": "To increase computational complexity",
            "B": "To make the model run slower",
            "C": "To ensure individual errors are less likely to overlap, allowing mistakes to cancel out through majority voting",
            "D": "To use more memory during training"
        },
        "correct": "C"
    },
    {
        "question": "If your priority is to catch as many spam emails as possible, even if it means sometimes marking legitimate emails as spam, which metric should you prioritize?",
        "options": {
            "A": "Precision",
            "B": "Accuracy",
            "C": "Recall",
            "D": "F1-score"
        },
        "correct": "C"
    },
    {
        "question": "If your priority is to minimize the number of legitimate emails wrongly classified as spam, which metric should you prioritize?",
        "options": {
            "A": "Recall",
            "B": "Precision",
            "C": "Accuracy",
            "D": "True Negative Rate"
        },
        "correct": "B"
    },
    {
        "question": "What is the main drawback of hill-climbing as a local search algorithm?",
        "options": {
            "A": "It uses too much memory",
            "B": "It can get stuck in local optima because it only accepts steps that improve the solution",
            "C": "It always finds the global optimum",
            "D": "It runs too slowly on large datasets"
        },
        "correct": "B"
    },
    {
        "question": "In a dataset where features are perfectly correlated, what can you observe about the relationship between features?",
        "options": {
            "A": "They are completely independent",
            "B": "They contain redundant information",
            "C": "They have no mathematical relationship",
            "D": "They cancel each other out"
        },
        "correct": "B"
    },
    {
        "question": "When features in a dataset are perfectly correlated, what is the minimum number of principal components needed to fully represent the data using PCA?",
        "options": {
            "A": "All original features are needed",
            "B": "Half of the original features",
            "C": "Two principal components",
            "D": "One principal component"
        },
        "correct": "D"
    },
    {
        "question": "What is a key difference between K-means and DBSCAN clustering algorithms?",
        "options": {
            "A": "K-means is faster than DBSCAN in all cases",
            "B": "K-means requires specifying the number of clusters in advance, while DBSCAN defines clusters based on density",
            "C": "DBSCAN always produces spherical clusters",
            "D": "K-means can handle noise better than DBSCAN"
        },
        "correct": "B"
    },
    {
        "question": "What type of cluster shapes can DBSCAN identify compared to K-means?",
        "options": {
            "A": "Only spherical clusters like K-means",
            "B": "Only rectangular clusters",
            "C": "Clusters with arbitrary shapes and sizes",
            "D": "Only linear clusters"
        },
        "correct": "C"
    },
    {
        "question": "If an anomaly detection model trained on year-round data consistently fails to detect unusual winter events, what type of anomaly is likely being missed?",
        "options": {
            "A": "Point anomalies",
            "B": "Collective anomalies",
            "C": "Contextual anomalies",
            "D": "Global anomalies"
        },
        "correct": "C"
    },
    {
        "question": "How could you adjust an anomaly detection model to better capture seasonal anomalies?",
        "options": {
            "A": "Use more training data from the same season",
            "B": "Incorporate seasonal factors as features or train separate models for different seasons",
            "C": "Reduce the number of features used",
            "D": "Increase the detection threshold"
        },
        "correct": "B"
    },
    {
        "question": "In confusion matrix terminology for spam detection, what does FP (False Positive) represent?",
        "options": {
            "A": "Spam emails correctly identified as spam",
            "B": "Legitimate emails correctly identified as not spam",
            "C": "Legitimate emails incorrectly identified as spam",
            "D": "Spam emails incorrectly identified as not spam"
        },
        "correct": "C"
    },
    {
        "question": "In confusion matrix terminology for spam detection, what does FN (False Negative) represent?",
        "options": {
            "A": "Spam emails correctly identified as spam",
            "B": "Legitimate spam emails incorrectly identified as not spam",
            "C": "Legitimate emails incorrectly identified as spam",
            "D": "Legitimate emails correctly identified as not spam"
        },
        "correct": "B"
    },
    {
        "question": "According to the product rule in probability, P(A, B) equals:",
        "options": {
            "A": "P(A) + P(B|A)",
            "B": "P(A) * P(B|A)",
            "C": "P(A) / P(B|A)",
            "D": "P(A) - P(B|A)"
        },
        "correct": "B"
    },
    {
        "question": "According to Bayes' theorem, P(Y|X) equals:",
        "options": {
            "A": "P(Y) * P(X|Y) / P(X)",
            "B": "P(Y) + P(X|Y) / P(X)",
            "C": "P(Y) / P(X|Y) * P(X)",
            "D": "P(Y) - P(X|Y) / P(X)"
        },
        "correct": "A"
    },
    {
        "question": "If events A and B are independent, then P(A, B) equals:",
        "options": {
            "A": "P(A) + P(B)",
            "B": "P(A) - P(B)",
            "C": "P(A) * P(B)",
            "D": "P(A) / P(B)"
        },
        "correct": "C"
    },
    {
        "question": "In Naive Bayes classification, what assumption is made about the features?",
        "options": {
            "A": "Features are perfectly correlated",
            "B": "Features are conditionally independent given the class",
            "C": "Features must be numerical",
            "D": "Features must be normally distributed"
        },
        "correct": "B"
    },
    {
        "question": "What is the correct order for Depth-First Search traversal of a binary tree starting from the root?",
        "options": {
            "A": "Visit all nodes at the same level before moving to the next level",
            "B": "Visit the root, then recursively visit left subtree, then right subtree",
            "C": "Visit nodes in random order",
            "D": "Visit leaf nodes first, then parent nodes"
        },
        "correct": "B"
    },
    {
        "question": "What is the correct order for Breadth-First Search traversal of a binary tree?",
        "options": {
            "A": "Visit the root, then recursively visit left subtree, then right subtree",
            "B": "Visit all nodes at the same level before moving to the next level",
            "C": "Visit leaf nodes first",
            "D": "Visit nodes in alphabetical order"
        },
        "correct": "B"
    },
    {
        "question": "What is a principal component in PCA?",
        "options": {
            "A": "The original feature with the highest variance",
            "B": "A linear combination of original features that captures maximum variance",
            "C": "The mean of all features",
            "D": "The feature with the most missing values"
        },
        "correct": "B"
    },
    {
        "question": "When would you choose a model with higher recall over higher precision?",
        "options": {
            "A": "When false positives are more costly than false negatives",
            "B": "When false negatives are more costly than false positives",
            "C": "When you want to minimize overall errors",
            "D": "When the dataset is perfectly balanced"
        },
        "correct": "B"
    },
    {
        "question": "In the context of clustering, what does 'density-based' mean for DBSCAN?",
        "options": {
            "A": "Clusters are formed based on the distance to centroids",
            "B": "Clusters are formed based on the density of data points in a region",
            "C": "All clusters must have the same number of points",
            "D": "Clusters are formed randomly"
        },
        "correct": "B"
    },
    {
        "question": "What happens when PCA is applied to perfectly correlated features?",
        "options": {
            "A": "All principal components explain equal variance",
            "B": "The first principal component explains 100% of the variance",
            "C": "PCA cannot be applied to correlated features",
            "D": "The number of components increases"
        },
        "correct": "B"
    },
    {
        "question": "In ensemble methods, what does 'majority voting' refer to?",
        "options": {
            "A": "Selecting the model with highest accuracy",
            "B": "Combining predictions by taking the class predicted by most models",
            "C": "Using only the first model's prediction",
            "D": "Averaging all model predictions"
        },
        "correct": "B"
    },
    {
        "question": "What is the key advantage of Random Forest over a single decision tree?",
        "options": {
            "A": "It runs faster",
            "B": "It uses less memory",
            "C": "It reduces overfitting by combining multiple diverse trees",
            "D": "It requires fewer features"
        },
        "correct": "C"
    },
    {
        "question": "In anomaly detection, what makes an anomaly 'contextual'?",
        "options": {
            "A": "It occurs frequently in the dataset",
            "B": "It is normal in some contexts but abnormal in others",
            "C": "It affects multiple features simultaneously",
            "D": "It is always the most extreme value"
        },
        "correct": "B"
    },
    {
        "question": "When features contain redundant information, what is the primary benefit of using PCA?",
        "options": {
            "A": "It increases the number of features",
            "B": "It reduces dimensionality while preserving most of the variance",
            "C": "It makes all features independent",
            "D": "It removes all noise from the data"
        },
        "correct": "B"
    },
{
    "question": "In job shop scheduling, what is a precedence constraint?",
    "options": {
        "A": "Jobs must arrive in numerical order",
        "B": "Operations within a job must follow a specific sequence and cannot start before the previous operation is complete",
        "C": "Machines must process jobs in arrival time order",
        "D": "All operations must finish at the same time"
    },
    "correct": "B"
},
{
    "question": "What is the resource constraint in job shop scheduling?",
    "options": {
        "A": "Each job can only use one machine total",
        "B": "Each machine can process at most one operation at a time",
        "C": "All machines must be used equally",
        "D": "Operations cannot be interrupted once started"
    },
    "correct": "B"
},
{
    "question": "In job shop scheduling, what does the makespan represent?",
    "options": {
        "A": "The total processing time of all operations",
        "B": "The average completion time of all jobs",
        "C": "The maximum completion time of all jobs",
        "D": "The minimum time required to complete one job"
    },
    "correct": "C"
},
{
    "question": "What does the Shortest Processing Time (SPT) dispatching rule prioritize?",
    "options": {
        "A": "Jobs that arrived first",
        "B": "Jobs with the longest total processing time",
        "C": "Operations with the shortest processing time",
        "D": "Jobs with the most operations"
    },
    "correct": "C"
},
{
    "question": "What does the First-Come-First-Serve (FCFS) dispatching rule prioritize?",
    "options": {
        "A": "Operations with shortest processing time",
        "B": "Jobs in order of their arrival time",
        "C": "Jobs with highest priority",
        "D": "Operations that use specific machines"
    },
    "correct": "B"
},
{
    "question": "In vehicle routing problems, what is the depot?",
    "options": {
        "A": "The node with the highest demand",
        "B": "The starting and ending point for all routes",
        "C": "The node furthest from all other nodes",
        "D": "Any node that requires service"
    },
    "correct": "B"
},
{
    "question": "In the nearest neighbor heuristic for vehicle routing, how is the next node selected?",
    "options": {
        "A": "The node with the highest demand",
        "B": "The node that arrived first",
        "C": "The closest unvisited node that can be served within capacity constraints",
        "D": "A randomly selected unvisited node"
    },
    "correct": "C"
},
{
    "question": "What is the capacity constraint in vehicle routing problems?",
    "options": {
        "A": "Each vehicle can visit at most a certain number of nodes",
        "B": "The total demand served by each vehicle cannot exceed its capacity",
        "C": "Each vehicle must return to the depot within a time limit",
        "D": "All vehicles must have the same route length"
    },
    "correct": "B"
},
{
    "question": "In the two-level bin packing problem, what happens at Level 1?",
    "options": {
        "A": "Large containers are packed into small containers",
        "B": "Items are packed into small containers of varying capacities",
        "C": "Small containers are packed into large containers",
        "D": "Items are sorted by size"
    },
    "correct": "B"
},
{
    "question": "In the two-level bin packing problem, what happens at Level 2?",
    "options": {
        "A": "Items are packed into small containers",
        "B": "Small containers are packed into large containers of uniform capacity",
        "C": "Items are redistributed among containers",
        "D": "Containers are sorted by utilization"
    },
    "correct": "B"
},
{
    "question": "What is the main principle of the First-Fit (FF) heuristic in bin packing?",
    "options": {
        "A": "Place each item in the bin with the most remaining space",
        "B": "Place each item in the first bin where it fits",
        "C": "Place each item in the bin with the least remaining space after placement",
        "D": "Sort items by size before packing"
    },
    "correct": "B"
},
{
    "question": "What is the main principle of the Best-Fit (BF) heuristic in bin packing?",
    "options": {
        "A": "Place each item in the first bin where it fits",
        "B": "Place each item in the largest available bin",
        "C": "Place each item in the bin with the least remaining space after placement",
        "D": "Place items randomly in available bins"
    },
    "correct": "C"
},
{
    "question": "In job shop scheduling, what determines the earliest starting time of an operation?",
    "options": {
        "A": "Only the arrival time of the job",
        "B": "Only the availability of the required machine",
        "C": "The later of the operation's earliest ready time and the machine's earliest idle time",
        "D": "The processing time of the operation"
    },
    "correct": "C"
},
{
    "question": "What is a Gantt chart used for in scheduling?",
    "options": {
        "A": "To calculate processing times",
        "B": "To visualize the schedule showing when operations are processed on each machine",
        "C": "To determine job arrival times",
        "D": "To calculate the total cost of scheduling"
    },
    "correct": "B"
},
{
    "question": "In vehicle routing, how is the cost typically calculated between two nodes?",
    "options": {
        "A": "The sum of their coordinates",
        "B": "The Manhattan distance",
        "C": "The Euclidean distance",
        "D": "The difference in their demands"
    },
    "correct": "C"
},
{
    "question": "What is a route in vehicle routing problems?",
    "options": {
        "A": "A path that visits all nodes exactly once",
        "B": "A sequence of nodes starting and ending at the depot",
        "C": "The shortest path between any two nodes",
        "D": "A path that maximizes the total demand served"
    },
    "correct": "B"
},
{
    "question": "In the context of Genetic Programming for Heuristic (GPHH), what does the terminal set represent?",
    "options": {
        "A": "The final solution to the problem",
        "B": "The basic building blocks or variables used in the GP trees",
        "C": "The stopping criteria for the algorithm",
        "D": "The mutation operators"
    },
    "correct": "B"
},
{
    "question": "What is the main difference between static and dynamic job shop scheduling?",
    "options": {
        "A": "Static uses more machines than dynamic",
        "B": "Static has all information known in advance, while dynamic handles real-time changes",
        "C": "Static is always faster than dynamic",
        "D": "Dynamic uses different machines than static"
    },
    "correct": "B"
},
{
    "question": "If one scheduling rule produces a better makespan than another on a specific instance, what can we conclude?",
    "options": {
        "A": "That rule is always better for all instances",
        "B": "That rule should never be used again",
        "C": "Nothing definitive - performance depends on the specific problem instance",
        "D": "The other rule is fundamentally flawed"
    },
    "correct": "C"
},
{
    "question": "In two-level bin packing, if n = 4k items are given, how are they distributed across the four capacity types?",
    "options": {
        "A": "Randomly distributed",
        "B": "All items go to the largest capacity containers first",
        "C": "Items 1 to k go to capacity c₁, k+1 to 2k go to capacity c₂, and so on",
        "D": "Items are sorted by size first, then distributed"
    },
    "correct": "C"
},
{
    "question": "What is the objective in bin packing problems?",
    "options": {
        "A": "Maximize the number of bins used",
        "B": "Minimize the number of bins used while satisfying capacity constraints",
        "C": "Maximize the utilization of the largest bin",
        "D": "Ensure all bins have equal utilization"
    },
    "correct": "B"
},
{
    "question": "In job shop scheduling, what happens if an operation tries to start before its job's arrival time?",
    "options": {
        "A": "The operation is cancelled",
        "B": "The operation must wait until the arrival time",
        "C": "The arrival time is adjusted",
        "D": "The operation can start immediately anyway"
    },
    "correct": "B"
},
{
    "question": "What distinguishes the Best-Fit heuristic from First-Fit in bin packing?",
    "options": {
        "A": "Best-Fit sorts items first",
        "B": "Best-Fit chooses the bin that minimizes waste after item placement",
        "C": "Best-Fit only uses the largest bins",
        "D": "Best-Fit places multiple items at once"
    },
    "correct": "B"
},
{
    "question": "What is the completion time of a job in job shop scheduling?",
    "options": {
        "A": "The sum of all its operation processing times",
        "B": "The time when its first operation starts",
        "C": "The finishing time of its last operation",
        "D": "The difference between start and end times"
    },
    "correct": "C"
},
{
        "question": "In anomaly detection using Isolation Forest and Local Outlier Factor, what is the primary reason for NOT using the provided labels during training?",
        "options": {
            "A": "The labels are incorrect and unreliable",
            "B": "Both algorithms are unsupervised methods",
            "C": "Labels would cause overfitting in the models",
            "D": "The dataset is too small for supervised learning"
        },
        "correct": "B"
    },
    {
        "question": "When preprocessing data for Isolation Forest and LOF algorithms, what must be done with nominal features like 'season'?",
        "options": {
            "A": "They can be used directly without any transformation",
            "B": "They must be converted to numerical values or removed",
            "C": "They should be normalized using z-score standardization",
            "D": "They need to be clustered into groups first"
        },
        "correct": "B"
    },
    {
        "question": "In the decision tree implementation using ID3 algorithm, what is the stopping criterion for tree growth?",
        "options": {
            "A": "When all instances belong to the same class",
            "B": "When the tree depth reaches 10 levels",
            "C": "When Information Gain is less than 0.00001",
            "D": "When there are fewer than 5 instances in a node"
        },
        "correct": "C"
    },
    {
        "question": "For the perceptron implementation, what happens when you increase the number of training epochs on linearly separable data?",
        "options": {
            "A": "Accuracy always decreases due to overfitting",
            "B": "Accuracy typically improves until convergence",
            "C": "The model becomes more complex with more parameters",
            "D": "Training time decreases exponentially"
        },
        "correct": "B"
    },
    {
        "question": "Why is a Multi-Layer Perceptron (MLP) expected to perform better than a simple perceptron on non-linearly separable data?",
        "options": {
            "A": "MLP has more training epochs available",
            "B": "MLP uses a different activation function",
            "C": "MLP can learn non-linear decision boundaries through hidden layers",
            "D": "MLP processes data faster than perceptrons"
        },
        "correct": "C"
    },
    {
        "question": "In anomaly detection evaluation, what does high recall indicate?",
        "options": {
            "A": "Most predicted anomalies are actually true anomalies",
            "B": "Most true anomalies are correctly identified by the model",
            "C": "The model has low computational complexity",
            "D": "The contamination parameter is set correctly"
        },
        "correct": "B"
    },
    {
        "question": "When implementing the decision tree from scratch, what information must be included for split nodes in the output?",
        "options": {
            "A": "Only the splitting feature name",
            "B": "Feature name, Information Gain, and entropy",
            "C": "Class distribution and leaf counts",
            "D": "Only the entropy value"
        },
        "correct": "B"
    },
    {
        "question": "In Part 2 of the assignment using scikit-learn, what is the recommended train-test split ratio?",
        "options": {
            "A": "50:50 split",
            "B": "80:20 split",
            "C": "70:30 split",
            "D": "90:10 split"
        },
        "correct": "C"
    },
    {
        "question": "For the breast cancer dataset preprocessing, how should missing values represented as 0 be handled?",
        "options": {
            "A": "Leave them as 0 since it's a valid numerical value",
            "B": "Remove all instances with missing values",
            "C": "Replace with meaningful numerical values using imputation",
            "D": "Convert them to categorical variables"
        },
        "correct": "C"
    },
    {
        "question": "In feature selection using Pearson correlation, what threshold is specified for retaining features?",
        "options": {
            "A": "Features with correlation > 0.5 with target variable",
            "B": "Features with correlation > 0.6 with target variable",
            "C": "Features with correlation > 0.8 with target variable",
            "D": "Features with correlation > 0.3 with target variable"
        },
        "correct": "B"
    },
    {
        "question": "What is the main limitation of a simple perceptron when applied to the RingSyn dataset?",
        "options": {
            "A": "Insufficient training data",
            "B": "Too many features in the dataset",
            "C": "Cannot learn non-linear decision boundaries",
            "D": "Requires more computational resources"
        },
        "correct": "C"
    },
    {
        "question": "Which of the following best describes supervised learning",
        "options": {
            "A": "Classification and regression are a few examples of supervised learning",
            "B": "Reinforcement learning is a form of supervised learning",
            "C": "Supervised learning involves clustering and association rules",
            "D": "Supervised learning is uhh"
        },
        "correct": "A"
    },
    {
        "question": "Which of the following best describes supervised learning",
        "options": {
            "A": "There is a training phase where instances with ground truth output are used to learn the model",
            "B": "Reinforcement learning is a form of supervised learning",
            "C": "I should have used AI to write this quiz, that would have been pretty funny",
            "D": "Supervised learning involves clustering and association ruels"
        },
        "correct": "A"
    },
    {
        "question": "What best describes classification",
        "options": {
            "A": "Weather, sunny or rainy?",
            "B": "Predicting the price of a used car based on values such as age, odometer etc",
            "C": "Forecasting sales of a product",
            "D": "Predicting energy consumption based off of factors such as weather"
        },
        "correct": "A"
    },
    {
        "question": "What best describes generalisation",
        "options": {
            "A": "Having good performance on unseen data (test data)",
            "B": "The model performs perfectly when tested on the training data",
            "C": "Learning the noise of the data",
            "D": "All are true"
        },
        "correct": "A"
    },
    {
        "question": "What best describes KNN",
        "options": {
            "A": "KNN is a lazy learner",
            "B": "KNN is a density based clustering algorithm",
            "C": "KNN stands for uhhh, something",
            "D": "KNN is very slow to learn"
        },
        "correct": "A"
    },
    {
        "question": "What best describes KNN",
        "options": {
            "A": "KNN usually uses euclidean distance under the hood to clasify",
            "B": "KNN is very similar to decision trees",
            "C": "KNN is considered one of the fastest algorithms for making predictions",
            "D": "P(Y) * P(X|Y) / P(X)\n(Bayes theorem jumpscare)"
        },
        "correct": "A"
    },
    {
        "question": "What best describes KNN",
        "options": {
            "A": "The higher the N value, the more likely the model is to be underfit",
            "B": "There is no point in normalizing data for KNN due to the nature of how it works",
            "C": "KNN is a density based clustering algorithm",
            "D": "KNN is a distance based clustering algorithm"
        },
        "correct": "A"
    },
    {
        "question": "How is accuracy calculated?",
        "options": {
            "A": "(TP + TN) / (TP + TN + FP + FN)",
            "B": "TP / (TP + FN)",
            "C": "TP / (FP + TP)",
            "D": "(2Recall * Precision) / (Recall + Precision)"
        },
        "correct": "A"
    },
    {
        "question": "How is recall calculated?",
        "options": {
            "A": "(TP + TN) / (TP + TN + FP + FN)",
            "B": "TP / (TP + FN)",
            "C": "TP / (FP + TP)",
            "D": "(2Recall * Precision) / (Recall + Precision)"
        },
        "correct": "B"
    },
    {
        "question": "How is precision calculated?",
        "options": {
            "A": "(TP + TN) / (TP + TN + FP + FN)",
            "B": "TP / (TP + FN)",
            "C": "TP / (FP + TP)",
            "D": "(2Recall * Precision) / (Recall + Precision)"
        },
        "correct": "C"
    },
    {
        "question": "How is f1-score calculated?",
        "options": {
            "A": "(TP + TN) / (TP + TN + FP + FN)",
            "B": "TP / (TP + FN)",
            "C": "TP / (FP + TP)",
            "D": "(2Recall * Precision) / (Recall + Precision)"
        },
        "correct": "D"
    },
    {
        "question": "What is gradient descent?",
        "options": {
            "A": "Gradient descent can be used for regression, it takes an iterative aproach (hopefully) improving accuracy with each iteration",
            "B": "Gradient descent is technically a classification algorithm",
            "C": "Gradient descent uses calculus black magic in order to figure out what direction to move to reduce error",
            "D": "all are true"
        },
        "correct": "D"
    },
    {
        "question": "In the context of DT's, what is a pure node?",
        "options": {
            "A": "A node that contains one class",
            "B": "A node that contains an even 50/50 mix of classes",
            "C": "A node with high entropy",
            "D": "A node with high information gain"
        },
        "correct": "A"
    },
    {
        "question": "What is true?",
        "options": {
            "A": "A pure node in a DT has an entropy value of 0",
            "B": "An impure node, when split, resulting in pure nodes will have an information gain of 0",
            "C": "Good attributes to split on usually result in minimal purity",
            "D": "How can we say anything is true? What is the meaning of it all? When presented with the meaninglessness of life how do you cope?"
        },
        "correct": "A"
    },
    {
        "question": "What is information gain?",
        "options": {
            "A": "Information gain represents the reduction of entropy when splitting",
            "B": "Information gain represents the increase of entropy when splitting",
            "C": "Information gain represents the increase in purity when splitting",
            "D": "Information gain represents the decrease in purity when splitting"
        },
        "correct": "A"
    },
    {
        "question": "When does a DT stop splitting?",
        "options": {
            "A": "A node is pure",
            "B": "An impurity threshold is reached",
            "C": "Maximum tree depth is reached",
            "D": "All of these"
        },
        "correct": "D"
    },
    {
        "question": "Which of the following is true",
        "options": {
            "A": "In the context of DT's, nominal features are easy to handle",
            "B": "Decision trees cant be adapated to regression",
            "C": "Underfitting is a common problem with DT's",
            "D": "Boundless and bare. The lone and level sands stretch far away."
        },
        "correct": "A"
    },
    {
        "question": "Why use ensembles?",
        "options": {
            "A": "Individual models often make mistakes, but the 'majority' is less liekly to make mistakes",
            "B": "Several simple classifiers can approximate complex classification boundries",
            "C": "When there is a small quantity data, bagging in conjunction with multiple learners can results in an accurate model",
            "D": "All of these are true"
        },
        "correct": "D"
    },
    {
        "question": "In the context of ensembles, what are some different ways of 'voting'",
        "options": {
            "A": "Majority vote",
            "B": "Weighted majority vote",
            "C": "Meta-learning",
            "D": "All are true"
        },
        "correct": "D"
    },
    {
        "question": "What is bagging?",
        "options": {
            "A": "Taking a sample from a sample with replacement, used often for ensembles",
            "B": "Taking data and randomly splitting so we have data for training and testing",
            "C": "Sampling without replacement",
            "D": ""
        },
        "correct": "A"
    },
    {
        "question": "In the context of search, what is state space",
        "options": {
            "A": "The set of all possible states reachable from the initial state by any sequence of actions, commonly represented in tree / graph form",
            "B": "A moment in time",
            "C": "State space relates to uh",
            "D": "State space is uhhhh, i cant really think of anything"
        },
        "correct": "A"
    },
    {
        "question": "Which best describes the following? Breadth first, iterative",
        "options": {
            "A": "Uninformed search",
            "B": "Informed (Heuristic) search",
            "C": "Hill climbing",
            "D": "Gradient descent"
        },
        "correct": "A"
    },
    {
        "question": "What is beam search?",
        "options": {
            "A": "BFS that limits the breadth or beam width for performance",
            "B": "DFS that has a maximum depth (beam depth)",
            "C": "Similar to a*",
            "D": "Arman?"
        },
        "correct": "A"
    },
    {
        "question": "Which of the following falls under uninformed search",
        "options": {
            "A": "A*",
            "B": "Best first search",
            "C": "Minimax algorithm",
            "D": "None of these"
        },
        "correct": "D"
    },
    {
        "question": "What is hill climbing?",
        "options": {
            "A": "An informed search algorithm that iteratively moves in the direction of increasing (or decreasing) value",
            "B": "An uninformed search algorithm that results in a sub optimal but 'good enough' result, considered to be incredibly fast",
            "C": "A regression algorithm",
            "D": "A best first search algorithm with a random aspect to it"
        },
        "correct": "A"
    },
    {
        "question": "What best describes simulated annealing?",
        "options": {
            "A": "it combines hill climbing with random walk, allowing the algorithm to excape from local minima and converge on a global minimum",
            "B": "It is similar to bootstrapping, it is used for sampling data and results in better model performance due to strong generalization",
            "C": "It relies entirely on deterministic steps, always choosing the best available option at each stage to ensure rapid convergence",
            "D": "Man would rather will nothingness than not will at all"
        },
        "correct": "A"
    },
    {
        "question": "What best aligns with dimensionality reduction?",
        "options": {
            "A": "PCA",
            "B": "Pearsons correlation coef",
            "C": "Hill climbing",
            "D": "Feature construction"
        },
        "correct": "A"
    },
    {
        "question": "What best aligns with data cleaning",
        "options": {
            "A": "Handling of missing data, outliers, and redundancy",
            "B": "Change the scale/distributions of variables",
            "C": "Create compact projections of data",
            "D": "Derive new variables from available data"
        },
        "correct": "A"
    },
    {
        "question": "What best aligns with data transformation",
        "options": {
            "A": "Change the scale/distribution of variables",
            "B": "Handling missing or noisy data",
            "C": "PCA",
            "D": "all are true"
        },
        "correct": "A"
    },
    {
        "question": "What best aligns with categorical data encoding",
        "options": {
            "A": "One hot encoding",
            "B": "Ordinal data simply given integer values, eg. xs = 0, s = 1, m = 2, l = 3...",
            "C": "None are true",
            "D": "All are true"
        },
        "correct": "A"
    },
    {
        "question": "How is 'normalization' different from 'standardization' for data pre-processing",
        "options": {
            "A": "Standardisation has values centered at 0, where normalization has values centered at 0.5",
            "B": "Standardisation has values centered at 0.5, where normalization has values centered at 0",
            "C": "",
            "D": ""
        },
        "correct": "A"
    },
    {
        "question": "What best describes normalization",
        "options": {
            "A": "MinMax scaling",
            "B": "Z-Score, eg. dividing by data mean",
            "C": "",
            "D": ""
        },
        "correct": "A"
    },
    {
        "question": "What best describes standardization",
        "options": {
            "A": "MinMax scaling",
            "B": "Z-Score, eg. dividing by data mean",
            "C": "",
            "D": ""
        },
        "correct": "B"
    },
    {
        "question": "Should normalization be done before or after splitting training and testing data",
        "options": {
            "A": "After",
            "B": "Before",
            "C": "",
            "D": ""
        },
        "correct": "A"
    },
    {
        "question": "what best describes imputation",
        "options": {
            "A": "Handling missing data",
            "B": "PCA",
            "C": "ensemble",
            "D": "IDPS"
        },
        "correct": "A"
    },
    {
        "question": "What best describes K-means",
        "options": {
            "A": "It is centroid based",
            "B": "It is density based",
            "C": "",
            "D": ""
        },
        "correct": "A"
    },
    {
        "question": "What best describes K-means",
        "options": {
            "A": "It is a unsupervised clustering algorithm",
            "B": "It is a supervised classification algorithm",
            "C": "It is a supervised regression algorithm",
            "D": "Is it? It thinks? it is."
        },
        "correct": "A"
    },
    {
        "question": "What best describes K-means",
        "options": {
            "A": "It is an iterative algorithm that repeats until convergence",
            "B": "It is an iterative algorithm that has a known number of iterations until it is done",
            "C": "It is a recursive algorithm",
            "D": "It closely aligns with how Arman works, in that it adopts 'learners' that have to perform an uninformed search."
        },
        "correct": "A"
    },
    {
        "question": "Which of the following clustering algorithms would perform best when presented with data in the shape of a ring contained inside another ring of data",
        "options": {
            "A": "DBSCAN",
            "B": "K-means",
            "C": "MeanShift",
            "D": "Gaussian mixture"
        },
        "correct": "A"
    },
    {
        "question": "What best describes LOF (Local outlier factor)",
        "options": {
            "A": "It is a density based algorithm, anomalies are identified as points with relative low density",
            "B": "It uses decision trees to separate data points, aiming to isolate anomalies which are few and distinct",
            "C": "It generates a DT, where anomaies are identified as being part of short branches",
            "D": "The higher we soar, the smaller we appear to those who cannot fly"
        },
        "correct": "A"
    },
    {
        "question": "what best describes Isolation forest",
        "options": {
            "A": "It uses decision trees to separate data points, aiming to isolate anomalies which are few and distinct",
            "B": "It is a density based algorithm, anomalies are identified as points with relative low density",
            "C": "It generates a KNN model, where anomalies are identified as being part of short branches",
            "D": "DBSCAN"
        },
        "correct": "A"
    },
    {
        "question": "In the context of Isolation Forest, what is most likely to be classified as an anomaly",
        "options": {
            "A": "Leaf nodes in the DT that occur close to the root node of the tree",
            "B": "Leaf nodes in the DT that occur at the bottom of the tree",
            "C": "Long branches in the DT",
            "D": "Anomalies are identified based on the ARMAN score of the node"
        },
        "correct": "A"
    },
    {
        "question": "What are some components of a NN",
        "options": {
            "A": "Node",
            "B": "Weights",
            "C": "Edges",
            "D": "All of these"
        },
        "correct": "D"
    },
    {
        "question": "Given that x is input, w is weights, b is bias and y is the prediction, what is the formula for the activation function of a perceptron",
        "options": {
            "A": "y = f(w * x + b)",
            "B": "y = f(b * w * x)",
            "C": "y = f(b / w * x)",
            "D": "y = f(x)"
        },
        "correct": "A"
    },
    {
        "question": "What is the output of a perceptron?",
        "options": {
            "A": "Binary value",
            "B": "Continuous value",
            "C": "A vector",
            "D": "a uh"
        },
        "correct": "A"
    },
    {
        "question": "What does the bias term do in a perceptron?",
        "options": {
            "A": "It shifts the decision boundry up or down",
            "B": "It adjusts the weights for each learning iteration of the perceptrion",
            "C": "It adjusts the epoch value up or down",
            "D": "It is multiplied by the ARMAN score in order to calculate the learning rate"
        },
        "correct": "A"
    },
    {
        "question": "In the context of a perceptron, what best aligns with how it learns?",
        "options": {
            "A": "if the prediction was incorect, it does something like the following: self.weights += (learningRate * (label - prediction)) * features",
            "B": "if the prediction was incorect, it does something like the following: self.weights += learningRate * prediction * features",
            "C": "if the prediction was incorect, it does something like the following: self.weights *= (learning rate * (label - prediction)) * features",
            "D": "I have no idea"
        },
        "correct": "A"
    },
    {
        "question": "What does the epoch value represent in the context of a perceptron",
        "options": {
            "A": "The number of learning iterations",
            "B": "The number of weights",
            "C": "The learning rate adjustment per learning iteration",
            "D": "A constant multiplier for the ARMAN score assosiated with the"
        },
        "correct": "A"
    },
    {
        "question": "When does a single layer perceptron perform well?",
        "options": {
            "A": "On linearly separable data",
            "B": "On data with a high ARMAN score",
            "C": "On data with polynomial features",
            "D": "a uh"
        },
        "correct": "A"
    },
    {
        "question": "What is the most common method to update weights in a MLP",
        "options": {
            "A": "Gradient descent",
            "B": "ARMAN score analysis",
            "C": "KNN",
            "D": "Hill climbing"
        },
        "correct": "A"
    },
]



def shuffle(items):
    shuffled = []
    list_copy = list(items)
    while len(list_copy) > 0:
        shuffled.append(list_copy.pop(random.randint(0, len(list_copy) - 1)))
    return shuffled

def clear_screen():
    """Clear the console screen based on operating system"""
    os.system('cls' if os.name == 'nt' else 'clear')

def display_question(q_data, q_num, q_count):
    """Display a question and its options"""
    print(f"\nQuestion {q_num} of {q_count}:")
    print(q_data["question"])
    print()
    q_data_shuffled = shuffle(q_data["options"].items())
    i = 0
    for key, value in q_data_shuffled:
        print(f"{['A', 'B', 'C', 'D'][i]}. {value}")
        i += 1
    print()
    return q_data_shuffled

def get_user_answer():
    """Get and validate user input"""
    while True:
        answer = input("Your answer (A/B/C/D): ").strip().upper()
        if answer in ['A', 'B', 'C', 'D']:
            return answer
        else:
            print("Invalid input. Please enter A, B, C, or D.")

def run_quiz():
    """Main function to run the quiz"""
    score = 0
    total_questions = 0
    incorrect_questions = []
    
    try:
        clear_screen()
        print("======================================")
        print("AIML320 PRACTICE QUIZ")
        print("======================================")
        #x = input("Press Enter to begin normal quiz, input 'mini' to start a trimmed version of the quiz\n")
        x = input("Press enter to begin")
        shuffled_questions = []

        if x == "mini":
            shuffled_questions = shuffle(questions_db[120:])
        else:
            shuffled_questions = shuffle(questions_db)
        
        for question in shuffled_questions:
            total_questions += 1
            
            clear_screen()
            answer_order = display_question(question, total_questions, len(shuffled_questions))
            user_answer = get_user_answer()
            
            # Check if answer is correct
            i = 0
            for item in answer_order:
                if item[0] == question["correct"]:
                    correct_answer = ['A', 'B', 'C', 'D'][i]
                i += 1
            is_correct = user_answer == correct_answer
            
            if is_correct:
                print("\n✓ Correct! Well done!")
                score += 1
            else:
                print(f"\n✗ Incorrect. The correct answer is {correct_answer}.")
                explanation = question['options'][question["correct"]]
                print(f"Explanation: {explanation}")
                incorrect_questions.append(f"Q: {question['question']}\nYour Answer: {user_answer}\nCorrect: {correct_answer}\nExplanation: {question['options'][correct_answer]}")
                input("Press enter to continue")
            
            print(f"\nCurrent score: {score}/{total_questions} ({score/total_questions*100:.1f}%)")
            
            time.sleep(2)
            
            
    except KeyboardInterrupt:
        clear_screen()
        print("\nQuiz interrupted.")
    
    finally:
        # Display final score
        if total_questions > 0:
            clear_screen()
            print("\n======================================")
            print("QUIZ COMPLETED")
            print("======================================")
            print(f"Final Score: {score}/{total_questions} ({score/total_questions*100:.1f}%)")
            
            if score/total_questions >= 0.9:
                print("Excellent! You're well prepared for the test!")
            elif score/total_questions >= 0.6:
                print("Good job! With a bit more study, you'll be well prepared.")
            elif score/total_questions >= 0.2:
                print("You might need some more study time to prepare for the test.")
            else:
                print("L.")


if __name__ == "__main__":
    run_quiz()
