# Annotated R script with recipes, notes, and examples for MATH 3339
#
# © Copyright 2021 Michael Moorman

##############################
# Useful commands in RStudio:#
##############################

# This command clears your workspace environment variables
# Useful when switching from one R script to another
rm(list=ls())

#######################################
# Creating tables and exporting plots #
#######################################
#
#### Creating LaTeX tables for insertion in a document
#
# Use xtable, create a dataframe object and give it the column names you desire.
# Example from HQ1, the schizophrenia dataset
library(xtable)
years <- c(1956, 1960, 1965)
schizo_births <- c(483, 192, 695)
total_births <- c(59088, 13748, 83536)

df <- data.frame("Year" = years, 
                 "Total births in region" = total_births, 
                 "Total incidence of schizophrenia" = schizo_births)
# Export LaTeX table for figure 2 as fig2
fig2 <- xtable(df)
digits(fig2) <- xdigits(fig2)
#
##### Create PNG images of plots for insertion in documents
#
# If you don't specify filename=, it will be saved in the R Workspace folder
# Width and height should be adjusted to suit your needs
#
png(filename = "/path/where/you/want/picture/to/be/saved", width = 480, height = 240)
# plotting commands goes here / plot() / boxplot() / mosaicplot() etc.
dev.off() # this closes the handle to filename and saves the file, so you can plot something else.
#
########################
#                      #
# Exploratory analysis #
#                      #
########################
#
#### Sample statistics functions:
#
mean(data) # sample mean
median(data) # median
mode(data) # mode
quantile(data, prob = (0.25)) # 1st quartile, for instance. Use 0.75 for 3rd
var(data) # sample variance
sd(data) # sample standard deviation
summary(data) # Get all the above quickly and easily
#
#### Miscellaneous other functions:
#
# How to use R to calculate binomial coefficients for combinations:
# example: 10 choose 5
x <- choose(10, 5)
#
############################################################################################################
#
# Which plot do you want?
#
# I have: 
#         1 continuous variable?
#              Use boxplot (preferred) or histogram
#              Use to assess distribution of values (is it normal, etc.)
#              We can do one-sample testing on means using the t-test or the T statistic as needed
#              Example from class: the John Wayne study from Lecture,
#         1 continous variable with possible time association?
#               Scatterplot against time variable to determine time association
#               Can then use linear model to subtract time dependence and examine residuals using boxplots
#               If the distribution of the residuals is normal, you can proceed with stationary analysis
#               Example from class: Problem 1 from Exam 1
#         1 categorical variable?
#               Use barchart of frequencies, or just a table.
#               Can use the X^2 goodness-of-fit test to see how it fits a given binomial model
#               Example from class: Whitlock Chapter 8, Example 1 about baby births by day of the week
#         2 categorical variables? 
#               Use mosaic plot to visualize the contingency table
#               Use to assess association between variables (pair with X^2 contingency test, 
#               Fisher exact test if it is 2x2)
#               Example from class: Remdesevir, USS Theodore Roosevelt datasets from HQ7
#         1 continous variable and 1 categorical? 
#               Use side-by-side boxplots, or strip chart
#               Use to assess association between variables
#         2 continous variables?
#               Scatterplot (for association), side-by-side boxplots (comparison of means)
#               Use linear model will test for association in scatterplot.
#               If the two variables are normal, we can test on the means using appropriate two-sample t-test
#               Examples from class: HQ11 (association of two CRVs), Stockholm question from HQ5 (unpaired data), 
#               Problem 1 from HQ5 (paired data)
#
#
####################################
#                                  #
# How to make and interpret plots: #
#                                  #
####################################
#
## Scatterplots:
#
# plot(x, y) does scatterplots pairing points from x and points from y
# When x is a time variable, this shows a trend in y over time.
plot(df_x, df_y)
# lm(y ~ x) creates a linear model between RVs y and x, with R^2 and other goodness-of-fit measures
l <- lm(df_y ~ df_x)
# abline(l) will plot a linear model on top of an existing scatterplot
abline(l)
# Examining residuals in linear model, and checking LM parameters:
# Residuals should be normally distributed with mean 0, check normality
boxplot(l$residuals) # gives you the distribution of residuals
qqnorm(l$residuals) # the usual tests on normality
shapiro.test(l$residuals)
# Get the R^2, p-value, and other goodness of fit parameters from LM
summary(l)
#
## Interpreting scatterplots:
#
# If there is no obvious trend in data points over the independent variable, consistent with no association
# We say there is positive association if there is a good linear model showing a strong positive slope
# Likewise, negative association if the model shows strong negative slope.
#
## Boxplots:
#
# Creating a boxplot and analyzing the distribution in the sample
systolic <- c(88, 88, 92, 96, 96, 100, 102, 102, 104, 104, 105, 105, 105, 107, 107, 
              108, 110, 110, 110, 111, 111, 112, 113, 114, 114, 114, 115, 115, 116, 
              116, 117, 117, 117, 117, 117, 117, 119, 119, 120, 121, 121, 121, 121, 
              121, 121, 122, 122, 123, 123, 123, 123, 123, 124, 124, 124, 125, 125, 
              125, 126, 126, 126, 126, 126, 127, 127, 128, 128, 128, 128, 129, 129, 
              130, 131, 131, 131, 131, 131, 131, 133, 133, 133, 134, 135, 136, 136, 
              136, 138, 138, 139, 141, 142, 142, 142, 143, 144, 146, 146, 147, 155, 
              156)
boxplot(systolic)
# List quantiles of dataset
quantile(systolic)
# IQR calculation
quantile(systolic,probs=(0.75)) - quantile(systolic, probs=(0.25))
# Side-by-side boxplots:
# Plot side-by-side boxplots of samples x and y:
boxplot(x, y)
#
## Interpreting boxplots:
#
# Checking the whiskers, median, and IQR for normality
# Is it obviously skewed?
# In side-by-side boxplots, how much do the boxplots overlap? 
# This can suggests an association of a categorical random variable
#
## Mosaic plots:
#
# Dataset from problem 12 on pg 55 of Whitlock
# We need to first put our values in a matrix,
# and then annotate rownames and columnnames to appear on the plot
data <- matrix(c(47, 43, 128, 57, 90, 30),
               ncol = 2,
               byrow = TRUE)
rownames(data) <- c('Inadequate',
                    'Adequate',
                    'Comfortable')
colnames(data) <- c('No convictions',
                    'Convicted')
caStudy <- as.table(data)
mosaicplot(x = caStudy)
#
# Interpreting mosaic plots:
#
# Association of variables is shown by a trend in the size 
# of the mosaic blocks accross the independent variable
# Then apply X^2 contingency test or Fisher Exact Test as necessary
#
## Bar plots:
#
# Dataset from problem 22 of pg 59.

convictions <- c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
frequency <- c(265, 49, 21, 19, 10, 10, 2, 2, 4, 2, 1, 4, 3, 1, 2)

df <- data.frame(convictions = convictions, frequency = frequency)
barplot(height = frequency, names.arg = convictions)
#
## Histograms:
#
# breaks= gives the number of bins to use
hist(x = systolic, breaks = 30)
#

##########################
#                        #
# Binomial Distributions #
#                        #
##########################
#
# If X is a Bernoulli RV with probability of success p, the variable Y
# given as the total number of successes in N samples X_i of X, 
# which are independent and identical,has binom(N,p) distribution.
#
# How to use R's tools for binomial distributions:
# The CDF of a binomial distributed RV: pbinom
# This represents the sum of the probabilities of all outcomes <= the input
# pbinom(input, number of samples, proportion)
pbinom(6, 25, 0.25)
# The density function: dbinom
# This represents the probability of a single specific outcome == the input
dbinom(6, 25, 0.25)
# binom.test - hypothesis testing and estimation of proportion on binomial RVs
# p= give the proportion of null hypothesis, p-value returned
# confidence interval gives range of 95% confidence for p's actual value
# alternative= lets you specify the alternative hypothesis range
# (two-sided means p different than null hypothesis, greater and lesser for p greater or lesser than in the null hypothesis)
binom.test(6, 25, p = 0.25, alternative="two.sided")
#
#########################
#                       #
# Uniform Distributions #
#                       #
#########################
#
# The class hasn't used this much, but it is used in Monte Carlo analysis
# Sampling from uniform distributions is quite common.
#
# PDF: dunif
dunif(3, 2, 10)
# CDF: punif
punif(3, 2, 10)
# Quantile function: qunif
qunif(0.5, 2, 10)
# Sample from a uniform distribution: runif
# This example returns a vector of 100 values sampled uniformly from (0,1)
runif(100, 0, 1)
#
#########################
#                       #
# Normal distributions  #
#                       #
#########################
#
## Theory: The Central Limit Theorem and Local Limit Theorems
#
# Normal distributions are the most important kind of continuous distribution
#
# By the CLT, the population mean mu of *any* random X variable satisfies:
#
#   (Xbar - mu) / sqrt(s/n) has approximately N(0,1) distribution
#      where Xbar and s are sample mean and std. dev., n is the sample size
#
# The left side is the T statistic and is an unbiased estimator for population means.
# This approximation requires "large enough" n and "not too much" skewness in X.
#
# By the LLT, when X is normal, we have an *exact* relationship of the T statistic 
# with the Student's t-distribution with n-1 degrees of freedom:
#
#   (Xbar - mu) / sqrt(s/n) has exactly t_{n-1} distribution
#
# And this allows us to use the t-test on means, provided we show X is normal and
# that all samples are independent.
#
# Normal distribution density function
# This gives the value of f_X(x), which is proportional to e^(-x^2).
# This is *not* the probability of X == x, which is 0 for CRVs, unlike DRVs.
dnorm(0)
# CDF:
# This is the same as the CDF for binomial DRVs, but with an integral instead of a sum.
pnorm(2, mean = 0, sd = 1)
# Quantile function - the inverse of the CDF
# We can use this to find values x of X for which P(x < X) is the given input.
# qnorm() returns the quantiles of the standard normal distribution, when given without options
# I think this is what is also called the Z score in the videos.
#
# How to get Z_{alpha/2} according to videos, for alpha = 0.05 (95% confidence)
#
qnorm(0.975)
#
# Sampling from normal distributions: rnorm
# This example samples 100 points from Z(0,1)
rnorm(100, 0, 1)
#
#####################################
#                                   #
# Assessing a dataset for normality #
#                                   #
#####################################
#
# First: view data in the boxplot, and get the mean, median, and quartiles.
# This is the *most important* way to evaluate normality in this class.
summary(data)
boxplot(data)
#        Does it have:
#             > a median that is near the center of the interquartile range?
#             > a median that is near the center of the whole range?
#             > whiskers of roughly equal width?
#             > median is "close" to the sample mean?
#        All these suggest a distribution that is unimodal without much skew (i.e. close to normal).
#        If it has most of these and there aren't a large number of samples, it is consistent with normality.
#        Small sample sets can deviate somewhat because they are small
#        If there are lots of samples and it doesn't have all three, suspect that it is not normal.
#        The more sample points, the closer it should be to normal.
# Second: normal plot (also called a QQ normal or quantile-to-quantile normal plot)
# For this, we want the points to be close to the line y=x, corresponding to matching quantiles
# But we allow for error at the tails; what we care about is the middle.
qqnorm(data)
# Third: Shapiro-Wilk test for normality.
# Remember: boxplot justification first, then QQ and Shapiro second.
# A p-value of at least 0.2 suggests normality, less than that and it is dicey
shapiro.test(data)
#
#####################################################
#                                                   #
# Student's t distribution and the t-test on means: #
#                                                   #  
#####################################################
#
# The t distribution is used for estimating population means 
# using the point estimates of mean and standard deviation.
# It requires the variable being normal, so we must justify normality first.
# Also must justify assuming independence of the samples, of course.
#
##############################################
# One-sample mean testing using t.test:      #
# Hypothesis testing and estimation of means #
##############################################
#
# How well does the input dataset support a population mean of mu (hypothesis testing)?
# Returns p-value for null hypothesis (true mean is equivalent to mu)
# Also provides a 95% CI for the population mean (estimation of mean)
# The first input is your dataset that you are examining.
# mu= the mean of the null hypothesis
# alternative= whether the test is one-sided or two-sided, just like binom.test
# We only use two-sided tests when both alternatives are being considered in our alternative hypothesis
# Sometimes this is not true. 
# For instance, if our null hypothesis for the mean of X is mu = 0, and X must be non-negative, 
# there is no point in considering mu < 0; use a right-sided test.
t.test(c(1,2,3,4), mu = 0, alternative="greater")
#
##############################################################################
# Paired difference of means testing using t.test: when X and Y are dependent#
##############################################################################
#
# Paired datasets often occur when we are measuring two variables on same subjects
# Examples: before and after of a given measure, longitudinal studies
# How well does the input datasets support the difference of means being 0?
# Returns p-value for null hypothesis (means are equal)
# The first input is the "after", Y
# The second input is the "before", X
# The test looks at whether the true mean of Y-X is less than, greater than, or not equal to mu
# Depending on alternative=
t.test(antibody_after, antibody_before, mu = 0, paired = TRUE, alternative = "less")
#
############################################################
# Unpaired two-sample testing: when X and Y are independent#
############################################################
#
# What if data is not paired? We can use the t-test with paired = FALSE.
# This happens when, for example, there are separate control and test groups
# We have to justify the independence of the control and test groups
# Consider variances of X and Y...are they equal or not? Do an F test on sample variances.
#
##########################
# The F Test on variances#
##########################
#
# Returns p-value for X and Y having the same population variances
# The F statistic is s_X^2 / s_Y^2
# Very unstable, not good results for small samples
#
var.test(antibody_after, antibody_before)
# If this gives us a very good p-value, we can try the t.test with var.equal = T or F
# The var.equal = F gives us the Welch version of the test, where we don't need equal variances.
# We compare the p-values to see how they differ
t.test(control_mouse_T, antibody_mouse_T, var.equal = T, alternative = "less")
t.test(control_mouse_T, antibody_mouse_T, var.equal = F, alternative = "less")
#
####################################################
#                                                  #
# What do we do if our dataset is not normal?      #
# Using the Central Limit Theorem and T statistic. #
#                                                  #
####################################################
#
#
# By the CLT, if X has any distribution, the T statistic (Xbar - mu)/(s / sqrt(n)) is approximately standard normal
# Here, s and Xbar are the sample standard deviation and mean, and n is the number of samples.
# This approximation improves as n increases.
#
# This allows us to construct an approximate confidence interval based on the values of the quantile function:
# mu is between [Xbar - Z_alpha/2 * s / sqrt(n), Xbar + Z_alpha/2 * s / sqrt(n) ]
# Z_alpha/2 is given by qnorm(0.975) if we want alpha = 0.05, for a 95% CI.
# This requires that X not be significantly skewed and n > 30, or X can be skewed if n > 50
# Otherwise the approximation is too poor.
xbar <- mean(data)
s <- sd(data)
n <- length(data)
z_alpha <- qnorm(0.975) # z_alpha is the value for which pnorm(z_alpha) = 0.975
ci_lower <- xbar - z_alpha*s/sqrt(n)
ci_upper <- xbar + z_alpha*s/sqrt(n)
#
########################################
#                                      #
# The chi-squared goodness-of-fit test #
#                                      #
########################################
#
# The chi-squared (X^2) statistic is useful for determining goodness-of-fit for categorical random variables
# This test is derived from the Local Limit Theorem (Video Lectures 22-22aux)
# Can use it to estimate proportion by analogy of estimation of means using the T statistic.
#
# A quintessential example would be: determining if a die is fair or not, given trials
# Our goal is to compare the observed probability density, and compare that to a null hypothesis one
# In the die example, the observed probability density is calculated by looking at the relative frequency
# of die rolls, and the null distribution is a uniform distribution (each die roll equally likely)
#
# Caveat: for chi-squared test, we must have the expected values for each outcome greater than 5, and n > 30.
#
# Given from Chapter 8 Example 1: are baby birthdays observed in 1990 matching 
# with a hypothetical model that all days of the week are equally likely?
# We find a hypothetical PDF for our categorical RV, with observed values
# How well does the data fit the hypothetical PDF?
# This are our observed values
days_obs <- c(33, 41, 63, 63, 47, 56, 47)
# Our hypothesized PDF for this problem: there are 53 Fridays in 1990, 
# so it is above the others.
# p_x[i] gives this PDF's value at a given outcome i
p_x <- c(52, 52, 52, 52, 52, 53, 52) / 365.0
# This is our n (must be over 30 for good predictive power)
n <- sum(days_obs)
# We have expected values for each outcome given the n trials, each should be above 5.
E_x <- n * p_x
# Now we run the actual test, yielding a p-value with the usual 0.05 significance level.
chisq.test(days_obs, p = p_x)
#
## The X^2 distributions:
#
# The X^2 distribution CDF is given by pchisq(q, df)
# Arguments q -> value, df = degrees of freedom (number of possible outcomes minus 1)
pchisq(15.24, 6)
# X^2 must be positive, so we can get a p-value by checking 1 - chisq(x,df)
#
###############################################
#                                             #
# Testing for independence of categorical RVs #
# using mosaic plots,                         #
# and the chi-squared contingency test        #
#                                             #
###############################################
#
# Indepence as seen in mosaic plots: how significant is the trend?
# Explored at https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test#Testing_for_statistical_independence
#
# We generate a mosaic plot of our problem and look at what it suggests.
# It generates a visualization of the  contingency table for us: 
# given a categorical X and Y, is there an association?

# For each value x_i that X achieves, and for each y_i that Y achieves, 
# there is a frequency given P[x_i,y_i]. Thus if X has N different outcomes, and
# Y has M different outcomes, the contingency table has MN entries.
# (Example from Lancet article in lectures 2021-11-15 and 2021-11-17)
fullvax_index <- matrix(t(c(12, 1, 4, 31, 8, 13)), byrow = F, ncol=2)
colnames(fullvax_index) <- c('PCR+', 'PCR-')
rownames(fullvax_index) <- c('Fully vaxxed contact',
                             'Partially-vaxxed contact',
                             'Unvaccinated contact')
mosaicplot(fullvax_index)
# We can use the chi-squared test for this to test against a null hypothesis, that
# each outcome P[x_i,y_i] is equally likely (there is no association between X and Y)
# We reject this hypothesis if our chi-squared test returns a p-value less than 0.05
# We need no parameters, the function assumes a contingency table when passed a matrix
chisq.test(fullvax_index)
# The caveats for the chi-squared test here are similar as before for testing against
# a given proportion. Each of the different events P(x_i and y_i) must have expected values
# greater than 5 for the given test, and the total number of samples must be greater than 30
# For N trials, this expected value E(x_i and y_j) = (rel. freq. of x_i) * (rel. freq. of y_i) * N
# R will warn us if chisq.test() is passed parameters that do not satisfy this.
#
# We can merge categories if necessary to provide this.
# But it must be sensible.
# Example: combine the small partially-vaccinated cohort in with unvaccinated
#          this creates a new "Not fully vaccinated" category that has sufficient samples
fullvax_merged <- matrix(t(c(12,31,5,21)), byrow = T, ncol = 2)
colnames(fullvax_merged) <- c('PCR+', 'PCR-')
rownames(fullvax_merged) <- c('Fully vaxxed contact',
                              'Not full vaxxed contact')
mosaicplot(x = fullvax_merged)
chisq.test(fullvax_merged)
#
#################################################
#                                               #
# Odds, odds ratios, and The Fisher Exact Test  #
#                                               #
#################################################
#
# In the easiest case of X and Y both Bernoulli RVs, the contingency table is 2x2
# The Fisher Exact Test is available to provide an easy p-value for us, it is exact
# and places no requirements on our samples, just requiring a 2x2 contingency table.
# It is ideal to go there first for small datasets.
#
# The FET's p-value tests whether the true odds ratios of X and Y are equal.
# The odds of X is given p_x/(1-p_x), and the odds of Y are p_y/(1-p_y)
# The odds ratio is given by the ratio of the odds of X and Y
# It is calculated using the point estimates of p_x and p_y 
# (the relative frequency of success observed)
# It is thus a posive real number

# Example: the above merged dataset is now 2x2, and we can use Fisher's test:
fisher.test(fullvax_merged)

###############################################################################
#                                                                             #
# Hypothesis testing for difference of proportions using the CLT Approximation#
#                                                                             #
###############################################################################
#
# Example available https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/binotest.htm
#
# When you have two Bernoulli variables X and Y, and we want to perform one-sided
# hypothesis tests on the proportions (is p_X > p_Y?)
# We can use the Z statistic mentioned in class:
#
# Z = (p_X - p_Y) / sqrt((p_0 * (1-p_0))/(n_x + n_y)) which is approximately N(0,1) by CLT
#
# p_0 is the pooled proportion, equivalent to (# of successes of X + # of successes of Y)/(n_x+n_y)
#
# Z here is analagous to the T statistic for estimating means, the denominator is a discrete analogue of pooled variance
#
# Because this is standard normal, we can use the value of Z to find a p-value.
# This p-value is 1 - pnorm(Z), which gives us the probability of Z
# sampled from a standard normal being at least as big as it is
#
# This is considered a waste of time for two-sided tests; just use the chi-squared contingency test
# Independence testing is equivalent to testing whether proportions are equal in this case.
