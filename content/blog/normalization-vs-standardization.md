---
title: "⚖ Normalisation vs Standardisation: What are the differences?"
date: 2023-02-20
published: false
tags: ['Machine Learning']
thumbnail: 'https://source.unsplash.com/0k9fu-P-110/640x360'
description: 'Normalisation and standardisation are often used interchangeably, but they are not the same. In this blog post, I will discuss the differences between the two.'
---

Data pre-processing is an essential step in any data science project, and it involves various techniques to prepare data for analysis. Two popular methods, data normalisation and standardisation, are often used interchangeably, but they are different. I will discuss the differences between the two techniques in this blog post.

## What is Normalization?

Normalisation is a technique that transforms the data values to a common scale, typically ranging from 0 to 1. Normalisation is useful when the scale of data varies significantly, and it is essential to bring all the data into a uniform range.

The formula for normalization is as follows:

```javascript
x_norm = (x - min(x)) / (max(x) - min(x))
```

Where `x_norm` is the normalised value of `x`, `min(x)` and `max(x)` are the minimum and maximum values of `x` in the dataset, respectively.

For example, suppose we have a dataset containing the heights and weights of people. The height is measured in centimetres, ranging from 150 to 200 cm, and the weight is measured in kilograms, ranging from 50 to 100 kg. We would use the above formula for each feature to normalise this dataset.

## What is Standardization?

Standardisation is a technique that transforms the data values to have a mean of zero and a standard deviation of one. Standardisation is useful when the scale of data is not important, but the distribution shape of the data is essential.

The formula for standardisation is as follows:

```javascript
x_std = (x - mean(x)) / std(x)
```

Where `x_std` is the standardised value of `x`, `mean(x)` and `std(x)` are the mean and standard deviation of `x` in the dataset, respectively.
For example, suppose we have a dataset containing a company's employees' salaries. The salaries range from 30,000 to 100,000 dollars, and the mean wage is 60,000, with a standard deviation of 20,000. We would use the above formula for the salary feature to standardise this dataset.

Normalisation and standardisation transform the data values to a common scale, but the approach differs. The significant differences are:

1. Normalisation scales the data between 0 and 1, while standardisation scales the data to have a mean of zero and a standard deviation of one.
2. Normalisation is useful when the scale of data varies significantly, and it is essential to bring all the data into a uniform range. Standardisation is useful when the scale of data is not important, but the distribution shape of the data is essential.
3. Normalisation preserves the original distribution shape of the data, while standardisation changes the distribution shape of the data to a normal distribution.

## Guidelines for choosing between normalisation and standardisation

Normalisation and standardisation are techniques used to transform data for analysis, but they have different use cases. Here are some guidelines for when to use each method:

### Normalisation:

Normalisation is useful when the scale of data varies significantly, and it is essential to bring all the data into a uniform range. For example, if you are working with a dataset that includes values ranging from 0 to 1000 and another that contains values ranging from 0 to 1, normalisation, it would be useful to bring them into a common range. Normalisation is also useful when comparing variables on equal footing, especially in machine learning algorithms that require input features to have the same scale.

One caveat to remember is that normalisation preserves the original distribution shape of the data. If the data's distribution shape is unimportant, standardisation may be a better choice.

### Standardisation:

Standardisation is useful when the scale of data is not important, but the distribution shape of the data is essential. Standardisation transforms the data to have a mean of zero and a standard deviation of one, which is useful when you want to compare variables with different measurement units. It is also useful when comparing variables on a relative basis, such as in principal component or factor analysis.

One important thing to remember is that standardisation changes the distribution shape of the data to a normal distribution. This can be useful in some cases but can also be problematic if the data has extreme outliers or a non-normal distribution shape. In those cases, normalisation or other techniques may be more appropriate.

## Conclusion

Normalisation and standardisation are essential techniques in data pre-processing. Normalisation transforms the data to a common scale between 0 and 1, while standardisation transforms the data to have a mean of zero and a standard deviation of one. The choice between normalisation and standardisation depends on the data type, the scale of the data, and the data distribution shape. It is essential to choose the right technique to prepare the data for analysis to ensure the accuracy and validity of the results.