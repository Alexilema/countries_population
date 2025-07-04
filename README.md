# Preparations for a Python Coding Challenge

## Introduction

With the technical interview we want to learn more about your
coding skills in Python.
We have created a small challenge that is similar to something you might
encounter in our team. This is *not* a difficult puzzle you might find in literature about coding interviews. Please have a look at the story below and prepare a solution for the interview. 
We will discuss your solution during the coding challenge and extend the task. 

## Story

The imaginary situation in the coding challenge will be as follows.
A scientist transferred parts of a historically grown code to you. This code contains functions
for reading and normalizing data based on the libraries Pandas and NumPy, see the attached file `script.py`.
Your task is to understand and if necessary debug, and extend some functionality.
Further, in our story the ownership of this code will be transferred to you, i.e., you
will have to maintain this code in the future. So you are allowed to change the Code
functions if you think it is a good idea.

Use, fix if necessary, and extend the Code to read and normalize and extend the data. 
More precisely:

- Read the data with `read_data`. Each column is a feature. Each row is a
  data point.
- Normalize the data with `normalize_mean_std`.
- Remove all boring features. A feature is boring if its data distribution is not
  Gaussian and its name is of the form `boring{i:03d}` for some integer `i`. 
- As an extension to the existing code the researchers would like to join data from a different data source to their data. Please implement a function to request the number of inhabitants of a country using the opean API https://restcountries.com/#api-endpoints-v3-all
- Make sure your code handle API downtimes gracefully
- Join the population to the original table via the common name.
- Is population a "boring" feature?
- Prepare a script to update the data when a new csv is coming in.

 
## Technical Setup
 
To prepare the solution you need to have Python 3.6+ and the libraries Numpy, and Pandas installed.
For the live session we would prefer it if you could work in your own environment on your machine and we can use screen sharing. To this end, you need to install the library Matplotlib in addition to Numpy and Pandas.

# Final Remark

In case of questions or issues, please do not hesitate to contact us before the interview.
