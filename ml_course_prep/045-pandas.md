
# Pandas

## tutorial

- [Data science best practices with pandas (video tutorial)](https://www.dataschool.io/data-science-best-practices-with-pandas/)
  - [jupyter notebook file with the solution for the tutorial](https://github.com/justmarkham/pycon-2019-tutorial/blob/master/tutorial.ipynb)

tutorial as questions:

1. Introduction to the TED Talks dataset
   1. load `ted.csv` as a pandas data frame
   2. print first 10 rows of talks
   3. what is number of rows?
   4. what is number of columns?
   5. what is the type for each column?
   6. count the number of missing values in each column
2. Which talks provoke the most online discussion?
   1. show top 5 talks with most comments
   2. create a new column representing comments per view
   3. show top 5 talks with most comments per view
   4. create a new column representing views per comment
   5. show top 5 talks with most views per comment
3. Visualize the distribution of comments
   1. plot a histogram of comments
   2. plot a histogram of comments of talks that have `1000` comments or less
   3. how many talks have `1000` comments or more.
4. Plot the number of talks that took place each year
   1. sample some values of `event` column to see how they look like
   2. sample some values of `film_date`
   3. create a new column `film_datetime` which is `datetime64` format. (The values are from `film_date` and they suppose to be relevant to their `event` values.)
   4. sample from `film_datetime` and show year values from the column
   5. plot number of talks with respect to year values.
5. What were the "best" events in TED history to attend?
   1. list `event` values and their counts
   2. group by `event` column and for each `event` show the count, mean of views, and sum.
6. Unpack the rating data
   1. Show first 10 values of `ratings` column
   2. Unpack the first value of `ratings` column using `ast` module's `literal_eval()` function
   3. define `str_to_list` function and apply it to `ratings` column
   4. create a new column `ratings_list` which contains `list` objects for `ratings` column values
   5. check that the data type of `ratings_list` column is `object`
7. Count the total number of ratings received by each talk
   1. define `get_num_ratings` function to apply to `ratings_list`
   2. apply it and create a new column `num_ratings`
8. Which occupations deliver the funniest TED talks on average?
   1. count the number of funny ratings
      1. define `get_funny_ratings` to get the number of funny ratings from `ratings_list
      2. apply it and create a new column `funny_ratings`
   2. calculate the percentage of ratings that are funny
      1. create a new column `funny_rate` which is `funny_ratings / num_ratings`
      2. sort with `funny_rate` and show top 20 `speaker_occupation` values with respect to `funny_rate`
   3. Analyze the funny rate by occupation
      1. show mean `funny_rate` for each `speaker_occupation`
   4. Focus on occupations that are well-represented in the data
      1. show counts of talks for each `speaker_occupation`
      2. list occupation where the counts are greater than or equal to `5`
   5. Re-analyze the funny rate by occupation (for top occupations only)
      1. show mean `funny_rate` for each `speaker_occupation` where the counts are greater than or equal to `5`

## Resources

- https://www.w3schools.com/python/pandas/
- [Data science best practices with pandas (video tutorial)](https://www.dataschool.io/data-science-best-practices-with-pandas/)
  - [jupyter notebook file with the solution for the tutorial](https://github.com/justmarkham/pycon-2019-tutorial/blob/master/tutorial.ipynb)
