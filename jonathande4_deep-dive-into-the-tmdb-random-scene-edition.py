#!/usr/bin/env python
# coding: utf-8



# Random Movie Picture function.
import requests
import bs4
import re

from IPython.display import Image, HTML, display

def get_movie_image(row):
    """
    Get Movie Image
    
    Description
    -----------
    The dataset already provides us with the poster url, but I thought it would be fun to extend this idea and
    get images from the movie gallery itself.
    
    Parameters
    ----------
    row: One row from the TMDB movie dataset. Needs an IMDB movie id to fetch an image correctly.
    
    """
    assert len(row) == 1, "Select one row from the dataset."
    
    title, release_date, revenue, ids = (
        "<center><h3>Title: " + row.title.values[0] + "</h3></center>",
        "<center><h3>Release Date: " + row.release_date.values[0] + "</h3></center>",
        "<center><h3>Revenue: " + '${:,}'.format(row.revenue.values[0]) + "</h3></center>",
        row.imdb_id.values[0]
    )
    res = requests.get("https://www.imdb.com/title/%s/mediaviewer" % (str(ids)))
    display(HTML(title)), display(HTML(revenue)), display(Image(re.findall("https://m.media.amazon.com/\w+/.+?\.jpg", str(res.content))[0]))




# Data
import numpy as np
import pandas as pd

# Plotting
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')




# Load the data.
train, test = pd.read_csv("../input/train.csv"), pd.read_csv("../input/test.csv")




train.head(3)




train.columns.values




get_movie_image(train.sample(1, random_state=24))




get_movie_image(train.sample(1, random_state=11233))




fig, axes = plt.figure(figsize=(12, 6)), plt.axes()

axes.vlines(ymin=0, ymax=2500, x=train.revenue.mean() / 1000000)
axes.hist(train.revenue / 1000000, bins=25)
axes.set(title="Distribution of Revenue")
axes.set(xlabel="Amount (Millions)")
plt.show()




# Scaled mean by millions.
train.revenue.mean() / 1000000




fig, axes = plt.figure(figsize=(12, 6)), plt.axes()

axes.vlines(ymin=0, ymax=2500, x=np.log1p(train.revenue.mean()))
axes.hist(np.log1p(train.revenue), bins=25)
axes.set(title="Distribution of Revenue")
axes.set(xlabel="Amount (Log)")
plt.show()




train.sort_values("revenue", ascending=True).head(5)[["title", "revenue", "budget", "popularity", "release_date", "runtime"]]




train.sort_values("revenue", ascending=False).head(5)[["title", "revenue", "budget", "popularity", "release_date", "runtime"]]




get_movie_image(train.sample(1, random_state=9087))




fig, axes = plt.figure(figsize=(12, 6)), plt.axes()

axes.vlines(ymin=0, ymax=2500, x=train.popularity.mean())
axes.hist(train.popularity, bins=25)
axes.set(title="Movie Popularity")
axes.set(xlabel="Popularity")
plt.show()




fig, axes = plt.figure(figsize=(12, 6)), plt.axes()

axes.vlines(ymin=0, ymax=2500, x=np.log(train.popularity.mean()))
axes.hist(np.log1p(train.popularity), bins=25)
axes.set(title="Movie Popularity")
axes.set(xlabel="Popularity")
plt.show()




train.popularity.describe()




get_movie_image(train.sample(1, random_state=71515))




fig, axes = plt.figure(figsize=(12, 6)), plt.axes()

axes.vlines(ymin=0, ymax=2500, x=train.budget.mean() / 1000000)
axes.hist(train.budget / 1000000, bins=25)
axes.set(title="Movie Budget")
axes.set(xlabel="Budget (Millions)")
plt.show()




train.sort_values("budget", ascending=True).head(5)[["title", "revenue", "budget", "popularity", "release_date", "runtime", "status"]]




train.sort_values("budget", ascending=False).head(5)[["title", "revenue", "budget", "popularity", "release_date", "runtime", "status"]]




fig, axes = plt.figure(figsize=(12, 6)), plt.axes()

movie_profit = (train.revenue - train.budget) / 1000000
avg_profit = (train.revenue.mean() - train.budget.mean()) / 1000000

axes.vlines(ymin=0, ymax=2500, x=avg_profit)
axes.hist(movie_profit[movie_profit > 0], color="blue", bins = 20)
axes.hist(movie_profit[movie_profit <= 0], color = "red", bins = 20)
axes.set(title="Movies Make an Average Profit of ${:,.0f}".format(avg_profit))
axes.set(xlabel="Budget (Millions)")

plt.show()

# Add profit to our training data.
train["profit"] = (train.revenue - train.budget)




print("Movies that made a profit: {}".format(sum((movie_profit >= 0))))

print("Movies that tanked: {}".format(sum((movie_profit < 0))))




get_movie_image(train.sample(1, random_state=1960))




def parse_date(x):
    x = tuple(str(x).split("/"))
    if len(x) == 3:
        month, day, year = tuple(x)
    else:
        return None
    
    # Ths works assuming there aren't any release dates lower than 1920.
    if int(year) > 19:
        year = "19" + str(year)
    else:
        year = "20" + str(year)
    
    return pd.Timestamp(year=int(year), month=int(month), day=int(day))




# Parse the release date.
train["release_date_parsed"] = train.release_date.apply(parse_date)




# Our groupby variable.
by_year = train.loc[train.status == "Released", "release_date_parsed"].apply(lambda x: pd.Period(x, "Y"))

# Add the year to our training data.
train["year"] = train["release_date_parsed"].apply(lambda x: x.year)

fig, axes = plt.figure(figsize=(12, 6)), plt.axes()

# Group by year and plot with pandas.
train     .loc[train.status == "Released", "title"]     .groupby(by_year)     .count()     .plot(kind="bar")


axes.set(title = "Movies Released by Year")
axes.set(xlabel = "")

# Get the default location.
loc, label = plt.xticks()

# Select every 4th label and loc. Add the right ticks.
ind = np.arange(0, len(loc), 4)
loc, label = loc[ind], np.array(list(label))[ind]
plt.xticks(loc, label, rotation=45)

plt.show()




# Result is .51.
# "{:.2f} of our movies were made after 2003.".format(sum(by_year > 2003) / len(train))




# Our groupby variable.
by_month = train.loc[train.status == "Released", "release_date_parsed"].apply(lambda x: pd.Period(x, "M").month)


# Add month to our dataset.
train["month"] = train["release_date_parsed"].apply(lambda x: x.month)

# Four are missing month values, we'll assign them a 9.

fig, axes = plt.figure(figsize=(12, 6)), plt.axes()

# Group by year and plot with pandas.
train     .loc[train.status == "Released", "title"]     .groupby(by_month)     .count()     .plot(kind="bar")


axes.set(title = "Movies Released by Month")
axes.set(xlabel = "")

# Get the default location.
loc, label = plt.xticks()

# Select every 4th label and loc. Add the right ticks.
ind = np.arange(0, len(loc), 1)
loc, label = loc, ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
plt.xticks(loc, label, rotation=45)

plt.show()




get_movie_image(train.sample(1, random_state=198568))




features = ["revenue", "budget", "popularity", "profit", "runtime", "year", "month", "id"]

# Display the table.
display(train[features].corr())

# Corr Plot.
fig = plt.figure(figsize=(10, 5))
plt.imshow(train[features].corr())
plt.title("Correlate Continuous Features")
plt.yticks(range(0, len(features)), features)
plt.xticks(range(0, len(features)), features, rotation=45)
plt.show()




features = ["revenue", "budget", "popularity", "runtime", "year", "month", "id"]

to_scale = ["revenue", "budget", "popularity", "runtime"]

train_scaled = train.copy()
for col in to_scale:
    train_scaled[col] = np.log1p(train_scaled[col])

# Display the table.
display(train_scaled[features].corr())

# Corr Plot.
fig = plt.figure(figsize=(10, 5))
plt.imshow(train_scaled[features].corr())
plt.title("Correlate Continuous Features")
plt.yticks(range(0, len(features)), features)
plt.xticks(range(0, len(features)), features, rotation=45)
plt.show()




get_movie_image(train.sample(1, random_state=1957))




train.select_dtypes("object").head(3)




def parse_content(x, key):
    """
    Parse Content
    
    Provide a string with a 'key': 'value' pair and the output will be a string value.
    """
    return re.findall(pattern=f"'{str(key)}': (.+?')", string=str(x))

def vectorize_list(x, unique_list):
    """
    Vectorize a list of values
    
    Returns a numpy nd-array. Similar to the output pd.get_dummies would give.
    """
    x = np.array([item in row_list for row_list in x for item in unique_list]).reshape(len(x), len(unique_list))
    x = pd.DataFrame(x)
    return unique_list, x

def make_column_names(prefix, x):
    return ["{}_{}".format(prefix, i) for i in range(0, len(x))]




# Get the name if a movie belongs in a collection.
belongs_to_collection = train.belongs_to_collection.apply(parse_content, key="name")

# How many belong to a series?
print(belongs_to_collection.apply(lambda x: x != []).sum())

# Add as an indicator.
train["movie_in_series"] = belongs_to_collection.apply(lambda x: x != [])




get_movie_image(train.sample(1, random_state=9761))




def plot_top_n_count(key, identity, n, plot_title):
    top_count = list(zip(key, identity.sum()))
    top_count.sort(key=lambda x: x[1], reverse=True)

    # Plot the top number of companies.
    top_count = np.array(top_count[:n])

    fig, ax = plt.figure(figsize=(12, 6)), plt.axes()

    # Note the float conversion. Otherwise the plot looked a weird.
    ax.bar(height=list(map(float, top_count[:, 1])), x=top_count[:, 0], align="center")
    plt.title(plot_title)
    plt.xticks(rotation=45)
    plt.show()




# Get the names of production companies.
production_companies = train.production_companies.apply(parse_content, key="name")

# Create an identity matrix.
unique_companies = set([item for item in production_companies for item in item])
company_key, company_identity = vectorize_list(production_companies, unique_companies)

# Plot.
plot_top_n_count(company_key, company_identity, 20, "What Companies Create the Most Movies?")




# Get Countries.
production_countries = train.production_countries.apply(parse_content, key="name")

# Create an identity matrix.
unique_countries = set([item for item in production_countries for item in item])
country_key, country_identity = vectorize_list(production_countries, unique_countries)

# Plot.
plot_top_n_count(country_key, country_identity, 20, "Where are Movies Made?")




cast_name = train.cast.apply(parse_content, key="name")

unique_cast = set([item for item in cast_name for item in item])
cast_key, cast_identity = vectorize_list(cast_name, unique_cast)

plot_top_n_count(cast_key, cast_identity, 20, "What Actors Participate in the Most Movies?")




keywords = train.Keywords.apply(parse_content, key="name")

unique_keyword = set([item for item in keywords for item in item])
keyword_key, keyword_identity = vectorize_list(keywords, unique_keyword)

plot_top_n_count(keyword_key, keyword_identity, 20, "What top Keywords are used to describe a Movie?")




crew_name = train.crew.apply(parse_content, key="name")
unique_crew = set([item for item in crew_name for item in item])
crew_key, crew_identity = vectorize_list(crew_name, unique_crew)
plot_top_n_count(crew_key, crew_identity, 20, "What Crewmember has Worked on the Most Movies?")




genre_name = train.genres.apply(parse_content, key="name")
unique_genre = set([item for item in genre_name for item in item])
genre_key, genre_identity = vectorize_list(genre_name, unique_genre)
plot_top_n_count(genre_key, genre_identity, 20, "What are the Top Movie Genres?")




# Rename Company
company_identity = company_identity.rename({i: "company_{}".format(i) for i in range(0, len(company_key))}, axis=1)

# Rename Country
country_identity = country_identity.rename({i: "country_{}".format(i) for i in range(0, len(country_key))}, axis=1)

# Rename Crew
crew_identity = crew_identity.rename({i: "crew_{}".format(i) for i in range(0, len(crew_key))}, axis=1)

# Rename Actor
cast_identity = cast_identity.rename({i: "cast_{}".format(i) for i in range(0, len(cast_key))}, axis=1)

# Rename Genre
genre_identity = genre_identity.rename({i: "genre_{}".format(i) for i in range(0, len(genre_key))}, axis=1)

# Create a categorical for train.
train_categorical = pd.concat([company_identity, country_identity, crew_identity, cast_identity, genre_identity], axis=1)




# Production Companies
production_companies = test.production_companies.apply(parse_content, key="name")
company_key, company_identity = vectorize_list(production_companies, unique_companies)
company_identity = company_identity.rename({i: "company_{}".format(i) for i in range(0, len(company_key))}, axis=1)

# Production Countries
production_countries = test.production_countries.apply(parse_content, key="name")
country_key, country_identity = vectorize_list(production_countries, unique_countries)
country_identity = country_identity.rename({i: "country_{}".format(i) for i in range(0, len(country_key))}, axis=1)

# Cast
cast_name = test.cast.apply(parse_content, key="name")
cast_key, cast_identity = vectorize_list(cast_name, unique_cast)
cast_identity = cast_identity.rename({i: "cast_{}".format(i) for i in range(0, len(cast_key))}, axis=1)

# Keywords
keywords = test.Keywords.apply(parse_content, key="name")
keyword_key, keyword_identity = vectorize_list(keywords, unique_keyword)
keyword_identity = keyword_identity.rename({i: "cast_{}".format(i) for i in range(0, len(keyword_key))}, axis=1)

# Crew
crew_name = test.crew.apply(parse_content, key="name")
crew_key, crew_identity = vectorize_list(crew_name, unique_crew)
crew_identity = crew_identity.rename({i: "crew_{}".format(i) for i in range(0, len(crew_key))}, axis=1)

# Genre
genre_name = test.genres.apply(parse_content, key="name")
genre_key, genre_identity = vectorize_list(genre_name, unique_genre)
genre_identity = genre_identity.rename({i: "genre_{}".format(i) for i in range(0, len(genre_key))}, axis=1)

test_categorical = pd.concat([company_identity, country_identity, crew_identity, cast_identity, genre_identity], axis=1)




train = pd.concat([train, train_categorical], axis=1)
test = pd.concat([test, train_categorical], axis=1)




fig, ax = plt.figure(figsize=(10, 6)), plt.axes()

# Data
train["original_language"]     .groupby(train["original_language"])     .count()     .sort_values(ascending=False)     .transform(lambda x: x / len(train))     .plot(kind="bar", color="salmon")

plt.title("Movies by Original Language")
plt.show()




from sklearn import linear_model, metrics, model_selection, impute




get_movie_image(train.sample(1, random_state= 1256))




# Parse the release date.
test["release_date_parsed"] = test.release_date.apply(parse_date)

# Add Year
test["year"] = test.release_date_parsed.apply(lambda x: x.year)

# Add Month
test["month"] = test.release_date_parsed.apply(lambda x: x.month)




features = ["budget", "popularity", "runtime", "year", "month", "id"]

# Simple impute uses the mean
imp = impute.SimpleImputer()

train[features] = imp.fit_transform(train[features])




test[features].isna().sum()




test[features] = imp.fit_transform(test[features])




# Small featureset for a linear model.
target, features = "revenue", ["budget", "popularity", "runtime", "year"]

# Do a train/test split.
X_train, X_test, y_train, y_test = model_selection.train_test_split(train[features], train[target], test_size=0.3, random_state=42)

# Init the estimator.
lm = linear_model.LinearRegression()

# KFold Cross validation.
cv = model_selection.KFold(n_splits=5)

# Fit a normal model.
lm.fit(X_train, y_train)

y_pred = lm.predict(X_train)

# Perform cross validation.
print("Cross Validation: ", model_selection.cross_val_score(estimator=lm, X=X_train, y=y_train, cv=cv, scoring="neg_mean_squared_error"))

print("Training Sample Performance: ", metrics.mean_squared_error(y_train, y_pred))

plt.figure(figsize=(10, 5))
plt.scatter(y_pred, y_train)
plt.show()




import sklearn




# Small featureset for a linear model.
target, features = "revenue", ["budget", "popularity", "runtime"]

# Do a train/test split.
X_train, X_test, y_train, y_test = model_selection.train_test_split(train[features], train[target], test_size=0.3, random_state=42)

# Feature scaling.
y_train = np.log1p(y_train)

X_train[features] = X_train[features].transform(np.log1p)

# Init the estimator.
lm = linear_model.LinearRegression()

# KFold Cross validation.
cv = model_selection.KFold(n_splits=5)

# Fit a normal model.
lm.fit(X_train, y_train)

y_pred = lm.predict(X_train)

# Use an exponential to back-transform our predictions, targets.
y_train, y_pred = np.expm1(y_train), np.expm1(y_pred)


plt.figure(figsize=(10, 5))
plt.scatter(y_pred, y_train)
plt.xlim(0, 1e8)
plt.ylim(0, 1e8)
plt.show()




get_movie_image(train.sample(1, random_state=8615))




from sklearn import ensemble




company_cols = list(company_identity.columns.values)
cast_cols = list(cast_identity.columns.values)

target, features = "revenue", ["budget", "popularity", "runtime", "year"] + company_cols + cast_cols

# Do a train/test split.
X_train, X_test, y_train, y_test = model_selection.train_test_split(train[features], train[target], test_size=0.3, random_state=42)

# Init the estimator.
lm = ensemble.RandomForestRegressor()

# KFold Cross validation.
cv = model_selection.KFold(n_splits=5)

# Fit a normal model.
lm.fit(X_train, y_train)

y_pred = lm.predict(X_train)

# Perform cross validation.
print("Cross Validation: ", model_selection.cross_val_score(estimator=lm, X=X_train, y=y_train, cv=cv, scoring="neg_mean_squared_error"))

print("Training Sample Performance: ", metrics.mean_squared_error(y_train, y_pred))

plt.figure(figsize=(10, 5))
plt.scatter(y_pred, y_train)
plt.show()




get_movie_image(train.sample(1, random_state=865241))

