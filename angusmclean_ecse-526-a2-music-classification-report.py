#!/usr/bin/env python
# coding: utf-8



from musicclassificationlib_py import *
ROOT_DIR = '/kaggle/input/music-classification/kaggle/'

dfLabelsAll = pd.read_csv(ROOT_DIR + 'labels.csv')
dfLabels = dfLabelsAll

DROP_COLS = dfLabels.columns
TARGET_COL = 'category'




def graphSong(songRow):
    display('Graphing Song : ', songRow)
    dfSong = loadSong(songRow.id)
    display(dfSong[:200].rolling(5).mean().plot.line(figsize=(16,8), title="Song Wave Form (by Channel)"))
    display(dfSong.plot.density(xlim=(-100,100), figsize=(12, 8), title="Distribution of values for each channel"))

graphSong(dfLabels.iloc[0])




## Load Song Data
dfSongVects = getSongMeans(dfLabels.id).set_index(dfLabels.id)
dfAll = pd.merge(dfSongVects, dfLabels, left_index=True, right_on='id', how='left')
dfAll




plot_cols = dfAll.columns[:4].tolist() + [TARGET_COL]
sns.pairplot(dfAll[plot_cols], hue=TARGET_COL, plot_kws={"s": 12})




get_ipython().run_cell_magic('time', '', 'numNeighboursResults = []\n\n## GridSearch k in kNN\nfor n in list(range(1, 20)):\n    ## Run CrossValidation Test\n    knnResults = testClassifier(dfAll, KnnClassifier(n=n))\n    \n    ## Calculate average score and append to results\n    avg_f1_score = np.mean([r[\'f1_score\'] for r in knnResults])\n    avg_accuracy_score = np.mean([r[\'accuracy_score\'] for r in knnResults])\n    numNeighboursResults.append([avg_accuracy_score, avg_f1_score])\n\n## Plot Results\npd.DataFrame(numNeighboursResults, columns=[\'accuracy\',\'f1_score\'])\\\n    .plot(title="kNN Performance for varying k", figsize=(16,8))')




gausResults = testClassifier(dfAll, GaussianSongClassifier(), verbose=True)
resultsToConfusionMat(gausResults)




knnResults = testClassifier(dfAll, KnnClassifier(n=9), verbose=True)
resultsToConfusionMat(knnResults)




from sklearn.ensemble import RandomForestClassifier

rfClf = RandomForestClassifier(max_features=0.8, n_estimators=50)
rfResults = testClassifier(dfAll, rfClf, verbose=True)
resultsToConfusionMat(rfResults)




dfSongVects_ext = getSongVectors(dfLabels.id, agg=['mean','std']).set_index(dfLabels.id)
dfAll_ext = pd.merge(dfSongVects_ext, dfLabels, left_index=True, right_on='id', how='left')




rfClf_ext = RandomForestClassifier(max_features=0.8, n_estimators=50)
rfResults_ext = testClassifier(dfAll_ext, rfClf_ext, verbose=True)
resultsToConfusionMat(rfResults_ext)




model_test_results = [gausResults, knnResults, rfResults, rfResults_ext]

dfModelResults = pd.DataFrame([resultsToAvgScore(r) for r in model_test_results], index=['Gaus','kNN','RandF', 'RandF_ext'])
display(dfModelResults.style.background_gradient())
_=dfModelResults.plot.bar(title="Model CV Performance Comparison")




loadSong(dfLabels.iloc[0].id).corr().style.background_gradient()




get_ipython().run_cell_magic('time', '', "genres = []\ncorrs = []\n\n# Iterate genres, load and average song correlations\nfor genre, dfGenre in dfLabels.sample(1000).groupby('category'):\n    songCorrs = []\n    # print(genre, dfGenre.shape)\n\n    for r,row in dfGenre.iterrows():\n        songCorrs.append(loadSong(row.id).corr().values)\n\n    arrGenreCor = np.mean(np.array(songCorrs), axis=0)\n    corrs.append(arrGenreCor)\n    genres.append(genre)\n    \ndef getCor(genreA):\n    return corrs[genres.index(genreA)]\n\ndef displayCor(genreA):\n        display(pd.DataFrame(getCor(genreA)).replace(1,0.3).style.background_gradient())")




display('----- rock -----')
displayCor('rock')

display('----- metal -----')
displayCor('metal')

display('----- classical -----')
displayCor('classical')




# Fetch and calculate absolute distance between correlation matrices
def getCorrDist(genreA, genreB):
    return np.sum(np.abs(corrs[genres.index(genreA)]-corrs[genres.index(genreB)]))

dfCorDist = pd.DataFrame(index=genres, columns=genres)

# Iterate genre combinations
for a in genres:
    for b in genres:
        dfCorDist[a][b] = getCorrDist(a, b)

display('----- Correlation Distances between Genres -----')
dfCorDist.astype(float).replace(0,dfCorDist.values.mean()).style.background_gradient()

