from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.pipeline import Pipeline
from stop_words import get_stop_words
import numpy as np
import os
import argparse


stopWords = get_stop_words('en')

def printPathsWithSimilarities(paths, cosineSimilarities):
    """
    prints each file path with its cosine similarity 
    :param: ordered list of paths
    :param: ordered list of cosine similarities for each path
    """
    for i in range(len(paths)): 
        print(paths[i], cosineSimilarities[i])


def getTop5FilesUsing_TfidfVectorizer(pathToFiles, query):
    """
    calculate the cosine similarity of all documents using TfidfVectorizer
    :param: path to files
    :param: the text query
    :return: sorted list of paths, corresponding sorted list of cosine similarities
    """
    listOfFiles = os.listdir(pathToFiles)
    files = [query]
    for i in range(len(listOfFiles)):
        files.append(open( pathToFiles + "/" + listOfFiles[i], 'r').read().replace('\n', ' '))
    
    tfXidf = TfidfVectorizer(stop_words = stopWords, lowercase=True, min_df=0.5, max_features=10).fit_transform(files)
    
    cosine_similarities = (linear_kernel(tfXidf[0:1], tfXidf[1:])).flatten()
    related_docs_indices = (-cosine_similarities).argsort()
    cosine_similarities =(cosine_similarities[(-cosine_similarities).argsort()])
    
    paths=[]
    for indice in related_docs_indices:
        paths.append(pathToFiles + "/" + listOfFiles[indice])
    return (paths, cosine_similarities)

def getTop5FilesUsing_TfidfTransformer(pathToFiles, query):
    """
    calculate the cosine similarity of all documents using TfidfTransformer
    :param: path to files
    :param: the text query
    :return: sorted list of paths, corresponding sorted list of cosine similarities
    """
    listOfFiles = os.listdir(pathToFiles)
    files = []
    for i in range(len(listOfFiles)):
        files.append(open( pathToFiles + "/" + listOfFiles[i], 'r').read().replace('\n', ' '))
    query=[query]
    
    vectorizer = CountVectorizer(stop_words = stopWords, lowercase=True, min_df=0.5, max_features=10)
    transformer = TfidfTransformer()
    
    filesVectors = vectorizer.fit_transform(files).toarray()
    queryVector = vectorizer.transform(query).toarray()
    
    transformer.fit(filesVectors)
    tfXidfFiles = (transformer.transform(filesVectors).toarray())

    transformer.fit(queryVector)
    tfXidfQuery = (transformer.transform(queryVector).todense())    

    cosine_similarities = linear_kernel(tfXidfFiles, tfXidfQuery).flatten()
    related_docs_indices = (-cosine_similarities).argsort()
    cosine_similarities = (cosine_similarities[(-cosine_similarities).argsort()])
    
    paths=[]
    for indice in related_docs_indices:
        paths.append(pathToFiles + "/" + listOfFiles[indice])
    return (paths, cosine_similarities)


if __name__ == '__main__':
    #read path and query from console while executing the program
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_files')
    parser.add_argument('query')
    args = parser.parse_args()

    #prepare the path for usage
    path = args.path_to_files
    if path.endswith("/"): path = path[:-1]

    #check if the path exists and is not empty before executing
    if (os.path.exists(path)==False):
        print("entered folder path doesn't exist")
    elif len(os.listdir(path)) == 0:
        print("folder is empty")
    else:
        print("the top 5 files using TfidfVectorizer:")
        paths, cosineSimilarities = (getTop5FilesUsing_TfidfVectorizer(path, args.query))
        printPathsWithSimilarities(paths[:5], cosineSimilarities[:5])

        print("the top 5 files using TfidfTransformer:")
        paths, cosineSimilarities = (getTop5FilesUsing_TfidfTransformer(path, args.query))
        printPathsWithSimilarities(paths[:5], cosineSimilarities[:5])

