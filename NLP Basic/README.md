# NLP_GNN_STUDY

### Members
    Keejun-Han, Cheonsol-Lee, Donghee-Han, Daehee-Kim, Jongwoo-Kim

### Dataset
1. NAVER sentiment movie corpus: https://github.com/e9t/nsmc
2. KORQUAD: https://korquad.github.io/

---------
### (2021.04.14) Introduction to Information Retrieval
- Presenter: Keejun Han

### (2021.05.06) Word2Vec, Doc2Vec
- (Word2Vec) Efficient Estimation of Word Representations in Vector Space
- (Doc2Vec) Distributed Representations of Sentences and Documents
- Practice: Predicting similar movies based on Naver reviews
- Presenter: Cheonsol Lee

### (2021.05.20) BERT 
- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- Practice: Predicting sentiment based on Naver reviews
- Presenter: Soyoung Cho

### (2021.06.09) Research paper review about BERT
- Purpose : We challenge the IEEE BicComp 2022 conference by reviewing papers using BERT.
- Presenter : Keejun Han / Cheonsol Lee / Soyoung Cho

### (2021.07.08) Graph Neural Network(GNN), OTT dataset building
- Purpose : OTT dataset building(Benchmark)
- Direction : 
    - (IEEE BigComp 2022) OTT dataset building(Movielens, Netflix, Watcher, etc)
    - (SCI Journal) Search and Recommender system Using GNN on SmartTV
- To do
    - Study : GNN
    - Search : movie dataset
    - Search : related works about linking of movie dataset
- Presenter : Cheonsol Lee

### (2021.07.15) Movie dataset linking
- Purpose : Robust Movie Dataset Matching
- Data : [MovieLens](https://www.kaggle.com/grouplens/movielens-20m-dataset), [IMDb top 1000](https://www.kaggle.com/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows), [MovieLens2019](https://grouplens.org/datasets/movielens/), [Rotten Tomato-2020](https://www.kaggle.com/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset)
- Direction : 
    - [MovieLens]-link.csv(right answer)
    - Movie contents profiling for recommender system
- Problem
    - Difficulty connecting OTT meta data
    - Difficulty in linking other forms' meta data
- Solution
    - Provision of integrated solutions for OTTs
- To do
    - Small data completion using IMDb top 30  ex) <title, I, M>
    - Case Study : Even if the title is different, judge the same movie by looking at the similarity of the feature
    - ex) Wiki, Title, Year, Genre, Director, Actor, Levenshtein Distance
 
 ### (2021.08.11) Recommender System based on Rating and Sentiment Analysis using GNN
- Purpose : Relational Recommendation using GNN
- Data : [Rotten Tomato-2020](https://www.kaggle.com/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset)
- Direction : 
    - Sentiment analysis based on review data using BERT
    - Review-Triple, Movie-Triple
    - Ranking films for any user to give positive reviews
    - Relational recommendation using movie meta data
