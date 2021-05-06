# Doc2Vec_tutorial

----------
### 1. 목적
* Document embedding을 이해하기 위해 네이버 영화 리뷰 데이터셋을 가지고 실습을 진행함.
* '한국어 임베딩'교재를 참고하며 진행한 소스코드

----------
### 2. 특징
1. 불용어 처리
2. 형태소 분석
3. 시각화: Bokeh API


----------
### 3. 설치할 것

* from konlpy.tag import Okt : Okt 형태소 분석기 사용(트위터 제작)
* from gensim.models.doc2vec import Doc2Vec, TaggedDocument : Gensim 사용


![image](https://user-images.githubusercontent.com/28869864/117263845-a3672600-ae8d-11eb-97c4-17412ee62695.png)
(Bokeh 시각화 예시 - '제1공화국'은  '써드 스타 제1공화국'과 유사함)

----------
### 4. 참고
* Document embedding : 문서 내에 등장하는 단어의 임베딩 벡터들을 평균하여 document 벡터로 간주함.
* 특징 : 특정 책과 유사한 줄거리를 가지는 책들을 추천함.
* 출처 : https://wikidocs.net/102705

![image](https://user-images.githubusercontent.com/28869864/117268236-06f35280-ae92-11eb-8079-0062eebda77a.png)
