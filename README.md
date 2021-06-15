# DRGR

DRGR: Deep Reinforcement learning based Group Recommender system.

A course project for Georgia
Tech [CSE 6240 Web Search and Text Mining](https://www.cc.gatech.edu/~srijan/teaching/cse6240/spring2021/) Spring 2021.

The [project report](https://arxiv.org/abs/2106.06900) is available on arXiv.

## Environment Settings

- python version: '3.7.10'
- pytorch version:  '1.8.1'

## Dataset

The original MovieLens dataset can be downloaded from [MovieLens](https://grouplens.org/datasets/movielens/), and the
one we use is [MovieLens 1M Dataset](https://files.grouplens.org/datasets/movielens/ml-1m.zip). You can generate the
group data `MovieLens-Rand` by using `data/MovieLens-1M.zip` and `drgr/generator.py`. (See "Run the DRGR".)

The `data/MovieLens-Rand` directory should include the following files:

movies.dat:

* Movie information file from MovieLens-1M.

users.dat:

* User information file from MovieLens-1M.

groupMember.dat:

* File including group members.
* Each line is a group instance: groupID userID1,userID2,...

group(user)RatingTrain.dat:

* Train file.
* Each line is a training instance: groupID(userID) itemID rating timestamp

group(user)RatingVal(Test).dat:

* group (user) validation (test) file (positive instances).
* Each line is a validation (test) instance: groupID(userID) itemID rating timestamp

group(user)RatingVal(Test)Negative.dat:

* group (user) validation (test) file (negative instances).
* Each line corresponds to the line of group(user)RatingVal(Test).dat, containing 100 negative samples.
* Each line is in the format: (groupID(userID),itemID) negativeItemID1 negativeItemID2 ...

## Run the DRGR

Change to the code directory (if not in):

```
cd drgr
```

Generate the group data from the MovieLens-1M dataset (`data/MovieLens-1M.zip`):

```
python generator.py
```

Run the DRGR:

```
python main.py
```

## Parameters

We put all parameters in `config.py`.

## References

During the code implementation, we reference the following repositories and articles:

* AGREE: https://github.com/LianHaiMiao/Attentive-Group-Recommendation
* LIRD: https://github.com/egipcy/LIRD
* Recsys-RL: https://github.com/shashist/recsys-rl
* GroupIM: https://github.com/CrowdDynamicsLab/GroupIM
* Deep Deterministic Policy Gradients
  Explained: https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b
