# Results





## 18 Dec 2024

### Configs

- 100k episodes - dqn_model/results/20241218103900-100k
- 250k episodes - dqn_model/results/-250k
- using data: data/ou/studentassessment_course_bbb2013b_with_activities.csv
- Simulator v8

Has the scores categorized in blocks of 15, with the num activities completed before the next assessment 

### Results 

- 100k episodes
    - with training data set
        - random 20.00, actual exact: 56.90, actual close(+/-1) 83.40
        - Exactly num correct 5384, num incorrect: 4078
        - Close(+/-1) num correct 7891, num incorrect: 1571
    - with test dataset (data/ou/studentassessment_course_bbb2013j_with_activities.csv)
        - random 20.00, actual exact: 54.43, actual close(+/-1) 82.34
        - Exactly num correct 6539, num incorrect: 5474
        - Close(+/-1) num correct 9891, num incorrect: 2122
        - 
- 250k episodes
    - with training data set
    - with test dataset (data/ou/studentassessment_course_bbb2013j_with_activities.csv)


### Conclusion

The "close" values here aren't so meaningful, since at random there's probability of 0.52 that the random number will 
be  +/-1 of a number in the set of actions (0-4) anyway





## 17 Dec 2024

### Configs

- 100k episodes - dqn_model/results/20241217175757-100k
- 250k episodes - dqn_model/results/20241217200224-250k
- using data: data/ou/studentassessment_course_bbb2013b_with_activities.csv
- Simulator v8

Has the scores categorized in blocks of 10, with the num activities completed before the next assessment 

### Results 

- 100k episodes
    - with training data set
        - random 14.29, actual exact: 24.09, actual close 69.43
        - Exactly num correct 2279, num incorrect: 7183
        - Close(+/-1) num correct 6569, num incorrect: 2893
    - with test dataset (data/ou/studentassessment_course_bbb2013j_with_activities.csv)
        - random 14.29, actual exact: 21.97, actual close 68.74
        - Exactly num correct 2639, num incorrect: 9374
        - Close(+/-1) num correct 8258, num incorrect: 3755

- 250k episodes
    - with training data set
        - random 14.29, actual exact: 25.66, actual close 71.55
        - Exactly num correct 2428, num incorrect: 7034
        - Close(+/-1) num correct 6770, num incorrect: 2692
    - with test dataset (data/ou/studentassessment_course_bbb2013j_with_activities.csv)
        - random 14.29, actual exact: 24.05, actual close 70.96
        - Exactly num correct 2889, num incorrect: 9124
        - Close(+/-1) num correct 8524, num incorrect: 3489


### Conclusion

Quite a lot better than random. Probability of 0.408 that a random number (from 0-6) with be within +/-1 of another
number 0-6 anyway