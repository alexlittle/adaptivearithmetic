# Results





## 18 Dec 2024

### Configs

* 100k episodes - dqn_model/results/-100k
* 250k episodes - dqn_model/results/-250k
* using data: data/ou/studentassessment_course_bbb2013b_with_activities.csv
* Simulator v8

Has the scores categorized in blocks of 15, with the num activities completed before the next assessment 

### Results 

* 100k episodes

** with training data set

** with test dataset (data/ou/studentassessment_course_bbb2013j_with_activities.csv)


* 250k episodes

** with training data set


** with test dataset (data/ou/studentassessment_course_bbb2013j_with_activities.csv)



### Conclusion

Basically very little better than random





## 17 Dec 2024

### Configs

* 100k episodes - dqn_model/results/20241217175757-100k
* 250k episodes - dqn_model/results/20241217200224-250k
* using data: data/ou/studentassessment_course_bbb2013b_with_activities.csv
* Simulator v8

Has the scores categorized in blocks of 10, with the num activities completed before the next assessment 

### Results 

* 100k episodes

** with training data set
*** random 14.29%, actual: 17.23%
*** Num correct 1630, num incorrect: 7832


** with test dataset (data/ou/studentassessment_course_bbb2013j_with_activities.csv)
*** random 14.29%, actual: 15.48%
*** Num correct 1860, num incorrect: 10153

* 250k episodes

** with training data set
*** random 14.29%, actual: 16.97%
*** Num correct 1606, num incorrect: 7856

** with test dataset (data/ou/studentassessment_course_bbb2013j_with_activities.csv)
*** random 14.29%, actual: 15.70%
*** Num correct 1886, num incorrect: 10127


### Conclusion

Basically very little better than random