# Teacher Action Quality Assessment Dataset (TAQADataset)

### About the dataset：

  The TAQA dataset covers four common actions of teachers during the teaching process, namely Inviting students to answer questions, Pointing to teaching devices, Walking around the classroom and Writing on the blackboard. Each action contains a detailed description of two different scenarios, positive and negative. This dataset contains a total of 3698 samples, all in mp4 format, with a resolution of 640 x 360 and a frame rate of 30fps. The detailed information is shown in Table 1.

  In addition, scores and grade labels are generated for the dataset. Among them,

(1) the classification label adopts a 15-point scale. For each action video, not only the final score, but also the independent scoring of all samples by 5 experts. Therefore, the dataset is capable of not only single-score prediction, but also multi-score prediction.

(2) After obtaining all the score labels, we divided the 15-point scale into 5 grades, which are Excellent, Good, Average, Failed and Poor. Finally, the labels for each video sample include the action category, grade, each judge's score, and final score.

Table 1. The detailed information of TAQA dataset.
| Action_type | #Samples | Avg.Seq.Len | Min.Seq.Len| Max.Seq.Len |
| :---: | :---: | :---: | :---: | :---: | 
|Inviting students to answer questions|	1062|	119	|50	|223|
|Pointing to teaching devices|	1132|	261	| 8 4|	1479|
|Walking around classroom|	585|	575|	77|	1461|
|Writing on the blackboard|	955	|436|	130|	1404|

1.**Action_type** represents the action category.

2.**#Samples** represents the number of samples in each category. 

3.**Avg.Seq.Len** represents the average number of video frames for each category. 

4.**Min.Seq.Len** represents the minimum number of video frames for each category.

5.**Max.Seq.Len** represents the maximum number of video frames for each category. 

The detailed partition of training set and test set is given in our paper.

### About the CoVL：
