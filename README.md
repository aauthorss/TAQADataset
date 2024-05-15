# Teacher Action Quality Assessment Dataset (TAQADataset)

## About the dataset：
![image](https://github.com/aauthorss/TAQADataset/blob/main/CoVL/fig/Overview%20of%20TAQA.png)
Figure 1 The comparison of the four types of teacher actions in two different contexts (positive on the left and negative on the right). (a) Inviting students to answer questions (b) Pointing to teaching devices (c) Walking around the classroom and (d) Writing on the blackboard.

  The TAQA dataset covers four common actions of teachers during the teaching process, namely Inviting students to answer questions, Pointing to teaching devices, Walking around the classroom and Writing on the blackboard. Each action contains a detailed description of two different scenarios, positive and negative, as shown in Figure 1. This dataset contains a total of 3698 samples, all in mp4 format, with a resolution of 640 x 360 and a frame rate of 30fps. The detailed information is shown in Table 1.

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

## About the CoVL model：

### Requirement
Python >= 3.6

Pytorch >=1.8.0

### Dataset Preparation
**1.TAQA dataset**

If the article is accepted for publication, you can download our prepared TAQA dataset demo from ["Google Drive"](https://drive.google.com/file/d/13Rr3XIo5t2QygmerOVCFn1pRiyg4wPVC/view?usp=sharing) . Then, please move the uncompressed data folder to `TAQA/data/frames`. We used the I3D backbone pretrained on Kinetics([Google Drive](https://drive.google.com/file/d/1M_4hN-beZpa-eiYCvIE7hsORjF18LEYU/)).

**2.MTL-AQA dataset**(["Google Drive"](https://drive.google.com/file/d/1T7bVrqdElRLoR3l6TxddFQNPAUIgAJL7/))

### Training & Evaluation

CoVL, as a plug and play module, has good versatility and can be easily integrated into other models. In this study, we first select classic action quality assessment models USDL, MUSDL, and DAE as baseline models, and then integrate CoVL into the above models, namely USDL-CoVL, MUSDL-CoVL and DAE-CoVL, respectively, to evaluate the action quality.Take **MUSDL-CoVL** as an example,To train and evaluate on TAQA:

` python -u main.py  --lr 1e-4 --weight_decay 1e-5 --gpu 0 `

If you use the TAQA dataset, please cite this paper: A New Assessment Dataset and Approach for Teacher Action Quality Assessment in Classroom Environment.
