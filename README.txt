code  for  "PPIformer :End-to-End Transformer-Based Siamese Network   with   attention-based 
BiGRU to Predict Interface Residue Pairs of  Pentamer Protein  Based on Sequence Only"
1.train:
python train.py --DATA "data/example.txt" --BATCH_SIZE 128 --MAXLEN 80 --EPOCHS 1
--LR   0.001
2.predict:
python  predict_test.py --DATA "data/example.txt"
3.files:
data:--example.txt : Part of the data set,all of data is on " "
Model:--0.pl :trained model
MODELS:--CNN.py   CNN model
       --DNN.py   DNN model
       --LR.py    Logistic Regression
       --LSTM.py  LSTM model
       --RandomForests.py  Random Forests
       --SVM.py   Support Vector Machine
holdonAUC.py  the code for drawing roc
read_loss.py  the code for drawing loss
train.py      the code for training
predict_test.py the code for predicting
4.Requirements:
python3.8
pytorch1.9.0+cu102
scikit-learn0.23
torchvision0.10.0+cpu
numpy1.20.1
pandas1.2.4
matplotlib3.3.4
5.References:
[1] Cheng, H. , et al. "PepFormer: End-to-End Transformer-Based Siamese Network to Predict and Enhance Peptide Detectability Based on Sequence Only." Analytical Chemistry (2021).
[2]When Does Label Smoothing Help?
[3]Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification
[4]Li, J. , et al. "Learning Small-Size DNN with Output-Distribution-Based Criteria." interspeech (2014).
[5]Lei, T. ,  R. Barzilay , and  T. Jaakkola . "Molding CNNs for text: non-linear, non-consecutive convolutions." Indiana University Mathematics Journal 58.3(2015):págs.  1151-1186.
[6]Gre Ff , K. , et al. "LSTM: A Search Space Odyssey." IEEE Transactions on Neural Networks & Learning Systems 28.10(2016):2222-2232.
[7]Peduzzi, P. , et al. "A simulation study of the number of events per variable in logistic regression analysis. " Journal of Clinical Epidemiology 49. 12(1996):1373-9.
[8]Liaw, A. , and  M. Wiener . "Classification and Regression by randomForest." R News 23.23(2002).
[9]Baesens, B. , et al. "Least squares support vector machine classifiers: an empirical evaluation." Access & Download Statistics (2000).
[10]A Visual Survey of Data Augmentation in NLP
[11]Improving Neural Machine Translation Models with Monolingual Data
[12]Zhe, et al. "SMOTETomek-Based Resampling for Personality Recognition." IEEE Access 7(2019):129678-129689.

