# information-retrieval HW2
University of Amsterdam

Project of Fije and Diede, master students Artificial Intelligence at the University of Amsterdam. 
Practial assignments for IR techniques.

Files included:
- Task 1
- Task 2
- Significancetesting
- /runfiles

**Task 1:**

In notebook "Task 1" and "Task 2" models are created by calling their functions. Running these functions could take a lot of time. So when running the cells and you don't want to call the functions, make sure you put those statements in comments. The lines to dump and load models through Pickle are in comments. So uncommenting this code, would result in the error of unabling to load the file because the models are not provided in this directory.

If you want to create run files, make sure you put the right data file in the parameters. Namely, calling the "write-run" function is not something we have copy-pasted for every model.

**Task 2:**

The trained latent semantic models are also not attached, so if you want to recreate the run files with the re-rankings you'll have to train the models again (the same parameters are in the code).

The models word2vec and doc2vec start to train immediately if you run the cells assigned to training these models (it is clear in the files which these are). To calculate the cosine similarity happens on loaded-in models, which have the same name as the file with the model that is saved after training. 

The models LSI and LDA train when you uncomment the represented lines of code at the end of the cell (very clear which cells are used for training the models as well).

All evaluation is done in the significanetesting ipython file.

**Significance Testing**

All runfiles used for the significance testing, and hyperparameter optimisation are included in the directory /runfiles. This code can be executed without a problem, will not take much time.

