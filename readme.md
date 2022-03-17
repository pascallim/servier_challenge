Welcome to the Servier Test Challenge!

Only the training phase has been treated (model 1 & 2) in this test due to lack of time.

You will find in the 'src' folder all the scripts concerning the training phase.

The training phase is using data files from the 'data' folder and save models in the 'models' folder.

Two models have been developped:

- The first one is using the 2048 features from the 'feature_extractor.py' script through a MultiLayerPerceptron architecture.
- The second one can be split in two parts. On one hand, a language model (Roberta) is trained to understand the smiles string and on the other hand the language model is fined-tuned on the classification task to recognize the property of the molecule.

The main.py script can be used with the following commands:
- python main.py --model_name classic (run the training of the MLP)
- python main.py --model_name lm (run the training of the LanguageModel Roberta)
- python main.py --model_name nlp (run the training of the pretrained LM model for the classification task)
- For more details on the available arguments, you can use python main.py -h

A lot of tasks have not been done concerning the Test Challenge. I would be very happy to discuss of the work done on my personal notebooks and how I would have implemented the rest of the tasks.

Thank you for reading!
Pascal LIM