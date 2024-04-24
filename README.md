dependencies:

- pip install torch transformers pandas

prepare_data.py:
  
- this cleans up and creates additional data sets, it needs to be run before all the other programs

machine_learning_multitask.py:

- the program has five arguments: train, save, load, accuracy, interactive
 1. train: train the model on the training sets
 2. save: save the model to model.pt
 3. load: load a previously saved model
 4. accuracy: measure the models accuracy against the test sets, on each individual task
 5. interactive: asks repeatedly for a prompt, which it will detect if the prompt is true or false, or neither

- arguments train or load needs to be used in order for save, accuracy and interactive to function.
- the order of the arguments does not matter.
- examples:
 - python machine_learning_multitask.py train save accuracy interactive
 - python machine_learning_multitask.py load interactive
 - python machine_learning_multitask.py train save
    
machine_learning_singletask.py:

- will measure the accuracies of the task as single task models, instead of multitask models.
- takes no arguments.