# deep-learning-tensorflow
Deep learning using Tensorflow<br> 
In the <strong>translation folder</strong> above, I trained a sequence to sequence (seq2seq) model for<br> Spanish to English translation based on Effective Approaches to Attention-based <br>Neural Machine Translation. This is an advanced example that assumes some knowledge of:
* Sequence to sequence models<br>
* TensorFlow fundamentals below the keras layer:<br>
* Working with tensors directly<br>
* Writing custom keras.Models and keras.layers<br>
While this architecture is somewhat outdated it is still a very useful project to work through to get a deeper understanding of attention mechanisms (before going on to Transformers).

After training the model in this notebook, you will be able to input a Spanish sentence, such as "¿todavia estan en casa?", and return the English translation: "are you still at home?"

The resulting model is exportable as a tf.saved_model, so it can be used in other TensorFlow environments.

The translation quality is reasonable for a toy example, but the generated attention plot is perhaps more interesting. This shows which parts of the input sentence has the model's attention while translating:
<image src="translation/image.png">

### Note: This example takes approximately 10 minutes to run on a single P100 GPU.
The data
We'll use a language dataset provided by <a href=">http://www.manythings.org/anki/">here</a> 
This dataset contains language translation pairs in the format:


### May I borrow this book? ¿Puedo tomar prestado este libro?

They have a variety of languages available, but we'll use the English-Spanish dataset.

Download and prepare the dataset
After downloading the dataset, here are the steps we'll take to prepare the data:

1. Add a start and end token to each sentence.
2. Clean the sentences by removing special characters.
3. Create a word index and reverse word index (dictionaries mapping from word → id and id → word).
4. Pad each sentence to a maximum length.
