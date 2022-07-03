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