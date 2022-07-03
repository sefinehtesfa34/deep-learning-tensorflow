<img src="https://www.tensorflow.org/images/seq2seq/attention_mechanism.jpg" width="500" alt="attention mechanism">

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
We'll use a language dataset provided by [here](http://www.manythings.org/anki/)
This dataset contains language translation pairs in the format:


### May I borrow this book? ¿Puedo tomar prestado este libro?

They have a variety of languages available, but we'll use the English-Spanish dataset.

Download and prepare the dataset
After downloading the dataset, here are the steps we'll take to prepare the data:

1. Add a start and end token to each sentence.
2. Clean the sentences by removing special characters.
3. Create a word index and reverse word index (dictionaries mapping from word → id and id → word).
4. Pad each sentence to a maximum length.

## The encoder/decoder model

The following diagram shows an overview of the model. At each time-step the decoder's output is combined with a weighted sum over the encoded input, to predict the next word. The diagram and formulas are from [Luong's paper](https://arxiv.org/abs/1508.04025v5).

<img src="https://www.tensorflow.org/images/seq2seq/attention_mechanism.jpg" width="500" alt="attention mechanism">


### The encoder

Start by building the encoder, the blue part of the diagram above.

The encoder:

1. Takes a list of token IDs (from `input_text_processor`).
3. Looks up an embedding vector for each token (Using a `layers.Embedding`).
4. Processes the embeddings into a new sequence (Using a `layers.GRU`).
5. Returns:
  * The processed sequence. This will be passed to the attention head.
  * The internal state. This will be used to initialize the decoder
The encoder returns its internal state so that its state can be used to initialize the decoder.

It's also common for an RNN to return its state so that it can process a sequence over multiple calls. You'll see more of that building the decoder.

### The attention head

The decoder uses attention to selectively focus on parts of the input sequence.
The attention takes a sequence of vectors as input for each example and returns an "attention" vector for each example. This attention layer is similar to a `layers.GlobalAveragePoling1D` but the attention layer performs a _weighted_ average.

Let's look at how this works:

<img src="images/attention_equation_1.jpg" alt="attention equation 1" width="800">

<img src="images/attention_equation_2.jpg" alt="attention equation 2" width="800">
