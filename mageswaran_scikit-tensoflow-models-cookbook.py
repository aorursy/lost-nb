#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
train_df.head()




category_col = "author"




from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer




le = LabelEncoder().fit(train_df["author"].as_matrix())
label_binarizer = LabelBinarizer().fit(train_df["author"].as_matrix()) # For TensorFlow




def get_one_hot_target_label(df):
    print('Labels and their document counts based on', end=' ')
    print(df.groupby(category_col)[category_col].count())

    return label_binarizer.transform(df[category_col].as_matrix())




get_one_hot_target_label(train_df)




import nltk
import spacy




def extract_lemmas(df: pd.DataFrame, text_col, nlp=spacy.load('en')):
    stopwords = nltk.corpus.stopwords.words('english')

    def cleaning(sentence):
        sentence = nlp(sentence)
        tokens = [token.lemma_ for token in sentence if not token.is_punct | token.is_space | token.is_bracket | (token.text in stopwords)]
        return ' '.join(tokens)

    df = df.assign(nlp_processed = lambda rows : rows[text_col].map(lambda row: cleaning(row)))

    return df
    
    




train_df = extract_lemmas(train_df, "text")
test_df = extract_lemmas(test_df, "text")




text_col = "nlp_processed"




train_df = train_df.assign(label = lambda rows : rows.author.map(lambda author: le.transform([author])[0]))




train_df = train_df.assign(length = lambda rows: rows.nlp_processed.map(lambda sent: len(sent.split(' '))) )




train_df.head()




# train_df = train_df[train_df["length"] > 10]




def _get_train_val_split(df, category_col='author'):
        print('Splitting the data set(stratified sampling)...')

        def train_validate_test_split(df, train_percent=.8, seed=42):
            np.random.seed(seed)
            perm = np.random.permutation(df.index)
            m = len(df)
            train_end = int(train_percent * m)
            train = df.loc[perm[:train_end]]
            validate = df.loc[perm[train_end:]]
            return train, validate

        #Make list of sampled dataframe for each category
        dfs = [train_validate_test_split(df[df[category_col] == label]) for label in le.classes_]

        #Now the values are grouped to form a Dataframe
        train_dfs = []
        val_dfs = []
        for train_df, val_df in dfs:
            train_dfs.append(train_df)
            val_dfs.append(val_df)

        train_df = pd.concat(train_dfs)
        val_df = pd.concat(val_dfs)

        #Shuffle the data
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        val_df = val_df.sample(frac=1).reset_index(drop=True)

        print('Done!')

        return train_df, val_df




train_df1, val_df = _get_train_val_split(train_df)




from sklearn.model_selection import train_test_split
train_df1, val_df = train_test_split(train_df,test_size=0.2) 




train_df.shape, train_df1.shape, val_df.shape
#((19579, 6), (15663, 6), (3916, 6))




train_df[[text_col, 'author']].groupby('author').count().plot(kind='bar')




z = {'EAP': 'Edgar Allen Poe', 'MWS': 'Mary Shelley', 'HPL': 'HP Lovecraft'}
data = [go.Bar(
            x = train_df.author.map(z).unique(),
            y = train_df.author.value_counts().values,
            marker= dict(colorscale='Jet',
                         color = train_df.author.value_counts().values
                        ),
            text='Text entries attributed to Author'
    )]

layout = go.Layout(
    title='Target variable distribution'
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='basic-bar')









from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

def evaluate(pipeline, val_df=val_df, text_col=text_col, label_col='label'):
    validation_prediction = pipeline.predict(val_df[text_col].as_matrix())
    validation_actual = val_df[label_col].as_matrix()
    val_acc = np.mean(validation_prediction == validation_actual)
    print("Model Accuracy is {}".format(val_acc))

    val_report = classification_report(validation_actual, validation_prediction, target_names=list(le.classes_))
    print(val_report)
    
def fit_n_evaluate(stages):
    pipeline = Pipeline(stages)
    pipeline.fit(train_df1['text'].as_matrix(), train_df1['label'].as_matrix())
    evaluate(pipeline)




from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer




# Define helper function to print top words
def print_top_words(model, feature_names, n_top_words):
    for index, topic in enumerate(model.components_):
        message = "\nTopic #{}:".format(index)
        message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1 :-1]])
        print(message)
        print("="*70)




from sklearn.decomposition import NMF, LatentDirichletAllocation

tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                stop_words='english')
tf = tf_vectorizer.fit_transform(train_df[text_col].as_matrix())

lda = LatentDirichletAllocation(n_components=3, max_iter=5,
                                learning_method = 'online',
                                learning_offset = 50.,
                                random_state = 0)

lda.fit(tf)




n_top_words = 20
print("\nTopics in LDA model: ")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)




first_topic = lda.components_[0]
second_topic = lda.components_[1]
third_topic = lda.components_[2]




first_topic_words = [tf_feature_names[i] for i in first_topic.argsort()[:-50 - 1 :-1]]
second_topic_words = [tf_feature_names[i] for i in second_topic.argsort()[:-50 - 1 :-1]]
third_topic_words = [tf_feature_names[i] for i in third_topic.argsort()[:-50 - 1 :-1]]




from wordcloud import WordCloud, STOPWORDS




# Generating the wordcloud with the values under the category dataframe
firstcloud = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          width=2500,
                          height=1800
                         ).generate(" ".join(first_topic_words))
plt.imshow(firstcloud)
plt.axis('off')
plt.show()




# Generating the wordcloud with the values under the category dataframe
firstcloud = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          width=2500,
                          height=1800
                         ).generate(" ".join(second_topic_words))
plt.imshow(firstcloud)
plt.axis('off')
plt.show()




# Generating the wordcloud with the values under the category dataframe
firstcloud = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          width=2500,
                          height=1800
                         ).generate(" ".join(third_topic_words))
plt.imshow(firstcloud)
plt.axis('off')
plt.show()




from sklearn.ensemble import RandomForestClassifier




rf_stages = [
                        ('counter', CountVectorizer(analyzer='word', ngram_range=(1,1), max_df=0.95, min_df=2, stop_words='english')),
                        ('vectorizer', TfidfTransformer()),
#                         ('vectorizer', TfidfVectorizer(min_df=0, max_df=1, ngram_range=(1, 3), stop_words='english')),
                        ('clf', RandomForestClassifier(n_jobs=8,
                                              n_estimators=100,
                                              min_samples_leaf=4,
                                              oob_score=True,
                                              max_depth=20,
                                              max_features=0.8, #Not much difference with log2
                                              random_state=42))
              ]




fit_n_evaluate(rf_stages)




from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest




xgboost_stages = [
    ('vectorizer', TfidfVectorizer(min_df=2, max_df=0.95, ngram_range=(1, 2), stop_words='english')),
    ("kbest",SelectKBest(k=300)), 
     ('clf', XGBClassifier())
]




fit_n_evaluate(xgboost_stages)




from sklearn import svm

svm_stages = [('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', svm.LinearSVC())
                    ]
fit_n_evaluate(svm_stages)




from sklearn.naive_bayes import MultinomialNB
mnb_stages = [('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf',  MultinomialNB())
                    ]

fit_n_evaluate(mnb_stages)














import tensorflow as tf
from tensorflow.python.platform import gfile
from tqdm import tqdm_notebook as tqdm
import tensorflow.contrib.learn as tflearn




# Define data loaders
class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        self.iterator_initializer_func(session)

def save_vocab(lines, outfilename, MAX_DOCUMENT_LENGTH, PADWORD='ZYXW'):
    # the text to be classified
    vocab_processor = tflearn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH,
                                                                min_frequency=0)
    vocab_processor.fit(lines)

    with gfile.Open(outfilename, 'wb') as f:
        f.write("{}\n".format(PADWORD))
        for word, index in tqdm(vocab_processor.vocabulary_._mapping.items()):
            f.write("{}\n".format(word))

    nwords = len(vocab_processor.vocabulary_)
    print('{} words into {}'.format(nwords, outfilename))

    return nwords + 2  # UNKNOWN + PADWORD

# Define the inputs
def setup_input_graph(features, labels, batch_size, scope='train-data'):
    """Return the input function to get the training data.

    Args:
        batch_size (int): Batch size of training iterator that is returned
                          by the input function.
        mnist_data (Object): Object holding the loaded mnist data.

    Returns:
        (Input function, IteratorInitializerHook):
            - Function that returns (features, labels) when called.
            - Hook to initialise input iterator.
    """
    iterator_initializer_hook = IteratorInitializerHook()


    def inputs():
        """Returns training set as Operations.

        Returns:
            (features, labels) Operations that iterate over the dataset
            on every evaluation
        """
        with tf.name_scope(scope):

            # Define placeholders
            features_placeholder = tf.placeholder(tf.string, features.shape)
            labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

            # Build dataset iterator
            dataset = tf.contrib.data.Dataset.from_tensor_slices((features_placeholder,
                                                                  labels_placeholder))
            dataset = dataset.repeat(None)  # Infinite iterations
            dataset = dataset.shuffle(buffer_size=10000)
            dataset = dataset.batch(batch_size)
            iterator = dataset.make_initializable_iterator()

            # Set runhook to initialize iterator
            iterator_initializer_hook.iterator_initializer_func =                 lambda sess: sess.run(
                    iterator.initializer,
                    feed_dict={features_placeholder: features,
                               labels_placeholder: labels})

            next_example, next_label = iterator.get_next()

            # Return batched (features, labels)
            return next_example, next_label

    # Return function and hook
    return inputs, iterator_initializer_hook




import tensorflow as tf
from tensorflow.contrib import lookup
from tensorflow.contrib.learn import learn_runner
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

# Set default flags for the output directories
FLAGS = tf.app.flags.FLAGS





class TFBaseEstimator(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, train_input_fn,
                 train_input_hook,
                 eval_input_fn,
                 eval_input_hook,
                 learning_rate,
                 train_steps,
                 min_eval_frequency):
        self.train_input_fn = train_input_fn
        self.train_input_hook = train_input_hook
        self.eval_input_fn = eval_input_fn
        self.eval_input_hook =  eval_input_hook

        self.learning_rate = learning_rate
        self.train_steps = train_steps
        self.min_eval_frequency = min_eval_frequency

    def estimator_spec(self, *args):
        return NotImplementedError

    def fit(self, X, y):
        return NotImplementedError

    def predict(self,X):
        return NotImplementedError

    def get_estimator(self, run_config, params):
        """Return the model as a Tensorflow Estimator object.

        Args:
             run_config (RunConfig): Configuration for Estimator run.
             params (HParams): hyperparameters.
        """
        return tf.estimator.Estimator(
            model_fn=self.estimator_spec,  # First-class function
            params=params,  # HParams
            config=run_config  # RunConfig
        )

    def experiment_fn(self, run_config, params):
        """Create an experiment to train and evaluate the model.
    
        Args:
            run_config (RunConfig): Configuration for Estimator run.
            params (HParam): Hyperparameters
    
        Returns:
            (Experiment) Experiment for training the mnist model.
        """
        # You can change a subset of the run_config properties as
        run_config = run_config.replace(save_checkpoints_steps=params.min_eval_frequency)

        # Define the mnist classifier
        estimator = self.get_estimator(run_config, params)

        # Define the experiment
        experiment = tf.contrib.learn.Experiment(
            estimator=estimator,  # Estimator
            train_input_fn=self.train_input_fn,  # First-class function
            eval_input_fn=self.eval_input_fn,  # First-class function
            train_steps=params.train_steps,  # Minibatch steps
            eval_steps=100,  # Use evaluation feeder until its empty
            min_eval_frequency=params.min_eval_frequency,  # Eval frequency
            train_monitors=[self.train_input_hook],  # Hooks for training
            eval_hooks=[self.eval_input_hook],  # Hooks for evaluation
            #         export_strategies=[saved_model_export_utils.make_export_strategy(
            #                                 serving_input_fn,
            #                                 default_output_alternative_key=None,
            #                                 exports_to_keep=1
            #                                 )],

        )

        return experiment

    def run_experiment(self, argv=None):
        """Run the training experiment."""
        # Define model parameters
        params = tf.contrib.training.HParams(
            learning_rate=self.learning_rate,
            train_steps=self.train_steps,
            min_eval_frequency=self.min_eval_frequency
        )

        # Set the run_config and the directory to save the model and stats
        run_config = tf.contrib.learn.RunConfig()
        run_config = run_config.replace(model_dir=FLAGS.model_dir)

        learn_runner.run(
            experiment_fn=self.experiment_fn,  # First-class function
            run_config=run_config,  # RunConfig
            schedule="train_and_evaluate",  # What to run
            hparams=params  # HParams
        )

    def run(self):
        tf.app.run(main=self.run_experiment)




def get_sequence_length(sequence):
    '''
    Returns the sequence length, droping out all the zeros if the sequence is padded
    :param sequence: Tensor(shape=[batch_size, doc_length, feature_dim])
    :return: Array of Document lengths of size batch_size
    '''
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used,1)
    length = tf.cast(length, tf.int32)
    return length




import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys
from tensorflow.contrib import lookup

# import tensorflow.contrib.rnn.LSTMStateTuple


class TextCNNRNN(TFBaseEstimator):
    def __init__(self,
                 vocab_file,
                 vocab_size,
                 train_input_fn,
                 train_input_hook,
                 eval_input_fn,
                 eval_input_hook,
                 max_doc_length,
                 learning_rate=0.001,
                 train_steps=40000,
                 min_eval_frequency=500):
        super().__init__(train_input_fn,
                         train_input_hook,
                         eval_input_fn,
                         eval_input_hook,
                         learning_rate,
                         train_steps,
                         min_eval_frequency)

        self.VOCAB_FILE = vocab_file
        self.VOCAB_SIZE = vocab_size
        self.PADWORD = 'PADXYZ'
        self.MAX_DOCUMENT_LENGTH = max_doc_length
        self.EMBEDDING_SIZE = 300

        self.WINDOW_SIZE = self.EMBEDDING_SIZE
        self.STRIDE = int(self.WINDOW_SIZE / 2)

        self.NUM_CLASSES = 3

        self.num_lstm_layers = 1
        self.output_keep_prob = 0.5

    def estimator_spec(self, features, labels, mode, params):
        """Model function used in the estimator.

        Args:
            features : Tensor(shape=[?], dtype=string) Input features to the model.
            labels : Tensor(shape=[?, n], dtype=Float) Input labels.
            mode (ModeKeys): Specifies if training, evaluation or prediction.
            params (HParams): hyperparameters.

        Returns:
            (EstimatorSpec): Model to be run by Estimator.
        """
        is_training = mode == ModeKeys.TRAIN


        # Define model's architecture
        with tf.variable_scope("sentence-2-words"):
            table = lookup.index_table_from_file(vocabulary_file=self.VOCAB_FILE,
                                                 num_oov_buckets=1,
                                                 default_value=-1,
                                                 name="table")
            tf.logging.info('table info: {}'.format(table))

            # string operations
            text_lines = tf.squeeze(features)
            words = tf.string_split(text_lines)
            densewords = tf.sparse_tensor_to_dense(words, default_value=self.PADWORD)
            numbers = table.lookup(densewords)
            sliced = numbers
            # padding = tf.constant([[0, 0], [0, self.MAX_DOCUMENT_LENGTH]])
            # padded = tf.pad(numbers, padding)
            # sliced = tf.slice(padded, [0, 0], [-1, self.MAX_DOCUMENT_LENGTH])

        with tf.device('/cpu:0'), tf.name_scope("embed-layer"):
            # layer to take the words and convert them into vectors (embeddings)
            # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
            # maps word indexes of the sequence into
            # [batch_size, MAX_DOCUMENT_LENGTH, EMBEDDING_SIZE].
            word_vectors = tf.contrib.layers.embed_sequence(sliced,
                                                      vocab_size=self.VOCAB_SIZE,
                                                      embed_dim=self.EMBEDDING_SIZE)

            # [?, self.MAX_DOCUMENT_LENGTH, self.EMBEDDING_SIZE]
            tf.logging.debug('words_embed={}'.format(word_vectors))

            # Split into list of embedding per word, while removing doc length dim.
            # word_list results to be a list of tensors [batch_size, EMBEDDING_SIZE].
            # word_list = tf.unstack(word_vectors, axis=1)

        with tf.name_scope("lstm-layer"):
                # LSTM cell
            lstm = tf.contrib.rnn.LSTMCell(self.EMBEDDING_SIZE, state_is_tuple=True)
            tf.logging.info('lstm: ------> {}'.format(lstm))

            # Add dropout to the cell
            # cell =  SwitchableDropoutWrapper(
            #     lstm,
            #     is_training,
            #     output_keep_prob=self.output_keep_prob)
            if is_training:
                cell = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=self.output_keep_prob)
            else:
                cell = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=1.0)

                # Stack up multiple LSTM layers, for deep learning
            cell = tf.contrib.rnn.MultiRNNCell([cell] * self.num_lstm_layers)
            tf.logging.info('cell: ------> {}'.format(cell))
            #
            outputs, encoding = tf.nn.dynamic_rnn(cell, word_vectors, dtype=tf.float32,
                                                  sequence_length=get_sequence_length(word_vectors))
            # LSTMStateTuple https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LSTMStateTuple
            encoding = encoding[0][0]

            # Create a Gated Recurrent Unit cell with hidden size of EMBEDDING_SIZE.
            # cell = tf.nn.rnn_cell.GRUCell(self.EMBEDDING_SIZE)

            # Create an unrolled Recurrent Neural Networks to length of
            # MAX_DOCUMENT_LENGTH and passes word_list as inputs for each unit.
            # _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)
            # [?, EMBEDDING_SIZE]
            tf.logging.info('encoding: ------> {}'.format(encoding))


        with tf.name_scope("hidden-mlp-layer"):
            # [batch_size, 100]
            hidden_layer = tf.contrib.layers.fully_connected(encoding, 100,
                                                                 activation_fn=tf.nn.relu)
            tf.logging.info('hidden_layer: ------> {}'.format(hidden_layer))
            hidden_layer = tf.contrib.layers.fully_connected(hidden_layer, 50,
                                                             activation_fn=tf.nn.relu)
            tf.logging.info('hidden_layer: ------> {}'.format(hidden_layer))

        with tf.name_scope("logits-layer"):
            # [?, self.NUM_CLASSES]
            logits = tf.contrib.layers.fully_connected(hidden_layer, self.NUM_CLASSES,
                                                                 activation_fn=tf.sigmoid)
            tf.logging.info('logits: ------> {}'.format(logits))

        with tf.name_scope("output-layer"):
            # [?,1]
            predictions = tf.argmax(logits, axis=-1)
            tf.logging.info('predictions: ------> {}'.format(predictions))

            # Loss, training and eval operations are not needed during inference.
        loss = None
        train_op = None
        eval_metric_ops = {}

        if mode != ModeKeys.INFER:
            loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=labels,
                logits=logits,
                name='softmax_entropy')

            loss = tf.reduce_mean(loss)

            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.contrib.framework.get_global_step(),
                optimizer=tf.train.AdamOptimizer,
                learning_rate=params.learning_rate)

            eval_metric_ops = {
                'RPXAccuracy': tf.metrics.accuracy(
                    labels=tf.argmax(labels, 1, name='target_argmax'),
                    predictions=predictions,
                    name='accuracy')
            }

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops
        )




# Show debugging output
tf.logging.set_verbosity(tf.logging.DEBUG)

# Set default flags for the output directories
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    flag_name='model_dir', 
    default_value='./spooky_tf_models_cnnrnn',
    docstring='Output directory for model and training stats.')

tf.app.flags.DEFINE_string(
    flag_name='data_dir', 
    default_value='./spooky_data_cnnrnn',
    docstring='Directory to download the data to.')

BATCH_SIZE = 64
MAX_DOCUMENT_LENGTH = 350




VOCAB_SIZE = save_vocab(train_df1[text_col].as_matrix(), 
                       outfilename='horror_vocab.tsv', 
                       MAX_DOCUMENT_LENGTH=MAX_DOCUMENT_LENGTH)




train_input_fn, train_input_hook = setup_input_graph(train_df1[text_col].as_matrix(), 
                                                     get_one_hot_target_label(train_df1),
                                                    batch_size=BATCH_SIZE, 
                                                     scope='train-data')




eval_input_fn, eval_input_hook =  setup_input_graph(val_df[text_col].as_matrix(), 
                                                     get_one_hot_target_label(val_df),
                                                    batch_size=BATCH_SIZE, 
                                                    scope='eval-data')




model = TextCNNRNN("horror_vocab.tsv", 
                 VOCAB_SIZE, 
                 train_input_fn, 
                 train_input_hook, 
                 eval_input_fn, 
                 eval_input_hook,
                  max_doc_length=MAX_DOCUMENT_LENGTH)




get_ipython().run_line_magic('time', '')
model.run()











