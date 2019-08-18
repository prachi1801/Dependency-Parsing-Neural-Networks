import collections
import tensorflow as tf
import numpy as np
import pickle
import math
from progressbar import ProgressBar


from DependencyTree import DependencyTree
from ParsingSystem import ParsingSystem
from Configuration import Configuration
import Config
import Util

"""
This script defines a transition-based dependency parser which makes
use of a classifier powered by a neural network. The neural network
accepts distributed representation inputs: dense, continuous
representations of words, their part of speech tags, and the labels
which connect words in a partial dependency parse.

This is an implementation of the method described in

Danqi Chen and Christopher Manning. A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

Author: Danqi Chen, Jon Gauthier
Modified by: Heeyoung Kwon (2017)
Modified by: Jun S. Kang (2018 Mar)
"""
'''train_weights_input2 = tf.Variable(tf.truncated_normal(shape=[Config.hidden_size, Config.hidden_size], stddev=0.1))
            train_bias_input2 = tf.Variable(tf.zeros([Config.hidden_size, 1]))

            # #First hidden layer propogation
            self.train_pred = self.forward_pass1(train_embed, weights_input, biases_input, weights_output)

            # #Second hidden layer propogation
            self.train_pred = self.forward_pass2(self.train_pred, train_weights_input2, train_bias_input2, weights_output)
            
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.train_pred, labels=train_labels)
            thetas2 = tf.nn.l2_loss(train_embed) + tf.nn.l2_loss(weights_input) + tf.nn.l2_loss(biases_input) + tf.nn.l2_loss(train_weights_input2) + tf.nn.l2_loss(train_bias_input2) + tf.nn.l2_loss(weights_output)
            self.loss = tf.reduce_mean(self.loss + Config.lam * thetas2)
            '''


class DependencyParserModel(object):

    def __init__(self, graph, embedding_array, Config):

        self.build_graph(graph, embedding_array, Config)

    def build_graph(self, graph, embedding_array, Config):
        """

        :param graph:
        :param embedding_array:
        :param Config:
        :return:
        """

        with graph.as_default():
            self.embeddings = tf.Variable(embedding_array, dtype=tf.float32)

            """
            ===================================================================

            Define the computational graph with necessary variables.
            
            1) You may need placeholders of:
                - Many parameters are defined at Config: batch_size, n_Tokens, etc
                - # of transitions can be get by calling parsing_system.numTransitions()
                
            self.train_inputs = 
            self.train_labels = 
            self.test_inputs =
            ...
            
            """


            """
                
            2) Call forward_pass and get predictions
            
            ...
            self.prediction = self.forward_pass(embed, weights_input, biases_input, weights_output)


            3) Implement the loss function described in the paper
             - lambda is defined at Config.lam
            
            ...
            self.loss = 
            
            ===================================================================
            """

            self.train_inputs = tf.placeholder(tf.int32, shape=[Config.batch_size, Config.n_Tokens])
            self.train_labels = tf.placeholder(tf.int32, shape=[Config.batch_size, parsing_system.numTransitions()])


            train_embedding_lookup = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
            train_embed = tf.reshape(train_embedding_lookup, [Config.batch_size, -1])


            # Masking out invalid -1 transitions in train_labels
            #train_labels = tf.nn.relu(self.train_labels)

            #weights_input = tf.Variable(tf.truncated_normal(shape=[Config.hidden_size, Config.embedding_size * Config.n_Tokens], stddev=1/math.sqrt(Config.embedding_size * Config.n_Tokens))

            '''biases_input = tf.Variable(tf.zeros([Config.hidden_size,1]))

            weights_output = tf.Variable(tf.truncated_normal(shape=[parsing_system.numTransitions(), Config.hidden_size], stddev= 1/ math.sqrt((Config.hidden_size))))




            #self.predictions = self.forward_pass_parallel(train_embed, weights_words, weights_tags, weights_labels, biases_words, biases_tags, biases_labels, weights_output)'''



            '''train_labels = tf.nn.relu(self.train_labels)

            


            self.predictions = self.forward_pass(train_embed, weights_input, biases_input, weights_output)


            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.predictions, labels=train_labels)
            thetas = tf.nn.l2_loss(train_embed) + tf.nn.l2_loss(weights_input) + tf.nn.l2_loss(biases_input) + tf.nn.l2_loss(weights_output)
            self.loss = tf.reduce_mean(self.loss + Config.lam * thetas)'''




            ######################################## Remove these comment for 2 hidden layer implementation
            '''weights_input = tf.Variable(tf.truncated_normal(shape=[Config.hidden_size, Config.embedding_size * Config.n_Tokens], stddev=1/math.sqrt(Config.embedding_size * Config.n_Tokens)))
            biases_input = tf.Variable(tf.zeros(shape = [Config.hidden_size, 1]))
            weights2 = tf.Variable(tf.truncated_normal(shape=[Config.hidden2_size, Config.hidden_size], stddev= 1/ math.sqrt((Config.hidden_size))))
            biases2 = tf.Variable(tf.zeros(shape = [Config.hidden2_size, 1]))
            weights_output = tf.Variable(tf.truncated_normal(shape=[parsing_system.numTransitions(), Config.hidden2_size], stddev=1 / math.sqrt(Config.hidden2_size)))

            self.predictions = self.forward_pass_2_hidden(train_embed, weights_input, biases_input,weights2, biases2, weights_output)

            train_labels = tf.nn.relu(self.train_labels)
            
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.predictions, labels=train_labels)
            thetas2 = tf.nn.l2_loss(train_embed) + tf.nn.l2_loss(weights_input) + tf.nn.l2_loss(biases_input) + tf.nn.l2_loss(weights2) + tf.nn.l2_loss(biases2) + tf.nn.l2_loss(weights_output)
            self.loss = tf.reduce_mean(self.loss + Config.lam * thetas2)'''


            ################### Alternate2 hidden layer

            '''weights_input = tf.Variable(tf.truncated_normal(shape=[Config.hidden_size, Config.embedding_size * Config.n_Tokens], stddev=0.1))
            biases_input = tf.Variable(tf.random_normal(stddev= 0.1, shape = [Config.hidden_size]))
            weights2 = tf.Variable(tf.truncated_normal(shape=[Config.hidden2_size, Config.hidden_size], stddev=0.1))
            biases2 = tf.Variable(tf.random_normal(stddev= 0.1, shape = [Config.hidden2_size]))
            weights_output = tf.Variable(tf.truncated_normal(shape=[parsing_system.numTransitions(), Config.hidden2_size], stddev=0.1))

            self.predictions = self.forward_pass_2_hidden_alt(train_embed, weights_input, biases_input,weights2, biases2, weights_output)

            print self.predictions

            #self.predictions = tf.Print(self.predictions, [self.predictions])

            train_labels = tf.nn.relu(self.train_labels)
            
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.predictions, labels=train_labels)
            thetas2 = tf.nn.l2_loss(train_embed) + tf.nn.l2_loss(weights_input) + tf.nn.l2_loss(biases_input) + tf.nn.l2_loss(weights2) + tf.nn.l2_loss(biases2) + tf.nn.l2_loss(weights_output)
            self.loss = tf.reduce_mean(self.loss + Config.lam * thetas2)'''


            #############################   Remove the comments for 3 hidden layer implementation   #################################

            '''weights_input = tf.Variable(tf.truncated_normal(shape=[Config.hidden_size, Config.embedding_size * Config.n_Tokens], stddev=1/math.sqrt(Config.embedding_size * Config.n_Tokens)))
            biases_input = tf.Variable(tf.zeros(shape = [Config.hidden_size, 1]))
            weights2 = tf.Variable(tf.truncated_normal(shape=[Config.hidden2_size, Config.hidden_size], stddev= 1/ math.sqrt((Config.hidden_size))))
            biases2 = tf.Variable(tf.zeros(shape = [Config.hidden2_size, 1]))
            weights3 = tf.Variable(tf.truncated_normal(shape=[Config.hidden3_size, Config.hidden2_size], stddev= 1/ math.sqrt((Config.hidden2_size))))
            biases3 = tf.Variable(tf.zeros([Config.hidden3_size, 1]))
            weights_output = tf.Variable(tf.truncated_normal(shape=[parsing_system.numTransitions(), Config.hidden3_size], stddev= 1/ math.sqrt((Config.hidden3_size))))
            
            self.predictions = self.forward_pass_3_hidden(train_embed, weights_input, biases_input,weights2, biases2, weights3, biases3, weights_output)

            train_labels = tf.nn.relu(self.train_labels)
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.predictions, labels=train_labels)
            thetas3 = tf.nn.l2_loss(train_embed) + tf.nn.l2_loss(weights_input) + tf.nn.l2_loss(biases_input) + tf.nn.l2_loss(weights2) + tf.nn.l2_loss(biases2) + tf.nn.l2_loss(weights3) + tf.nn.l2_loss(biases3) + tf.nn.l2_loss(weights_output)    
            self.loss = tf.reduce_mean(self.loss + Config.lam * thetas3)'''


            ##################### Use below commented code for 3 parralel layers for words, tags and labels ###############

            
            train_embed_words = tf.slice(train_embedding_lookup, [0, 0, 0], [Config.batch_size, Config.n_Tokens_word, Config.embedding_size])
            train_embed_words = tf.reshape(train_embed_words, [Config.batch_size, -1])

            train_embed_pos = tf.slice(train_embedding_lookup, [0, 18, 0], [Config.batch_size, Config.n_Tokens_pos, Config.embedding_size])
            train_embed_pos = tf.reshape(train_embed_pos, [Config.batch_size, -1])
            
            train_embed_labels = tf.slice(train_embedding_lookup, [0, 36, 0], [Config.batch_size, Config.n_Tokens_labels, Config.embedding_size])
            train_embed_labels = tf.reshape(train_embed_labels, [Config.batch_size, -1])


            weights_output_words = tf.Variable(tf.random_normal(shape=[parsing_system.numTransitions(), Config.hidden_size], stddev=1.0/math.sqrt(Config.hidden_size)))
            weights_output_pos = tf.Variable(tf.random_normal(shape=[parsing_system.numTransitions(), Config.hidden_size], stddev=1.0/math.sqrt(Config.hidden_size)))
            weights_output_labels = tf.Variable(tf.random_normal(shape=[parsing_system.numTransitions(), Config.hidden_size], stddev=1.0/math.sqrt(Config.hidden_size)))



            weights_input_words = tf.Variable(tf.truncated_normal(shape=[Config.hidden_size, Config.n_Tokens_word * Config.embedding_size], stddev=0.1))
            biases_input_words = tf.Variable(tf.zeros([Config.hidden_size, 1]))

            weights_input_pos = tf.Variable(tf.truncated_normal(shape=[Config.hidden_size, Config.n_Tokens_pos * Config.embedding_size], stddev=0.1))
            biases_input_pos = tf.Variable(tf.zeros([Config.hidden_size, 1]))

            weights_input_labels = tf.Variable(tf.truncated_normal(shape=[Config.hidden_size, Config.n_Tokens_labels * Config.embedding_size], stddev=0.1))
            biases_input_labels = tf.Variable(tf.zeros([Config.hidden_size, 1]))



            self.prediction_words = self.forward_pass(train_embed_words, weights_input_words, biases_input_words, weights_output_words)
            self.prediction_pos = self.forward_pass(train_embed_pos, weights_input_pos, biases_input_pos, weights_output_pos)
            self.prediction_labels = self.forward_pass(train_embed_labels, weights_input_labels, biases_input_labels, weights_output_labels)


            train_labels = tf.nn.relu(self.train_labels)

            loss_words = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.prediction_words, labels=train_labels)
            loss_pos = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.prediction_pos, labels=train_labels)
            loss_labels = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.prediction_labels, labels=train_labels)


            l2_input_words = Config.lam * tf.nn.l2_loss(weights_input_words)
            l2_biases_words = Config.lam * tf.nn.l2_loss(biases_input_words)

            l2_input_pos = Config.lam * tf.nn.l2_loss(weights_input_pos)
            l2_biases_pos = Config.lam * tf.nn.l2_loss(biases_input_pos)

            l2_input_labels = Config.lam * tf.nn.l2_loss(weights_input_labels)
            l2_biases_labels = Config.lam * tf.nn.l2_loss(biases_input_labels)

            l2_output_words = Config.lam * tf.nn.l2_loss(weights_output_words)
            l2_output_pos = Config.lam * tf.nn.l2_loss(weights_output_pos)
            l2_output_labels = Config.lam * tf.nn.l2_loss(weights_output_labels)

            l2_embed_words = Config.lam * tf.nn.l2_loss(train_embed_words)
            l2_embed_pos = Config.lam * tf.nn.l2_loss(train_embed_words)
            l2_embed_labels = Config.lam * tf.nn.l2_loss(train_embed_words)


            l2_loss = (loss_words + l2_input_words + l2_biases_words + l2_output_words + l2_embed_words) + \
                        (loss_pos + l2_input_pos + l2_biases_pos + l2_output_pos + l2_embed_pos) + \
                        (loss_labels + l2_input_labels + l2_biases_labels + l2_output_labels + l2_embed_labels)

            #------------------------------------------------------------------------------------------------------------------#
  
            # Take average loss over the entire batch
            self.loss = tf.reduce_mean(l2_loss)





            #################====================================================##########################

            ##############  gradient descent computation with gradient clipping ##############
            optimizer = tf.train.GradientDescentOptimizer(Config.learning_rate)
            grads = optimizer.compute_gradients(self.loss)
            clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
            self.app = optimizer.apply_gradients(clipped_grads)


            ################### Test Predictions #######################################

            self.test_inputs = tf.placeholder(tf.int32, shape=[Config.n_Tokens])

            test_embed = tf.nn.embedding_lookup(self.embeddings, self.test_inputs)
            test_embed = tf.reshape(test_embed, [1, -1])




            #self.test_pred = self.forward_pass(test_embed, weights_input, biases_input, weights_output)


            ############ Use below commented code to run for 2 hidden layers ##########

            #self.test_pred = self.forward_pass_2_hidden(test_embed, weights_input, biases_input, weights2, biases2, weights_output)


            ############ Use below commented code for 2 hidden alternate ################

            #self.test_pred = self.forward_pass_2_hidden_alt(test_embed, weights_input, biases_input, weights2, biases2, weights_output)


            ########### Use below commented code for 3 hidden layer implementation

            #self.test_pred = self.forward_pass_3_hidden(test_embed, weights_input, biases_input, weights2, biases2, weights3, biases3, weights_output)







            # Prediction for the test data

            test_embed_words = tf.slice(test_embed, [0, 0], [Config.n_Tokens_words, test_embed.get_shape()[1]])
            test_embed_words = tf.reshape(test_embed_words, [1, -1])

            test_embed_pos = tf.slice(test_embed, [18, 0], [Config.n_Tokens_pos, test_embed.get_shape()[1]])
            test_embed_pos = tf.reshape(test_embed_pos, [1, -1])

            test_embed_labels = tf.slice(test_embed, [36, 0], [Config.n_Tokens_labels, test_embed.get_shape()[1]])
            test_embed_labels = tf.reshape(test_embed_labels, [1, -1])


            test_pred_words = self.forward_pass(test_embed_words, weights_input_words, biases_input_words, weights_output_words)
            test_pred_pos = self.forward_pass(test_embed_pos, weights_input_pos, biases_input_pos, weights_output_pos)
            test_pred_labels = self.forward_pass(test_embed_labels, weights_input_labels, biases_input_labels, weights_output_labels)

            self.test_pred = (test_pred_words + test_pred_pos + test_pred_labels) / 3





            # intializer
            self.init = tf.global_variables_initializer()

    def train(self, sess, num_steps):
        """

        :param sess:
        :param num_steps:
        :return:
        """
        self.init.run()
        print "Initailized"

        average_loss = 0
        for step in range(num_steps):
            start = (step * Config.batch_size) % len(trainFeats)
            end = ((step + 1) * Config.batch_size) % len(trainFeats)
            if end < start:
                start -= end
                end = len(trainFeats)
            batch_inputs, batch_labels = trainFeats[start:end], trainLabels[start:end]

            feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels}
            _, loss_val = sess.run([self.app, self.loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % Config.display_step == 0:
                if step > 0:
                    average_loss /= Config.display_step
                print "Average loss at step ", step, ": ", average_loss
                average_loss = 0
            if step % Config.validation_step == 0 and step != 0:
                print "\nTesting on dev set at step ", step
                predTrees = []
                for sent in devSents:
                    numTrans = parsing_system.numTransitions()

                    c = parsing_system.initialConfiguration(sent)
                    while not parsing_system.isTerminal(c):
                        feat = getFeatures(c)
                        pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                        optScore = -float('inf')
                        optTrans = ""

                        for j in range(numTrans):
                            if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                                optScore = pred[0, j]
                                optTrans = parsing_system.transitions[j]

                        c = parsing_system.apply(c, optTrans)

                    predTrees.append(c.tree)
                result = parsing_system.evaluate(devSents, predTrees, devTrees)
                print result

        print "Train Finished."

    def evaluate(self, sess, testSents):
        """

        :param sess:
        :return:
        """

        print "Starting to predict on test set"
        predTrees = []
        for sent in testSents:
            numTrans = parsing_system.numTransitions()

            c = parsing_system.initialConfiguration(sent)
            while not parsing_system.isTerminal(c):
                # feat = getFeatureArray(c)
                feat = getFeatures(c)
                pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                optScore = -float('inf')
                optTrans = ""

                for j in range(numTrans):
                    if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                        optScore = pred[0, j]
                        optTrans = parsing_system.transitions[j]

                c = parsing_system.apply(c, optTrans)

            predTrees.append(c.tree)
        print "Saved the test results."
        Util.writeConll('result_test.conll', testSents, predTrees)


    def forward_pass(self, embed, weights_input, biases_input, weights_output):
        """
        :param embed:
        :param weights:
        :param biases:
        :return:
        """
        """
        =======================================================

        Implement the forwrad pass described in
        "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

        =======================================================
        """

        out = (tf.matmul(embed, weights_input, transpose_b=True) + biases_input)

        h = tf.pow(out, 3)

        #h = tf.nn.sigmoid(out)

        #h = tf.tanh(out)

        #h = tf.nn.relu(out)
        p = tf.matmul(h, weights_output, transpose_b= True)



        return p

    def forward_pass_2_hidden(self, embed, weights_input, biases_input, weights2, biases2, weights_output):


        h = tf.pow((tf.matmul(weights_input, embed, transpose_b=True) + biases_input) , 3)

        h2 = tf.pow(tf.transpose((tf.matmul(weights2, h) + biases2)) , 3)

        p = tf.transpose(tf.matmul(weights_output, h2, transpose_b = True))


        return p



    '''def forward_pass_2_hidden_alt(self, embed, weights_input, biases_input, weights2, biases2, weights_output):
        """
        :param embed:
        :param weights:
        :param biases:
        :params only1: This is to indicate that there are more hidden layers in network
        :return:
        """

        out = tf.matmul(embed, weights_input, transpose_b=True) + biases_input
        h = tf.tanh(out)

        out = tf.matmul(h, weights2, transpose_b=True) + biases2
        h2 = tf.tanh(out)

        p = tf.matmul(h2, weights_output, transpose_b = True)


        return p'''

    def forward_pass_3_hidden(self, embed, weights_input, biases_input, weights2, biases2, weights3, biases3, weights_output):



        h = tf.pow((tf.matmul(weights_input, embed, transpose_b=True) + biases_input) , 3)

        h2 = tf.pow(tf.matmul(weights2, h) + biases2, 3)

        h3 = tf.pow(tf.matmul(weights3, h2) + biases3 , 3)

        p = tf.transpose(tf.matmul(weights_output, h3))


        return p

def genDictionaries(sents, trees):
    word = []
    pos = []
    label = []
    for s in sents:
        for token in s:
            word.append(token['word'])
            pos.append(token['POS'])

    rootLabel = None
    for tree in trees:
        for k in range(1, tree.n + 1):
            if tree.getHead(k) == 0:
                rootLabel = tree.getLabel(k)
            else:
                label.append(tree.getLabel(k))

    if rootLabel in label:
        label.remove(rootLabel)

    index = 0
    wordCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    wordCount.extend(collections.Counter(word))
    for word in wordCount:
        wordDict[word] = index
        index += 1

    posCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    posCount.extend(collections.Counter(pos))
    for pos in posCount:
        posDict[pos] = index
        index += 1

    labelCount = [Config.NULL, rootLabel]
    labelCount.extend(collections.Counter(label))
    for label in labelCount:
        labelDict[label] = index
        index += 1

    return wordDict, posDict, labelDict


def getWordID(s):
    if s in wordDict:
        return wordDict[s]
    else:
        return wordDict[Config.UNKNOWN]


def getPosID(s):
    if s in posDict:
        return posDict[s]
    else:
        return posDict[Config.UNKNOWN]


def getLabelID(s):
    if s in labelDict:
        return labelDict[s]
    else:
        return labelDict[Config.UNKNOWN]


def getFeatures(c):

    """
    =================================================================

    Implement feature extraction described in
    "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

    =================================================================
    """

    features = []

    ########### top 3 stack words #####
    s1 = c.getStack(0)
    s2 = c.getStack(1)
    s3 = c.getStack(2)

    ######### top 3 buffer words ######
    b1 = c.getBuffer(0)
    b2 = c.getBuffer(1)
    b3 = c.getBuffer(2)


    ############ word ids for top 3 stack and buffer words #########
    features.append(getWordID(c.getWord(s1)))
    features.append(getWordID(c.getWord(s2)))
    features.append(getWordID(c.getWord(s3)))
    features.append(getWordID(c.getWord(b1)))
    features.append(getWordID(c.getWord(b2)))
    features.append(getWordID(c.getWord(b3)))

    ########### word ids for 1st and 2nd left child  of top 2 stack words ######
    features.append(getWordID(c.getWord(c.getLeftChild(s1,1))))
    features.append(getWordID(c.getWord(c.getLeftChild(s1,2))))
    features.append(getWordID(c.getWord(c.getLeftChild(s2,1))))
    features.append(getWordID(c.getWord(c.getLeftChild(s2,2))))

    ########### word ids for 1st and 2nd right child  of top 2 stack words ######
    features.append(getWordID(c.getWord(c.getRightChild(s1,1))))
    features.append(getWordID(c.getWord(c.getRightChild(s1,2))))
    features.append(getWordID(c.getWord(c.getRightChild(s2,1))))
    features.append(getWordID(c.getWord(c.getRightChild(s2,2))))

    ####### word ids for left of left child of top two stack words #######
    features.append(getWordID(c.getWord(c.getLeftChild(c.getLeftChild(s1, 1), 1))))
    features.append(getWordID(c.getWord(c.getLeftChild(c.getLeftChild(s2, 1), 1))))

    ####### word ids for right of right child of top two stack words #######
    features.append(getWordID(c.getWord(c.getRightChild(c.getRightChild(s1, 1), 1))))
    features.append(getWordID(c.getWord(c.getRightChild(c.getRightChild(s2, 1), 1))))

    ############ pos ids for top 3 stack and buffer words #########
    features.append(getPosID(c.getPOS(s1)))
    features.append(getPosID(c.getPOS(s2)))
    features.append(getPosID(c.getPOS(s3)))
    features.append(getPosID(c.getPOS(b1)))
    features.append(getPosID(c.getPOS(b2)))
    features.append(getPosID(c.getPOS(b3)))


    ########### pos ids for 1st and 2nd left child  of top 2 stack words ######
    features.append(getPosID(c.getPOS(c.getLeftChild(s1,1))))
    features.append(getPosID(c.getPOS(c.getLeftChild(s1,2))))
    features.append(getPosID(c.getPOS(c.getLeftChild(s2,1))))
    features.append(getPosID(c.getPOS(c.getLeftChild(s2,2))))

    ########### pos ids for 1st and 2nd right child  of top 2 stack words ######
    features.append(getPosID(c.getPOS(c.getRightChild(s1,1))))
    features.append(getPosID(c.getPOS(c.getRightChild(s1,2))))
    features.append(getPosID(c.getPOS(c.getRightChild(s2,1))))
    features.append(getPosID(c.getPOS(c.getRightChild(s2,2))))

    ####### pos ids for left of left child of top two stack words #######
    features.append(getPosID(c.getPOS(c.getLeftChild(c.getLeftChild(s1, 1), 1))))
    features.append(getPosID(c.getPOS(c.getLeftChild(c.getLeftChild(s2, 1), 1))))

    ####### pos ids for right of right child of top two stack words #######
    features.append(getPosID(c.getPOS(c.getRightChild(c.getRightChild(s1, 1), 1))))
    features.append(getPosID(c.getPOS(c.getRightChild(c.getRightChild(s2, 1), 1))))



    ########### label ids for 1st and 2nd left child  of top 2 stack words ######
    features.append(getLabelID(c.getLabel(c.getLeftChild(s1,1))))
    features.append(getLabelID(c.getLabel(c.getLeftChild(s1,2))))
    features.append(getLabelID(c.getLabel(c.getLeftChild(s2,1))))
    features.append(getLabelID(c.getLabel(c.getLeftChild(s2,2))))


    ########### label ids for 1st and 2nd right child  of top 2 stack words ######
    features.append(getLabelID(c.getLabel(c.getRightChild(s1,1))))
    features.append(getLabelID(c.getLabel(c.getRightChild(s1,2))))
    features.append(getLabelID(c.getLabel(c.getRightChild(s2,1))))
    features.append(getLabelID(c.getLabel(c.getRightChild(s2,2))))

    ####### label ids for left of left child of top two stack words #######
    features.append(getLabelID(c.getLabel(c.getLeftChild(c.getLeftChild(s1, 1), 1))))
    features.append(getLabelID(c.getLabel(c.getLeftChild(c.getLeftChild(s2, 1), 1))))

    ####### label ids for right of right child of top two stack words #######
    features.append(getLabelID(c.getLabel(c.getRightChild(c.getRightChild(s1, 1), 1))))
    features.append(getLabelID(c.getLabel(c.getRightChild(c.getRightChild(s2, 1), 1))))

    return features


def genTrainExamples(sents, trees):
    numTrans = parsing_system.numTransitions()

    features = []
    labels = []
    pbar = ProgressBar()
    for i in pbar(range(len(sents))):
        if trees[i].isProjective():
            c = parsing_system.initialConfiguration(sents[i])

            while not parsing_system.isTerminal(c):
                oracle = parsing_system.getOracle(c, trees[i])
                feat = getFeatures(c)
                label = []
                for j in range(numTrans):
                    t = parsing_system.transitions[j]
                    if t == oracle:
                        label.append(1.)
                    elif parsing_system.canApply(c, t):
                        label.append(0.)
                    else:
                        label.append(-1.)

                if 1.0 not in label:
                    print i, label
                features.append(feat)
                labels.append(label)
                c = parsing_system.apply(c, oracle)
    return features, labels


def load_embeddings(filename, wordDict, posDict, labelDict):
    dictionary, word_embeds = pickle.load(open(filename, 'rb'))

    embedding_array = np.zeros((len(wordDict) + len(posDict) + len(labelDict), Config.embedding_size))
    knownWords = wordDict.keys()
    foundEmbed = 0
    for i in range(len(embedding_array)):
        index = -1
        if i < len(knownWords):
            w = knownWords[i]
            if w in dictionary:
                index = dictionary[w]
            elif w.lower() in dictionary:
                index = dictionary[w.lower()]
        if index >= 0:
            foundEmbed += 1
            embedding_array[i] = word_embeds[index]
        else:
            embedding_array[i] = np.random.rand(Config.embedding_size) * 0.02 - 0.01
    print "Found embeddings: ", foundEmbed, "/", len(knownWords)
    return embedding_array


if __name__ == '__main__':

    wordDict = {}
    posDict = {}
    labelDict = {}
    parsing_system = None

    trainSents, trainTrees = Util.loadConll('train.conll')
    devSents, devTrees = Util.loadConll('dev.conll')
    testSents, _ = Util.loadConll('test.conll')
    genDictionaries(trainSents, trainTrees)

    embedding_filename = 'word2vec.model'

    embedding_array = load_embeddings(embedding_filename, wordDict, posDict, labelDict)

    labelInfo = []
    for idx in np.argsort(labelDict.values()):
        labelInfo.append(labelDict.keys()[idx])
    parsing_system = ParsingSystem(labelInfo[1:])
    print parsing_system.rootLabel

    print "Generating Traning Examples"
    trainFeats, trainLabels = genTrainExamples(trainSents, trainTrees)

    print "Done."

    # Build the graph model
    graph = tf.Graph()
    model = DependencyParserModel(graph, embedding_array, Config)

    num_steps = Config.max_iter
    with tf.Session(graph=graph) as sess:

        model.train(sess, num_steps)

        model.evaluate(sess, testSents)


