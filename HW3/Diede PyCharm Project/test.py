_author_ = 'agrotov'

import itertools
import numpy as np
import lasagne
import theano
theano.config.floatX = 'float32'
import theano.tensor as T
import time
from itertools import count
import query
import pickle
import math


NUM_EPOCHS = 5
FOLDS = 5
nr_epoch = 0
nr_fold = 0

ALGO = 'pair'
BATCH_SIZE = 1000
NUM_HIDDEN_UNITS = 100
LEARNING_RATE = 0.00005
MOMENTUM = 0.95

# TODO: Implement the lambda loss function = weights
def lambda_loss(output, lambdas):
    return T.dot(output, lambdas)

class LambdaRankHW:

    NUM_INSTANCES = count()

    def _init_(self, feature_count):
        self.feature_count = feature_count
        self.output_layer = self.build_model(feature_count,1,BATCH_SIZE)
        self.iter_funcs = self.create_functions(self.output_layer)

    # train_queries are what load_queries returns - implemented in query.py
    def train_with_queries(self, train_queries, num_epochs):
        try:
            for epoch in self.train(train_queries):
                if epoch['number'] % 10 == 0:
                    now = time.time()
                    print("Epoch {} of {} took {:.3f}s".format(
                    epoch['number'], num_epochs, time.time() - now))
                    print("training loss:\t\t{:.6f}\n".format(epoch['train_loss']))
                if epoch['number'] >= num_epochs:
                    break
        except KeyboardInterrupt:
            pass

    def score(self, query):
        feature_vectors = query.get_feature_vectors()
        scores = self.iter_funcs['out'](feature_vectors)
        return scores


    def build_model(self,input_dim, output_dim,
                    batch_size=BATCH_SIZE):
        """Create a symbolic representation of a neural network with `intput_dim`
        input nodes, `output_dim` output nodes and `num_hidden_units` per hidden
        layer.

        The training function of this model must have a mini-batch size of
        `batch_size`.

        A theano expression which represents such a network is returned.
        """
        print "input_dim",input_dim, "output_dim",output_dim
        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, input_dim),
        )

        l_hidden = lasagne.layers.DenseLayer(
            l_in,
            num_units=200,
            nonlinearity=lasagne.nonlinearities.tanh,
        )

        l_out = lasagne.layers.DenseLayer(
            l_hidden,
            num_units=output_dim,
            nonlinearity=lasagne.nonlinearities.linear,
        )

        return l_out

    # Create functions to be used by Theano for scoring and training
    def create_functions(self, output_layer,
                          X_tensor_type=T.matrix,
                          batch_size=BATCH_SIZE,
                          learning_rate=LEARNING_RATE, momentum=MOMENTUM, L1_reg=0.0000005, L2_reg=0.000003):
        """Create functions for training, validation and testing to iterate one
           epoch.
        """
        X_batch = X_tensor_type('x')
        y_batch = T.fvector('y')

        output_row = lasagne.layers.get_output(output_layer, X_batch, dtype="float32")
        output = output_row.T

        output_row_det = lasagne.layers.get_output(output_layer, X_batch,deterministic=True, dtype="float32")

        if ALGO == 'point':
            loss_train = lasagne.objectives.squared_error(output,y_batch)
        elif ALGO == 'pair':
            loss_train = lambda_loss(output, y_batch)
        loss_train = loss_train.mean()


        # TODO: (Optionally) You can add regularization if you want - for those interested
        # L1_loss = lasagne.regularization.regularize_network_params(output_layer,lasagne.regularization.l1)
        # L2_loss = lasagne.regularization.regularize_network_params(output_layer,lasagne.regularization.l2)
        # loss_train = loss_train.mean() + L1_loss * L1_reg + L2_loss * L2_reg

        # Parameters you want to update
        all_params = lasagne.layers.get_all_params(output_layer)

        # Update parameters, adam is a particular "flavor" of Gradient Descent
        updates = lasagne.updates.adam(loss_train, all_params)

        # Create two functions:

        # (1) Scoring function, deterministic, does not update parameters, outputs scores
        score_func = theano.function(
            [X_batch],output_row_det,
        )

        # (2) Training function, updates the parameters, outputs loss
        train_func = theano.function(
            [X_batch,y_batch], loss_train,
            updates=updates,
            # givens={
            #     X_batch: dataset['X_train'][batch_slice],
            #     # y_batch: dataset['y_valid'][batch_slice],
            # },
        )

        print "finished create_iter_functions"
        return dict(
            train=train_func,
            out=score_func,
        )

    def calculate_S(self, labels, u, v):
        if labels[u] > labels[v]:
            return 1
        elif labels[u] == labels[v]:
            return 0
        else:
            return -1

    # TODO: Implement the aggregate (i.e. per document) lambda function
    def lambda_function(self, labels, scores):


       # relevant = np.where(labels==1)
        #irrelevant = np.where(labels==0)


      #  diff = scores[relevant, np.newaxis] - scores[irrelevant]


        sigma = 1
        lambdasplus = 0
        lambdasminus = 0
        lambdas = []

        for u in range(len(scores)):  # per document u
            for v in range(len(scores)):  # per document v
                if u != v:
                    #S = self.calculate_S(labels, u, v)

                    if (labels[u] > labels[v]):
                        lambda_uv = sigma * - (1 / (1 + math.exp(sigma * (scores[u] - scores[v]))))
                        lambdasplus += lambda_uv

                    elif (labels[u] < labels[v]):
                        lambda_uv = sigma * (1 - (1 / (1 + math.exp(sigma * (scores[u] - scores[v])))))
                        lambdasminus +=  lambda_uv

            lambda_u = lambdasplus - lambdasminus
            lambdas.append(lambda_u)
        lambdas = np.asarray(lambdas, dtype = np.float32)
        return lambdas

    def compute_lambdas_theano(self,query, labels):
        scores = self.score(query).flatten()
        result = self.lambda_function(labels, scores[:len(labels)])
        return result

    def train_once(self, X_train, query, labels):

        resize_value = BATCH_SIZE
        resize_value=min(resize_value,len(labels))

        X_train.resize((resize_value, self.feature_count),refcheck=False)

        if ALGO == 'pair':
            lambdas = self.compute_lambdas_theano(query, labels)
            lambdas.resize((resize_value,))
            batch_train_loss = self.iter_funcs['train'](X_train, lambdas)
        elif ALGO == 'point':
            batch_train_loss = self.iter_funcs['train'](X_train, labels)

        return batch_train_loss


    def train(self, train_queries):
        X_trains = train_queries.get_feature_vectors()
        queries = train_queries.values()

        for epoch in itertools.count(1):
            batch_train_losses = []
            random_batch = np.arange(len(queries))
            np.random.shuffle(random_batch)
            for index in xrange(len(queries)):

                print '\rquery: ' + str(index) + '/' + str(len(queries)), 'epoch: ' + str(
                    nr_epoch) + '/' + str(NUM_EPOCHS), 'fold: ' + str(nr_fold) + '/' + str(FOLDS),

                random_index = random_batch[index]
                labels = queries[random_index].get_labels()

                batch_train_loss = self.train_once(X_trains[random_index],queries[random_index],labels)
                batch_train_losses.append(batch_train_loss)

            avg_train_loss = np.mean(batch_train_losses)

            yield {
                'number': epoch,
                'train_loss': avg_train_loss,
            }

def get_fold_queries(nr_folds, type_set, type_algo):
    for fold in range(1,nr_folds+1):
        filename = './HP2003/Fold' + str(fold) + '/'+type_set + '.txt'
        train_queries = query.load_queries(filename, 64)
        pickle.dump(train_queries, open("models/" +type_set + '-'+ type_algo + "-fold" + str(fold) +'-'+str(64), "wb"))

def get_query_ndcg(labels, nr_relev, k):
    dcg = 0
    for r in range(1, k+1):
        dcg += ((2 ** labels[r-1]) - 1) / (math.log(1 + r, 2))
    ideal_dcg = 0
    for r in range(1, nr_relev+1):
        ideal_dcg += ((2 ** 1) - 1) / (math.log(1 + r, 2))
    return dcg/ideal_dcg

def get_ndcg(queries, model):
    ndcgs = []
    for i in queries.keys(): # per query
        q = queries.get_query(i)
        labels = q.get_labels()
        nr_relev = np.count_nonzero(labels)
        if nr_relev != 0:
            score = model.score(q)
            ranking = np.argsort(score.flatten())[::-1]
            relevs = []
            for rank in ranking:
                relevs.append(labels[rank])
            ndcgs.append(get_query_ndcg(relevs, nr_relev, 10))
    return np.mean(ndcgs)

def do_k_fold_validation(k, num_epochs, type_algo):
    global nr_epoch, nr_fold
    obj = LambdaRankHW(64)
    ndcgs = []
    for fold in range(1, k + 1):
        nr_epoch = 0
        nr_fold += 1
        best_fold_ndcg = 0
        best_fold_model = obj
        train_queries = pickle.load(open("models/train-point-fold" + str(fold) + "-64", "rb"))
        vali_queries = pickle.load(open("models/vali-point-fold" + str(fold) + "-64", "rb"))
        test_queries = pickle.load(open("models/test-point-fold" + str(fold) + "-64", "rb"))
        for epoch in range(num_epochs):
            nr_epoch += 1
            obj.train_with_queries(train_queries, 1)
            ndcg = get_ndcg(vali_queries, obj)
            if ndcg > best_fold_ndcg:
                best_fold_ndcg = ndcg
                best_fold_model = obj
        ndcgs.append(get_ndcg(test_queries, best_fold_model))

    print ''
    print 'type:', type_algo + 'wise'
    print 'ndcgs of', k, 'folds:', ndcgs
    print 'mean ndcg:', np.mean(ndcgs)
    return ndcgs

#get_fold_queries(5, 'train', 'point')
#get_fold_queries(5, 'vali', 'point')
#get_fold_queries(5, 'test', 'point')

# Run with 1 fold
ndcgs = do_k_fold_validation(5, 5,'pair')