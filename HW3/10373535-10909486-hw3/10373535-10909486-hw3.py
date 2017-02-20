import itertools
import numpy as np
import lasagne
import theano
import theano.tensor as T
from itertools import count
import query
import pickle
import math
np.random.seed()


# epoch and fold settings:

NUM_EPOCHS = 5
FOLDS = 5
nr_epoch = 0
nr_fold = 0
ALGO = 'list'

BATCH_SIZE = 1000
NUM_HIDDEN_UNITS = 100
LEARNING_RATE = 0.00005
MOMENTUM = 0.95

# The lambda loss function
def lambda_loss(output, lambdas):
    return output * lambdas


class LambdaRankHW:

    NUM_INSTANCES = count()

    def __init__(self, feature_count):
        self.feature_count = feature_count
        self.output_layer = self.build_model(feature_count,1,BATCH_SIZE)
        self.iter_funcs = self.create_functions(self.output_layer)


    # train_queries are what load_queries returns - implemented in query.py
    def train_with_queries(self, train_queries, num_epochs):
        loss = 0
        try:
            for epoch in self.train(train_queries):
                loss += epoch['train_loss']
                if epoch['number'] >= num_epochs:
                    break

        except KeyboardInterrupt:
            pass
        return loss

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
        print "\ninput_dim",input_dim, "output_dim",output_dim
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

        # Loss function

        if ALGO == 'point':  # Point-wise loss function (squared error)
            loss_train = lasagne.objectives.squared_error(output, y_batch)

        else:  # Pairwise and listwise loss function
            loss_train = lambda_loss(output, y_batch)

        loss_train = loss_train.sum() # mean

        # Regularization
        L1_loss = lasagne.regularization.regularize_network_params(output_layer,lasagne.regularization.l1)
        L2_loss = lasagne.regularization.regularize_network_params(output_layer,lasagne.regularization.l2)
        loss_train = loss_train.mean() + L1_loss * L1_reg + L2_loss * L2_reg

        # Parameters you want to update
        all_params = lasagne.layers.get_all_params(output_layer)

        # Update parameters, adam is a particular "flavor" of Gradient Descent
        updates = lasagne.updates.adam(loss_train, all_params)


        # Create two functions:

        # (1) Scoring function, deterministic, does not update parameters, outputs scores
        score_func = theano.function(
            [X_batch],output_row_det,
        )

        # (2) Training function, updates the parameters, outpust loss
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

    # The aggregate (i.e. per document) lambda function
    def lambda_function(self, labels, scores):

        sigma = 1
        lambdas = []

        for u in range(len(scores)):  # per document u
            lambdasplus = 0
            lambdasminus = 0
            for v in range(len(scores)):  # per document v

                if labels[u] > labels[v] and scores[u] < scores[v]:
                    if ALGO == 'pair':
                        lambda_uv = sigma * - (1 / (1 + math.exp(sigma * (scores[u] - scores[v]))))

                    else:  # listwise
                        if labels[u] != labels[v]:  # if labels are different:
                            # calculate the delta of the NDCG if u and v would be flipped:
                            delta = np.abs(1 / (np.log2(1 + u + 1)) - 1 / (np.log2(1 + v + 1)))
                            # multiply lambda with this delta
                            lambda_uv = (sigma * - (1 / (1 + math.exp(sigma * (scores[u] - scores[v]))))) * delta

                    lambdasplus += lambda_uv


                elif labels[u] < labels[v] and scores[u] > scores[v]:
                    if ALGO == 'pair':
                        lambda_uv = sigma * (1 - (1 / (1 + math.exp(sigma * (scores[u] - scores[v])))))

                    else:  # listwise
                        if labels[u] != labels[v]:  # if labels are different:

                            # calculate the delta of the NDCG if u and v would be flipped:
                            delta = np.abs(1 / (np.log2(1 + u + 1)) - 1 / (np.log2(1 + v + 1)))
                            # multiply lambda with this delta
                            lambda_uv = (sigma * (1 - (1 / (1 + math.exp(sigma * (scores[v] - scores[u])))))) * delta

                    lambdasminus += lambda_uv

            # aggregate the lambdas
            lambda_u = lambdasplus - lambdasminus
            lambdas.append(lambda_u)
        lambdas = np.asarray(lambdas, dtype=np.float32)
        return lambdas

    def compute_lambdas_theano(self,query, labels):
        scores = self.score(query).flatten()
        result = self.lambda_function(labels, scores[:len(labels)])
        return result

    def train_once(self, X_train, query, labels):

        resize_value = BATCH_SIZE
        resize_value = min(resize_value, len(labels))
        X_train.reshape((resize_value, self.feature_count))

        if ALGO == 'point':
            batch_train_loss = self.iter_funcs['train'](X_train, labels)

        else:  # pair and list
            lambdas = self.compute_lambdas_theano(query, labels)
            batch_train_loss = self.iter_funcs['train'](X_train[:len(labels)], lambdas)
        return batch_train_loss


    def train(self, train_queries):
        X_trains = train_queries.get_feature_vectors()
        queries = train_queries.values()

        for epoch in itertools.count(1):
            batch_train_losses = []
            random_batch = np.arange(len(queries))
            np.random.shuffle(random_batch)
            for index in xrange(len(queries)): # per query
                random_index = random_batch[index]
                labels = queries[random_index].get_labels()

                # print the progress:
                print '\ralgo:', str(ALGO)+'wise', 'query: ' + str(index+1) + '/' + str(len(queries)), 'epoch: ' + str(nr_epoch) + '/' + str(NUM_EPOCHS), 'fold: ' + str(nr_fold) + '/' + str(FOLDS),

                batch_train_loss = self.train_once(X_trains[random_index], queries[random_index], labels)
                batch_train_losses.append(batch_train_loss)

            avg_train_loss = np.mean(batch_train_losses)

            yield {
                'number': epoch,
                'train_loss': avg_train_loss,
            }

## GET QUERIES

def get_fold_queries(nr_folds, type_set, type_algo):
    for fold in range(1,nr_folds+1):
        filename = './HP2003/Fold' + str(fold) + '/'+type_set + '.txt'
        train_queries = query.load_queries(filename, 64)
        pickle.dump(train_queries, open("models/" +type_set + '-'+ type_algo + "-fold" + str(fold) +'-'+str(64), "wb"))



## GET NDCGS

def max_ndcg(labels, k):
    nr_relev = np.count_nonzero(labels)
    ideal_dcg = 0
    for r in range(1, min(nr_relev+1, k)):
        ideal_dcg += ((2 ** 1) - 1) / (math.log(1 + r, 2))
    return ideal_dcg

def real_ndcg(labels, k):
    dcg = 0
    dcg_list = []
    for r in range(1, k+1):
        dcg_point = ((2 ** labels[r-1]) - 1) / (math.log(1 + r, 2)) # nominator could just be label
        dcg += dcg_point
        dcg_list.append(dcg_point)
    return [dcg, dcg_list]

def get_ndcg(queries, model):
    ndcgs = []
    for i in queries.keys():
        q = queries.get_query(i)
        labels = q.get_labels()
        nr_relev = np.count_nonzero(labels)
        if nr_relev != 0:
            score = model.score(q)
            ranking = np.argsort(score.flatten())[::-1]
            relevs = []
            for rank in ranking:
                relevs.append(labels[rank])
            ndcgs.append(real_ndcg(relevs, 10)[0]/max_ndcg(relevs, 10))

    return np.mean(ndcgs)

## K VOLD VALIDATION

def do_k_fold_validation(k, type_algo):
    global nr_fold, nr_epoch

    # Printing the final ndcgs
    ndcgs = []

    for fold in range(1, k + 1):
        f.write('\nFOLD' + str(fold))
        f.flush()
        nr_epoch = 0
        nr_fold += 1

        # Initialise object
        obj = LambdaRankHW(64)

        # Get the queries according to the fold
        train_queries = pickle.load(open("models/train-fold" + str(fold), "rb"))
        vali_queries = pickle.load(open("models/vali-fold" + str(fold), "rb"))
        test_queries = pickle.load(open("models/test-fold" + str(fold), "rb"))

        best_fold_ndcg = 0
        best_fold_model = obj


        # Lists for logging
        losses = []
        train_ndcgs = []
        val_ndcgs = []
        test_ndcgs = []

        # Get the pre-train NDCGS
        train_ndcgs.append(get_ndcg(train_queries, obj))
        val_ndcgs.append(get_ndcg(vali_queries, obj))

        # Start training
        for epoch in range(NUM_EPOCHS):
            nr_epoch += 1

            losses.append(obj.train_with_queries(train_queries, 1))
            train_ndcgs.append(get_ndcg(train_queries, obj))
            valndcg = get_ndcg(vali_queries, obj)
            val_ndcgs.append(valndcg)

            # Safe the best model according to the validation set
            if valndcg > best_fold_ndcg:
                best_fold_ndcg = valndcg
                best_fold_model = obj

        # after NUM_EPOCHS: evaluate the test queries
        test_eval = get_ndcg(test_queries, best_fold_model)
        test_ndcgs.append(test_eval)
        ndcgs.append(test_eval)

        f.write('\nlosses, ')
        for loss in losses:
            f.write("%s," % loss)
        f.write('\ntrain_ndcgs, ')
        for tndcg in train_ndcgs:
            f.write("%s," % tndcg)
        f.write('\nval_ndcgs, ')
        for vndcg in val_ndcgs:
            f.write("%s," % vndcg)
        f.write('\ntest_ndcgs, ')
        for tendcg in test_ndcgs:
            f.write("%s," % tendcg)
        f.flush()


    print ''
    print 'type:', type_algo + 'wise'
    print 'ndcgs of', k, 'folds:', ndcgs
    print 'mean ndcg:', np.mean(ndcgs)
    return ndcgs


# Saving the queries of every fold in a file
get_fold_queries(5, 'train')
get_fold_queries(5, 'vali')
get_fold_queries(5, 'test')


# Saving losses, train, val and test NDCGS in a file
f = open(ALGO+'final'+'results.txt', 'w')

# Execute the k Fold validation
do_k_fold_validation(FOLDS, ALGO)

f.close()