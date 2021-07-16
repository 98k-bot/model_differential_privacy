from __future__ import division

import abc
import collections
import math
import sys

import numpy as np
import tensorflow as tf

EpsDelta = collections.namedtuple("EpsDelta", ["spent_eps", "spent_delta"])


class MomentsAccountant(object):
  """Privacy accountant which keeps track of moments of privacy loss.
  Note: The constructor of this class creates tf.Variables that must
  be initialized with tf.global_variables_initializer() or similar calls.
  MomentsAccountant accumulates the high moments of the privacy loss. It
  requires a method for computing differenital moments of the noise (See
  below for the definition). So every specific accountant should subclass
  this class by implementing _differential_moments method.
  Denote by X_i the random variable of privacy loss at the i-th step.
  Consider two databases D, D' which differ by one item. X_i takes value
  log Pr[M(D')==x]/Pr[M(D)==x] with probability Pr[M(D)==x].
  In MomentsAccountant, we keep track of y_i(L) = log E[exp(L X_i)] for some
  large enough L. To compute the final privacy spending,  we apply Chernoff
  bound (assuming the random noise added at each step is independent) to
  bound the total privacy loss Z = sum X_i as follows:
    Pr[Z > e] = Pr[exp(L Z) > exp(L e)]
              < E[exp(L Z)] / exp(L e)
              = Prod_i E[exp(L X_i)] / exp(L e)
              = exp(sum_i log E[exp(L X_i)]) / exp(L e)
              = exp(sum_i y_i(L) - L e)
  Hence the mechanism is (e, d)-differentially private for
    d =  exp(sum_i y_i(L) - L e).
  We require d < 1, i.e. e > sum_i y_i(L) / L. We maintain y_i(L) for several
  L to compute the best d for any give e (normally should be the lowest L
  such that 2 * sum_i y_i(L) / L < e.
  We further assume that at each step, the mechanism operates on a random
  sample with sampling probability q = batch_size / total_examples. Then
    E[exp(L X)] = E[(Pr[M(D)==x / Pr[M(D')==x])^L]
  By distinguishign two cases of wether D < D' or D' < D, we have
  that
    E[exp(L X)] <= max (I1, I2)
  where
    I1 = (1-q) E ((1-q) + q P(X+1) / P(X))^L + q E ((1-q) + q P(X) / P(X-1))^L
    I2 = E (P(X) / ((1-q) + q P(X+1)))^L
  In order to compute I1 and I2, one can consider to
    1. use an asymptotic bound, which recovers the advance composition theorem;
    2. use the closed formula (like GaussianMomentsAccountant);
    3. use numerical integration or random sample estimation.
  Dependent on the distribution, we can often obtain a tigher estimation on
  the moments and hence a more accurate estimation of the privacy loss than
  obtained using generic composition theorems.
  """

  #__metaclass__ = abc.ABCMeta

  def __init__(self, total_examples=5040, moment_orders=1):
    """Initialize a MomentsAccountant.
    Args:
      total_examples: total number of examples.
      moment_orders: the order of moments to keep.
    """

    assert total_examples > 0
    self._total_examples = total_examples
    self._moment_orders = (moment_orders
                           if isinstance(moment_orders, (list, tuple))
                           else range(1, moment_orders + 1))
    self._max_moment_order = max(self._moment_orders)
    assert self._max_moment_order < 100, "The moment order is too large."
    self._log_moments = [tf.Variable(np.float64(0.0),
                                     trainable=False,
                                     name=("log_moments-%d" % moment_order))
                         for moment_order in self._moment_orders]

  def _compute_log_moment(self, num_examples=5040):
    """Compute high moment of privacy loss.
    Args:
      sigma: the noise sigma, in the multiples of the sensitivity.
      q: the sampling ratio.
      moment_order: the order of moment.
    Returns:
      log E[exp(moment_order * X)]
    """
    q = tf.cast(num_examples, tf.float64) * 1.0 / self._total_examples
    mu_1, sigma_1 = 0, 4  # mean and standard deviation
    s_1 = np.random.normal(mu_1, sigma_1, 1000)
    mu_2, sigma_2 = 1, 4  # mean and standard deviation
    s_2 = np.random.normal(mu_2, sigma_2, 1000)
    s = (1-q)*s_1 + q*s_2
    moment_1 =[0]*len(self._log_moments)
    moment_2 = [0]*len(self._log_moments)
    log_moment = [0] * len(self._log_moments)
    for i in range(len(self._log_moments)):
        for j in range(len(s_1)):
            moment_1[i] += ((s_1[j]/s[j])**self._moment_orders[i])/len(s_1)
            moment_2[i] += ((s[j] / s_1[j]) ** self._moment_orders[i]) / len(s_1)
    for i in range(len(self._log_moments)):
        log_moment[i] = math.log(abs(max(moment_1[i],moment_2[i])))
    return log_moment

  def accumulate_privacy_spending(self, sigma=1, num_examples=5040):
        """Accumulate privacy spending.
        In particular, accounts for privacy spending when we assume there
        are num_examples, and we are releasing the vector
        (sum_{i=1}^{num_examples} x_i) + Normal(0, stddev=l2norm_bound*sigma)
        where l2norm_bound is the maximum l2_norm of each example x_i, and
        the num_examples have been randomly selected out of a pool of
        self.total_examples.
        Args:
          unused_eps_delta: EpsDelta pair which can be tensors. Unused
            in this accountant.
          sigma: the noise sigma, in the multiples of the sensitivity (that is,
            if the l2norm sensitivity is k, then the caller must have added
            Gaussian noise with stddev=k*sigma to the result of the query).
          num_examples: the number of examples involved.
        Returns:
          a TensorFlow operation for updating the privacy spending.
        """
        q = tf.cast(num_examples, tf.float64) * 1.0 / self._total_examples

        moments_accum_ops = []
        for i in range(len(self._log_moments)):
            moment = self._compute_log_moment()
            moments_accum_ops.append(tf.compat.v1.assign_add(self._log_moments[i], moment[i]))
        #print(moments_accum_ops)
        return tf.group(*moments_accum_ops)

  def _compute_delta(self, log_moments, eps):
        """Compute delta for given log_moments and eps.
        Args:
          log_moments: the log moments of privacy loss, in the form of pairs
            of (moment_order, log_moment)
          eps: the target epsilon.
        Returns:
          delta
        """
        min_delta = 1.0
        for moment_order, log_moment in log_moments:
            if math.isinf(log_moment) or math.isnan(log_moment):
                sys.stderr.write("The %d-th order is inf or Nan\n" % moment_order)
                continue
            if log_moment < moment_order * eps:
                min_delta = min(min_delta,
                                math.exp(log_moment - moment_order * eps))
        return min_delta

  def _compute_eps(self, log_moments, delta=1e-5):
        min_eps = float("inf")
        eps = []
        for i in range(len(log_moments)):
            if math.isinf(log_moments[i]) or math.isnan(log_moments[i]):
                sys.stderr.write("The %d-th order is inf or Nan\n" % self._moment_orders[i])
                continue
            min_eps = min(min_eps, (log_moments[i] - math.log(delta)) / self._moment_orders[i])
            eps.append((log_moments[i] - math.log(delta)) / self._moment_orders[i])
            order = eps.index(min(eps))
        return min_eps,order

  def get_privacy_spent(self, target_eps=None, target_deltas=None):
        """Compute privacy spending in (e, d)-DP form for a single or list of eps.
        Args:
          target_eps: a list of target epsilon's for which we would like to
            compute corresponding delta value.
          target_deltas: a list of target deltas for which we would like to
            compute the corresponding eps value. Caller must specify
            either target_eps or target_delta.
        Returns:
          A list of EpsDelta pairs.
        """
        assert (target_eps is None) ^ (target_deltas is None)
        eps_deltas = []
        log_moments = self._log_moments
        log_moments_with_order = zip(self._moment_orders, log_moments)
        if target_eps is not None:
            for eps in target_eps:
                eps_deltas.append(
                    EpsDelta(eps, self._compute_delta(log_moments_with_order, eps)))
        else:
            assert target_deltas
            for delta in target_deltas:
                eps_deltas.append(
                    EpsDelta(self._compute_eps(log_moments_with_order, delta), delta))
        return eps_deltas


if __name__ == '__main__':
    accountant = MomentsAccountant()
    accountant.accumulate_privacy_spending()
    log_mo = accountant._compute_log_moment()
    eps = accountant._compute_eps(log_mo)
    print(eps)