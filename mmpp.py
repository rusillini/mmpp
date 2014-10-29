#!/usr/bin/env python

#########################################################################################
#
# Implementation of the Markov-Modulated Poisson Process (MMPP)
# Author: Alexander Kotov (akotov2@illinois.edu), 2010
# Company: University of Illinois at Urbana-Champaign
#
#########################################################################################

"""Markov-Modulated Poisson Process"""

__version__ = "1.0.2"

import sys

from math import fabs, ceil, exp, log
from random import random
from copy import copy

CONSTR_THRESHOLD=1.0e-10  # threshold for checking constraints
CONV_THRESHOLD=1.0e-10    # threshold for checking convergence
EPSILON=1.0e-100          # smoothing parameter

# functional for computing factorials
fact=lambda n: [1,0][n>0] or reduce(lambda x,y: x*y, xrange(1,n+1))

def fac(n):
  res = 1
  for i in range(2,n+1):
    res *= i
  return res

def print_matrix(A, fd=sys.stdout):
   """Prints matrix A to file descriptor fd"""
   for i in range(len(A)):
      if isinstance(A[i], list):
         for j in range(len(A[i])):
            if isinstance(A[i][j], float):
               fd.write("%.4f " % A[i][j])
            else:
               fd.write(A[i][j])
               fd.write(" ")
         fd.write("\n")
      elif isinstance(A[i], float):
         fd.write("%.4E " % A[i])
      else:
         fd.write(A[i])
         fd.write('\n')

   fd.write('\n')
   fd.flush()

def create_matrix(rows, cols=1, val=0.0):
   """Create a new rows x cols matrix, whose entries are equal to the specified value"""
   A = []
   if cols != 1 or cols != 0:
      for i in range(rows):
         A.append([])
      for i in range(rows):
         for j in range(cols):
            A[i].append(val)
   else:
      for i in range(rows):
         A.append(val)
   return A

class MMPP(object):
   """Implementation of Markov-Modulated Poisson Process.
   Methods are provided for computing the probability of a sequence of
   observations, the most probable state transitions leading to a
   sequence of observations"""

   def __init__(self, N, O):
      """Builds a new Poisson Hidden Markov Model with the given
      number of states (N) and observation sequence (O)"""
      
      self.N = N
      self.O = O
      self.T = len(O)
      self.plambda = []
      self.A = []
      self.pi = []
      self.alpha = create_matrix(N, self.T)
      self.beta = create_matrix(N, self.T)
      self.ksi = create_matrix(N, N)
      self.gamma = create_matrix(N)
      self.lambda_sum = create_matrix(N)
      self.C = create_matrix(self.T)
      self.cur_L = 0.0
      self.prev_L = 0.0
      sum_pi = 0.0
      rand = 0.0
      for i in range(N):
         rand = random()
         sum_pi = sum_pi + rand
         self.pi.append(rand)
         self.plambda.append(random()*N)
         self.A.append([])
         sum_A = 0.0
         for j in range(N):
            rand = random()
            sum_A = sum_A + rand
            self.A[i].append(rand)
         for j in range(N):
            self.A[i][j] = self.A[i][j] / sum_A

      for i in range(N):
         self.pi[i] = self.pi[i] / sum_pi

   def check_constraints(self):
      """Check to see if all constraints are satisfied""" 
      sum_pi = 0.0
      for i in range(self.N):
         sum_pi = sum_pi + self.pi[i]
         sum = 0.0
         for j in range(self.N):
            sum = sum + self.A[i][j]
         if fabs(1.0-sum) > CONSTR_THRESHOLD:
            return False
      return fabs(1.0-sum_pi) < CONSTR_THRESHOLD

   def converged(self):
      """Check for convergence"""
      if self.cur_L == 0.0:
         return False
      else:
         self.diff_L = self.cur_L - self.prev_L
         return fabs(self.diff_L) < CONV_THRESHOLD

   def smooth_transition_probs(self):
      """Smooth transition probabilities"""
      sum = 0.0
      u = create_matrix(self.N, val=0)
      for i in range(self.N):
         sum = 0.0
         cnt = 0
         for j in range(self.N):
            u[j] = 0
         for j in range(self.N):
            if self.A[i][j] < EPSILON:
               self.A[i][j] = EPSILON
               u[j] = 0
               cnt = cnt + 1
            else:
               u[j] = 1
               sum = sum + self.A[i][j]
         if cnt != 0:
            for j in range(self.N):
               if u[j] == 1:
                  self.A[i][j] = (1-cnt*EPSILON) * self.A[i][j] / sum 

   def fac(self, n):
      return fac(n)

   def poisson_ex(self, plambda, n):
      return exp(-plambda)*(plambda**n)/fac(n)
   
   def poisson(self, plambda, n):
      try:
         if n == 0 or plambda == 0.0:
            prob = exp(-plambda)*(plambda**n)/fac(n)
         else:
            prob = exp(-plambda+n*log(plambda)-log(fac(n)))
      except OverflowError, e:
         print >> sys.stderr, "Overflow when computing Poisson probability!"
         print >> sys.stderr, "lambda=", plambda
         print >> sys.stderr, "n=", n
         exit(1)
      else:
         return prob 

   def ksi_ij(self, i, j, t):
      nomin = self.alpha[i][t] * self.A[i][j] * self.poisson(self.plambda[j], self.O[t]) * self.beta[j][t+1]
      denom = 0.0

      for i in range(self.N):
         for j in range(self.N):
            denom = denom + self.alpha[i][t] * self.A[i][j] * self.poisson(self.plambda[j], self.O[t]) * self.beta[j][t+1]
      # ADDED
      if denom < 1.0e-308:
         denom = 1.0e-308
      return nomin/denom

   def gamma_i(self, i, t):
      denom = 0.0
      nomin = self.alpha[i][t] * self.beta[i][t]

      for i in range(self.N):
         denom = denom + self.alpha[i][t] * self.beta[i][t]
      # ADDED
      if denom < 1.0e-308:
         denom = 1.0e-308
      return nomin/denom

   def iterate(self):
      """Perform one iteration of an EM algorithm"""
      P = 0.0
      # Computing forward probabilities
      for t in range(self.T):
         if t == 0:
            self.C[0] = 0.0
            for i in range(self.N):
               self.alpha[i][0] = self.pi[i] * self.poisson(self.plambda[i], self.O[t])
               self.C[0] =  self.C[0] + self.alpha[i][0]
         else:
            self.C[t] = 0.0
            for j in range(self.N):
               self.alpha[j][t] = 0.0
               for i in range(self.N):
                  self.alpha[j][t] = self.alpha[j][t] + self.alpha[i][t-1] * self.A[i][j] * self.poisson(self.plambda[j], self.O[t])
               self.C[t] = self.C[t] + self.alpha[j][t]
         # ADDED
         if self.C[t] < 1.0e-308:
            self.C[t] = 1.0e-308
         
         # DEBUG
         try:
            P = P + log(1/self.C[t])
         except OverflowError, e:
            print >> sys.stderr, "Overflow when computing log-likelihood!\n"
            print >> sys.stderr, 'P=',P
            print >> sys.stderr, 'C=',self.C[t]
            exit(1)  
         
         # updating forward probabilities by the scaling factor
         for i in range(self.N):
            self.alpha[i][t] = self.alpha[i][t] / self.C[t]

      # Computing backward probabilities
      for t in range(self.T):
         if t == 0:
            for i in range(self.N):
               self.beta[i][self.T-1] = 1.0
         else:
            for j in range(self.N):
               self.beta[j][self.T-t-1] = 0.0
               for i in range(self.N):
                  self.beta[j][self.T-t-1] = self.beta[j][self.T-t-1] + self.A[j][i] * self.poisson(self.plambda[i], self.O[self.T-t])
         # updating backward probabilities by the scaling factor
         for i in range(self.N):
            self.beta[i][self.T-t-1] = self.beta[i][self.T-t-1] / self.C[self.T-t-1]

      for i in range(self.N):
         for j in range(self.N):
            self.ksi[i][j] = 0.0

      for i in range(self.N):
         self.lambda_sum[i] = 0.0

      # Computing expected number of transitions matrix
      for i in range(self.N):
         self.gamma[i] = 0.0
         self.pi[i] = self.gamma_i(i, 0)
         for t in range(self.T):
            if t != self.T-1:
               for j in range(self.N):
                  self.ksi[i][j] = self.ksi[i][j] + self.ksi_ij(i, j, t)
               self.gamma[i] = self.gamma[i] + self.gamma_i(i, t)
            self.lambda_sum[i] = self.lambda_sum[i] + self.gamma_i(i, t) * self.O[t]
         
      for i in range(self.N):
         # ADDED
         if self.gamma[i] < 1.0e-308:
            self.gamma[i] = 1.0e-308
         for j in range(self.N):
            self.A[i][j] = self.ksi[i][j] / self.gamma[i]

      for i in range(self.N):
         self.gamma[i] = self.gamma[i] + self.gamma_i(i, self.T-1)

      for i in range(self.N):
         # ADDED
         if self.gamma[i] < 1.0e-308:
            self.gamma[i] = 1.0e-308
         self.plambda[i] = self.lambda_sum[i] / self.gamma[i]
         # ADDED
         if self.plambda[i] < 1.0e-308:
            self.plambda[i] = 1.0e-308        

      self.prev_L = self.cur_L
      self.cur_L = -P

   def viterbi(self):
      """Obtain a sequence of hidden states corresponding to observations by Viterbi algorithm"""
      labels = []
      
      # DEBUG
      #print >> sys.stderr, "Lambdas:"
      #print >> sys.stderr, self.plambda
      
      # dynamic programming table:
      #   * column 1 - old probabilities
      #   * column 2 - new probabilities
      DPT = create_matrix(self.N, 2, val=0)
      # best previous states grid
      BPS = create_matrix(self.N, self.T)

      for i in range(self.N):
         if 1.0-self.pi[i] < 1.0e-10:
            self.pi[i] = 1.0-1.0e-100*(self.N-1)
         else:
            self.pi[i] = 1.0e-100

      # constructing dynamic-programming table
      for t in range(self.T):
         if t == 0:
            for i in range(self.N):
               prob = self.poisson(self.plambda[i], self.O[0])
               # ADDED
               if prob < 1.0e-308:
                  prob = 1.0e-308
               DPT[i][0] = log(self.pi[i]) + log(prob)
               BPS[i][0] = i
         else:
            for j in range(self.N):
               prob = self.poisson(self.plambda[j], self.O[t])
               # ADDED
               if prob < 1.0e-308:
                 prob = 1.0e-308
               max_L = -1.7e308
               best_state = 0
               for i in range(self.N):
                  L = DPT[i][0] + log(self.A[i][j]) + log(prob)
                  if L > max_L:
                     max_L = L
                     best_state = i
               BPS[j][t] = best_state
               DPT[j][1] = max_L
            for j in range(self.N):
               DPT[j][0] = DPT[j][1]

      # ordering states by the value of lambda
      lambdas = copy(self.plambda)
      lambdas.sort()
      # creating a mapping from some states to the others
      state_map = {}
      for i in range(self.N):
         prev_ind = 0
         while 1:
            ind = self.plambda.index(lambdas[i], prev_ind)
            if state_map.has_key(ind):
               prev_ind = ind + 1
            else:
               break 
         state_map[ind] = i+1
         
      max_L = -1.7e308
      best_state = 0
      
      for i in range(self.N):
         if DPT[i][1] > max_L:
            best_state = i

      t = self.T-1
      while t >= 0:
         #labels.append(int(ceil(self.plambda[best_state]))) #uncomment, if you want to label by expectations
         #labels.append(best_state+1) #uncomment, if you want to label by original state number
         labels.append(state_map[best_state]) #uncomment, if you want to label by frequency rank
         best_state = BPS[best_state][t]
         t = t - 1
      
      labels.reverse()
      return labels
       

   def get_cur_likelihood(self):
      return self.cur_L

   def get_prev_likelihood(self):
      return self.prev_L

   def print_transition_probs(self, fd=sys.stdout, header=True):
      """Print state transition probabilities matrix"""
      
      if header:
         fd.write('Transition probabilities:\n')
      print_matrix(self.A)

   def print_poisson_lambdas(self, fd=sys.stdout, header=True):
      print 'Poisson parameters:'
      print_matrix(self.plambda)

   def print_init_probs(self, fd=sys.stdout, header=True):
      print 'Initial state probabilities:'
      print_matrix(self.pi)

   def print_alpha(self, fd=sys.stdout, header=True):
      print 'Forward probabilities:'
      print_matrix(self.alpha)

   def print_betas(self, fd=sys.stdout, header=True):
      print 'Backward probabilities:'
      print_matrix(self.beta)

   



