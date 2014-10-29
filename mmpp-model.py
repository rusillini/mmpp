#!/usr/bin/env python

#########################################################################################
#
# Program to model the behavior of time series by the Markov-Modulated Poisson Process 
# Author: Alexander Kotov (akotov2@illinois.edu), 2010
# Company: University of Illinois at Urbana-Champaign
#
#########################################################################################

import sys
import mmpp

from mmpp import create_matrix, print_matrix
from getopt import getopt
from getopt import GetoptError
from math import floor

# the default maximum number of iterations of EM algorithm
max_iter_def = 1000
# the number of stars in the progress line
star_count = 20

def usage(file_name):
   print 'Usage:\n\t%s -i file | --input=file -o file | --output=file -s num | --states=num [-m num | -max-iter=num] [-p | --progress] [-d | --debug]' % file_name
   print 'Parameters:\n\t-i file or --input=file - input file with entity citation counts'
   print '\t-o file or --output=file - output file with state labels'
   print '\t-s or --states - number of states in MMPP'
   print '\t-m or --max-iter - maximum number of iterations'
   print '\t-p or --progress - show progress'
   print '\t-d or --debug - print debug information to stdout'
   sys.exit(1)
   
def get_num_lines(fname):
   line_cnt = 0
   fd = open(fname, 'r')
   for line in fd.xreadlines():
      line_cnt = line_cnt + 1
   fd.close()
   return line_cnt

def main(argv):
   global max_iter_def
   global star_count
   
   input_fname = None
   output_fname = None
   max_iter = None
   num_states = None
   debug_mode = False
   show_prog = False
   
   total_ent_count = 0
   ent_count = 0
   one_star_ent_count = 0
   
   if len(argv) == 1:
      usage(argv[0])

   try:
      opts, args = getopt(argv[1:], "hdpi:o:m:s:", ['help', 'debug', 'progress', 'input=', 'output=', 'max-iter=', 'states='])
   except GetoptError, e:
      print >> sys.stderr, "Error: option %s: %s\n" % (e.opt, e.msg)
      usage(argv[0])

   for opt, arg in opts:
      if opt in ('-h', '--help'):
         usage(argv[0])
      elif opt in ('-d', '--debug'):
         debug_mode = True
      elif opt in ('-p', '--progress'):
         show_prog = True
      elif opt in ('-i', '--input'):
         input_fname = arg
      elif opt in ('-o', '--output'):
         output_fname = arg
      elif opt in ('-m', '--max-iter'):
         max_iter = int(arg)
      elif opt in ('-s', '--states'):
         num_states = int(arg)

   if input_fname is None:
      print >> sys.stderr, "Error: input file name is not provided"
      usage(argv[0])
   if output_fname is None:
      print >> sys.stderr, "Error: output file name is not provided"
      usage(argv[0])
   if num_states is None:
      print >> sys.stderr, "Error: number of states is not provided"
      usage(argv[0])
   
   if max_iter is None:
      max_iter = max_iter_def
   
   ent_id = None
   in_fd = open(input_fname, 'r')
   out_fd = open(output_fname, 'w')
   
   if show_prog:
      total_ent_count = get_num_lines(input_fname)
      one_star_ent_count = int(floor(float(total_ent_count) / star_count))
      sys.stdout.write("%s%s%s\n" % ('+', '-' * star_count, '+'))
      sys.stdout.write("|")
      sys.stdout.flush()
      
   for line in in_fd.xreadlines():
      line = line.strip()
      if len(line) == 0:
         continue
      vals = line.split(' ')
      O = []
      for i in range(len(vals)):
         if i == 0:
            ent_id = vals[i]
         else:
            O.append(int(vals[i]))
      
      try:
        model = mmpp.MMPP(num_states, O)
        iter_cnt = 0
  
        while not model.converged() and iter_cnt < max_iter:
           model.iterate()
           if debug_mode:
              model.print_init_probs()
              model.print_transition_probs()
              model.print_poisson_lambdas()
              print 'log(L): %.4e' % model.get_cur_likelihood()
           model.smooth_transition_probs()
           iter_cnt = iter_cnt + 1
  
        if debug_mode:
           if iter_cnt < max_iter:
              print 'PHMM converged in %d iterations' % iter_cnt
           else:
              print "PHMM didn't converge in %d iterations" % max_iter
  
        labels = model.viterbi()
    
      except Exception, e:
        print >> sys.stderr, "entity with ID %s is skipped" % ent_id
        ent_count = ent_count + 1
      else:
        ent_count = ent_count + 1
        out_fd.write('%s ' % ent_id)
        for i in range(len(labels)):
           out_fd.write('%d ' % labels[i])
        out_fd.write('\n')
        out_fd.flush()
        
      if show_prog:
         if ent_count % one_star_ent_count == 0:
            star_count = star_count - 1
            sys.stdout.write("*")
            sys.stdout.flush()
      
   if show_prog:
      while star_count > 0:
         sys.stdout.write("*")
         star_count = star_count - 1
      sys.stdout.write("|\n")
      sys.stdout.flush()   

   out_fd.close()
   in_fd.close()

if __name__ == '__main__':
   main(sys.argv)
