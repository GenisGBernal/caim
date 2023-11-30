"""
.. module:: MRKmeans

MRKmeans
*************

:Description: MRKmeans

    Iterates the MRKmeansStep script

:Authors: bejar
    

:Version: 

:Created on: 17/07/2017 10:16 

"""

from MRKmeansStep import MRKmeansStep
import shutil
import argparse
import os
import time
from mrjob.util import to_lines

__author__ = 'bejar'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prot', default='prototypes.txt', help='Initial prototpes file')
    parser.add_argument('--docs', default='documents.txt', help='Documents data')
    parser.add_argument('--iter', default=5, type=int, help='Number of iterations')
    parser.add_argument('--ncores', default=2, type=int, help='Number of parallel processes to use')

    args = parser.parse_args()
    assign = {}

    # Copies the initial prototypes
    cwd = os.getcwd()
    shutil.copy(cwd + '/' + args.prot, cwd + '/prototypes0.txt')

    nomove = False  # Stores if there has been changes in the current iteration
    for i in range(args.iter):
        tinit = time.time()  # For timing the iterations

        # Configures the script
        print('Iteration %d ...' % (i + 1))
        # The --file flag tells to MRjob to copy the file to HADOOP
        # The --prot flag tells to MRKmeansStep where to load the prototypes from
        mr_job1 = MRKmeansStep(args=['-r', 'local', args.docs,
                                     '--file', cwd + '/prototypes%d.txt' % i,
                                     '--prot', cwd + '/prototypes%d.txt' % i,
                                     '--num-cores', str(args.ncores)])

        # Runs the script
        with mr_job1.make_runner() as runner1:
            runner1.run()
            new_assign = {}
            new_proto = {}
            # Process the results of the script iterating the (key,value) pairs
            for key, value in mr_job1.parse_output(runner1.cat_output()):
                # You should store things here probably in a datastructure
                new_assign[key] = value[0]
                new_proto[key] = value[1]

            # If your scripts returns the new assignments you could write them in a file here
            nomove = (assign == new_assign)

            assigStr = ""
            for k,v in new_assign.items():
                assigStr += k + ":"
                for doc in v:
                    assigStr += doc + " "
                assigStr += "\n"

            protoStr = ""
            for k,v in new_proto.items():
                protoStr += k + ":"
                for term in v:
                    protoStr += term[0] + "+" + str(term[1]) + " "
                protoStr += "\n"


            assigFileName = "/assignments%d.txt"%(i+1)
            protoFileName = "/prototypes%d.txt"%(i+1)
            if (i+1 == args.iter or nomove):
                assigFileName = "/assignments-final.txt"
                protoFileName = "/prototypes-final.txt"

            outputAssign = open(cwd + assigFileName, 'w')
            outputAssign.write(assigStr)
            outputAssign.close()
            outputProtos = open(cwd + protoFileName, 'w')
            outputProtos.write(protoStr)
            outputProtos.close()



            # You should store the new prototypes here for the next iteration
            # If you have saved the assignments, you can check if they have changed from the previous iteration
            assign = new_assign


        print("Time= %f seconds" % (time.time() - tinit))

        if nomove:  # If there is no changes in two consecutive iteration we can stop
            print("Algorithm converged")
            break

    # Now the last prototype file should have the results
