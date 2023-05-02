# ClearML - example of multiple sub-processes interacting and reporting to a single master experiment

import multiprocessing
import os
import subprocess
import sys
import time
from argparse import ArgumentParser
from random import randint

from clearml import Task


def mp_worker(arguments):
    print('sub process', os.getpid())
    data, extra_dict = arguments
    inputs, the_time = data
    Task.current_task().connect(extra_dict)
    print(" Process %s\tWaiting %s seconds" % (inputs, the_time))
    time.sleep(int(the_time))
    print(" Process %s\tDONE" % inputs)


def mp_handler(use_subprocess, data, additional_parameters):
    if use_subprocess:
        process = multiprocessing.Pool(4)
    else:
        process = multiprocessing.pool.ThreadPool(4)

    process.map(mp_worker, zip(data, additional_parameters))
    process.close()
    print('DONE main !!!')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--num-workers', help='integer value', type=int, default=3)
    parser.add_argument('--use-subprocess', help="Use sub processes", dest='subprocess', action='store_true')
    parser.add_argument('--no-subprocess', help="Use threads", dest='subprocess', action='store_false')
    parser.add_argument('--additional-parameters', help='task parameters', type=str, nargs="+")
    parser.set_defaults(subprocess=True)
    # this argument we will not be logging, see below Task.init
    parser.add_argument('--counter', help='integer value', type=int, default=-1)

    args = parser.parse_args()
    print(os.getpid(), 'ARGS:', args)

    # Fake data for us to "process"
    data = (
        ['a', '2'], ['b', '4'], ['c', '6'], ['d', '8'],
        ['e', '1'], ['f', '3'], ['g', '5'], ['h', '7'],
    )

    if not args.additional_parameters:
        # Random task parameters
        args.additional_parameters = [
            f"stuff_{str(randint(0, 100))}:some stuff {str(randint(0, 100))}" for _ in range(len(data))]
    task_parameters = (dict([p.partition(":")[::2]]) for p in args.additional_parameters)

    # We have to initialize the task in the master process,
    # it will make sure that any sub-process calling Task.init will get the master task object
    # notice that we exclude the `counter` argument, so we can launch multiple sub-processes with clearml-agent
    # otherwise, the `counter` will always be set to the original value.
    task = Task.init('examples', 'Popen example', auto_connect_arg_parser={'counter': False})

    # we can connect multiple dictionaries, each from different process, as long as the keys have different names
    param = {'args_{}'.format(args.num_workers): 'some value {}'.format(args.num_workers)}
    task.connect(param)

    # check if we need to start the process, meaning counter is negative
    counter = args.num_workers if args.counter < 0 else args.counter

    p = None
    # launch sub-process, every subprocess will launch the next in the chain, until we launch them all.
    # We could also launch all of them here, but that would have been to simple for us J
    if counter > 0:
        cmd = [sys.executable, sys.argv[0],
               '--counter', str(counter - 1),
               '--num-workers', str(args.num_workers),
               '--use-subprocess' if args.subprocess else '--no-subprocess',
               '--additional-parameters', *args.additional_parameters]

        print(cmd)
        p = subprocess.Popen(cmd, cwd=os.getcwd())

    # the actual "processing" is done here
    mp_handler(args.subprocess, data, task_parameters)
    print('Done logging')

    # wait for the process we launched
    # this means every subprocess will be waiting for the process it launched and
    # the master process will exit after all of them are completed
    if p and counter > 0:
        p.wait()
    print('Exiting')
