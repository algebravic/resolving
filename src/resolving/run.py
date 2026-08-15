"""
Command line for the ping_pong calculation.
"""
import argparse
from .pingpong import ping_pong

def run_ping_pong():

    parser = argparse.ArgumentParser(description="CEGAR solver for Hypercube Metric Dimension")
    parser.add_argument('dim', type=int, default=10,
                        help='The dimension of the hypercube')
    parser.add_argument('mdim' type=int, default=6,
                        help='The number of rows in the detecting matrix')
    parser.add_argument('--trace', type=int, default=0,
                        help='The trace interval')

    args = parser.parse_args()
    
