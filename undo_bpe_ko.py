import os
import argparse


if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--i", type=str, default='', help="Type the input file")
    args = parser.parse_args()

    input_path = args.i

    origin = open( input_path, "r" ).readlines()

    for d in origin :
        print(' '.join(d).replace(' ', '').replace('▁▁', ' ').replace('▁', ' ')[1:].replace("\n", "")) # 마지막에서 두번째 replace에서 ' ', '' 에 따라서 tok 처리가 달라짐.
        