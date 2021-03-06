from data_preperation import read_data, download_dependencies
from strategy_1 import main_strategy_1
from strategy_2 import main_strategy_2
from strategy_3 import main_strategy_3
from argparse import ArgumentParser


parser = ArgumentParser(description='Machine Translation Quality Estimation')

parser.add_argument("--strategy",
                    type=int,
                    choices=[1, 2, 3, 4],
                    default=4)



if __name__ == '__main__':
    """
    Starting point of the project, this runs the two experiements we have.
    """

    print('#'*10)
    print('IN PROGRESS: Downloading Dependencies')
    print('#'*10)

    download_dependencies()

    print('#'*10)
    print('IN PROGRESS: Reading data')
    print('#'*10)

    read_data()

    print('DONE: Reading data')
    print('#'*10)

    config = parser.parse_args()

    if config.strategy == 1 or config.strategy == 4:
        print('IN PROGRESS: training strategy 1')
        print('#'*10)
        main_strategy_1()
        print('#'*10)
    if config.strategy == 2 or config.strategy == 4:    
        print('IN PROGRESS: training strategy 2')
        print('#'*10)
        main_strategy_2()
        print('DONE: Strategy 2')
        print('#'*10)
    if config.strategy == 3 or config.strategy == 4:
        print('IN PROGRESS: training strategy 3')
        main_strategy_3()
        print('DONE: Strategy 3')
        print('#'*10)



