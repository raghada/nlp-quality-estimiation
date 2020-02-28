from data_preperation import read_data
from strategy_2 import main_strategy_2
from strategy_3 import main_strategy_3

if __name__ == '__main__':
    """
    Starting point of the project, this runs the two experiements we have.
    """
    print('#'*10)
    print('IN PROGRESS: Reading data')
    print('#'*10)
    read_data()
    print('DONE: Reading data')
    print('#'*10)
    print('IN PROGRESS: training strategy 2')
    print('#'*10)
    main_strategy_2()
    print('DONE: Strategy 2')
    print('#'*10)
    print('IN PROGRESS: training strategy 3')
    #main_strategy_3()
    print('DONE: Strategy 3')
    print('#'*10)
