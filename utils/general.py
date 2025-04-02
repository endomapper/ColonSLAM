import os

def get_root():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

def get_data_root():
    return os.path.join(get_root(), 'data')

if __name__ == '__main__':
    print(get_root())
    print(get_data_root())
