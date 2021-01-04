import os

import systems

def main():
    shell_test = systems.ShellSystem(None)
    symbols = shell_test.get_data()

    for x in symbols:
        print(x)

if __name__ == '__main__':
    main()
