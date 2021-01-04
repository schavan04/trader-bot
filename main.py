import os

import auth
import interfaces
import strategies

def main():
    paper = interfaces.Alpaca(auth.paper)
    shell = strategies.ShellSystemTest(paper)
    basic = strategies.BasicStrategy(paper)

    basic.system_loop()

if __name__ == '__main__':
    main()
