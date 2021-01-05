import os

import auth
import interfaces
import strategies

def main():
    print(auth.paper.get_account().status)

    paper = interfaces.Alpaca(auth.paper) # Temporary line to ensure API functionality

    shell = strategies.ShellSystemTest(paper)
    basic = strategies.BasicStrategy(paper)

    basic.system_loop()

if __name__ == '__main__':
    main()
