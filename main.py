import os

import auth
import interfaces
import strategies

def main():
    print(auth.paper.get_account().status) # Temporary line to ensure API functionality

    main_inter = interfaces.MainInterface(auth.paper)

    shell = strategies.ShellSystemTest(main_inter)
    basic = strategies.Basic(main_inter)
    moving = strategies.MovingAvgDay(main_inter)
    bollinger = strategies.BollingerShortTerm(main_inter)

    while True:
        bollinger.system_loop()

if __name__ == '__main__':
    main()
