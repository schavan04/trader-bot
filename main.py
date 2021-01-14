import os
import sys

import auth
import interfaces
import strategies

def main():
    os.system('clear')

    print(auth.paper.get_account().status) # Temporary line to ensure API functionality

    main_inter = interfaces.MainInterface(auth.paper)

    basic = strategies.Basic(main_inter)
    movingavg = strategies.MovingAvgDay(main_inter)
    bollinger = strategies.BollingerShortTerm(main_inter)

    args = None
    if len(sys.argv) >= 2:
        args = sys.argv[1]

    if args == '-t' or args == '--test':
        print("Executing in test mode...")
        bollinger.system_test()
        print("Exiting...")

    else:
        print("Executing normally...")
        while True:
            try:
                bollinger.system_loop()
            except KeyboardInterrupt:
                break
            finally:
                print("\nExiting...")

if __name__ == '__main__':
    main()
