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
        try:
            bollinger.system_test()
        except KeyboardInterrupt:
            print("\nExiting...")

    elif args == '-g' or args == '--graph':
        print("Executing graphs...")
        focus = ['BILL', 'SAM', 'ZM', 'SGEN', 'LGND']
        bollinger.system_graph(focus)
        print("Exiting...")

    else:
        print("Executing...")
        while True:
            try:
                bollinger.system_loop()
                # Execute system graph using subprocess and popen OR use threads
            except KeyboardInterrupt:
                print("\nExiting...")
                break

if __name__ == '__main__':
    main()
