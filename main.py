import os
import sys

import auth
import interfaces
import loop

def main():
    os.system('clear')

    print(auth.paper.get_account().status) # Temporary line to ensure API functionality

    main_inter = interfaces.MainInterface(auth.paper)

    main = loop.Loop(main_inter)

    args = None
    if len(sys.argv) >= 2:
        args = sys.argv[1]

    if args == '-t' or args == '--test':
        print("Executing in test mode...")
        try:
            main.system_test()
        except KeyboardInterrupt:
            print("\nExiting...")

    elif args == '-g' or args == '--graph':
        print("Executing graphs...")
        focus = ['TYL', 'TWLO', 'FUTU', 'NVDA', 'MDB']
        main.system_graph(focus)
        print("Exiting...")

    else:
        print("Executing...")
        while True:
            try:
                main.system_loop()
                # Execute system graph simultaneously
            except KeyboardInterrupt:
                print("\nExiting...")
                break

if __name__ == '__main__':
    main()
