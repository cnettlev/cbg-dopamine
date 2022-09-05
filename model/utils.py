#!/usr/bin/python3
import os

def checkFolder(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)

class askFlag():
    def __init__(self, msg = '', default=''):
        self.flag = default[0].lower() if default else default
        self.msg = msg

    def askOnce(self, msg, default=True, doExit=True, always=False):
        print(msg+" [y/Y]")
        choice = input().lower()
        if always and choice != '' and (choice == 'none'):
            return 'n'
        if always and choice != '' and (choice[0] == 'a'):
            return 'a'
        if choice == '':
            return default
        elif choice[0] == 'y':
            return True 
        elif doExit:
            exit(0)
        return False
    
    def ask(self,msg = ''):
        if not msg:
            msg = self.msg
        if (self.flag != 'a' and self.flag !='n'):
            self.flag = self.askOnce(msg,doExit=False, always=True)
        # else:
        #     return self.flag

    def check(self):
        return self.flag and self.flag != 'n'

    def askAndCheck(self, msg=''):
        self.ask(msg)
        return self.check()

