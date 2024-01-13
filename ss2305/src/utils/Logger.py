import sys

class OutputLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # これは flush() を持たないストリームに対してエラーを防ぐために必要です
        pass