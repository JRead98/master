
class Loggers:

    def __init__(self, filename, columns):
        self.filename = filename
        with open(self.filename, 'w') as f:
            f.write(','.join(columns) + '\n')

    def log(self, arguments):
        with open(self.filename, 'a') as f:
            f.write(','.join(map(lambda k: str(k), arguments)) + '\n')
