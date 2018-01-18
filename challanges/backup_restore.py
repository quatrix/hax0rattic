from challange import challange
from base64 import b64decode
from subprocess import call
import re


starts_with_number = re.compile('^\d+')

def solver(data):
    data = b64decode(data['dump'])
    open('dump.gz', 'wb').write(data)
    call(['gunzip', '-f', 'dump.gz'])
    dump = open('dump').read()


    ssns = []

    for l in dump.split('\n'):
        if starts_with_number.match(l):
            record = l.split('\t')

            if record[7] == 'alive':
                ssns.append(record[3])

    return {'alive_ssns': ssns}

def main():
    challange('backup_restore', solver, post=True)


if __name__ == '__main__':
    main()
