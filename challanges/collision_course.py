from challange import challange
from base64 import b64encode
from subprocess import call
import re


starts_with_number = re.compile('^\d+')

def solver(data):
    open('input.txt', 'w').write(data['include'])

    call('docker run --rm -it -v $PWD:/work -w /work -u $UID:$GID brimstone/fastcoll --prefixfile input.txt -o res1.bin res2.bin', shell=True)

    output_files = ['res1.bin', 'res2.bin']

    files = [b64encode(open(f, 'rb').read()).decode('utf-8') for f in output_files]
    return {'files': files}

def main():
    challange('collision_course', solver, post=True)


if __name__ == '__main__':
    main()
