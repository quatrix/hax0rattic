import os
import subprocess

def main():
    d = './chars/'

    for f in filter(lambda x: x.endswith('.png'), os.listdir(d)):
        p = os.path.join(d, f)
        cmd = 'tesseract --psm 15 {} out'.format(p)
        subprocess.call(cmd, shell=True)
        res = open('out.txt').read().strip()


        if res == 'x' or res == 'X':
            res = 'mul'

        if res == '-' or res == '_' or res == '.':
            res = 'minus'

        if res == 'O' or res == 'o':
            res = '0'

        if res == 'q':
            res = '9'

        if res == '‘7':
            res = '7'

        if res == '—:':
            res = 'div'

        if res == 'z.' or res == '1.' or res == 'l.' or res == 'A':
            res = '4'

        print(p)
        print('got {}'.format(res))

        chars = [
            '0', '1', '2', '3',
            '4', '5', '6', '7',
            '8', '9', 'minus', 'mul', 'div'
        ]


        if res in chars:
            dest = './chars/{}/{}'.format(res, f)
            print('mv {} -> {}'.format(p, dest))
            os.rename(p, dest)




if __name__ == '__main__':
    main()
