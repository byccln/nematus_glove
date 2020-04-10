import os
import sys
import random

import tempfile
from subprocess import call


def main(files, temporary=False):

    fds = [open(ff) for ff in files]

    lines = []
    for l in fds[0]:
        l = l.strip()
        #将源句子中的单词转为utf8
        #ll = []
        #for word in l.split():
        #    ll.append("".join([hex(ord(char))[2:] for char in word]))
        #l = " ".join(ll)
        line = [l] + [ff.readline().strip() for ff in fds[1:]]
        lines.append(line)

    [ff.close() for ff in fds]

    random.shuffle(lines)

    if temporary: #默认
        fds = []
        for ff in files:
            path, filename = os.path.split(os.path.realpath(ff))
            fd = tempfile.TemporaryFile(prefix=filename+'.shuf',
                                        dir=path,
                                        mode='w+')
            fds.append(fd)
    else:
        fds = [open(ff+'.shuf', mode='w') for ff in files]

    for l in lines:
        for ii, fd in enumerate(fds):
            print(l[ii], file=fd)

    if temporary:
        [ff.seek(0) for ff in fds]
    else:
        [ff.close() for ff in fds]

    return fds

if __name__ == '__main__':
    main(sys.argv[1:])
