import numpy as np
from os import listdir
from os.path import isdir, isfile, join, splitext, split


def get_kernels_pathes(path):
    files = listdir(path)
    files = filter(lambda f: f.endswith('.ocl'), files)
    files = [join(path, f) for f in files]
    files = filter(lambda f: isfile(f), files)
    return list(files)

def copy_file_content(dhfile, shfile):
    for line in shfile:
        dhfile.write(line)

def compose_header(dstpath, srcpathes):
    with open(dstpath, 'w') as dh:
        dh.write("#pragma once\n\n");

        for spath in srcpathes:
            _,name = split(spath)
            name,_ = splitext(name)
            dh.write(f'inline char const {name}[] = R"(')

            with open(spath) as sh:
                copy_file_content(dh, sh)

            dh.write(f')";\n')

def main():
    files = get_kernels_pathes('.')
    compose_header('kernels.h', files)

if __name__ == '__main__':
    main()
