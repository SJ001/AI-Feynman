import os
import re
import shutil
import urllib.request


def get_demos(dataset="example_data"):
    base = "https://space.mit.edu/home/tegmark/aifeynman/" + dataset
    with urllib.request.urlopen(base) as base_response:
        string = base_response.read().decode('utf-8')

        # the pattern actually creates duplicates in the list
        pattern = re.compile('>.*\.txt<')
        filelist = map(lambda x: x[1:-1], pattern.findall(string))

        print("downloading...")

        try:
            os.mkdir(dataset)
        except FileExistsError:
            pass

        for fname in filelist:
            print(fname)
            print(base + '/' + fname)
            print(dataset + '/' + fname)
            with urllib.request.urlopen(base + '/' + fname) as response, \
                    open(dataset + '/' + fname, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)


if __name__ == "__main__":
    get_demos()
