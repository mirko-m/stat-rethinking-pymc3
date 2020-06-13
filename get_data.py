import urllib.request
import re
import os
import sys

def get_data_urls(url, prefix='https://github.com'):

    html = urllib.request.urlopen(url)
    html = html.read().decode('utf-8')
    lines = html.splitlines()

    data_urls = []
    for line in lines:
        if '.csv' in line:
            match = re.findall('href=".*"', line)
            if len(match) == 1:
                data_urls.append(prefix + match[0][6:-1])

    return data_urls

if __name__ == '__main__':

    if not os.path.isdir('data'):
        print('Error: You need to create a data folder first')
        sys.exit(1)

    data_urls = get_data_urls('https://github.com/rmcelreath/rethinking/tree/master/data',
            prefix='https://raw.githubusercontent.com')
    for i, url in enumerate(data_urls):
        url = url.replace('blob/','')
        fname = url.split('/')[-1]
        html = urllib.request.urlopen(url)
        with open(os.path.join('data', fname), 'w') as f:
                f.write(html.read().decode('utf-8'))
                f.close()
