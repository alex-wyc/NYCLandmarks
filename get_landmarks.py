################################################################################
# A script to download images for each of the landmarks specified in LANDMARKS #
# using google, change the constants and run after cloning                     #
#                                                                              #
# Authors                                                                      #
#  Authors                                                                     #
#                                                                              #
# Description                                                                  #
#  In Depth Description                                                        #
#                                                                              #
################################################################################

# TODO
#  TODO-List

# Dev Log
#  Project Created: 2017-06-29 22:55 - Yicheng W.

import requests
import time
import os

from PIL import Image
from StringIO import StringIO

GOOGLE_IMG_URL="https://www.google.com/search?tbm=isch&q={query}"
IMG_PER_LANDMARK=300 # to change

landmarks = open('landmarks.txt', 'r').read().split('\n')

landmarks = map(lambda s: s.strip(), landmarks)
landmarks = filter(lambda x: x != "", landmarks)

# given raw html of a google img search page find the 'Next Image'
def get_next_item(page):
    start = page.find('rg_di')

    if start == -1:
        end_quote = 0
        link = None
        return link, end_quote

    else:
        start = page.find('class="rg_meta"')
        start_content = page.find('"ou"', start + 1)
        end_content = page.find(',"ow"', start_content + 1)
        content = str(page[start_content+6:end_content-1])
        return content, end_content

def get_first_n_results(keyword, n):
    url = GOOGLE_IMG_URL.format(query=keyword)
    print "Downloading from " + url
    page = requests.get(url, headers={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"})
    page = page.text
    results = []

    link, index = get_next_item(page)
    print link

    while link is not None and len(results) <= n:
        results.append(link)
        page = page[index:]
        link, index = get_next_item(page)

    return results

os.system('mkdir dataset')

failures = 0
successes = 0

for i in xrange(len(landmarks)):

    print "Processing %dth Landmark: %s" % (i, landmarks[i])

    os.system('mkdir dataset/' + str(i))

    query = landmarks[i].replace(' ', '%20')
    
    links = get_first_n_results(query, IMG_PER_LANDMARK)
    print links

    for j in xrange(len(links)):
        time.sleep(0.01)
        try:
            r = requests.get(links[j])
            img = Image.open(StringIO(r.content))
            img.save('dataset/%d/%d.png' % (i, j), 'PNG')
            successes += 1
        except KeyboardInterrupt:
            exit(1)
        except:
            os.system("cp dataset/%d/%d.png dataset/%d/%d.png" % (i, j-1, i, j))
            failures += 1

print "Done! Downloaded %d images, %d failures" % (successes, failures)
