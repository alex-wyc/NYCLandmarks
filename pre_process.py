################################################################################
# does cropping, black-whiting of picture, etc.                                #
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
#  Project Created: 2017-07-01 02:29 - Yicheng W.

import PIL
from PIL import Image
import os

width = 600
height = 600

successes = 0
failures = 0

f = open('landmarks.txt', 'r').read().split('\n')
f = map(lambda s: s.strip(), f)
f = filter(lambda x: x != "", f)

try:
    os.mkdir("./dataset_processed")
except OSError:
    pass

for subdir in xrange(len(f)):
    print "Processing landmark %d: %s" % (subdir, f[subdir])
    subdir = str(subdir)
    file_location = "./dataset/" + subdir + "/"
    output_location = "./dataset_processed/" + subdir + "/"

    try:
        os.mkdir(output_location)
    except OSError:
        pass

    for file in os.listdir(file_location):
        try:
            img = Image.open(file_location + file)
            img = img.convert('1')
            img = img.resize((width, height), PIL.Image.ANTIALIAS)
            img.save(output_location + file)
            successes += 1
        except IOError:
            failures += 1

print "Processed %d files, %d success, %d failures" % ((successes + failures),
        successes, failures)
