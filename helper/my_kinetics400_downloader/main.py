import numpy as np
import os
import json
from utilities.utils import opt_mkdir, opt_makedirs


'''

TODO
make a list of succeeded downloads

AT INIT
(in case the script abruptly ends we can continue without issues)
make list of downloadeds
remove anything that isn't complete


MODES 
'only_failed': download videos on failed list
'og_list': download things that haven't downloaded yet and that aren't on failed


WHICH
'test', 'train', 'valid'


ONLY_FAILED
create failed_reason file
read failed list
read downloaded list
remove ids that intersect
for the indicated amount of videos:
    download the video
    if success:
        add successful ids on succeeded list
        remove ids from failed
    else:
        get reason of failure
        remove from failed (so that it doesn't try it twice)
        add to failed_reason file


OG_LIST
read og list
read success list
read failed list
make list that removes intersections
for the indicated amount of videos:
    download the video
    if success:
        add successful ids on succeeded list
    else:
        get reason of failure
        add to failed_reason file
    

'''