import sys
import glob
import os.path
import rnn3_utils as utils

irodsdir = sys.argv[1]
outputfilename = sys.argv[2]
fext = sys.argv[3]
start_seq = sys.argv[4]
maxlen = int(sys.argv[5])

totlen = 0
nfiles = 0
with open(outputfilename, "w") as outputfile:
    for filename in glob.iglob(os.path.join(irodsdir, "**", "*."+fext), recursive=True):
        with open(filename, "r") as inputfile:
            input = start_seq + "\n" + inputfile.read()
            input += " " * ((utils.step - len(input) % utils.step) % utils.step)
            outputfile.write(input)

            totlen += len(input)
            nfiles += 1
            if totlen > maxlen:
                break

print(nfiles, "files")
