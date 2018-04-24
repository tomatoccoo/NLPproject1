from practnlptools.tools import Annotator
import os
import time
import re
import sys

def rawfile_to_sentencefile_dir():

    indir = sys.argv[1]
    outdir = sys.argv[2]
    counter = 1
    try:
        os.makedirs(outdir)
    except:
        print('dir existed')
        pass
    time_start = time.time()
    annotator = Annotator()
    a = os.listdir(indir)
    part_data = os.listdir(indir)[int(sys.argv[3]):int(sys.argv[4])] # devide data into several parts,we need to set the start-end
    #part_data = os.listdir(indir)[0:8]
    for fname in part_data:
        if os.path.splitext(fname)[1] == '.summary':
           if not os.path.exists(os.path.join(outdir, fname.split('.')[0]+'.summary'+'.new')): #determine whether the file has been processed
                print(fname)
                #time_start = time.time()
                rawfile_to_sentencefile(annotator,os.path.join(indir, fname), os.path.join(outdir, fname+'.new'))
                counter = counter+1
                #time_end = time.time()
                #print('totally cost', time_end - time_start, 'Number', counter)
           else:
               print('skip', fname )

    time_end = time.time()
    print('totally cost: ', time_end - time_start, 'file number: ', counter-1)



def rawfile_to_sentencefile(annotator,filename_in, filename_out):
    fout = open(filename_out,'w')
    with open(filename_in, 'r') as f:
        g = f.read().split('\n\n')

    out = ""
    for line in g[1].split('\n'):
        sentence = re.sub(r"[\*,\@,\-,\+,\!,\&,\:,\<,\>,\{,\},\[,\],\~,\=,\_,\^,\#,\/,\?,\`,\|]",'',line[:-4]) #delete special symbols
        Phrases = sentence_to_Phrase(annotator,sentence)
        for Ph in Phrases:
            for k in ['A1', 'A2', 'V', 'TMP', 'LOC']:
                out = out+(' '.join(Ph[k])+' ')
            out = out+'\n'

    out = out + '\n'
    for line in g[2].split('\n'):
        sentence = re.sub(r"[\*,\@,\-,\+,\!,\&,\:,\<,\>,\{,\},\[,\],\~,\=,\_,\^,\#,\/,\?,\`,\|]",'',line[:-4]) #delete special symbols
        Phrases = sentence_to_Phrase(annotator,sentence)
        for Ph in Phrases:
            for k in ['A1', 'A2', 'V', 'TMP', 'LOC']:
                out = out+(' '.join(Ph[k])+' ')
            out = out+'\n'
    fout.write(out)
    fout.close()


def sentence_to_Phrase(annotator,sentence):

    try:
        r = annotator.getAnnotations(sentence)
    except:
        return []

    nn = [x[0] for x in r['pos'] if ('NN' in x[1]) or ('RB' in x[1])]
    result = []
    for phrase in r['srl']:
        A1_A2_V_TMP_LOC = {}
        first = True
        for k, v in phrase.items():
            if 'V' in k:
                A1_A2_V_TMP_LOC['V'] = ([v.split(' ')[0]])
            elif (first == True) & (len(k) == 2): # A?
                A1_A2_V_TMP_LOC['A1'] = [x for x in v.split(' ') if x in nn]
                first = False
            elif (first == False) & (len(k) == 2):
                A1_A2_V_TMP_LOC['A2'] = [x for x in v.split(' ') if x in nn]
            elif k == 'AM-TMP':
                A1_A2_V_TMP_LOC['TMP'] = [x for x in v.split(' ') if x in nn]
            elif k == 'AM-LOC':
                A1_A2_V_TMP_LOC['LOC'] = [x for x in v.split(' ') if x in nn]
            else:
                pass

        for key in ['A1', 'A2', 'TMP', 'LOC']:
            if A1_A2_V_TMP_LOC.has_key(key):
                plen =  len(A1_A2_V_TMP_LOC[key])
                if plen < 3:
                    A1_A2_V_TMP_LOC[key].extend(['__' for _ in range(3-plen)])
                else:
                    A1_A2_V_TMP_LOC[key] = A1_A2_V_TMP_LOC[key][0:3]
            else:
                A1_A2_V_TMP_LOC[key] = ['__', '__', '__']

        result.append(A1_A2_V_TMP_LOC)
    return result

# rawfile_to_sentencefile('./data/0a0a7140b649fb724b60086c3f914c16f2a9625e.summary', 'output.txt')


# dmarticle="when it comes to racing the entity4 's horses , one man is her majesty 's knight in shining armor. entity7 is seemingly the entity9 monarch 's jockey of choice and no wonder. the consensus is that the entity13 is the best in the business , a view enhanced by his win at the prestigious entity18 in november. however , he is quick to downplay his regular meetings with entity4 , treating her like he would any other owner. \" it 's not really different and it 's very easy to ride for her , \" entity7 told entity2. \" there 's no pressure.\""
# annotator = Annotator()
# time_start = time.time()
# annotator.getAnnotations(dmarticle)
# time_end = time.time()
# print('totally cost', time_end - time_start)
if __name__ == '__main__':
    rawfile_to_sentencefile_dir() # 4 parameter './training' './traning_new' datastart  dataend
