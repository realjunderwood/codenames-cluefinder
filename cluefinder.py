import numpy as np
from scipy import spatial
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE





embeddings_dict = {}

with open("glove.6B.300d.txt", 'r', encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector



def find_closest_embeddings(embedding):
    return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))


def find_closest_embeddings_cos(embedding):
    return sorted( embeddings_dict.keys(), key=lambda word: spatial.distance.cosine(embeddings_dict[word], embedding) )[:50]




globContendorRatings = {}

goodWords = {"crime","riot","don"}

from itertools import combinations

goodWordSets = []

for n in range(len(goodWords) + 1):
    for w in (list(combinations(goodWords, n))):
        goodWordSets.append (set(w))
goodWordSets.pop(0)
print(goodWordSets)



badWordSet = {"spoon","king","wrist","iron","font","montana"}
        
contendors = set()



for gWPre in goodWords:
    print(gWPre)
    for gah in find_closest_embeddings_cos(embeddings_dict[gWPre]):
        contendors.add(gah)

for gWPre in goodWords:
    contendors.discard(gWPre)



for baddie in badWordSet:
    contendors.discard(baddie)





print(contendors)

tempN = 0


for c in contendors:
    if c in embeddings_dict:
        # print(c)
        contendorBadScore = 0

        for badWord in badWordSet:
            try:
                contendorBadScore = contendorBadScore + 1 - (spatial.distance.cosine(embeddings_dict[badWord], embeddings_dict[c]))
            except:
                continue
            
        for goodWordSet in goodWordSets:
                if len(goodWordSet) == 2:
                    tempN = contendorBadScore
                    tempN = 0
                    for goodWord in goodWordSet:
                        try:
                            # if (spatial.distance.cosine(embeddings_dict[goodWord], embeddings_dict[c]) < 0.9):
                                tempN = tempN + ( 1 * (spatial.distance.cosine(embeddings_dict[goodWord], embeddings_dict[c]))  )
                        except:
                            continue
                    print("tempN of " + c + " for the specific goodwordset being evaluated rn: " + str(tempN))

                    # globContendorRatings[(c,frozenset(goodWordSet))] = tempN * np.sqrt(len(goodWordSet))
                    globContendorRatings[(c,frozenset(goodWordSet))] = tempN / (len(goodWordSet))**1

for blahh in  sorted(globContendorRatings.items(), key=lambda x:x[1] * 1)[:100]:
    print(blahh)




