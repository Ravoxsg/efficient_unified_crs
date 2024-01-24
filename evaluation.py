def distinct_metrics(outs):
    # outputs is a list which contains several sentences, each sentence contains several words
    unigram_count = 0
    bigram_count = 0
    trigram_count = 0
    quagram_count = 0
    unigram_set = set()
    bigram_set = set()
    trigram_set = set()
    quagram_set = set()

    for sen in outs:
        for word in sen:
            unigram_count += 1
            unigram_set.add(word)
        for start in range(len(sen) - 1):
            bg = str(sen[start]) + ' ' + str(sen[start + 1])
            bigram_count += 1
            bigram_set.add(bg)
        for start in range(len(sen) - 2):
            trg = str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2])
            trigram_count += 1
            trigram_set.add(trg)
        for start in range(len(sen) - 3):
            quag = str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2]) + ' ' + str(sen[start + 3])
            quagram_count += 1
            quagram_set.add(quag)

    dis1 = len(unigram_set) / len(outs) #unigram_count
    dis2 = len(bigram_set) / len(outs) #bigram_count
    dis3 = len(trigram_set) / len(outs) #trigram_count
    dis4 = len(quagram_set) / len(outs)  # quagram_count

    return dis1, dis2, dis3, dis4
